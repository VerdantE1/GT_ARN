import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange
from dynamic_graph_conv import DynamicGraphSpatialConv

# ========= Utils =========
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # eq: rearrange(x, '... d j -> ... (d j)')

def theta_shift(x, sin, cos):
    assert x.size(-1) % 2 == 0, "Last dimension must be even for rotary embedding"
    return (x * cos) + (rotate_every_two(x) * sin)

# ========= RMSNorm (custom,避免冲突) =========
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


# ====== Graph-based Token Extractor (GTE) ======
class GraphTokenExtractor(nn.Module):
    """
    Graph-based Token Extractor (GTE).
    Converts raw EEG sequences into informative tokens
    by combining temporal convolution, graph-based spatial modeling,
    and patch-based tokenization.
    """

    def __init__(self, embed_dim=40, num_channels=22, dropout=0.5):
        super().__init__()

        # Shallow feature extractor
        self.shallow_net = nn.Sequential(
            # Local temporal convolution
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            # Graph-based spatial convolution across EEG channels
            DynamicGraphSpatialConv(40, 40, num_channels, K=3),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # Pooling acts as slicing along the time dimension -> tokens
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            nn.Dropout(dropout),
        )

        # Projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Conv2d(40, embed_dim, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e")  # (B, N_tokens, D_embed)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input EEG tensor of shape (B, 1, C, T),
               where B = batch size,
                     C = number of EEG channels,
                     T = sequence length.

        Returns:
            tokens: Tensor of shape (B, N, D),
                    where N = number of tokens,
                          D = embedding dimension.
        """
        features = self.shallow_net(x)   # (B, 40, C', T')
        tokens = self.projection(features)  # (B, N, D)
        return tokens


import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= FFN =========
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, activation_fn=F.gelu, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ========= Relative Positional Encoding =========
class RetentionRelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for Retention mechanism.
    """
    def __init__(self, embed_dim, num_heads, gamma_min=0.5, gamma_max=0.99):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()

        self.gamma_param = nn.Parameter(torch.zeros(num_heads))
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.register_buffer("angle", angle)

    def get_gamma(self):
        return torch.sigmoid(self.gamma_param) * (self.gamma_max - self.gamma_min) + self.gamma_min

    def forward(self, slen):
        device = self.angle.device
        index = torch.arange(slen, device=device)

        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])

        dist = (index[:, None] - index[None, :]).abs()
        gamma = self.get_gamma()
        mask = gamma[:, None, None] ** dist[None, :, :]
        mask = torch.nan_to_num(mask)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()

        return (sin, cos), mask


# ========= Retention Unit =========
class RetentionUnit(nn.Module):
    """
    Core Retention mechanism = gated attention-like interaction
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.key_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.group_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.scaling = self.key_dim ** -0.5

    def forward(self, x, rel_pos, gate=None):
        bsz, tgt_len, _ = x.size()
        (sin, cos), mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Rotary positional shift
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat * mask

        if gate is not None:
            qk_mat = qk_mat * gate

        attn = torch.softmax(qk_mat, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = self.group_norm(out).reshape(bsz, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        return out


# ========== Retention Block ==========
class RetentionBlock(nn.Module):
    """
    One block of Adaptive Retention Module = RetentionUnit + FFN + gating.
    """
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.retention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.retention = RetentionUnit(embed_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_heads),
            nn.Sigmoid()
        )

    def forward(self, x, retention_rel_pos=None):
        gate_token = self.gate_mlp(self.retention_norm(x))     # (B, L, num_heads)
        gate_token = gate_token.permute(0, 2, 1)               # (B, num_heads, L)
        gate = gate_token.unsqueeze(-1) * gate_token.unsqueeze(-2)  # (B, num_heads, L, L)

        x = x + self.retention(self.retention_norm(x), retention_rel_pos, gate)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ========== ARM 主体 ==========
class AdaptiveRetentionModule(nn.Module):
    """
    Adaptive Retention Module (ARM).
    Stacks multiple RetentionBlocks with relative position encodings.
    """
    def __init__(self, embed_dim, depth, num_heads, ffn_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.rel_pos_enc = RetentionRelativePositionalEncoding(embed_dim, num_heads)

        self.blocks = nn.ModuleList([
            RetentionBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(depth)
        ])

    def forward(self, x):
        _, seq_len, _ = x.size()
        rel_pos = self.rel_pos_enc(seq_len)

        for blk in self.blocks:
            x = blk(x, retention_rel_pos=rel_pos)
        return x


# ========== GT-ARN ==========
class GT_ARN(nn.Module):
    """
    Graph-Tokenized Adaptive Retentive Network (GT-ARN).

    Structure:
        1. GraphTokenExtractor (GTE)
        2. AdaptiveRetentionModule (ARM)
        3. Classification Head
    """

    def __init__(self,
                 in_chans=22,
                 num_classes=4,
                 embed_dim=40,
                 depth=5,
                 num_heads=4,
                 mlp_ratio=3):
        super().__init__()

        ffn_dim = int(embed_dim * mlp_ratio)

        # ---- GTE ----
        self.gte = GraphTokenExtractor(embed_dim=embed_dim, num_channels=in_chans)

        # ---- ARM ----
        self.arm = AdaptiveRetentionModule(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim
        )

        # ---- Normalization & Pooling ----
        self.norm = nn.BatchNorm1d(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ---- Classifier ----
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        Args:
            x: EEG input tensor (B, 1, C, T)

        Returns:
            features: (B, D)
        """
        tokens = self.gte(x)  # (B, N, D) Step 1: GTE
        reps = self.arm(tokens)  # (B, N, D) Step 2: ARM

        reps = reps.permute(0, 2, 1)  # (B, D, N)
        reps = self.norm(reps)
        reps = self.avgpool(reps)  # (B, D, 1)
        reps = torch.flatten(reps, 1)  # (B, D)
        return reps

    def forward(self, x):
        """
        Returns:
            features: (B, D)
            logits: (B, num_classes)
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return features, logits