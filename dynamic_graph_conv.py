import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dense_to_sparse


class DynamicGraphSpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=22, K=3):
        super().__init__()
        self.K = K
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adj_param = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.cheb_conv = ChebConv(in_channels, out_channels, K)

        # 添加空间聚合层，将22维降到1维
        self.spatial_aggregation = nn.Conv2d(out_channels, out_channels, (num_nodes, 1), (1, 1))

    def forward(self, x):
        # x: (batch, in_channels, num_nodes, time)
        batch, in_ch, num_nodes, time = x.shape
        device = x.device
        adj = torch.sigmoid(self.adj_param)  # 限制到[0,1]
        adj = (adj + adj.t()) / 2  # 对称化
        adj = adj * (1 - torch.eye(num_nodes, device=device))
        edge_index, edge_weight = dense_to_sparse(adj)
        x = x.permute(0, 3, 2, 1).contiguous()  # (batch, time, num_nodes, in_ch)
        x_reshaped = x.view(batch * time, num_nodes, in_ch)
        x_flat = x_reshaped.view(-1, in_ch)  # (batch*time*num_nodes, in_ch)
        batch_vector = torch.arange(batch * time, device=device).repeat_interleave(num_nodes)
        out = self.cheb_conv(x_flat, edge_index, edge_weight, batch=batch_vector)

        # 重塑回原始格式
        out = out.view(batch * time, num_nodes, self.out_channels)
        out = out.view(batch, time, num_nodes, self.out_channels)
        out = out.permute(0, 3, 2, 1)  # (batch, out_channels, num_nodes, time)

        # 空间聚合：将22维降到1维，模拟原来的空间卷积效果
        out = self.spatial_aggregation(out)  # (batch, out_channels, 1, time)

        return out
