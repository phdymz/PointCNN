from typing import Optional

from math import ceil

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN
from torch.nn import ELU, Conv1d, Dropout
from torch_geometric.nn import Reshape
from torch_geometric.nn import fps, global_mean_pool, XConv


try:
    from torch_cluster import knn_graph, knn
except ImportError:
    knn_graph = None


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class XConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim: int,
                 kernel_size: int, hidden_channels: Optional[int] = None,
                 dilation: int = 1, bias: bool = True, num_workers: int = 1,
                 with_global: bool = False, bn_momentum: float = 0.5):
        super(XConv, self).__init__()

        if knn_graph is None:
            raise ImportError('`XConv` requires `torch-cluster`.')

        # print(with_global)
        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_workers = num_workers
        self.with_global = with_global

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        self.mlp1 = Seq(
            Lin(dim, C_delta),
            ELU(),
            BN(C_delta, momentum=bn_momentum),
            Lin(C_delta, C_delta),
            ELU(),
            BN(C_delta, momentum=bn_momentum),
            Reshape(-1, K, C_delta),
        )

        self.mlp2 = Seq(
            Lin(D * K, K ** 2),
            ELU(),
            BN(K ** 2, momentum=bn_momentum),
            Reshape(-1, K, K),
            Conv1d(K, K ** 2, K, groups=K),
            ELU(),
            BN(K ** 2, momentum=bn_momentum),
            Reshape(-1, K, K),
            Conv1d(K, K ** 2, K, groups=K),
            BN(K ** 2, momentum=bn_momentum),
            Reshape(-1, K, K),
        )

        if with_global:
            self.global_mlp = Seq(Lin(3, out_channels // 4, bias=True),
                                  ELU(),
                                  BN(out_channels // 4, momentum=bn_momentum),
                                  Lin(out_channels // 4, out_channels // 4, bias=True),
                                  ELU(),
                                  BN(out_channels // 4, momentum=bn_momentum),
                                  )

        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = Seq(
            Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier),
            Lin(C_in * depth_multiplier, C_out, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        torch.nn.init.xavier_normal(self.mlp1[0].weight)
        torch.nn.init.xavier_normal(self.mlp1[3].weight)
        reset(self.mlp2)
        torch.nn.init.xavier_normal(self.mlp2[0].weight)
        torch.nn.init.xavier_normal(self.mlp2[4].weight)
        torch.nn.init.xavier_normal(self.mlp2[8].weight)
        reset(self.conv)

    def forward(self, x: Tensor, pos: Tensor, batch: Optional[Tensor] = None,
                pos_query: Optional[Tensor] = None, batch_query: Optional[Tensor] = None):
        """"""
        pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
        pos_backup = pos.clone()
        (N, D), K = pos.size(), self.kernel_size

        if pos_query is None:
            edge_index = knn_graph(pos, K * self.dilation, batch, loop=True,
                                   flow='target_to_source',
                                   num_workers=self.num_workers)
            row, col = edge_index[0], edge_index[1]
        else:
            # print('deconv')
            edge_index = knn(pos_query, pos, K * self.dilation, batch_x=batch_query, batch_y=batch)
            row, col = edge_index[0], edge_index[1]
            # print(row, col)
            # print(row.max(), col.max(), row.shape, col.shape)

        if self.dilation > 1:
            dil = self.dilation
            index = torch.randint(K * dil, (N, K), dtype=torch.long,
                                  device=row.device)
            arange = torch.arange(N, dtype=torch.long, device=row.device)
            arange = arange * (K * dil)
            index = (index + arange.view(-1, 1)).view(-1)

            # print(row, index)
            row = row[index]
            col = col[index]

        pos = pos[col] - pos[row]

        x_star = self.mlp1(pos.view(N * K, D))
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[col].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()
        x_star = x_star.view(N, self.in_channels + self.hidden_channels, K, 1)

        transform_matrix = self.mlp2(pos.view(N, K * D))
        transform_matrix = transform_matrix.view(N, 1, K, K)

        x_transformed = torch.matmul(transform_matrix, x_star)
        x_transformed = x_transformed.view(N, -1, K)

        out = self.conv(x_transformed)
        # print(out.shape)

        if self.with_global:
            # print(pos.shape)
            fts_global = self.global_mlp(pos_backup)
            # print(fts_global.shape)
            return torch.cat([fts_global, out], axis=-1)
        else:
            return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class PointCNN(torch.nn.Module):
    def __init__(self, num_classes=2, phase = 'train'):
        super(PointCNN, self).__init__()

        self.conv1 = XConv(
            0, 64, dim=3, kernel_size=8, hidden_channels=32, dilation=1)
        self.conv2 = XConv(
            64, 64, dim=3, kernel_size=12, hidden_channels=64, dilation=2)
        self.conv3 = XConv(
            64, 128, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        self.conv4 = XConv(
            128, 256, dim=3, kernel_size=16, hidden_channels=256, dilation=2)

        self.deconv1 = XConv(
            256, 128, dim=3, kernel_size=16, hidden_channels=256, dilation=4)
        self.deconv2 = XConv(
            128 + 128, 64, dim=3, kernel_size=12, hidden_channels=256, dilation=2)
        self.deconv3 = XConv(
            64 + 64, 64, dim=3, kernel_size=8, hidden_channels=256, dilation=2)
        self.deconv4 = XConv(
            128, 64, dim=3, kernel_size=8, hidden_channels=256, dilation=2)


        self.lin1 = Lin(128, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, num_classes)

        self.training = (phase == 'train')

    def forward(self, pos, batch):
        x1 = F.relu(self.conv1(None, pos, batch))
        x1_sub, pos1_sub, batch1_sub = x1, pos, batch
        #2048->2048

        x2 = F.relu(self.conv2(x1_sub, pos1_sub, batch1_sub))
        idx2 = fps(pos1_sub, batch1_sub, ratio=0.375)
        x2_sub, pos2_sub, batch2_sub = x2[idx2], pos1_sub[idx2], batch1_sub[idx2]
        #2048 -> 764

        x3 = F.relu(self.conv3(x2_sub, pos2_sub, batch2_sub))
        idx3 = fps(pos2_sub, batch2_sub, ratio=0.5)
        x3_sub, pos3_sub, batch3_sub = x3[idx3], pos2_sub[idx3], batch2_sub[idx3]
        #764 -> 384

        x4 = F.relu(self.conv4(x3_sub, pos3_sub, batch3_sub))
        idx4 = fps(pos3_sub, batch3_sub, ratio=1/3)
        x4_sub, pos4_sub, batch4_sub = x4[idx4], pos3_sub[idx4], batch3_sub[idx4]
        # 384 -> 128

        x = F.relu(self.deconv1(x4_sub, pos3_sub, batch3_sub, pos_query=pos4_sub, batch_query=batch4_sub))
        #128 -> 384

        x = torch.cat([x, x3_sub], dim = -1)
        x = F.relu(self.deconv2(x, pos2_sub, batch2_sub, pos_query=pos3_sub, batch_query=batch3_sub))
        #384 -> 764

        x = torch.cat([x, x2_sub], dim=-1)
        x = F.relu(self.deconv3(x, pos1_sub, batch1_sub, pos_query=pos2_sub, batch_query=batch2_sub))
        #764 -> 2048

        x = torch.cat([x, x1_sub], dim=-1)
        x = F.relu(self.deconv4(x, pos1_sub, batch1_sub))
        #2048 -> 2048

        #mlp
        x = torch.cat([x, x1_sub], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)




# class PointCNN(torch.nn.Module):
#     def __init__(self, num_classes, bn_momentum=0.01):
#         super(PointCNN, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.conv1 = XConv(0, 256, dim=3, kernel_size=8, hidden_channels=256 // 2, dilation=1)
#         self.conv2 = XConv(256, 256, dim=3, kernel_size=12, hidden_channels=256 // 4, dilation=2)
#         self.conv3 = XConv(256, 512, dim=3, kernel_size=16, hidden_channels=512 // 4, dilation=2)
#         self.conv4 = XConv(512, 1024, dim=3, kernel_size=16, hidden_channels=1024 // 4, dilation=6, with_global=True)
#
#         self.deconv1 = XConv(1024 + 1024 // 4, 1024, dim=3, kernel_size=16, hidden_channels=512 // 4, dilation=6)
#         self.deconv2 = XConv(1024, 512, dim=3, kernel_size=16, hidden_channels=256 // 4, dilation=6)
#         self.deconv3 = XConv(512, 256, dim=3, kernel_size=12, hidden_channels=256 // 4, dilation=6)
#         self.deconv4 = XConv(256, 256, dim=3, kernel_size=8, hidden_channels=256 // 4, dilation=6)
#         self.deconv5 = XConv(256, 256, dim=3, kernel_size=8, hidden_channels=256 // 4, dilation=4)
#
#         self.fuse1 = Seq(Lin(2048 + 1024 // 4, 1024, bias=True), ELU(), BN(1024, momentum=bn_momentum))
#         self.fuse2 = Seq(Lin(1024, 512, bias=True), ELU(), BN(512, momentum=bn_momentum))
#         self.fuse3 = Seq(Lin(512, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
#         self.fuse4 = Seq(Lin(512, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
#         self.fuse5 = Seq(Lin(512, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
#
#         self.lin1 = Seq(Lin(256, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
#         torch.nn.init.xavier_uniform(self.lin1[0].weight)
#         self.lin2 = Seq(Lin(256, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
#         torch.nn.init.xavier_uniform(self.lin1[0].weight)
#         self.lin3 = Lin(256, num_classes)
#         torch.nn.init.xavier_uniform(self.lin1[0].weight)
#
#     def forward(self, x, pos, batch, edge_index=None):
#         x1 = F.relu(self.conv1(None, pos, batch))
#
#         idx1 = fps(pos, batch, ratio=0.375)
#         x1_sub, pos1_sub, batch1_sub = x1[idx1], pos[idx1], batch[idx1]
#
#         x2 = F.relu(self.conv2(x1_sub, pos1_sub, batch1_sub))
#
#         idx2 = fps(pos1_sub, batch1_sub, ratio=0.5)
#         x2_sub, pos2_sub, batch2_sub = x2[idx2], pos1_sub[idx2], batch1_sub[idx2]
#
#         x3 = F.relu(self.conv3(x2_sub, pos2_sub, batch2_sub))
#
#         idx3 = fps(pos2_sub, batch2_sub, ratio=1 / 3)
#         x3_sub, pos3_sub, batch3_sub = x3[idx3], pos2_sub[idx3], batch2_sub[idx3]
#         x4 = F.relu(self.conv4(x3_sub, pos3_sub, batch3_sub))
#
#
#         x = F.relu(self.deconv1(x4, pos3_sub, batch3_sub, pos_query=pos3_sub, batch_query=batch3_sub))
#         x = torch.cat([x, x4], axis=-1)
#
#         x = self.fuse1(x)
#         x = F.relu(self.deconv2(x, pos2_sub, batch2_sub, pos_query=pos3_sub, batch_query=batch3_sub))
#
#         x = torch.cat([x, x3], axis=-1)
#         x = self.fuse2(x)
#
#         x = F.relu(self.deconv3(x, pos1_sub, batch1_sub, pos_query=pos2_sub, batch_query=batch2_sub))
#
#         x = torch.cat([x, x2], axis=-1)
#         x = self.fuse3(x)
#
#         x = F.relu(self.deconv4(x, pos, batch, pos_query=pos1_sub, batch_query=batch1_sub))
#
#         x = torch.cat([x, x1], axis=-1)
#         x = self.fuse4(x)
#
#         x = F.relu(self.deconv5(x, pos, batch))
#
#         x = torch.cat([x, x1], axis=-1)
#         x = self.fuse5(x)
#
#         x = F.relu(self.lin1(x))
#         x = F.relu(self.lin2(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin3(x)
#         return F.log_softmax(x, dim=-1)

