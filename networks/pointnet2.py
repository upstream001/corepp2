import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnext import farthest_point_sample, index_points, knn_point


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def group_norm(channels):
    return nn.GroupNorm(_group_count(channels), channels)


class SharedMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SharedMLP1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample

        layers = []
        last_channels = in_channels + 3
        for out_channels in mlp_channels:
            layers.append(SharedMLP2d(last_channels, out_channels))
            last_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        group_idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)

        if points is None:
            grouped_points = grouped_xyz
        else:
            grouped_points = index_points(points, group_idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(grouped_points).max(dim=-1)[0]
        return new_xyz, new_points.transpose(1, 2).contiguous()


class PointNetGlobalAbstraction(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        layers = []
        last_channels = in_channels + 3
        for out_channels in mlp_channels:
            layers.append(SharedMLP1d(last_channels, out_channels))
            last_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        if points is None:
            features = xyz.transpose(1, 2).contiguous()
        else:
            features = torch.cat([xyz.transpose(1, 2).contiguous(), points.transpose(1, 2).contiguous()], dim=1)
        return self.mlp(features).max(dim=-1)[0]


class PointNet2Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        sa1_npoint=512,
        sa1_nsample=32,
        sa2_npoint=128,
        sa2_nsample=32,
        dropout=0.2,
    ):
        super().__init__()
        feature_dims = max(0, in_channels - 3)
        self.sa1 = PointNetSetAbstraction(sa1_npoint, sa1_nsample, feature_dims, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(sa2_npoint, sa2_nsample, 128, [128, 128, 256])
        self.sa3 = PointNetGlobalAbstraction(256, [256, 512, 1024])
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels),
        )

    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        points = None
        if x.shape[1] > 3:
            points = x[:, 3:, :].transpose(1, 2).contiguous()

        xyz, points = self.sa1(xyz, points)
        xyz, points = self.sa2(xyz, points)
        global_feat = self.sa3(xyz, points)
        return self.head(global_feat)


def build_pointnet2_encoder(out_channels, cfg=None, in_channels=3):
    cfg = cfg or {}
    return PointNet2Encoder(
        in_channels=in_channels,
        out_channels=out_channels,
        sa1_npoint=cfg.get("pointnet2_sa1_npoint", 512),
        sa1_nsample=cfg.get("pointnet2_sa1_nsample", 32),
        sa2_npoint=cfg.get("pointnet2_sa2_npoint", 128),
        sa2_nsample=cfg.get("pointnet2_sa2_nsample", 32),
        dropout=cfg.get("pointnet2_dropout", 0.2),
    )
