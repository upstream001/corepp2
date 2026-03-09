import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def group_norm(channels):
    return nn.GroupNorm(_group_count(channels), channels)


def square_distance(src, dst):
    src_norm = torch.sum(src ** 2, dim=-1, keepdim=True)
    dst_norm = torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    dist = src_norm + dst_norm - 2 * torch.matmul(src, dst.transpose(1, 2))
    return torch.clamp(dist, min=0.0)


def index_points(points, idx):
    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (idx.dim() - 1)
    batch_indices = torch.arange(batch_size, device=points.device).view(*view_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    npoint = min(npoint, num_points)

    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(dist, k=min(nsample, xyz.shape[1]), dim=-1, largest=False, sorted=False)
    return group_idx


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


class InvResMLP(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            group_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            group_norm(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, npoint, nsample):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = nn.Sequential(
            SharedMLP2d(in_channels + 3, out_channels),
            SharedMLP2d(out_channels, out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            group_norm(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, features):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        center_features = index_points(features.transpose(1, 2), fps_idx).transpose(1, 2)

        group_idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        grouped_features = index_points(features.transpose(1, 2), group_idx).permute(0, 3, 1, 2)
        grouped_features = torch.cat((grouped_xyz.permute(0, 3, 1, 2), grouped_features), dim=1)

        aggregated = self.mlp(grouped_features).max(dim=-1)[0]
        shortcut = self.skip(center_features)
        new_features = self.act(aggregated + shortcut)
        return new_xyz, new_features


class PointNeXtEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, width=48, nsample=24, dropout=0.05):
        super().__init__()
        self.stem = nn.Sequential(
            SharedMLP1d(in_channels, width),
            SharedMLP1d(width, width),
        )

        self.sa1 = SetAbstraction(width, width * 2, npoint=512, nsample=nsample)
        self.stage1 = nn.Sequential(InvResMLP(width * 2), InvResMLP(width * 2))

        self.sa2 = SetAbstraction(width * 2, width * 4, npoint=128, nsample=nsample)
        self.stage2 = nn.Sequential(InvResMLP(width * 4), InvResMLP(width * 4))

        final_dim = width * 4
        self.head = nn.Sequential(
            nn.Linear(final_dim * 2, 512, bias=False),
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
        xyz = x.transpose(1, 2).contiguous()
        features = self.stem(x)

        xyz, features = self.sa1(xyz, features)
        features = self.stage1(features)

        xyz, features = self.sa2(xyz, features)
        features = self.stage2(features)

        max_feat = F.adaptive_max_pool1d(features, 1).squeeze(-1)
        avg_feat = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
        global_feat = torch.cat((max_feat, avg_feat), dim=1)
        return self.head(global_feat)


def build_pointnext_encoder(out_channels, cfg=None, in_channels=3):
    cfg = cfg or {}
    return PointNeXtEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        width=cfg.get("pointnext_width", 48),
        nsample=cfg.get("pointnext_nsample", 24),
        dropout=cfg.get("pointnext_dropout", 0.05),
    )
