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


class AttentionPooling1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            group_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


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


class ResidualMLP1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
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


class Stage(nn.Module):
    def __init__(self, channels, depth=2, expansion=4):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(InvResMLP(channels, expansion=expansion))
            blocks.append(ResidualMLP1d(channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            SharedMLP1d(in_channels, out_channels),
            ResidualMLP1d(out_channels),
            ResidualMLP1d(out_channels),
        )

    def forward(self, target_xyz, source_xyz, target_features, source_features):
        interpolated = interpolate_features(target_xyz, source_xyz, source_features)

        if target_features is not None:
            fused = torch.cat((target_features, interpolated), dim=1)
        else:
            fused = interpolated
        return self.fuse(fused)


def interpolate_features(target_xyz, source_xyz, source_features, k=3):
    if source_xyz.shape[1] == 1:
        return source_features.expand(-1, -1, target_xyz.shape[1])

    k = min(k, source_xyz.shape[1])
    dist = square_distance(target_xyz, source_xyz)
    dists, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    dists = torch.clamp(dists, min=1e-10)
    weights = 1.0 / dists
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    neighbor_features = index_points(source_features.transpose(1, 2), idx)
    interpolated = torch.sum(neighbor_features * weights.unsqueeze(-1), dim=2)
    return interpolated.transpose(1, 2).contiguous()


class PointNeXtEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        width=64,
        nsample=32,
        dropout=0.1,
        sa1_npoint=512,
        sa2_npoint=128,
        sa3_npoint=32,
        stage_depth=2,
        expansion=4,
        feature_fusion="multi_scale",
        global_pool="max_avg",
    ):
        super().__init__()
        self.feature_fusion = str(feature_fusion).lower()
        self.global_pool = str(global_pool).lower()

        self.stem = nn.Sequential(
            SharedMLP1d(in_channels, width),
            SharedMLP1d(width, width),
            ResidualMLP1d(width),
        )

        self.sa1 = SetAbstraction(width, width * 2, npoint=sa1_npoint, nsample=nsample)
        self.stage1 = Stage(width * 2, depth=stage_depth, expansion=expansion)

        self.sa2 = SetAbstraction(width * 2, width * 4, npoint=sa2_npoint, nsample=nsample)
        self.stage2 = Stage(width * 4, depth=stage_depth, expansion=expansion)

        self.sa3 = SetAbstraction(width * 4, width * 8, npoint=sa3_npoint, nsample=nsample)
        self.stage3 = Stage(width * 8, depth=stage_depth, expansion=expansion)

        if self.feature_fusion == "multi_scale":
            fused_dim = width * 2 + width * 4 + width * 8
        elif self.feature_fusion == "stage3":
            fused_dim = width * 8
        else:
            raise ValueError(
                f"Unsupported pointnext feature fusion mode: {feature_fusion}. "
                "Expected one of ['multi_scale', 'stage3']."
            )

        if self.global_pool == "max_avg":
            head_in_dim = fused_dim * 2
        elif self.global_pool in {"max", "avg"}:
            head_in_dim = fused_dim
        else:
            raise ValueError(
                f"Unsupported pointnext global pool mode: {global_pool}. "
                "Expected one of ['max', 'avg', 'max_avg']."
            )

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 512, bias=False),
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

        xyz1, features1 = self.sa1(xyz, features)
        features1 = self.stage1(features1)

        xyz2, features2 = self.sa2(xyz1, features1)
        features2 = self.stage2(features2)

        _, features3 = self.sa3(xyz2, features2)
        features3 = self.stage3(features3)

        if self.feature_fusion == "multi_scale":
            multi_scale = torch.cat(
                (features1, F.interpolate(features2, size=features1.shape[-1], mode="nearest")),
                dim=1,
            )
            coarse_scale = F.interpolate(features3, size=features1.shape[-1], mode="nearest")
            fused = torch.cat((multi_scale, coarse_scale), dim=1)
        else:
            fused = features3

        pooled = []
        if self.global_pool in {"max", "max_avg"}:
            pooled.append(F.adaptive_max_pool1d(fused, 1).squeeze(-1))
        if self.global_pool in {"avg", "max_avg"}:
            pooled.append(F.adaptive_avg_pool1d(fused, 1).squeeze(-1))
        global_feat = pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=1)
        return self.head(global_feat)


class PointNeXtUNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        width=64,
        nsample=32,
        dropout=0.1,
        sa1_npoint=512,
        sa2_npoint=128,
        sa3_npoint=32,
        stage_depth=2,
        expansion=4,
        global_pool="max_avg",
    ):
        super().__init__()
        self.global_pool = str(global_pool).lower()
        branch_width = max(256, width * 4)

        self.stem = nn.Sequential(
            SharedMLP1d(in_channels, width),
            SharedMLP1d(width, width),
            ResidualMLP1d(width),
        )

        self.sa1 = SetAbstraction(width, width * 2, npoint=sa1_npoint, nsample=nsample)
        self.stage1 = Stage(width * 2, depth=stage_depth, expansion=expansion)

        self.sa2 = SetAbstraction(width * 2, width * 4, npoint=sa2_npoint, nsample=nsample)
        self.stage2 = Stage(width * 4, depth=stage_depth, expansion=expansion)

        self.sa3 = SetAbstraction(width * 4, width * 8, npoint=sa3_npoint, nsample=nsample)
        self.stage3 = Stage(width * 8, depth=stage_depth, expansion=expansion)

        self.fp3 = FeaturePropagation(width * 8 + width * 4, width * 4)
        self.fp2 = FeaturePropagation(width * 4 + width * 2, width * 2)
        self.fp1 = FeaturePropagation(width * 2 + width, width * 2)
        fused_dense_dim = width * 2 + width * 2 + width * 4 + width * 8
        self.dense_fuse = nn.Sequential(
            SharedMLP1d(fused_dense_dim, width * 4),
            ResidualMLP1d(width * 4),
            SharedMLP1d(width * 4, width * 2),
        )
        self.refine = Stage(width * 2, depth=max(1, stage_depth + 1), expansion=expansion)
        self.attn_pool = AttentionPooling1d(width * 2)
        self.coarse_attn_pool = AttentionPooling1d(width * 8)

        if self.global_pool == "max_avg":
            dense_pool_dim = width * 6
            coarse_pool_dim = width * 16
        elif self.global_pool in {"max", "avg"}:
            dense_pool_dim = width * 2
            coarse_pool_dim = width * 8
        elif self.global_pool == "attn":
            dense_pool_dim = width * 2
            coarse_pool_dim = width * 8
        else:
            raise ValueError(
                f"Unsupported pointnext_unet global_pool mode: {global_pool}. "
                "Expected one of ['max', 'avg', 'max_avg', 'attn']."
            )

        self.shape_head = nn.Sequential(
            nn.Linear(dense_pool_dim, branch_width, bias=False),
            nn.LayerNorm(branch_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(branch_width, branch_width, bias=False),
            nn.LayerNorm(branch_width),
            nn.ReLU(inplace=True),
        )
        self.size_head = nn.Sequential(
            nn.Linear(coarse_pool_dim + 6, branch_width, bias=False),
            nn.LayerNorm(branch_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(branch_width, branch_width, bias=False),
            nn.LayerNorm(branch_width),
            nn.ReLU(inplace=True),
        )
        self.branch_gate = nn.Sequential(
            nn.Linear(branch_width * 2, branch_width, bias=False),
            nn.LayerNorm(branch_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(branch_width, branch_width * 2),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(branch_width * 2, 512, bias=False),
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
        xyz0 = x.transpose(1, 2).contiguous()
        features0 = self.stem(x)

        xyz1, features1 = self.sa1(xyz0, features0)
        features1 = self.stage1(features1)

        xyz2, features2 = self.sa2(xyz1, features1)
        features2 = self.stage2(features2)

        xyz3, features3 = self.sa3(xyz2, features2)
        features3 = self.stage3(features3)

        up2 = self.fp3(xyz2, xyz3, features2, features3)
        up1 = self.fp2(xyz1, xyz2, features1, up2)
        up0 = self.fp1(xyz0, xyz1, features0, up1)
        dense_mid = interpolate_features(xyz0, xyz1, up1)
        dense_coarse = interpolate_features(xyz0, xyz2, up2)
        dense_global = interpolate_features(xyz0, xyz3, features3)
        dense_feat = torch.cat((up0, dense_mid, dense_coarse, dense_global), dim=1)
        dense_feat = self.dense_fuse(dense_feat)
        dense_feat = self.refine(dense_feat)

        dense_pooled = []
        if self.global_pool in {"max", "max_avg"}:
            dense_pooled.append(F.adaptive_max_pool1d(dense_feat, 1).squeeze(-1))
        if self.global_pool in {"avg", "max_avg"}:
            dense_pooled.append(F.adaptive_avg_pool1d(dense_feat, 1).squeeze(-1))
        if self.global_pool in {"attn", "max_avg"}:
            dense_pooled.append(self.attn_pool(dense_feat))
        shape_feat = dense_pooled[0] if len(dense_pooled) == 1 else torch.cat(dense_pooled, dim=1)

        coarse_pooled = []
        if self.global_pool in {"max", "max_avg"}:
            coarse_pooled.append(F.adaptive_max_pool1d(features3, 1).squeeze(-1))
        if self.global_pool in {"avg", "max_avg"}:
            coarse_pooled.append(F.adaptive_avg_pool1d(features3, 1).squeeze(-1))
        if self.global_pool == "attn":
            coarse_pooled.append(self.coarse_attn_pool(features3))
        coarse_feat = coarse_pooled[0] if len(coarse_pooled) == 1 else torch.cat(coarse_pooled, dim=1)

        xyz_extent = xyz0.max(dim=1)[0] - xyz0.min(dim=1)[0]
        xyz_std = xyz0.std(dim=1, unbiased=False)
        size_stats = torch.cat((xyz_extent, xyz_std), dim=1)

        shape_branch = self.shape_head(shape_feat)
        size_branch = self.size_head(torch.cat((coarse_feat, size_stats), dim=1))
        fused_branch = torch.cat((shape_branch, size_branch), dim=1)
        gate = self.branch_gate(fused_branch)
        fused_branch = fused_branch * gate
        return self.head(fused_branch)


def build_pointnext_encoder(out_channels, cfg=None, in_channels=3):
    cfg = cfg or {}
    return PointNeXtEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        width=cfg.get("pointnext_width", 64),
        nsample=cfg.get("pointnext_nsample", 32),
        dropout=cfg.get("pointnext_dropout", 0.1),
        sa1_npoint=cfg.get("pointnext_sa1_npoint", 512),
        sa2_npoint=cfg.get("pointnext_sa2_npoint", 128),
        sa3_npoint=cfg.get("pointnext_sa3_npoint", 32),
        stage_depth=cfg.get("pointnext_stage_depth", 2),
        expansion=cfg.get("pointnext_expansion", 4),
        feature_fusion=cfg.get("pointnext_feature_fusion", "multi_scale"),
        global_pool=cfg.get("pointnext_global_pool", "max_avg"),
    )


def build_pointnext_unet_encoder(out_channels, cfg=None, in_channels=3):
    cfg = cfg or {}
    return PointNeXtUNetEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        width=cfg.get("pointnext_width", 64),
        nsample=cfg.get("pointnext_nsample", 32),
        dropout=cfg.get("pointnext_dropout", 0.1),
        sa1_npoint=cfg.get("pointnext_sa1_npoint", 512),
        sa2_npoint=cfg.get("pointnext_sa2_npoint", 128),
        sa3_npoint=cfg.get("pointnext_sa3_npoint", 32),
        stage_depth=cfg.get("pointnext_stage_depth", 2),
        expansion=cfg.get("pointnext_expansion", 4),
        global_pool=cfg.get("pointnext_global_pool", "max_avg"),
    )
