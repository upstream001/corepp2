import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnext import farthest_point_sample, index_points, knn_point


class PointTokenAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = max(dim // 2, 64)
        self.score = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        weights = torch.softmax(self.score(tokens), dim=1)
        return torch.sum(tokens * weights, dim=1)


class PointMAEPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=384, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, grouped_xyz):
        # grouped_xyz: [B, G, K, 3]
        features = grouped_xyz.permute(0, 3, 1, 2).contiguous()
        features = self.mlp(features)
        return features.max(dim=-1)[0].transpose(1, 2).contiguous()


class PointMAETransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PointMAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        num_groups=64,
        group_size=32,
        embed_dim=384,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size

        self.patch_embed = PointMAEPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            hidden_dim=max(embed_dim // 3, 64),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList(
            [
                PointMAETransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_pool = PointTokenAttentionPooling(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels),
        )

    def _group_points(self, xyz):
        # xyz: [B, N, 3]
        center_idx = farthest_point_sample(xyz, self.num_groups)
        centers = index_points(xyz, center_idx)  # [B, G, 3]
        group_idx = knn_point(self.group_size, xyz, centers)  # [B, G, K]
        grouped_xyz = index_points(xyz, group_idx)  # [B, G, K, 3]
        grouped_xyz = grouped_xyz - centers.unsqueeze(2)
        return centers, grouped_xyz

    def forward(self, x):
        # x: [B, 3, N]
        xyz = x.transpose(1, 2).contiguous()
        centers, grouped_xyz = self._group_points(xyz)
        tokens = self.patch_embed(grouped_xyz)
        tokens = tokens + self.pos_embed(centers)
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        cls_token = cls_token + self.cls_pos
        tokens = torch.cat((cls_token, tokens), dim=1)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        cls_feat = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        attn_pool = self.attn_pool(patch_tokens)
        max_pool = patch_tokens.max(dim=1)[0]
        avg_pool = patch_tokens.mean(dim=1)
        global_feat = torch.cat((cls_feat, attn_pool, max_pool, avg_pool), dim=1)
        return self.head(global_feat)


def build_point_mae_encoder(out_channels, cfg=None, in_channels=3):
    cfg = cfg or {}
    return PointMAEEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        num_groups=cfg.get("point_mae_num_groups", 64),
        group_size=cfg.get("point_mae_group_size", 32),
        embed_dim=cfg.get("point_mae_embed_dim", 384),
        depth=cfg.get("point_mae_depth", 8),
        num_heads=cfg.get("point_mae_num_heads", 8),
        mlp_ratio=cfg.get("point_mae_mlp_ratio", 4.0),
        dropout=cfg.get("point_mae_dropout", 0.1),
    )
