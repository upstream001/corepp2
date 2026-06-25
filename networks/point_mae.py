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
        use_cls_token=True,
        use_attention_pooling=True,
        pooling_modes=None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.use_cls_token = bool(use_cls_token)
        if pooling_modes is None:
            pooling_modes = []
            if use_attention_pooling:
                pooling_modes.append("attn")
            pooling_modes.extend(["max", "avg"])
        self.pooling_modes = [str(mode).lower() for mode in pooling_modes]
        if not self.pooling_modes:
            raise ValueError("PointMAEEncoder requires at least one pooling mode.")
        valid_pooling_modes = {"attn", "max", "avg"}
        invalid_modes = sorted(set(self.pooling_modes) - valid_pooling_modes)
        if invalid_modes:
            raise ValueError(f"Unsupported pooling modes: {invalid_modes}")
        self.use_attention_pooling = "attn" in self.pooling_modes

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
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
            self.cls_pos = None
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_pool = PointTokenAttentionPooling(embed_dim) if self.use_attention_pooling else None
        num_global_chunks = len(self.pooling_modes) + int(self.use_cls_token)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * num_global_chunks, 512),
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
        if self.use_cls_token:
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            cls_token = cls_token + self.cls_pos
            tokens = torch.cat((cls_token, tokens), dim=1)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        if self.use_cls_token:
            cls_feat = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            cls_feat = None
            patch_tokens = tokens

        global_chunks = []
        if cls_feat is not None:
            global_chunks.append(cls_feat)
        for mode in self.pooling_modes:
            if mode == "attn":
                global_chunks.append(self.attn_pool(patch_tokens))
            elif mode == "max":
                global_chunks.append(patch_tokens.max(dim=1)[0])
            elif mode == "avg":
                global_chunks.append(patch_tokens.mean(dim=1))
        global_feat = torch.cat(global_chunks, dim=1)
        return self.head(global_feat)


def build_point_mae_encoder(out_channels, cfg=None, in_channels=3):
    cfg = cfg or {}
    pooling_modes = cfg.get("point_mae_pooling_modes")
    if pooling_modes is None:
        pooling_modes = None
    elif isinstance(pooling_modes, str):
        pooling_modes = [token.strip() for token in pooling_modes.split("+") if token.strip()]
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
        use_cls_token=cfg.get("point_mae_use_cls_token", True),
        use_attention_pooling=cfg.get("point_mae_use_attention_pooling", True),
        pooling_modes=pooling_modes,
    )
