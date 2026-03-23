import os
import random

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import ConvexHull, Delaunay, cKDTree


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        pad_size,
        pretrain=None,
        split=None,
        use_partial=False,
        supervised_3d=True,
        norm_scale=45.54,
        sdf_loss=False,
        grid_density=20,
        sdf_trunc=0.015,
    ):
        """
        专用点云数据集加载类，可用于加载草莓数据集、20260301_dataset 等任意仅包含 complete/partial 结构的数据集。

        Args:
            data_source: 数据集根目录, 下方应包含 `complete` 目录 (或 `partial` 目录)
            pad_size: 采样后的点集大小（即喂入 Encoder 的点数）
            pretrain: 包含由 DeepSDF 预先计算好的 Latent Code (.pth) 的文件夹路径
            split: 当前使用的数据划分名称，会读取数据集根目录下的 `split.json`
            use_partial: 如果为 True, 则读取 `partial` 文件夹而不是 `complete` 文件夹
            supervised_3d: 如果为 True, 会读取 pretrain 下的 latent code 作为 ground truth
            norm_scale: 外部传入的用于从网络虚拟比例映射回现实世界毫米空间的最大范围缩放乘数
            sdf_loss: 是否为样本生成规则网格上的 SDF 监督
            grid_density: 3D 监督使用的规则网格分辨率
            sdf_trunc: 将真实距离归一化成 [-1, 1] TSDF 目标时使用的截断距离
        """
        self.data_source = data_source
        self.pad_size = pad_size
        self.split = split
        self.use_partial = use_partial
        self.supervised_3d = supervised_3d
        self.norm_scale = norm_scale
        self.sdf_loss = sdf_loss
        self.grid_density = int(grid_density)
        self.sdf_trunc = float(sdf_trunc)
        self._sdf_cache = {}

        if self.supervised_3d and pretrain is not None:
            self.latents_dict = self.get_latents_dict(pretrain)
        else:
            self.latents_dict = {}

        self.files = self.get_instance_filenames()

    def get_latents_dict(self, path):
        latent_dictionary = {}
        if not os.path.exists(path):
            print(f"[Warning] Latent codes path {path} does not exist.")
            return latent_dictionary

        # Support Unified Latent Matrix (.pth file instead of directory)
        if os.path.isfile(path) and path.endswith(".pth"):
            try:
                latents_file = torch.load(path)
                if "latent_codes" in latents_file and "weight" in latents_file["latent_codes"]:
                    latents_matrix = latents_file["latent_codes"]["weight"]
                    import json

                    base_exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
                    ds_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
                    split_type = self.split if self.split else "train"
                    split_file = os.path.join(base_exp_dir, "splits", f"{ds_name}_{split_type}.json")

                    if os.path.exists(split_file):
                        with open(split_file, "r") as f:
                            splits_data = json.load(f)

                        instance_names = []
                        if ds_name in splits_data:
                            class_dict = splits_data[ds_name]
                            if len(class_dict) == 1:
                                class_key = list(class_dict.keys())[0]
                                instance_names = class_dict[class_key]
                            elif "fruit" in class_dict:
                                instance_names = class_dict["fruit"]

                        if len(instance_names) == latents_matrix.shape[0]:
                            for i, inst in enumerate(instance_names):
                                latent_dictionary[inst] = latents_matrix[i].squeeze()
                            print(
                                f"[Info] Successfully mapped {len(instance_names)} latent codes "
                                f"from matrix {path} via {split_type} split."
                            )
                        else:
                            print(
                                f"[Warning] Matrix rows ({latents_matrix.shape[0]}) mismatch "
                                f"DeepSDF split size ({len(instance_names)})."
                            )
                    else:
                        print(f"[Warning] Cannot find DeepSDF split definition at {split_file} to map the latent matrix.")
                return latent_dictionary
            except Exception as e:
                print(f"[Error] Failed to load unified latent file {path}: {e}")
                return latent_dictionary

        # Support directory with individual .pth files.
        for fname in os.listdir(path):
            if fname.endswith(".pth"):
                latent = torch.load(os.path.join(path, fname))
                key = fname[:-4]
                latent_dictionary[key] = latent
        return latent_dictionary

    def get_instance_filenames(self):
        import json

        subfolder = "partial" if self.use_partial else "complete"
        pcd_dir = os.path.join(self.data_source, subfolder)

        allowed_keys = None
        if self.split is not None:
            split_file = os.path.join(self.data_source, "split.json")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    splits = json.load(f)
                    if self.split in splits:
                        allowed_keys = set(splits[self.split])
                    else:
                        print(f"[Warning] Split '{self.split}' not found in {split_file}")
            else:
                print(f"[Warning] Split file {split_file} not found, using all files.")

        files = []
        if os.path.exists(pcd_dir):
            for fname in sorted(os.listdir(pcd_dir)):
                if not fname.endswith(".ply"):
                    continue
                key = fname[:-4]

                if allowed_keys is not None and key not in allowed_keys:
                    continue

                if self.supervised_3d:
                    if key in self.latents_dict:
                        files.append(os.path.join(pcd_dir, fname))
                    else:
                        print(f"[Warning] Found {fname} but no corresponding latent code in {key}.pth")
                else:
                    files.append(os.path.join(pcd_dir, fname))
        else:
            print(f"[Error] Data directory {pcd_dir} does not exist!")

        return files

    def _generate_grid_points(self, bbox_min, bbox_max):
        grid_density_complex = self.grid_density * 1j
        X, Y, Z = np.mgrid[
            bbox_min[0]:bbox_max[0]:grid_density_complex,
            bbox_min[1]:bbox_max[1]:grid_density_complex,
            bbox_min[2]:bbox_max[2]:grid_density_complex,
        ]
        grid = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1).reshape((-1, 3))

        # Match Grid3D.generate_point_from_bbox().
        grid[:, :2][1::2] += ((X.max() - X.min()) / self.grid_density / 2.0)
        return grid.astype(np.float32)

    def _compute_convex_hull_sdf(self, points, grid_points):
        hull = ConvexHull(points)
        hull_vertices = points[hull.vertices]
        delaunay = Delaunay(hull_vertices)
        kdtree = cKDTree(points)

        unsigned_distance = kdtree.query(grid_points)[0].astype(np.float32)
        inside_mask = delaunay.find_simplex(grid_points) >= 0
        signed_distance = unsigned_distance
        signed_distance[inside_mask] *= -1.0
        return signed_distance.reshape(-1, 1)

    def _compute_mesh_sdf(self, points, grid_points):
        if not hasattr(o3d, "t") or not hasattr(o3d.t, "geometry"):
            raise RuntimeError("Open3D tensor geometry is unavailable.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        mesh = None

        # Alpha-shape is numerically unstable for sparse / near-planar point sets.
        # Gate it aggressively to avoid Open3D tetra warnings during training.
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        non_flat_axes = np.count_nonzero(bbox_extent > 1e-4)
        if points.shape[0] >= 256 and non_flat_axes == 3:
            try:
                pcd.estimate_normals()
                bbox_diag = float(np.linalg.norm(bbox_extent))
                alpha = max(bbox_diag / 18.0, 1e-3)
                tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha, tetra_mesh, pt_map
                )
            except Exception:
                mesh = None

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            mesh, _ = pcd.compute_convex_hull()

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)

        query = o3d.core.Tensor(grid_points.astype(np.float32))
        sdf = scene.compute_signed_distance(query).numpy().astype(np.float32)
        return sdf.reshape(-1, 1)

    def _get_sdf_targets(self, fruit_id, points_centered, bbox_min, bbox_max):
        if fruit_id in self._sdf_cache:
            return self._sdf_cache[fruit_id]

        grid_points = self._generate_grid_points(bbox_min, bbox_max)

        try:
            signed_distance = self._compute_mesh_sdf(points_centered, grid_points)
        except Exception:
            signed_distance = self._compute_convex_hull_sdf(points_centered, grid_points)

        target_sdf = np.clip(signed_distance / self.sdf_trunc, -1.0, 1.0).astype(np.float32)
        target_weights = np.ones_like(target_sdf, dtype=np.float32)

        cached = (
            torch.from_numpy(target_sdf),
            torch.from_numpy(target_weights),
        )
        self._sdf_cache[fruit_id] = cached
        return cached

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        fruit_id = os.path.basename(file_path)[:-4]

        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        num_points = points.shape[0]

        if num_points >= self.pad_size:
            random_list = np.random.choice(num_points, size=self.pad_size, replace=False)
            sampled_points = points[random_list, :]
        else:
            random_list = np.random.choice(num_points, size=self.pad_size, replace=True)
            sampled_points = points[random_list, :]

        center = np.mean(points, axis=0)
        points_centered = points - center
        sampled_points = sampled_points - center
        scale = 1.0

        bbox_min = np.min(sampled_points, axis=0)
        bbox_max = np.max(sampled_points, axis=0)
        margin = (bbox_max - bbox_min) * 0.1
        margin = np.maximum(margin, 1e-4)
        bbox_min -= margin
        bbox_max += margin

        item = {
            "fruit_id": fruit_id,
            "target_pcd": torch.Tensor(sampled_points).float(),
            "partial_pcd": torch.Tensor(sampled_points).float(),
            "center": torch.Tensor(center).float(),
            "scale": torch.tensor(scale).float(),
            "bbox": {
                "xmin": torch.tensor(bbox_min[0]).float(),
                "xmax": torch.tensor(bbox_max[0]).float(),
                "ymin": torch.tensor(bbox_min[1]).float(),
                "ymax": torch.tensor(bbox_max[1]).float(),
                "zmin": torch.tensor(bbox_min[2]).float(),
                "zmax": torch.tensor(bbox_max[2]).float(),
            },
        }

        item["rgb"] = torch.zeros((3, self.pad_size, self.pad_size))
        item["depth"] = torch.zeros((1, self.pad_size, self.pad_size))
        item["mask"] = torch.zeros((1, self.pad_size, self.pad_size))

        if self.supervised_3d:
            trained_latent = self.latents_dict[fruit_id]
            item["latent"] = trained_latent.squeeze().float()

        if self.sdf_loss:
            target_sdf, target_sdf_weights = self._get_sdf_targets(
                fruit_id, points_centered, bbox_min, bbox_max
            )
            item["target_sdf"] = target_sdf.clone()
            item["target_sdf_weights"] = target_sdf_weights.clone()

        item["frame_id"] = str(fruit_id)
        return item


if __name__ == "__main__":
    ds = PointCloudDataset("/home/tianqi/corepp2/data/strawberry", pad_size=2048, supervised_3d=False)
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        item = ds[0]
        print(f"Sample item partial_pcd shape: {item['partial_pcd'].shape}")
