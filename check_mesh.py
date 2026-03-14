import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh('/home/tianqi/corepp2/logs/strawberry/output/00252.ply')
mesh_points = np.asarray(mesh.vertices)
print("Mesh min:", mesh_points.min(axis=0))
print("Mesh max:", mesh_points.max(axis=0))

import torch
import json
from torch.utils.data import DataLoader
from dataloaders.pointcloud_dataset import PointCloudDataset

with open('./configs/strawberry.json') as json_file:
    param = json.load(json_file)

ds = PointCloudDataset(
    data_source=param["data_dir"],
    pad_size=param["input_size"],
    pretrain=None,
    split='test',
    use_partial=False,
    supervised_3d=False
)
for item in ds:
    if item['fruit_id'] == '00252':
        gt_points = item['target_pcd'].numpy()
        print("GT min:", gt_points.min(axis=0))
        print("GT max:", gt_points.max(axis=0))
        break
