import open3d as o3d
import numpy as np
import torch
import json
from dataloaders.pointcloud_dataset import PointCloudDataset

with open('./configs/strawberry.json') as json_file:
    param = json.load(json_file)

cl_dataset = PointCloudDataset(
    data_source=param["data_dir"],
    pad_size=param["input_size"],
    pretrain=None,
    split='test',
    use_partial=False,
    supervised_3d=False
)

item = cl_dataset[0]
from metrics_3d.metric import Metrics3D
m = Metrics3D()

mesh_file = '/home/tianqi/corepp2/logs/strawberry/output/' + item['frame_id'] + '.ply'
mesh = o3d.io.read_triangle_mesh(mesh_file)
gt = o3d.geometry.PointCloud()
gt.points = o3d.utility.Vector3dVector(item['target_pcd'].numpy())

gt_pts = np.asarray(gt.points)
mesh_pts = np.asarray(mesh.vertices)

local_center = np.mean(gt_pts, axis=0)
shifted_gt = gt_pts - local_center
local_scale = np.max(np.linalg.norm(shifted_gt, axis=1)) 
if local_scale == 0: local_scale = 1.0

eval_gt = o3d.geometry.PointCloud()
eval_gt.points = o3d.utility.Vector3dVector(shifted_gt / local_scale)

eval_mesh = o3d.geometry.TriangleMesh()
eval_mesh.vertices = o3d.utility.Vector3dVector((mesh_pts - local_center) / local_scale)
eval_mesh.triangles = mesh.triangles

print("eval_mesh prediction is empty:", m.prediction_is_empty(eval_mesh))
print("eval_mesh has vertices after check?", len(eval_mesh.vertices))
print("eval_mesh has triangles after check?", len(eval_mesh.triangles))
