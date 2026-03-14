import open3d as o3d
import numpy as np

gt = o3d.io.read_point_cloud('/home/tianqi/corepp2/data/20260312_dataset/complete/00252.ply')
mesh = o3d.io.read_triangle_mesh('/home/tianqi/corepp2/logs/strawberry/output/00252.ply')

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

from metrics_3d.metric import Metrics3D
m = Metrics3D()
print("eval_mesh prediction is empty:", m.prediction_is_empty(eval_mesh))
print("eval_gt prediction is empty:", m.prediction_is_empty(eval_gt))
