import open3d as o3d
import numpy as np

mesh_file = '/home/tianqi/corepp2/logs/strawberry/output/00252.ply'
mesh = o3d.io.read_triangle_mesh(mesh_file)
print(f"Original verts: {len(mesh.vertices)}, faces: {len(mesh.triangles)}")

mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_degenerate_triangles()
print(f"After cleaning verts: {len(mesh.vertices)}, faces: {len(mesh.triangles)}")

# Now try the copied eval_mesh
gt = o3d.io.read_point_cloud('/home/tianqi/corepp2/data/20260312_dataset/complete/00252.ply')
gt_pts = np.asarray(gt.points)
mesh_pts = np.asarray(mesh.vertices)
local_center = np.mean(gt_pts, axis=0)
shifted_gt = gt_pts - local_center
local_scale = np.max(np.linalg.norm(shifted_gt, axis=1)) 

eval_mesh = o3d.geometry.TriangleMesh()
eval_mesh.vertices = o3d.utility.Vector3dVector((mesh_pts - local_center) / local_scale)
eval_mesh.triangles = mesh.triangles

eval_mesh.remove_duplicated_vertices()
eval_mesh.remove_duplicated_triangles()
eval_mesh.remove_degenerate_triangles()
print(f"Eval mesh after cleaning verts: {len(eval_mesh.vertices)}, faces: {len(eval_mesh.triangles)}")
