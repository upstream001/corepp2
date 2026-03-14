import open3d as o3d
import numpy as np
import torch

def compute(gt_file, mesh_file):
    gt = o3d.io.read_point_cloud(gt_file)
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
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

    eval_mesh_pcd = eval_mesh.sample_points_uniformly(1000000)

    # precision: predicted --> ground truth
    dist_pt_2_gt = np.asarray(eval_mesh_pcd.compute_point_cloud_distance(eval_gt))
    # recall: ground truth --> predicted
    dist_gt_2_pt = np.asarray(eval_gt.compute_point_cloud_distance(eval_mesh_pcd))

    print(f"dist_pt_2_gt mean: {np.mean(dist_pt_2_gt)}")
    print(f"dist_gt_2_pt mean: {np.mean(dist_gt_2_pt)}")
    
    t = 0.005
    p = np.sum(dist_pt_2_gt < t) / len(dist_pt_2_gt) * 100
    r = np.sum(dist_gt_2_pt < t) / len(dist_gt_2_pt) * 100
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
        
    print(f"p: {p}, r: {r}, f1: {f1}")
    
    # Save the aligned pcds for visual inspection
    eval_gt.paint_uniform_color([0, 1, 0])
    eval_gt.estimate_normals()
    eval_mesh_pcd.paint_uniform_color([1, 0, 0])
    o3d.io.write_point_cloud("debug_gt.ply", eval_gt)
    o3d.io.write_point_cloud("debug_pred.ply", eval_mesh_pcd)

compute('/home/tianqi/corepp2/data/20260312_dataset/complete/00252.ply', '/home/tianqi/corepp2/logs/strawberry/output/00252.ply')
