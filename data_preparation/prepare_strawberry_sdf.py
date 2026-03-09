import os
import numpy as np
import open3d as o3d
import argparse

def generate_tsdf_samples(swl_points: np.ndarray, no_samples_per_point: int,
                          tsdf_positive: float, tsdf_negative: float) -> tuple:
    no_points = swl_points.shape[0]

    offset_vectors = swl_points[:,3:6] - swl_points[:,0:3]
    offset_vectors = offset_vectors / np.expand_dims(np.linalg.norm(offset_vectors, axis=1), axis=-1)
    offset_vectors = np.repeat(offset_vectors, no_samples_per_point, axis=0)

    tsdf_positive_offset = tsdf_positive * np.random.rand(no_points*no_samples_per_point, 1)
    tsdf_positive_offset_env = 0.1 * np.random.rand(no_points*no_samples_per_point, 1)
    tsdf_negative_offset = (-1.0) * tsdf_negative * np.random.rand(no_points*no_samples_per_point, 1)

    sample_points = np.repeat(swl_points[:,0:3], no_samples_per_point, axis=0)

    pos = sample_points + tsdf_positive_offset * offset_vectors
    env = sample_points + tsdf_positive_offset_env * offset_vectors
    neg = sample_points + tsdf_negative_offset * offset_vectors

    pos = np.concatenate ((pos, tsdf_positive_offset), axis=1)
    env = np.concatenate ((env, tsdf_positive_offset_env), axis=1)
    pos = np.concatenate((pos,env), axis=0)
    neg = np.concatenate ((neg, tsdf_negative_offset), axis=1)

    return (pos,neg)


def sample_freespace(points, normals, n_samples=20000):
    """
    在表面外侧较远处采样 free space 点，给网络提供远场正 SDF 约束。
    参考原版 o3d_utils.py 中的 sample_freespace 函数。
    """
    xyz_free = []
    mu_free = []
    for _ in range(n_samples):
        mu_freespace = np.random.uniform(0.01, 0.1)  # 比近表面采样更远
        idx = np.random.randint(len(points))
        sample = points[idx] + mu_freespace * normals[idx]
        xyz_free.append(sample)
        mu_free.append(mu_freespace)
    return np.asarray(xyz_free), np.asarray(mu_free)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare deep sdf training data for generalized complete pointclouds")
    parser.add_argument("--src", required=True, type=str, 
                        help="Path to the dataset directory which contains a 'complete/' folder with .ply models")
    parser.add_argument("--no_of_samples", default=100000, type=int)
    parser.add_argument("--tsdf_positive", default=0.04, type=float)
    parser.add_argument("--tsdf_negative", default=0.01, type=float)
    args = parser.parse_args()
    
    complete_dir = os.path.join(args.src, "complete")
    if not os.path.exists(complete_dir):
        print(f"Error: Directory {complete_dir} not found. Ensure --src points to your dataset root.")
        exit(1)
        
    print(f"Processing dataset in {args.src} ...")
    files = sorted([f for f in os.listdir(complete_dir) if f.endswith(".ply")])
    if len(files) == 0:
        print("No .ply files found in the 'complete' folder.")
        exit(1)
        
    print(f"Found {len(files)} point clouds. Generating SDF samples...")
    for idx, fname in enumerate(files):
        fid = fname[:-4]
        # read point cloud
        pcd = o3d.io.read_point_cloud(os.path.join(complete_dir, fname))
        
        # 估计法线并确保法线朝外
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # ==================== 关键修复：法线方向 ====================
        # 原代码使用 orient_normals_towards_camera_location(np.zeros(3))
        # 对于居中在原点的点云，这会让法线全部指向内部，导致 SDF 内外翻转
        # 
        # 修复：逐点检查法线是否朝外，如果法线与"从中心到该点"的方向点积为负则翻转
        centroid = np.asarray(pcd.points).mean(axis=0)
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        directions_from_center = points - centroid
        dot_products = np.sum(normals * directions_from_center, axis=1)
        # 翻转指向内部的法线
        normals[dot_products < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)
            
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        num_points = points.shape[0]
        
        if num_points == 0:
            print(f"Skipping empty pointcloud {fname}")
            continue
            
        viewpoints = points + normals
        swl_points = np.concatenate([points, viewpoints], axis=1)
        
        # calculate sampling logic
        no_samples_per_point = int(np.ceil(args.no_of_samples / num_points))
        pos, neg = generate_tsdf_samples(swl_points, no_samples_per_point=no_samples_per_point,
                                tsdf_positive=args.tsdf_positive, tsdf_negative=args.tsdf_negative)
        
        # ==================== 关键修复：添加 free space 采样 ====================
        # 参考原版 o3d_utils.py 的 sample_freespace 函数
        # 在表面外侧添加远场正 SDF 样本，避免网络在远场产生无约束的零交叉面
        xyz_free, sdf_free = sample_freespace(points, normals, n_samples=20000)
        free_samples = np.column_stack([xyz_free, sdf_free])
        pos = np.vstack([pos, free_samples])
                                
        # subsampling appropriately
        pos = pos[np.random.choice(pos.shape[0], args.no_of_samples, replace=False), :]
        neg = neg[np.random.choice(neg.shape[0], args.no_of_samples, replace=False), :]
        
        # Because DeepSDF loader rigidly expects `data_dir/class_name/instance_id/laser/samples.npz`
        # and our make split creates `data_dir -> "fruit"`
        # We output it at `{args.src}/fruit/{fid}/laser/samples.npz`
        out_dir = os.path.join(args.src, "fruit", fid, "laser")
        os.makedirs(out_dir, exist_ok=True)
        
        out_npz = os.path.join(out_dir, "samples.npz")
        np.savez(out_npz, pos=pos, neg=neg)
        
        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1}/{len(files)}")
        
    print(f"SDF Data successfully exported to {os.path.join(args.src, 'fruit')}!")
