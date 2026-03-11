import os
import argparse
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm
import torch

def upsample_point_cloud(points, target_points=16384, k=5, jitter_scale=0.0):
    """
    使用基于密度感知的邻近点插值算法进行上采样。
    改进点：
    1. 根据局部稀疏程度分配采样概率：较稀疏的区域会产生更多的新点，避免密集区过度重叠。
    2. 加入微小抖动：避免生成的新点呈现出不自然的明显连线伪像。
    """
    n_points = points.shape[0]
    
    if n_points >= target_points:
        # 如果点数已经达到或超过目标数值，严格满足目标输出要求，使用随机降采样：
        idx = np.random.choice(n_points, target_points, replace=False)
        return points[idx]
        
    n_added = target_points - n_points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    actual_k = min(k + 1, n_points)
    weights = np.zeros(n_points)
    
    # 1. 计算所有点的局部分布密度权重 (稀疏的地方权重大)
    if actual_k > 1:
        for i in range(n_points):
            [_, _, sq_dist] = pcd_tree.search_knn_vector_3d(points[i], actual_k)
            # 使用周边距离的均值作为这个点生成新点的权重（即距离越远越容易被挑中）
            weights[i] = np.mean(np.array(sq_dist)[1:])
    else:
        weights = np.ones(n_points)
        
    # 转换为采样概率律
    if np.sum(weights) > 0:
        probs = weights / np.sum(weights)
    else:
        probs = np.ones(n_points) / n_points

    # 2. 依据概率批量采样被抽中的源点索引
    src_indices = np.random.choice(n_points, size=n_added, p=probs)
    new_points = np.zeros((n_added, 3))
    
    for i in range(n_added):
        src_idx = src_indices[i]
        src_point = points[src_idx]
        
        [_, idx, _] = pcd_tree.search_knn_vector_3d(src_point, actual_k)
        if actual_k > 1:
            # 随机挑选除自己外的一个近邻
            neighbor_idx = np.random.choice(idx[1:])
            neighbor_point = points[neighbor_idx]
            
            # 在连线上插值
            t = np.random.random()
            new_point = src_point + t * (neighbor_point - src_point)
            
            # 添加微小的不规则表面扰动，打破绝对直线的生成人造感
            if jitter_scale > 0:
                dist = np.linalg.norm(neighbor_point - src_point)
                jitter = np.random.randn(3) * dist * jitter_scale
                new_point += jitter
        else:
            new_point = src_point
            
        new_points[i] = new_point
        
    upsampled_points = np.vstack((points, new_points))
    return upsampled_points

def main():
    parser = argparse.ArgumentParser(description="使用临近点线性插值将目标文件夹内的点云上采样/下采样到固定点数 (默认 16384)")
    parser.add_argument('--input', type=str, default='/home/tianqi/corepp2/data/test_ply', help='输入的旧点云文件夹路径')
    parser.add_argument('--output', type=str, default='/home/tianqi/corepp2/data/test_ply_upsample', help='上采样后保存的新点云文件夹路径')
    parser.add_argument('--num', type=int, default=16384, help='目标输出点数大小')
    parser.add_argument('--k', type=int, default=10, help='用于插值的 K 近邻数量')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    ply_files = glob(os.path.join(args.input, '*.ply'))
    
    if not ply_files:
        print(f"未在文件夹 {args.input} 内找到 .ply 文件！")
        return
        
    for ply_path in tqdm(ply_files, desc=f"Upsampling to {args.num} points"):
        filename = os.path.basename(ply_path)
        out_path = os.path.join(args.output, filename)
        
        # 读取点云
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            print(f"警告：点云 {filename} 为空！已跳过。")
            continue
            
        # 进行插值上采样
        upsampled_points = upsample_point_cloud(points, target_points=args.num, k=args.k)
        
        # 保存点云
        out_pcd = o3d.geometry.PointCloud()
        if isinstance(upsampled_points, torch.Tensor):
            upsampled_points = upsampled_points.cpu().numpy()
        out_pcd.points = o3d.utility.Vector3dVector(upsampled_points)
        o3d.io.write_point_cloud(out_path, out_pcd)
        
    print(f"\n全部处理完成，结果已保存至：{args.output}")

if __name__ == "__main__":
    main()
