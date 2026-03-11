#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云重采样脚本
支持将点云重采样为指定点数（默认 16384）并进行归一化。
支持单个文件或整个文件夹的处理。
"""

import open3d as o3d
import numpy as np
import os
import argparse
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pathlib import Path

def resample_pcd(pcd, target_n=16384):
    points = np.asarray(pcd.points)
    n_pts = points.shape[0]
    
    if n_pts == target_n:
        print(f"点数已经是 {target_n}，无需重采样。")
        return pcd
    
    if n_pts > target_n:
        print(f"当前点数 ({n_pts}) > {target_n}，执行随机下采样...")
        idx = np.random.choice(n_pts, target_n, replace=False)
        resampled_points = points[idx]
        
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(resampled_points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            new_pcd.colors = o3d.utility.Vector3dVector(colors[idx])
    else:
        print(f"当前点数 ({n_pts}) < {target_n}，执行临近点插值上采样...")
        # 需要增加的点数
        num_to_add = target_n - n_pts
        
        # 使用 KNN 查找最近邻
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # 随机选择基础点进行插值
        add_idx = np.random.choice(n_pts, num_to_add, replace=True)
        
        base_points = points[add_idx]
        neighbor_points = points[indices[add_idx, 1]]
        
        # 计算中点插值
        new_points = (base_points + neighbor_points) / 2.0
        resampled_points = np.concatenate([points, new_points], axis=0)
        
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(resampled_points)
        
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            new_colors = (colors[add_idx] + colors[indices[add_idx, 1]]) / 2.0
            new_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors, new_colors], axis=0))
            
    return new_pcd

def pc_norm(pc):
    """ 
    归一化点云到 [-0.5, 0.5]
    pc: (N, 3) numpy array
    returns: (normalized_pc, centroid, scale)
    """
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    # 计算到中心的最远距离
    dist = np.max(np.sqrt(np.sum(pc_centered**2, axis=1)))
    scale = dist * 2
    if scale > 0:
        pc_normed = pc_centered / scale
    else:
        pc_normed = pc_centered
    return pc_normed, centroid, scale

def process_single_file(input_path, output_path, target_n=16384, normalize=True):
    """处理单个点云文件"""
    print(f"\n--- 正在处理: {os.path.basename(input_path)} ---")
    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty():
        print(f"警告: 文件为空或无法读取: {input_path}")
        return

    # 1. 执行重采样
    new_pcd = resample_pcd(pcd, target_n)
    
    # 2. 执行归一化
    if normalize:
        print("执行归一化...")
        points = np.asarray(new_pcd.points)
        points_norm, centroid, scale = pc_norm(points)
        new_pcd.points = o3d.utility.Vector3dVector(points_norm)
        print(f"归一化参数: 质心={centroid}, 缩放因子={scale:.4f}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), new_pcd)
    print(f"已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="点云重采样工具 (16384点)")
    parser.add_argument("--input", type=str, default="/home/tianqi/corepp2/data/test_ply", help="输入文件或文件夹路径 (.ply, .pcd)")
    parser.add_argument("--output", type=str, default="/home/tianqi/corepp2/data/test_ply_resample", help="输出文件或文件夹路径 (如果是文件夹，则在其中生成结果)")
    parser.add_argument("--target", type=int, default=16384, help="目标点数 (默认: 16384)")
    parser.add_argument("--no_norm", action="store_true", help="禁用归一化处理")
    parser.add_argument("--ext", type=str, default="ply", help="批量处理时的扩展名 (默认: ply)")

    args = parser.parse_args()

    input_path = Path(args.input)
    
    # 如果没指定输出，默认在输入同级或原位处理
    if args.output:
        output_root = Path(args.output)
    else:
        output_root = input_path.parent / (input_path.name + "_resampled")

    if input_path.is_file():
        if not args.output:
            # 单文件默认输出名
            target_file = input_path.parent / f"{input_path.stem}_{args.target}_normalized.ply"
        else:
            target_file = Path(args.output)
            if target_file.suffix == '': # 如果 output 是个目录
                target_file = target_file / f"{input_path.stem}_{args.target}_normalized.ply"
        
        process_single_file(input_path, target_file, args.target, not args.no_norm)
        
    elif input_path.is_dir():
        files = list(input_path.glob(f"*.{args.ext}"))
        if not files:
            print(f"在目录 {args.input} 中未找到 .{args.ext} 文件。")
            return
        
        print(f"在 {args.input} 找到 {len(files)} 个文件。开始批量处理...")
        for f in tqdm(files):
            # 保持子目录结构（如果有的话）
            rel_path = f.relative_to(input_path)
            # 修改文件名以示区别
            new_name = f"{f.stem}_{args.target}_normalized.ply"
            target_path = output_root / rel_path.parent / new_name
            
            process_single_file(f, target_path, args.target, not args.no_norm)
            
        print(f"\n批量任务处理完成！输出文件夹: {output_root}")
    else:
        print(f"错误: 找不到输入路径 {args.input}")

if __name__ == "__main__":
    main()
