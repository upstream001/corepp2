#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算单个点云文件的体积 (使用 scipy.spatial.ConvexHull 算法)。
适用于评估 .ply 等点云文件的物理包围盒占据体积。
"""

import argparse
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import sys

def calculate_pcd_volume(ply_path, unit_scale=1.0):
    """
    计算给定点云的凸包体积。
    
    参数:
        ply_path: 点云文件路径
        unit_scale: 如果点云是以毫米(mm)为单位，需要转换为毫升(ml/cm³)，
                    可以使用系数。如果要从 mm³ -> ml，可除以 1000。
    
    返回:
        体积值
    """
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
    except Exception as e:
        print(f"[{ply_path}] 读取点云失败: {e}")
        return None
        
    points = np.asarray(pcd.points)
    
    if len(points) < 4:
        print(f"[{ply_path}] 警告: 点数少于4个，无法计算三维凸包！")
        return 0.0
        
    # 计算凸包
    try:
        hull = ConvexHull(points)
        volume = hull.volume * unit_scale
        
        # 简单输出物体的范围极值（用于对比分析实际长度）
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        spans = max_bounds - min_bounds
        
        print("-" * 50)
        print(f"分析文件: {ply_path}")
        print(f"包含点数: {len(points)}")
        print(f"三维跨度 (X, Y, Z): {spans[0]:.2f}, {spans[1]:.2f}, {spans[2]:.2f}")
        print(f"绝对凸包体积: {volume:.4f} (按照 unit_scale={unit_scale} 缩放后)")
        print("-" * 50)
        
        return volume
    except Exception as e:
        print(f"[{ply_path}] 计算凸包时发生错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="使用 ConvexHull 计算点云文件的体积")
    parser.add_argument("ply_file", type=str, help="要计算的 .ply 点云文件路径")
    parser.add_argument("--to_ml", action="store_true", help="如果模型坐标位是 mm，开启此项自动转换 mm³ 到 ml(毫升)")
    
    args = parser.parse_args()
    
    # mm³ 到 ml 的换算系数是 1 / 1000
    scale = 1/1000.0 if args.to_ml else 1.0
    
    calculate_pcd_volume(args.ply_file, unit_scale=scale)

if __name__ == "__main__":
    main()
