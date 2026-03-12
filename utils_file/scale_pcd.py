#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="对文件夹中的点云进行体积缩放")
    parser.add_argument("input_dir", help="输入点云文件夹路径")
    parser.add_argument("output_dir", help="输出点云文件夹路径")
    parser.add_argument("--scale", type=float, default=100.0, help="体积缩放倍数，默认为 1000.0 (边长将放大约 10 倍)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    scale_factor = args.scale

    if not os.path.exists(input_dir):
        print(f"Error: 找不到输入文件夹 {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    ply_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.ply')]

    if not ply_files:
        print(f"在 {input_dir} 中没有找到任何 .ply 文件！")
        return

    # 根据体积倍数计算一维的边长缩放倍数（开三次方）
    linear_scale_factor = scale_factor ** (1/3)
    
    print(f"开始缩放点云...")
    print(f" -> 目标体积缩放倍数: {scale_factor} 倍")
    print(f" -> 实际应用到的边长(坐标)缩放倍数: {linear_scale_factor:.6f} 倍")
    
    for fname in tqdm(ply_files):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # 读取点云
        pcd = o3d.io.read_point_cloud(in_path)
        
        # 提取坐标并直接按比例相乘 (此操作保证以原点[0,0,0]为缩放中心，不会引入额外的平移量)
        points = np.asarray(pcd.points)
        scaled_points = points * linear_scale_factor
        pcd.points = o3d.utility.Vector3dVector(scaled_points)

        # 保存点云
        o3d.io.write_point_cloud(out_path, pcd)

    print(f"\n缩放处理完成！输出已保存在: {output_dir}")

if __name__ == "__main__":
    main()
