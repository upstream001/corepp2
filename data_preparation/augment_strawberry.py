#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import open3d as o3d
import numpy as np
import copy
import json
from glob import glob
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="专为草莓数据集独立扩增点云 (仅扩增坐标尺寸/形变，不管真值 SDF)")
    parser.add_argument("--json_config", default="data_preparation/augment.json", help="json filename with the parameters for augmentation")
    parser.add_argument("--src", default="/home/tianqi/corepp2/data/20260301_dataset", help="源数据集路径")
    parser.add_argument("--dst", default="/home/tianqi/corepp2/data/20260301_dataset_aug", help="输出增强数据集的路径")
    
    args = parser.parse_args()

    if not os.path.exists(args.json_config):
        print(f"配置文件 {args.json_config} 不存在！")
        return

    with open(args.json_config) as json_file:
        config = json.load(json_file)

    src_complete = os.path.join(args.src, "complete")
    dst_complete = os.path.join(args.dst, "complete")
    os.makedirs(dst_complete, exist_ok=True)

    src_partial = os.path.join(args.src, "partial")
    dst_partial = os.path.join(args.dst, "partial")
    os.makedirs(dst_partial, exist_ok=True)

    ply_files = glob(os.path.join(src_complete, "*.ply"))
    if not ply_files:
        print(f"未在 {src_complete} 找到 .ply 文件！")
        return

    for ply_path in tqdm(ply_files, desc="Augmenting point clouds"):
        fname = os.path.basename(ply_path)
        fid = fname[:-4]
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # 尝试读取对应的 partial 点云 (如果存在的话)
        partial_path = os.path.join(src_partial, fname)
        has_partial = os.path.exists(partial_path)
        if has_partial:
            pcd_partial = o3d.io.read_point_cloud(partial_path)

        # 始终保存一个完全未经修改的原始版本 (00 编号)
        o3d.io.write_point_cloud(os.path.join(dst_complete, f"{fid}_aug_00.ply"), pcd)
        if has_partial:
            o3d.io.write_point_cloud(os.path.join(dst_partial, f"{fid}_aug_00.ply"), pcd_partial)

        for jdx in range(1, config['no_of_augmentations']):
            tmp = copy.deepcopy(pcd)
            if has_partial:
                tmp_p = copy.deepcopy(pcd_partial)
            
            # --- 构建统一的变换矩阵 T ---
            # 1. Scale 缩放
            I = np.eye(4)
            scale = np.random.uniform(config['min_scalefactor'], config['max_scalefactor'], size=(1, 4)) 
            scale[0, -1] = 1 # 齐次坐标位保持 1
            T = scale * I

            # 2. Rotation around Z 绕 Z 轴旋转
            angle = np.random.uniform(-config['max_rotation_angle_degree'] * np.pi/180.0,
                                      +config['max_rotation_angle_degree'] * np.pi/180.0)
            R = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray([[0, 0, angle]]).T)
            T_R = np.eye(4)
            T_R[0:3, 0:3] = R
            T = T_R @ T
            
            # 3. Shear 倾斜形变
            shear = np.random.uniform(-config['max_shear'], +config['max_shear'], size=(2,)) 
            T_shear = np.eye(4)
            T_shear[0, 1] = shear[0]
            T_shear[0, 2] = shear[1]
            T = T_shear @ T

            # 施加同一套矩阵 T
            tmp.transform(T)
            if has_partial:
                 tmp_p.transform(T)
            
            out_name = f"{fid}_aug_{jdx:02d}.ply"
            o3d.io.write_point_cloud(os.path.join(dst_complete, out_name), tmp)
            if has_partial:
                 o3d.io.write_point_cloud(os.path.join(dst_partial, out_name), tmp_p)

    # --- 这里新增自动生成新的 mapping.json 以备需要 ---
    print("正在生成增强数据集专用的 mapping.json...")
    src_mapping_path = os.path.join(args.src, "mapping.json")
    dst_mapping_path = os.path.join(args.dst, "mapping.json")
    if os.path.exists(src_mapping_path):
        with open(src_mapping_path, 'r') as f:
            old_mapping = json.load(f)
        new_mapping = {}
        for old_k, old_v in old_mapping.items():
            base_id = old_k[:-4]  # 去掉 .ply
            for j in range(config['no_of_augmentations']):
                new_key = f"{base_id}_aug_{j:02d}.ply"
                # 用户要求所有形变变种依然对应到最原生的完整精模名称 (如 sw9_mesh_16384_normalized.ply)
                new_mapping[new_key] = old_v
        with open(dst_mapping_path, 'w') as f:
            json.dump(new_mapping, f, indent=4)
    else:
        print(f"Warning: 原目录中未找到 {src_mapping_path}，已跳过新映射表生成。")

    print(f"\n全部点云增广完成！扩增数据已被释放到：{args.dst}")

if __name__ == '__main__':
    main()
