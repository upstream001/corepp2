#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import open3d as o3d
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="同时渲染并对比同一空间下的残缺点云与完整点云，左为残缺，右为完整。")
    parser.add_argument("partial_dir", help="残缺点云文件夹路径")
    parser.add_argument("complete_dir", help="完整点云文件夹路径")
    args = parser.parse_args()

    partial_dir = args.partial_dir
    complete_dir = args.complete_dir

    if not os.path.exists(partial_dir):
        print(f"Error: 找不到残缺文件夹 {partial_dir}")
        return
    if not os.path.exists(complete_dir):
        print(f"Error: 找不到完整文件夹 {complete_dir}")
        return

    # 读取文件列表并按名称排序
    partial_files = [f for f in sorted(os.listdir(partial_dir)) if f.endswith('.ply')]

    if not partial_files:
        print(f"在 {partial_dir} 中没有找到任何 .ply 文件！")
        return

    print("操作提示：")
    print(" - 鼠标左键拖动：旋转")
    print(" - 鼠标滚轮：缩放")
    print(" - 鼠标右键拖动：平移")
    print(" - 按下键盘 [N] 键显示下一组数据")
    print("-" * 50)

    for i, fname in enumerate(partial_files):
        partial_path = os.path.join(partial_dir, fname)
        complete_path = os.path.join(complete_dir, fname)

        if not os.path.exists(complete_path):
            print(f"[Warning] 找不到对应的完整点云文件，跳过: {fname}")
            continue

        print(f"正在显示 ({i+1}/{len(partial_files)}): {fname}")

        pcd_partial = o3d.io.read_point_cloud(partial_path)
        pcd_complete = o3d.io.read_point_cloud(complete_path)
        
        # 判断并获取包围盒来计算安全平移距离
        bbox_p = pcd_partial.get_axis_aligned_bounding_box()
        bbox_c = pcd_complete.get_axis_aligned_bounding_box()
        
        extent_p = bbox_p.get_extent()
        extent_c = bbox_c.get_extent()
        
        # 以它们 X 轴最大跨度再加一点点余量作为平移距离，确保左右不会重叠
        max_extent_x = max(extent_p[0], extent_c[0])
        if max_extent_x == 0:
            max_extent_x = 0.5
            
        offset = np.array([max_extent_x * 1.5, 0, 0])

        # 将残缺模型向负X轴（左边）平移，完整模型向正X轴（右边）平移
        pcd_partial.translate(-offset / 2)
        pcd_complete.translate(offset / 2)

        # 默认给它们着色以更方便区分（残缺：偏红，完整：偏绿）
        # 如果你希望保留它们本来的颜色，把下面这两行注释掉即可
        pcd_partial.paint_uniform_color([0.9, 0.2, 0.2])
        pcd_complete.paint_uniform_color([0.2, 0.8, 0.2])

        # 在统一窗口中展示两个点云
        def next_view(vis):
            vis.close()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"左: Partial (红) | 右: Complete (绿) - {fname}", width=1280, height=720, left=100, top=100)
        vis.add_geometry(pcd_partial)
        vis.add_geometry(pcd_complete)
        # 注册按键 'N' (ASCII: 78) 来关闭当前窗口进而走向循环的下一项
        vis.register_key_callback(ord('N'), next_view)
        
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    main()
