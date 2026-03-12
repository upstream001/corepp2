import open3d as o3d
import argparse
import numpy as np
import os

def load_point_cloud(file_path):
    """
    加载点云文件，支持 .ply, .pcd, .npy 格式
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension in ['.pcd', '.ply']:
        pcd = o3d.io.read_point_cloud(file_path)
    elif file_extension == '.npy':
        pts = np.load(file_path)
        # 如果是 (B, N, 3) 形状，取第一个 batch
        if pts.ndim == 3:
            pts = pts[0]
        # 如果是 (N, 3) 形状
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")
    return pcd

def visualize(files, offset=False):
    pcds = []
    
    # 颜色调色板
    colors = [
        [1, 0, 0],       # 红色
        [0, 1, 0],       # 绿色
        [0, 0, 1],       # 蓝色
        [1, 0.706, 0],   # 橙色/黄色
        [0, 0.651, 0.929], # 青色/蓝色
        [1, 0, 1],       # 紫红色
        [0, 1, 1],       # 青色
    ]

    total_geometries = []
    current_x_offset = 0.0
    
    print(f"正在显示 {len(files)} 个点云:")
    
    for i, file_path in enumerate(files):
        try:
            pcd = load_point_cloud(file_path)
            if pcd.is_empty():
                print(f"警告: 点云文件 {file_path} 为空，跳过。")
                continue
            
            color = colors[i % len(colors)]
            pcd.paint_uniform_color(color)
            
            # 打印点信息
            print(f"  [{i}] 文件: {file_path}")
            print(f"      颜色: {color} | 点数: {len(pcd.points)}")
            
            if offset:
                # 累加偏移
                pcd.translate([current_x_offset, 0, 0])
                bbox = pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                current_x_offset += extent[0] * 1.2
            
            total_geometries.append(pcd)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    if not total_geometries:
        print("未成功加载任何点云。")
        return

    if offset:
        print("已启用偏移: 并排显示。")
    else:
        print("未启用偏移: 重叠显示。")

    print("\n操作提示: 使用鼠标左键旋转，右键平移，滚轮缩放。按 'q' 键退出。")

    # 创建坐标轴
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    total_geometries.append(axis_frame)

    # 启动可视化窗口
    o3d.visualization.draw_geometries(total_geometries, 
                                      window_name="PoinTr 多点云可视化",
                                      width=1280, 
                                      height=720,
                                      left=50, 
                                      top=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoinTr 项目多点云可视化脚本")
    # 支持传入多个文件
    parser.add_argument("files", nargs="*", help="点云文件路径列表 (.ply, .pcd, .npy)")
    
    # 为了向后兼容，保留 --file1 和 --file2 但标记为可选
    parser.add_argument("--file1", type=str, help="第一个点云文件路径")
    parser.add_argument("--file2", type=str, help="第二个点云文件路径")
    parser.add_argument("--offset", action="store_true", help="是否并排显示点云 (默认重叠显示)")
    
    args = parser.parse_args()
    
    # 汇总所有输入文件
    input_files = []
    if args.files:
        input_files.extend(args.files)
    if args.file1:
        input_files.append(args.file1)
    if args.file2:
        input_files.append(args.file2)
        
    # 如果没有任何输入，使用默认值（用于快速测试）
    if not input_files:
        print("未指定文件，使用默认示例文件...")
        input_files = [
            "/home/tianqi/corepp2/data/render_output_perspective/sw2_mesh_16384_normalized/perp_000.ply",
            "/home/tianqi/corepp2/data/scanned_straw_meshed_resize_ml/sw2_mesh_16384_normalized.ply",
            ""
        ]
    
    visualize(input_files, args.offset)
