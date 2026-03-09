import open3d as o3d
import os
import argparse
import glob
import numpy as np

def load_point_cloud(file_path):
    """
    加载点云文件，支持 .ply, .pcd, .npz
    """
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.npz':
        data = np.load(file_path)
        # DeepSDF样本文件包含 'pos' 和 'neg'，形状都是 (N, 4)，前三维是坐标
        pos_pts = data['pos'][:, :3]
        neg_pts = data['neg'][:, :3]
        
        # 为了可视化效果更好，我们可以按比例抽样，因为点通常非常多(10W+)
        sample_rate = 10
        pos_pts = pos_pts[::sample_rate]
        neg_pts = neg_pts[::sample_rate]
        
        pcd_pos = o3d.geometry.PointCloud()
        pcd_pos.points = o3d.utility.Vector3dVector(pos_pts)
        pcd_pos.paint_uniform_color([0, 1, 0]) # 正距离(外部)为绿色
        
        pcd_neg = o3d.geometry.PointCloud()
        pcd_neg.points = o3d.utility.Vector3dVector(neg_pts)
        pcd_neg.paint_uniform_color([1, 0, 0]) # 负距离(内部)为红色
        
        # 将内外部点云合并以便后续统一处理
        pcd = pcd_pos + pcd_neg
        return pcd
    else:
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

class PointCloudVisualizer:
    def __init__(self, file_list):
        self.file_list = sorted(file_list)
        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # 创建坐标轴
        self.axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
    def update_visualizer(self):
        if self.index >= len(self.file_list):
            print("已经到达最后一个文件。")
            self.vis.close()
            return

        file_path = self.file_list[self.index]
        print(f"正在显示 ({self.index + 1}/{len(self.file_list)}): {file_path}")
        
        pcd = load_point_cloud(file_path)
        if pcd.is_empty():
            print(f"警告: 文件 {file_path} 为空。")
            self.next_callback(self.vis)
            return

        self.vis.clear_geometries()
        
        _, ext = os.path.splitext(file_path)
        if ext.lower() != '.npz':
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        # 计算最远点坐标
        points = np.asarray(pcd.points)
        if len(points) > 0:
            dists = np.linalg.norm(points, axis=1)
            max_idx = np.argmax(dists)
            farthest_point = points[max_idx]
            print(f"      点数: {len(points)} | 最远点坐标: {farthest_point} | 最远距离: {dists[max_idx]:.4f}")

        self.vis.add_geometry(pcd)
        self.vis.add_geometry(self.axis_frame) # 添加坐标轴
        self.vis.reset_view_point(True)

    def next_callback(self, vis):
        self.index += 1
        if self.index < len(self.file_list):
            self.update_visualizer()
        else:
            print("所有文件已查看完毕。")
            self.vis.close()

    def run(self):
        if not self.file_list:
            print("没有找到符合条件的点云文件。")
            return

        self.vis.create_window(window_name="点云批量可视化 (按 N 下一个, Q 退出)", width=1280, height=720)
        
        # 注册按键回调
        # 'N' 键的 GLFW 键值通常是 78
        self.vis.register_key_callback(ord('N'), self.next_callback)
        self.vis.register_key_callback(ord(' '), self.next_callback) # 空格也可以
        
        self.update_visualizer()
        self.vis.run()
        self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="批量可视化目录中的点云文件")
    parser.add_argument("dir", default="/home/tianqi/PoinTr/data/straw_test3/partial", type=str, help="点云文件夹路径")
    parser.add_argument("--ext", default="ply", type=str, help="文件扩展名 (如 ply, pcd, npz)")
    
    args = parser.parse_args()
    
    # 默认搜索所有支持的扩展名
    if args.ext:
        extensions = [args.ext]
    else:
        extensions = ["ply", "pcd", "npz"]
        
    file_list = []
    for ext in extensions:
        search_pattern = os.path.join(args.dir, "**", f"*.{ext}")
        file_list.extend(glob.glob(search_pattern, recursive=True))
        
    if not file_list:
        print(f"在目录 {args.dir} 及其子目录中未找到扩展名为 {', '.join(extensions)} 的文件。")
        return

    visualizer = PointCloudVisualizer(file_list)
    visualizer.run()

if __name__ == "__main__":
    main()
