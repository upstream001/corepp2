import open3d as o3d
import os
import argparse
import glob
import numpy as np

def load_geometry(file_path):
    """
    加载几何文件，优先尝试作为 Mesh 加载，如果没面片则作为点云加载。
    """
    _, ext = os.path.splitext(file_path)
    
    # 特殊处理 .npz (DeepSDF 采样点)
    if ext.lower() == '.npz':
        data = np.load(file_path)
        pos_pts = data['pos'][:, :3]
        neg_pts = data['neg'][:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate([pos_pts, neg_pts], axis=0))
        colors = np.zeros((len(pcd.points), 3))
        colors[:len(pos_pts)] = [0, 1, 0] # 绿
        colors[len(pos_pts):] = [1, 0, 0] # 红
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # 尝试加载为 Mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.has_triangles():
        # 如果没有法线，计算法线以便正确着色
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7]) # 给点高级感的灰色
        return mesh
    
    # 如果不是 Mesh，作为点云加载
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_colors():
        pcd.paint_uniform_color([0, 0.6, 0.9]) # 默认蓝色
    return pcd

class MeshVisualizer:
    def __init__(self, file_list):
        self.file_list = sorted(file_list)
        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.current_geometry = None

    def show_current(self):
        if self.current_geometry:
            self.vis.remove_geometry(self.current_geometry, reset_bounding_box=False)
        
        file_path = self.file_list[self.index]
        print(f"\n正在重建并显示 ({self.index + 1}/{len(self.file_list)}): {file_path}")
        
        self.current_geometry = load_geometry(file_path)
        
        # 打印简单统计
        if isinstance(self.current_geometry, o3d.geometry.TriangleMesh):
            print(f"      类型: Mesh | 顶点数: {len(self.current_geometry.vertices)} | 面片数: {len(self.current_geometry.triangles)}")
        else:
            print(f"      类型: PointCloud | 点数: {len(self.current_geometry.points)}")

        self.vis.add_geometry(self.current_geometry, reset_bounding_box=(self.index == 0))
        
        # 更新标题
        self.vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1]) # 深色背景更有质感
        self.vis.get_render_option().mesh_show_back_face = True

    def next_callback(self, vis):
        self.index = (self.index + 1) % len(self.file_list)
        self.show_current()

    def prev_callback(self, vis):
        self.index = (self.index - 1) % len(self.file_list)
        self.show_current()

    def run(self):
        if not self.file_list:
            print("没有找到可显示的文件。")
            return

        self.vis.create_window(window_name="3D 几何可视化 (N:下一个, P:上一个, Q:退出)", width=1280, height=720)
        self.vis.add_geometry(self.axis_frame)
        
        # 绑定快捷键
        self.vis.register_key_callback(ord("N"), self.next_callback)
        self.vis.register_key_callback(ord("P"), self.prev_callback)
        self.vis.register_key_callback(ord("Q"), lambda vis: vis.destroy_window())

        self.show_current()
        self.vis.run()
        self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="支持 Mesh 和点云的批量可视化工具")
    parser.add_argument("dir", help="包含几何文件的目录")
    parser.add_argument("--ext", default=None, help="指定扩展名 (如 ply, obj, npz)")
    args = parser.parse_args()

    # 搜索文件
    if args.ext:
        extensions = [args.ext]
    else:
        extensions = ["ply", "obj", "npz"]
    
    file_list = []
    for ext in extensions:
        search_pattern = os.path.join(args.dir, "**", f"*.{ext}")
        file_list.extend(glob.glob(search_pattern, recursive=True))

    if not file_list:
        print(f"在目录 {args.dir} 中未找到相关文件。")
        return

    visualizer = MeshVisualizer(file_list)
    visualizer.run()

if __name__ == "__main__":
    main()
