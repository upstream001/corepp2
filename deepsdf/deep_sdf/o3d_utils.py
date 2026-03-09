import open3d as o3d
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import copy
import getpass

import os
from diskcache import FanoutCache


cache = FanoutCache(
    directory=os.path.join("/tmp", "fanoutcache_" + getpass.getuser() + "/"),
    shards=64,
    timeout=1,
    size_limit=3e11,
)


def visualize_sdf(sdf_data):
    xyz = sdf_data[:, :-1]
    val = sdf_data[:, -1]

    xyz_min = xyz[val<0] + np.array([0,0,0])
    xyz_max = xyz[val>0]

    val_min = val[val<0]
    val_max = val[val>0]

    val_min += np.min(val_min)
    val_min /= np.max(val_min)

    val_max -= np.min(val_max)
    val_max /= np.max(val_max)

    colors_min = np.zeros(xyz_min.shape)
    colors_min[:, 0] =  val_min

    colors_max = np.zeros(xyz_max.shape)
    colors_max[:, 2] =  val_max

    pcd_min = o3d.geometry.PointCloud()
    pcd_min.points = o3d.utility.Vector3dVector(xyz_min)
    pcd_min.colors = o3d.utility.Vector3dVector(colors_min)

    pcd_max = o3d.geometry.PointCloud()
    pcd_max.points = o3d.utility.Vector3dVector(xyz_max)
    pcd_max.colors = o3d.utility.Vector3dVector(colors_max)

    o3d.visualization.draw_geometries([pcd_max, pcd_min])

def sample_freespace(pcd, n_samples=20000):
    xyz = np.asarray(pcd.points)
    n_xyz = np.asarray(pcd.normals)

    xyz_free = []
    mu_free = []
    for _ in range(n_samples):

        mu_freespace = np.random.uniform(0.01, 0.1)
        idx = np.random.randint(len(xyz))

        p = xyz[idx]
        pn = n_xyz[idx]

        sample = p + mu_freespace*pn

        xyz_free.append(sample)
        mu_free.append(mu_freespace)

    # free = o3d.geometry.PointCloud()
    # free.points = o3d.utility.Vector3dVector(np.asarray(xyz_free))
    # free = free.paint_uniform_color([0,0.5,1])

    # frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([pcd, free, frame], point_show_normal=False, window_name='Free space')

    return np.asarray(xyz_free), np.asarray(mu_free)

def generate_pcd_from_virtual_depth(filename):
    """
    从完整点云模拟残缺观测（为草莓点云重写）。
    
    原版通过 Poisson 重建 → 虚拟深度渲染 → 提取部分点云，
    这对居中的草莓点云效果很差（法线、Poisson 精度、单视角等问题）。
    
    新版直接从完整点云出发，使用 Open3D hidden_point_removal 
    从随机视角模拟遮挡，得到"从某个角度看过去能看到的点"。
    """
    # 从 SDF 路径推断出原始完整点云路径
    raw_pcd_filename = filename.replace('laser/samples.npz', '').replace('fruit/', 'complete/')
    if raw_pcd_filename.endswith('/'):
        raw_pcd_filename = raw_pcd_filename[:-1]
    raw_pcd_filename += '.ply'
    
    pcd = o3d.io.read_point_cloud(raw_pcd_filename)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        return pcd
    
    # 随机生成一个视角方向（单位球面上均匀采样）
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0.3, np.pi - 0.3)  # 避免正上方/正下方
    view_dir = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # 将视点放在物体中心外足够远的地方
    centroid = points.mean(axis=0)
    camera_pos = centroid + view_dir * 2.0  # 距中心 2.0 的位置
    
    # 使用 hidden point removal 模拟该视角下的可见点
    try:
        # radius 参数控制投影球的半径，值越大保留的点越多
        _, visible_indices = pcd.hidden_point_removal(camera_pos, radius=100.0)
        partial_pcd = pcd.select_by_index(visible_indices)
    except Exception:
        # 如果 hidden_point_removal 失败，退化为随机保留 50-70% 的点
        keep_ratio = np.random.uniform(0.5, 0.7)
        n_keep = int(len(points) * keep_ratio)
        indices = np.random.choice(len(points), n_keep, replace=False)
        partial_pcd = pcd.select_by_index(indices)
    
    return partial_pcd

def generate_deepsdf_target(pcd, mu=0.001, align_with=np.array([0.0, 1.0, 0.0])):
    """
    从点云生成 DeepSDF 优化目标（pos/neg SDF 采样）。
    
    重写要点：
    1. 法线方向：使用质心外向检测，不再强制对齐到单一方向
    2. SDF 采样距离：与训练数据一致（正样本 0.04，负样本 0.01），
       不再使用原版极小的 mu=0.001
    3. 包含 free space 远场采样
    """
    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 使用质心外向法线（与 prepare_strawberry_sdf.py 保持一致）
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)
    normals = np.asarray(pcd.normals)
    directions_from_center = points - centroid
    dot_products = np.sum(normals * directions_from_center, axis=1)
    normals[dot_products < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    xyz = np.asarray(pcd.points)
    n_xyz = np.asarray(pcd.normals)
    
    # 与训练数据一致的 SDF 采样距离
    tsdf_positive = 0.04
    tsdf_negative = 0.01
    
    # 正样本：在法线方向随机偏移 [0, tsdf_positive]
    n_pts = len(xyz)
    pos_offsets = tsdf_positive * np.random.rand(n_pts, 1)
    xyz_pos = xyz + pos_offsets * n_xyz
    sdf_val_pos = pos_offsets.flatten()
    
    # 负样本：在法线反方向随机偏移 [0, tsdf_negative]
    neg_offsets = tsdf_negative * np.random.rand(n_pts, 1)
    xyz_neg = xyz - neg_offsets * n_xyz
    sdf_val_neg = -neg_offsets.flatten()

    # free space 远场采样
    xyz_free, sdf_val_free = sample_freespace(pcd)

    # 合并正样本和 free space
    xyz_pos = np.vstack((xyz_pos, xyz_free))
    sdf_val_pos = np.concatenate((sdf_val_pos, sdf_val_free))

    # 打包为 [x, y, z, sdf_val] 格式
    sdf_pos = np.empty((len(xyz_pos), 4))
    sdf_pos[:, :3] = xyz_pos
    sdf_pos[:, 3] = sdf_val_pos 

    sdf_neg = np.empty((len(xyz_neg), 4))
    sdf_neg[:, :3] = xyz_neg
    sdf_neg[:, 3] = sdf_val_neg

    return sdf_pos, sdf_neg

def read_depth_as_pcd(filename, pose=True):

    frame_id = int(filename.split('/')[-1][:-4])

    # getting depth
    depth = np.load(filename, allow_pickle=True)
    depth = o3d.geometry.Image(depth)

    # getting rgb
    image_path = filename.replace('depth', 'color') # os.path.join('/',*filename.split('/')[:-3])#, 'color/', filename.replace('npy', 'png'))
    color_file = image_path.replace('npy', 'png')
    mask_file = color_file.replace('color','masks')

    color = o3d.io.read_image(color_file)
    mask = cv.imread(mask_file, cv.IMREAD_GRAYSCALE) // 255
    rgb_np = np.asarray(color)
    rgb_np = np.copy(rgb_np[:,:,0:3])
    rgb_np[np.where(mask==0)] = 0
    color = o3d.geometry.Image(rgb_np)

    # getting pose
    invpose = np.eye(4)
    if pose:
        posesfilename = os.path.join('/',*filename.split('/')[:-3], 'tf/tf_allposes.npz')
        poses = np.load(posesfilename)
        invpose = np.linalg.inv(poses['arr_0'][frame_id-1])

    # getting bbox
    bbfilename = os.path.join('/',*filename.split('/')[:-3], 'tf/bounding_box.npz')
    bb_coordinates = np.load(bbfilename)['arr_0'] #* 1000

    bb = o3d.geometry.AxisAlignedBoundingBox()
    bb = bb.create_from_points(o3d.utility.Vector3dVector(bb_coordinates[:, :3]))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    # getting intrinsic
    intrinsicfilename = os.path.join('/',*filename.split('/')[:-2], 'intrinsic.json')
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsicfilename)
  
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale = 1000,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, invpose,
                                                          project_valid_depth_only=True)
    pcd_colors = np.asarray(pcd.colors)
    valid_mask = pcd_colors.sum(axis=1)
    pcd = pcd.select_by_index(np.where(valid_mask!=0)[0])
    pcd.translate(-pcd.get_center())
    pcd = pcd.crop(bb)
    return pcd

if __name__ == "__main__":
    filename = './data/cameralaser/peppers/p17/realsense/depth/00050.npy'
    read_depth_as_pcd(filename)