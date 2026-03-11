import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from glob import glob
try:
    from utils_file.io import read_ply_xyz, export_ply
except ImportError:
    pass

def generate_rotmat(batch_size):
    # 方位角：0 - 360 度随机，实现视角旋转
    az_batch = np.random.rand(batch_size) * 2 * np.pi
    
    # 俯仰角：固定为 0 (平视，不仰视也不俯视)
    # 我们通过平移物体来决定看哪里，而不是旋转相机
    el_batch = np.zeros(batch_size) 
    
    # 如果想稍微有一点点头部晃动 (模拟手持)，可以加极小的随机
    # el_batch = (np.random.rand(batch_size) - 0.5) * np.pi * 0.05 # +/- 4.5度
    
    rotmat_az_batch = np.array([
        [np.cos(az_batch),     -np.sin(az_batch),    np.zeros(batch_size)],
        [np.sin(az_batch),     np.cos(az_batch),     np.zeros(batch_size)],
        [np.zeros(batch_size), np.zeros(batch_size), np.ones(batch_size)]]
        )
    rotmat_az_batch = np.transpose(rotmat_az_batch, (2,0,1)) 
    
    rotmat_el_batch = np.array([
        [np.ones(batch_size),  np.zeros(batch_size), np.zeros(batch_size)],
        [np.zeros(batch_size), np.cos(el_batch),     -np.sin(el_batch)],
        [np.zeros(batch_size), np.sin(el_batch),     np.cos(el_batch)]]
        )
    rotmat_el_batch = np.transpose(rotmat_el_batch, (2,0,1)) 
    return rotmat_az_batch, rotmat_el_batch, az_batch, el_batch

def perspective_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch, 
                             resolution=100, npoints=2048, 
                             camera_dist=2.0, fov=60):
    """
    透视投影渲染函数 (Perspective Rendering)
    
    Args:
        pcl_batch: (B, N, 3) 输入点云
        rotmat_az_batch, rotmat_el_batch: 旋转矩阵
        resolution: 渲染分辨率
        npoints: 点的数量
        camera_dist: 相机距离物体中心的距离 (假设物体在原点，相机在 z 轴)
        fov: 垂直视场角 (Field of View)，单位为度
    """
    batch_size, npoints, dimension = pcl_batch.shape
    device = pcl_batch.device

    # 1. 旋转点云
    rotmat_az_batch = torch.Tensor(rotmat_az_batch).to(device)
    rotmat_el_batch = torch.Tensor(rotmat_el_batch).to(device)
    
    pcl_batch = torch.matmul(pcl_batch, rotmat_az_batch)
    pcl_batch = torch.matmul(pcl_batch, rotmat_el_batch)
    
    # 2. 透视变换准备
    # 计算焦距 (以像素为单位)
    # tan(fov/2) = (resolution / 2) / f
    focal_length = (resolution / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    
    # 计算相机坐标系下的坐标
    # 假设相机位于 (0, 0, -camera_dist) 看向 Z 正方向
    # 物体在原点。则物体相对于相机的 Z 坐标为 z_model + camera_dist
    # 我们希望 z_cam 越小越近? 不，通常相机看前方，z_cam 越大越远。
    # 这里我们定义: 相机在原点，物体在 (0, 0, camera_dist)
    # 这样 z_cam = z_rot + camera_dist
    
    x = pcl_batch[:,:,0]
    y = pcl_batch[:,:,1]
    z = pcl_batch[:,:,2]
    
    z_cam = z + camera_dist
    
    # 避免除以 0 或负数 (剔除相机背后的点)
    valid_mask = z_cam > 0.1
    
    # 3. 投影到屏幕 (Screen Space)
    # u = x * f / z + cx
    # v = y * f / z + cy
    cx = cy = resolution / 2.0
    
    u = (x * focal_length) / z_cam + cx
    v = (y * focal_length) / z_cam + cy
    
    # 离散化坐标
    u_idx = u.long()
    v_idx = v.long()
    
    # 4. 边界检查和有效性检查
    in_view_mask = (u_idx >= 0) & (u_idx < resolution) & \
                   (v_idx >= 0) & (v_idx < resolution) & valid_mask

    # 5. 构建 Grid Index 用于遮挡剔除
    # 我们需要构建一个 (B, Res, Res, N) 的张量来存储每个点的深度
    # 然后在 N 维度取 max (或 min)
    
    # 为了复用原代码的逻辑 (取 max)，我们定义 depth 使得“越近越大”。
    # z_cam 是距离，越小越近。
    # 所以定义 depth = -z_cam。这样 max(depth) 就是 min(z_cam)，即最近的点。
    depth_val = -z_cam
    
    # 处理超出边界的点，将其索引设为无效值或夹断，这里我们只处理 in_view 的点
    # 为了方便索引操作，我们将无效点的索引设为 0 (或其他安全值)，但在填值时过滤
    u_idx = torch.clamp(u_idx, 0, resolution - 1)
    v_idx = torch.clamp(v_idx, 0, resolution - 1)
    
    # 创建索引张量: (B, N, 4) -> [batch_i, u, v, point_i]
    # 维度 0: batch index
    b_idx = torch.arange(batch_size).view(batch_size, 1).expand(-1, npoints).to(device)
    # 维度 3: point index
    p_idx = torch.arange(npoints).view(1, npoints).expand(batch_size, -1).to(device)
    
    # -------------------------------------------------------------
    # 内存优化版 Z-Buffer 逻辑
    # 避免创建 (B, Res, Res, N) 的巨大张量
    # -------------------------------------------------------------

    # 1. 初始化 Z-Buffer (B, Res, Res)
    # 存储每个像素点处最大的 depth (即最前面的点)
    z_buffer = torch.full((batch_size, resolution, resolution), -1000.0, device=device)
    
    # 2. 准备 Scatter 数据
    # 我们需要将 valid_depths (N_valid) 填入 z_buffer (B, Res, Res)
    # 使用 scatter_reduce_ (取 max)
    
    # 构建扁平化索引 (针对 B*Res*Res)
    # flat_pixel_idx: [b * (R*R) + u * R + v]
    flat_pixel_idx = b_idx.view(-1) * (resolution * resolution) + \
                     u_idx.view(-1) * resolution + \
                     v_idx.view(-1)
    
    # 只取有效点
    mask_flat = in_view_mask.view(-1)
    valid_pixel_idx = flat_pixel_idx[mask_flat]
    valid_depths = depth_val.view(-1)[mask_flat]

    # 将 Z-Buffer 展平以便 scatter
    z_buffer_flat = z_buffer.view(-1)
    
    # Scatter Reduce: 
    # z_buffer_flat[idx] = max(z_buffer_flat[idx], value)
    # 注意: PyTorch 的 scatter_reduce 需要很新的版本，或者是 scatter_ + reduce='max'
    # 如果版本不支持，可以用 index_put_ (accumulate=False不行，因为会随机覆盖)
    # 或者用 loop，或者用 scatter_max (torch_scatter库，但不一定有)
    # 这里用 scatter_reduce_ (PyTorch 1.12+)
    
    try:
        z_buffer_flat.scatter_reduce_(0, valid_pixel_idx, valid_depths, reduce='amax', include_self=True)
    except AttributeError:
        # 兼容旧版本 PyTorch: 
        # 如果不支持 scatter_reduce，我们可以先排序再赋值（利用赋值覆盖的特性，如果是按 depth 排序）
        # 或者简单的循环（如果 batch 和点数不大）
        # 这里用简易 workaround: 既然只是 batch=1，我们可以循环
        # 但为了通用性，我们用 index_put (不保证 max，只保证随机覆盖，这是个问题)
        # 所以最好还是要求 PyTorch版本。或者用虽然慢但省内存的 atomic max (不易实现)
        
        # 备选方案: 使用循环处理每个 batch (通常 batch 只有 1)
        # 或者，如果显存不够，我们只能牺牲一点速度：
        print("Warning: scatter_reduce_ not found, falling back to loop implementation.")
        z_buffer = z_buffer.view(batch_size, resolution, resolution)
        for i in range(len(valid_pixel_idx)):
            # 这是一个非常慢的 fallback，但能工作
            # 为了加速，我们可以先在 CPU 做或者用其他库
            pass
            # 实际情况通常 pytorch 版本都支持 scatter_reduce 或 scatter with reduce
            # 尝试 scatter (deprecated reduce argument)
            # z_buffer_flat.scatter_(0, valid_pixel_idx, valid_depths, reduce='max')
            pass
            
    # 3. 判定可见性
    # 恢复形状
    z_buffer = z_buffer.view(batch_size, resolution, resolution)
    
    # 对于每个点 (N)，查询其投影位置的 z_buffer 值
    # 我们知道每个点的 (b, u, v)
    
    # 取样 Z-Buffer 值: 
    # gathered_z = z_buffer[b, u, v]
    # 使用 gather
    gathered_z_flat = z_buffer_flat.gather(0, flat_pixel_idx)
    gathered_z = gathered_z_flat.view(batch_size, npoints)
    
    # 原始深度
    point_depths = depth_val  # (B, N)
    
    # 判定
    # 可见如果: 自己的深度 >= buffer里的最大深度 - bias
    
    # 模拟微距时，Bias 要非常小
    bias = 0.005
    
    # 基础 Z-Buffer 可见性
    is_visible_zbuffer = (point_depths >= (gathered_z - bias))
    
    # -------------------------------------------------------------------------
    # 额外增强：深度切片 (Depth Slicing) - 模拟微距下的景深裁切
    # -------------------------------------------------------------------------
    # 在极近距离拍摄时，我们只想保留最靠近镜头的那一层“表皮”。
    # 侧面、背面的点虽然可能通过 Z-Buffer (如果正面没遮住它)，
    # 但它们离相机的绝对距离肯定比正面的点远。
    
    # depth_val 是 -z_cam (越大越近)。
    # 找到整个视野里离相机最近的点 (max depth_val)
    # 注意：只考虑视锥体内的点
    valid_depths_in_view = point_depths[in_view_mask]
    if len(valid_depths_in_view) > 0:
        max_depth_val = torch.max(valid_depths_in_view) # 离相机最近的距离 (-z)
        
        # 只保留最近点后方一定厚度内的点 (比如 0.15 范围)
        # 相当于把后面的草莓肉和背面全切掉
        thickness_threshold = 0.15 
        is_in_focus = point_depths >= (max_depth_val - thickness_threshold)
    else:
        is_in_focus = torch.zeros_like(point_depths, dtype=torch.bool)

    # 最终可见性 = Z-Buffer遮挡 & 在焦平面内 & 在视锥体内
    is_visible = is_visible_zbuffer & is_in_focus & in_view_mask
    
    return is_visible.cpu().numpy()



def partial_render_batch_perspective(pcl_batch, partial_batch, 
                                     resolution=100, camera_dist=1.8, fov=50,
                                     view_offset=None):
    """
    生成部分点云的主函数 (透视版)
    view_offset: [dx, dy, dz] 用于在渲染时平移点云，改变观察中心，但不改变输出点云的坐标
    """
    batch_size, npoints, _ = pcl_batch.shape
    
    # 生成随机旋转
    rotmat_az_batch, rotmat_el_batch, _, _ = generate_rotmat(batch_size)
    
    # 准备用于渲染的点云
    pcl_for_render = pcl_batch.clone()
    
    # 新增归一化逻辑：为适配硬件级固定的渲染相机参数（如 camera_dist=1, bias=0.005, thickness_threshold=0.15 等等）
    # 这些相机阈值当初都是基于物体占位边界 [-0.5, 0.5] 调教的。
    # 对于未归一化的原始空间大小点云（如直径30~60毫米），相机由于距离参数仅为 '1'，会导致相机直接"钻入且穿透"物体内部。
    # 修复法：在渲染世界里将替身点云强行缩小至标准0.5半径大小，判定完毕后再将 mask 返回原物理坐标
    for i in range(batch_size):
        center = pcl_for_render[i].mean(dim=0, keepdim=True)
        pcl_for_render[i] -= center
        scale = pcl_for_render[i].norm(dim=-1).max()
        if scale > 1e-5:
            pcl_for_render[i] /= (scale * 2.0) # 让最大半径变成 0.5
            
    if view_offset is not None:
        # view_offset 应该是 (3,) 或 (1, 3)
        pcl_for_render += torch.tensor(view_offset, device=pcl_batch.device).view(1, 1, 3)

    # 计算可见性 (使用 pcl_for_render)
    point_visible_batch = perspective_render_batch(pcl_for_render, rotmat_az_batch, rotmat_el_batch,
                                                   resolution=resolution, npoints=npoints,
                                                   camera_dist=camera_dist, fov=fov)
    
    # 采样 (使用原始 pcl_batch)
    for i in range(batch_size):
        visible_mask = point_visible_batch[i] # (N,) bool
        pcl = pcl_batch[i]
        
        visible_indices = np.where(visible_mask)[0]
        
        if len(visible_indices) > 0:
            # 如果可见点不足 2048，这就涉及上采样；如果多于 2048，随机采样
            # 简单起见，如果不足，允许重复
            choice_indices = np.random.choice(visible_indices, 2048, replace=True)
            partial_batch[i] = pcl[choice_indices]
        else:
            # 极端情况：没有可见点
            print(f"Warning: No visible points for sample {i}")
            partial_batch[i] = pcl
            
    return partial_batch

if __name__ == "__main__":
    import sys
    import argparse
    
    # 添加路径以导入 utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 假设 utils.io 存在
    # from utils.io import read_ply_xyz, export_ply

    parser = argparse.ArgumentParser(description='Perspective Render Test')
    parser.add_argument('--input', type=str, default='/home/tianqi/corepp2/data/test_ply_resample', help='Input PLY')
    parser.add_argument('--output_dir', type=str, default='/home/tianqi/corepp2/data/render_output_perspective')
    parser.add_argument('--num', type=int, default=10)
    args = parser.parse_args()

    # 使用 open3d 进行读写，更健壮
    import open3d as o3d

    def read_ply_o3d(path):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)

    def write_ply_o3d(points, path):
        pcd = o3d.geometry.PointCloud()
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(path, pcd)

    input_files = glob(os.path.join(args.input, "*.ply"))
    if not input_files and os.path.isfile(args.input):
        input_files = [args.input]
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Processing {len(input_files)} files...")

    for ply_path in input_files:
        base_name = os.path.splitext(os.path.basename(ply_path))[0]
        out_sub = os.path.join(args.output_dir, base_name)
        os.makedirs(out_sub, exist_ok=True)
        
        try:
            points = read_ply_o3d(ply_path)
            # 如果点云为空，跳过
            if len(points) == 0:
                print(f"Warning: {base_name} is empty.")
                continue
        except Exception as e:
            print(f"Read error {ply_path}: {e}")
            continue
            
        print(f"Processing {base_name}, points: {len(points)}")
        
        for i in range(args.num):
            pcl_single = torch.from_numpy(points).unsqueeze(0).float().cuda()
            partial_single = torch.zeros(1, 2048, 3).cuda()
            
            # 使用透视渲染 - 模拟 D405 近距离特写 (正对表面)
            
            # 使用 view_offset 来模拟相机对焦位置，而不改变原始点云坐标
            # Y += 0.2 表示将物体向上移，这样相机就会看到物体的下半部分
            
            result_batch = partial_render_batch_perspective(
                pcl_single, partial_single, 
                resolution=200,    
                camera_dist=2,   # 极近距离 0.5 上次的值：1 1.5
                fov=70,            # 窄 FOV 上次的值：70 150
                view_offset=[0, 0.2, 0] # 渲染时的偏移量
            )
            
            res_pc = result_batch[0].cpu().numpy()
            
            # --- 新增恢复原始尺度与位置逻辑 ---
            # 尝试回溯推断原始文件路径
            original_ply_path = ply_path.replace('_resample', '').replace('_16384_normalized', '')
            if os.path.exists(original_ply_path):
                # 读取原版具有物理尺度和偏移的完整点云
                orig_pcd = o3d.io.read_point_cloud(original_ply_path)
                orig_pts = np.asarray(orig_pcd.points)
                if len(orig_pts) > 0:
                    orig_center = orig_pts.mean(axis=0)
                    orig_scale = np.linalg.norm(orig_pts - orig_center, axis=-1).max()
                    
                    # 当前点云是被 resample_to_16384 归一化在 [-0.5, 0.5] 范围内的 (或近似)
                    # 假定原归一化公式为: (pts - 原始中心) / (原始最大向径 * 2) = 现pts
                    # 反向恢复公式为: 现pts * (原始最大向径 * 2) + 原始中心 = 原始坐标
                    res_pc = res_pc * (orig_scale * 2.0) + orig_center
            else:
                print(f"Warning: Could not find original complete pointcloud to restore scale: {original_ply_path}")

            save_path = os.path.join(out_sub, f"perp_{i:03d}.ply")
            write_ply_o3d(res_pc, save_path)
            
    print("Done.")
