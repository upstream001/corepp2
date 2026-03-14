import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def ortho_render(pcl, rotmat_az, rotmat_el, resolution=50, npoints=2048, box_size=1.):
    pcl = torch.Tensor(pcl).cuda()
    rotmat_az = torch.Tensor(rotmat_az).cuda()
    rotmat_el = torch.Tensor(rotmat_el).cuda()
    pcl = torch.matmul(pcl, rotmat_az)
    pcl = torch.matmul(pcl, rotmat_el)
    depth = -box_size - pcl[:,2]
    grid_idx = (pcl[:,0:2] + box_size)/(2*box_size/resolution)
    grid_idx = grid_idx.long()
    grid_idx = torch.cat((grid_idx,torch.arange(npoints).view(npoints,-1).cuda()),1)
    grid_idx = grid_idx[:,0]*resolution*npoints + grid_idx[:,1]*npoints + grid_idx[:,2]
    plane_distance = torch.ones((resolution*resolution*npoints)).cuda() * -box_size*2
    plane_distance[grid_idx] = depth
    plane_distance = plane_distance.view(resolution,resolution,npoints)
    plane_depth,_ = torch.max(plane_distance,2)
    plane_mask = (plane_depth <= (-box_size * 2 + 1e-6))
    plane_mask = plane_mask.float() * box_size * 2
    plane_depth = plane_depth + plane_mask
    plane_depth -= box_size*2/50
    plane_depth = plane_depth.view(resolution,resolution,1)
    #print(plane_distance.shape)
    #print(plane_depth.shape)
    point_visible = (plane_distance >= plane_depth)
    point_visible, _ = torch.max(torch.max(point_visible.int(),0)[0],0)
    return point_visible.cpu().numpy()
def ortho_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch, resolution=100, npoints=2048, box_size=1.0):
    """
    超高效 Z-Buffer 版正交渲染：彻底解决 OOM，显存占用极低。
    """
    batch_size, npoints, _ = pcl_batch.shape
    device = pcl_batch.device
    
    # 1. 旋转点云映射到相机坐标系
    rotmat_az_batch = torch.Tensor(rotmat_az_batch).to(device)
    rotmat_el_batch = torch.Tensor(rotmat_el_batch).to(device)
    pcl_batch = torch.matmul(pcl_batch, rotmat_az_batch)
    pcl_batch = torch.matmul(pcl_batch, rotmat_el_batch)
    
    # 2. 计算投影坐标和深度 (-z 轴作为深度，越大越靠前)
    depth_val = -pcl_batch[:, :, 2]
    grid_xy = (pcl_batch[:, :, 0:2] + box_size) / (2 * box_size / resolution)
    u_idx = grid_xy[:, :, 0].long()
    v_idx = grid_xy[:, :, 1].long()
    
    # 获取视口内的点掩码
    in_view_mask = (u_idx >= 0) & (u_idx < resolution) & (v_idx >= 0) & (v_idx < resolution)
    
    # 3. 初始化 Z-Buffer (B, R, R) —— 显存占用极小
    z_buffer = torch.full((batch_size, resolution, resolution), -box_size * 2, device=device)
    
    # 构建扁平化像素索引用于索引 scatter
    b_idx = torch.arange(batch_size, device=device).view(batch_size, 1).expand(-1, npoints)
    flat_pixel_idx = (b_idx * (resolution * resolution) + u_idx * resolution + v_idx).view(-1)
    
    # 仅处理视口内的点
    mask_flat = in_view_mask.view(-1)
    valid_pixel_idx = flat_pixel_idx[mask_flat]
    valid_depths = depth_val.view(-1)[mask_flat]
    
    # 核心：使用 scatter_reduce 完成深度竞争（只保留最前面的点）
    z_buffer_flat = z_buffer.view(-1)
    if len(valid_pixel_idx) > 0:
        z_buffer_flat.scatter_reduce_(0, valid_pixel_idx, valid_depths, reduce='amax', include_self=True)
    
    # 4. 判定可见性
    import torch.nn.functional as F
    z_buffer = z_buffer.view(batch_size, 1, resolution, resolution)
    
    # 两次膨胀以填补离散点云之间的空隙，防止背面点漏光
    z_buffer = F.max_pool2d(z_buffer, kernel_size=3, stride=1, padding=1)
    z_buffer = F.max_pool2d(z_buffer, kernel_size=3, stride=1, padding=1)
    
    z_buffer_flat = z_buffer.view(-1)
    
    # 为每个原始点查询对应像素位置的遮盖深度
    sampling_idx = (b_idx * resolution * resolution + 
                    torch.clamp(u_idx, 0, resolution-1) * resolution + 
                    torch.clamp(v_idx, 0, resolution-1)).view(-1)
    
    gathered_z = z_buffer_flat.gather(0, sampling_idx).view(batch_size, npoints)
    
    # 判定布尔可见性 (将容差从 0.05 提升至 0.15，允许捕捉更多侧面/边缘的点)
    is_visible = (depth_val >= (gathered_z - 0.15)) & in_view_mask
    
    return is_visible.cpu().numpy()


def partial_render_batch(pcl_batch, partial_batch, resolution=100, box_size=1.0):
    """
    生成部分点云的主函数。
    核心原则：使用临时归一化的‘替身’计算遮挡，使用原始坐标的‘本体’采样输出。
    """
    batch_size, npoints, dimension = pcl_batch.shape 
    rotmat_az_batch, rotmat_el_batch, az_batch, el_batch = generate_rotmat(batch_size)
    
    az_batch, el_batch = az_batch.reshape(batch_size, 1), el_batch.reshape(batch_size, 1)
    azel_batch = np.concatenate([az_batch, el_batch], 1)
    
    # 1. 准备归一化替身进行可见性判定
    pcl_for_render = pcl_batch.clone()
    for i in range(batch_size):
        c = pcl_for_render[i].mean(dim=0)
        pcl_for_render[i] -= c
        s = pcl_for_render[i].norm(dim=-1).max()
        if s > 1e-6:
            # 将物体强制放入 [-0.5, 0.5] 视窗内，防止座標越界崩溃
            pcl_for_render[i] /= (s * 2.0) 
    
    # 2. 计算可见性 (此时所有渲染参数基于 box_size=1.0 这个标准范围)
    point_visible_batch_res = ortho_render_batch(
        pcl_for_render, rotmat_az_batch, rotmat_el_batch, 
        resolution=resolution, npoints=npoints, box_size=1.0
    )
    point_visible_batch = point_visible_batch_res.reshape(batch_size, npoints, 1)
    
    # 3. 采样原始物理坐标点
    for i in range(batch_size):
        point_visible = point_visible_batch[i, :, :]
        pcl = pcl_batch[i, :, :] # 原始物体点
        
        point_visible_idx, _ = np.where(point_visible > 0.5)
        
        if len(point_visible_idx) > 0:
            # 随机选择 2048 个可见点
            choice_indices = np.random.choice(point_visible_idx, 2048, replace=True)
            new_pcl = pcl[choice_indices]
            partial_batch[i, :, :] = new_pcl
        else:
            # 兜底：如果没看出来可见点，返回原点
            print(f"Warning: No visible points found for sample {i}")
            partial_batch[i, :, :] = pcl[:2048]
            
    return partial_batch, rotmat_az_batch, rotmat_el_batch, azel_batch

def generate_rotmat(batch_size):
    # 方位角
    az_batch = np.random.rand(batch_size)
    az_batch = az_batch * 2 * np.pi

    # az_batch = np.zeros(batch_size)
    # 俯仰角
    el_batch = np.random.rand(batch_size)
    # el_batch = (el_batch - 0.5) * np.pi*0.4 # [-0.5, 0.5] * 0.5π -> [-0.25π, 0.25π]
    # 俯仰角 (仰角)：放宽范围以模拟更多变的拍摄角度
    # 例如让相机在 [0, 80] 度之间随机，即从侧面到近乎顶部的范围
    min_angle = 0 / 180 * np.pi
    max_angle = 80 / 180 * np.pi
    el_batch = np.random.rand(batch_size) * (max_angle - min_angle) + min_angle

    rotmat_az_batch = np.array([
        [np.cos(az_batch),     -np.sin(az_batch),    np.zeros(batch_size)],
        [np.sin(az_batch),     np.cos(az_batch),     np.zeros(batch_size)],
        [np.zeros(batch_size), np.zeros(batch_size), np.ones(batch_size)]]
        )
    #print(rotmat_az_batch.shape)
    rotmat_az_batch = np.transpose(rotmat_az_batch, (2,0,1)) 
    rotmat_el_batch = np.array([
        [np.ones(batch_size),  np.zeros(batch_size), np.zeros(batch_size)],
        [np.zeros(batch_size), np.cos(el_batch),     -np.sin(el_batch)],
        [np.zeros(batch_size), np.sin(el_batch),     np.cos(el_batch)]]
        )
    rotmat_el_batch = np.transpose(rotmat_el_batch, (2,0,1)) 
    return rotmat_az_batch, rotmat_el_batch, az_batch, el_batch


# if __name__ == "__main__":
    # def visualize_pc(points):
    #     xs = points[:,0]
    #     ys = points[:,1]
    #     zs = points[:,2]
    #     fig = plt.figure()
    #     ax=fig.add_subplot(projection='3d')

    #     ax.scatter(xs,ys,zs,marker='o')
    #     ax.set_xlim(-0.5,0.5)
    #     ax.set_ylim(-0.5,0.5)
    #     ax.set_zlim(-0.5,0.5)
    #     plt.show()
    # pcl1 = np.load("datasets/ModelNet40_Completion/table/train/table_0294_3_complete.npy")
    # pcl2 = np.load("datasets/ModelNet40_Completion/table/train/table_0290_3_complete.npy")
    # pcl_batch = np.concatenate((pcl1.reshape(1,2048,3),pcl2.reshape(1,2048,3)),axis=0)
    # pcl_batch = np.repeat(pcl_batch, 5, axis=0)
    # #ys=pcl1[:,1].copy()
    # #zs=pcl1[:,2].copy()
    # #pcl1[:,1]=zs
    # #pcl1[:,2]=ys
    # #visualize_pc(pcl1)
    # #pcl_batch = pcl1.resize(1,2048,3)#xzy
    # #az = 1.
    # #el = 0.5
    # #rotmat_az = np.array([
    # #    [np.cos(az), -np.sin(az), 0],
    # #    [np.sin(az), np.cos(az),  0],
    # #    [0,          0,           1]]
    # #    )
    # #rotmat_el = np.array([
    # #    [1, 0, 0],
    # #    [0, np.cos(el), -np.sin(el)],
    # #    [0, np.sin(el), np.cos(el)]]
    # #    )
    # #rotmat_az_batch = rotmat_az.reshape(1,3,3).repeat(2,axis=0)
    # #rotmat_el_batch = rotmat_el.reshape(1,3,3).repeat(2,axis=0)
    # #point_visible = ortho_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch).reshape(2,2048,1)[1]
    # #depth = ortho_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch)[0]
    # #print(pcl1.shape)
    # #pcl1 = np.matmul(pcl1, rotmat_az)
    # #pcl1 = np.matmul(pcl1, rotmat_el)
    # #visualize_pc(pcl1)
    # #print(rotmat_az)
    # #plt.imshow(depth)
    # #plt.show()
    # #print(np.sum(point_visible))
    # #point_visible_idx, _ = np.where(point_visible > 0.5)
    # #point_visible_idx = np.random.choice(point_visible_idx, 2048)
    # #pcl2 = pcl2[point_visible_idx]
    # #print(pcl2.shape)
    # pcl_batch = torch.Tensor(pcl_batch).cuda()
    # batch_size, npoints, dimension = pcl_batch.shape
    # partial_batch = torch.Tensor(np.zeros((batch_size,npoints,dimension))).cuda()

    # new_pcl_batch, _, _, _ = partial_render_batch(pcl_batch, partial_batch)

    # new_pcl_batch = new_pcl_batch.cpu().numpy()
    # for i in range(10):
    #     new_pcl = new_pcl_batch[i]
    #     visualize_pc(new_pcl)

if __name__ == "__main__":
    import sys
    import os
    import argparse
    from glob import glob
    # 将项目根目录添加到 sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils_file.io import read_ply_xyz, export_ply

    parser = argparse.ArgumentParser(description='Realtime Render Test')
    parser.add_argument('--input', type=str, default='/home/tianqi/corepp2/data/scanned_straw_meshed_resize_ml', help='Input PLY file or directory')
    parser.add_argument('--vis', action='store_true', help='Visualize the results')
    parser.add_argument('--save', action='store_true', default=True, help='Save the results to disk')
    parser.add_argument('--output_dir', type=str, default='data/render_output', help='Directory to save results')
    parser.add_argument('--num', type=int, default=10, help='Number of partial point clouds to generate per input file')
    args = parser.parse_args()

    def visualize_pc(points, title="Point Cloud"):
        xs = points[:,0]
        ys = points[:,1]
        zs = points[:,2]
        fig = plt.figure()
        ax=fig.add_subplot(projection='3d')

        ax.scatter(xs,ys,zs,marker='o', s=1) # s=1 for better visibility
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(-0.5,0.5)
        plt.title(title)
        plt.show()

    # 1. 确定输入文件列表
    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        input_files = glob(os.path.join(args.input, "*.ply"))
        input_files.sort()
    else:
        print(f"Error: Input {args.input} not found.")
        sys.exit(1)

    if not input_files:
        print(f"No .ply files found in {args.input}")
        sys.exit(0)

    print(f"Total input files to process: {len(input_files)}")

    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")

    # 2. 循环处理每个输入文件
    for file_idx, ply_path in enumerate(input_files):
        base_name = os.path.splitext(os.path.basename(ply_path))[0]
        print(f"\n[{file_idx+1}/{len(input_files)}] Processing: {base_name}")
        
        # 为每个输入文件创建独立的子文件夹
        if args.save:
            file_output_dir = os.path.join(args.output_dir, base_name)
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)
                print(f"  Created subfolder: {file_output_dir}")
        
        # 加载点云
        try:
            points = read_ply_xyz(ply_path)
        except Exception as e:
            print(f"Failed to read {ply_path}: {e}")
            continue
            
        print(f"Loaded {len(points)} points.")

        # 串行生成以节省显存
        for i in range(args.num):
            # 准备单样本 Batch [1, N, 3]
            pcl_single = torch.from_numpy(points).unsqueeze(0).float().cuda()
            partial_single = torch.zeros(1, 2048, 3).cuda()

            # 执行渲染生成
            # batch_size 为 1
            new_pcl_single, _, _, _ = partial_render_batch(pcl_single, partial_single)
            
            # 转换为 numpy
            res_pc = new_pcl_single[0].cpu().numpy()

            # 保存
            if args.save:
                # 存入对应的子文件夹
                save_path = os.path.join(file_output_dir, f"part_{i:03d}.ply")
                export_ply(res_pc, save_path)
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  Generated {i+1}/{args.num} samples...")

            # 可视化 (仅第一个文件的第一个视角)
            if args.vis and file_idx == 0 and i == 0:
                print(f"Visualizing first sample...")
                visualize_pc(res_pc, title=f"Partial: {base_name}")
            
            # 释放显存
            del pcl_single, partial_single, new_pcl_single
            torch.cuda.empty_cache()

    print(f"\nAll tasks completed. Results saved to {args.output_dir}")
