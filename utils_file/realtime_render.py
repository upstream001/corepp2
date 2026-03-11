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
def ortho_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch, resolution=100, npoints=2048, box_size=1):
    batch_size, npoints, dimension = pcl_batch.shape
    rotmat_az_batch = torch.Tensor(rotmat_az_batch).cuda()#B*3*3
    rotmat_el_batch = torch.Tensor(rotmat_el_batch).cuda()#B*3*3
    pcl_batch = torch.matmul(pcl_batch, rotmat_az_batch)#B*N*3
    pcl_batch = torch.matmul(pcl_batch, rotmat_el_batch)#B*N*3
    depth = -box_size - pcl_batch[:,:,2]#B*N
    grid_idx = (pcl_batch[:,:,0:2] + box_size)/(2*box_size/resolution)#B*N*2
    grid_idx = grid_idx.long()#B*N*2
    grid_idx = torch.cat((grid_idx,torch.arange(npoints).view(1,npoints,-1).cuda().repeat(batch_size, 1,1)),2)#B*N*3
    grid_idx = torch.cat(((torch.arange(batch_size).view(batch_size,1,-1).cuda().repeat(1, npoints, 1)),grid_idx), 2)#B*N*4
    grid_idx = grid_idx[:,:,0]*resolution*resolution*npoints + grid_idx[:,:,1]*resolution*npoints + grid_idx[:,:,2]*npoints + grid_idx[:,:,3]#B*N
    grid_idx = grid_idx.view(batch_size*npoints)#(B*N)
    device = torch.device('cuda:0')
    #plane_distance = torch.ones((batch_size*resolution*resolution*npoints)).cuda() * -box_size*2#(B*R*R*N)
    plane_distance = torch.ones((batch_size*resolution*resolution*npoints), device=device) * -box_size*2#(B*R*R*N)
    depth = depth.view(batch_size*npoints)#(B*N)
    plane_distance[grid_idx] = depth#(B*R*R*N)
    plane_distance = plane_distance.view(batch_size,resolution,resolution,npoints)#B*R*R*N
    plane_depth,_ = torch.max(plane_distance,3)#B*R*R
    plane_mask = (plane_depth <= (-box_size * 2 + 1e-6))#B*R*R
    plane_mask = plane_mask.float() * box_size * 2#B*R*R
    plane_depth = plane_depth + plane_mask#B*R*R
    plane_depth -= box_size*2/50 * 1#B*R*R
    plane_depth = plane_depth.view(batch_size,resolution,resolution,1)#B*R*R*1
    point_visible = (plane_distance >= plane_depth)
    point_visible,_ = torch.max(point_visible.int(),1)
    point_visible,_ = torch.max(point_visible.int(),1)
    #print(point_visible.shape)
    #point_visible, _ = torch.max(torch.max(point_visible.int(),1)[0],1)
    return point_visible.cpu().numpy()
def partial_render_batch(pcl_batch, partial_batch, resolution=50, box_size=1):    #gt, partial
    batch_size, npoints, dimension = pcl_batch.shape #[10,2048,3]   #获取gt的bs和Number of points 10,2048,3
    rotmat_az_batch, rotmat_el_batch, az_batch, el_batch = generate_rotmat(batch_size)
    #生成一个batch size的随机旋转矩阵对于方位角绕y轴旋转rotmat_az_batch, 对于高度角沿x轴旋转rotmat_el_batch
    #和对应的方位角和高度角az_batch, el_batch，模拟从不同视角观察每个点云的情况
    az_batch,el_batch = az_batch.reshape(batch_size,1), el_batch.reshape(batch_size,1)
    azel_batch = np.concatenate([az_batch,el_batch],1)#调整角度形状，沿第二维合并，得到[batchsize,2]
    point_visible_batch = ortho_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch,resolution, npoints, box_size).reshape(batch_size,npoints,1)
    #批量正交渲染，对一个批次的3D点云进行处理，生成每个点在特定视角下是否可见的信息。
    #这个过程包括将点云转换到新的视角、创建深度图，并确定每个点是否被遮挡
    for i in range(batch_size):
        point_visible = point_visible_batch[i,:,:]
        pcl = pcl_batch[i,:,:]
        point_visible_idx, _ = np.where(point_visible > 0.5)
        point_visible_idx = np.random.choice(point_visible_idx, 2048)
        new_pcl = pcl[point_visible_idx]
        partial_batch[i,:,:] = new_pcl
    """
    生成部分点云:
    对于批次中的每个点云：
    获取该点云的可见点索引（point_visible_idx），这些索引表示从当前视角可以看到的点。
    通过阈值（0.5）过滤可见点，选择那些大于0.5的点作为可见点。
    从可见点中随机选择2048个点（使用np.random.choice），生成部分观察到的点云 new_pcl。
    将生成的部分点云 new_pcl 存储在 partial_batch 对应的位置上。

    这个函数为后续的3D点云处理提供了基础，特别是在模拟不同视角下的点云观察时。
    通过应用这些旋转矩阵，可以从不同角度“观察”点云，这对于某些应用（如数据增强、3D重建等）非常有用。
    """
    return partial_batch , rotmat_az_batch, rotmat_el_batch, azel_batch
def generate_rotmat(batch_size):
    # 方位角
    az_batch = np.random.rand(batch_size)
    az_batch = az_batch * 2 * np.pi

    # az_batch = np.zeros(batch_size)
    # 俯仰角
    el_batch = np.random.rand(batch_size)
    # el_batch = (el_batch - 0.5) * np.pi*0.4 # [-0.5, 0.5] * 0.5π -> [-0.25π, 0.25π]
    # 假设你想让视角限制在 [-30度, +10度] 之间（+10度即限制了上方角度）
    min_angle = 60 / 180 * np.pi
    max_angle = 70 / 180 * np.pi
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
    parser.add_argument('--input', type=str, default='/home/tianqi/DAPoinTr/data/scanned_straw_meshed', help='Input PLY file or directory')
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
