#!/usr/bin/env python3

from re import I
from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import ToTensor, Compose, Resize

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
import deepsdf.deep_sdf.o3d_utils as o3d_utils


from sdfrenderer.grid import Grid3D
from dataloaders.transforms import Pad, Rotate, RandomHorizontalFlip, RandomVerticalFlip
from dataloaders.cameralaser_w_masks import MaskedCameraLaserData
from dataloaders.pointcloud_dataset import PointCloudDataset

from networks.models import Encoder, EncoderBig, ERFNetEncoder, EncoderBigPooled, EncoderPooled, DoubleEncoder, PointCloudEncoder, PointCloudEncoderLarge, FoldNetEncoder
import networks.utils as net_utils

import open3d as o3d
import numpy as np

import time
import json

from utils import sdf2mesh_cuda, tensor_dict_2_float_dict
import skimage.measure
from scipy.spatial import ConvexHull
from metrics_3d import chamfer_distance, precision_recall

cd = chamfer_distance.ChamferDistance()
pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

torch.autograd.set_detect_anomaly(True)
import pandas as pd
from sklearn.metrics import mean_squared_error


def main_function(decoder, pretrain, cfg, latent_size):
    torch.manual_seed(133)
    np.random.seed(133)
    
    df = pd.DataFrame()
    columns = ['fruit_id',
                'frame_id',
                'mesh_volume_ml',
                'chamfer_distance',
                'precision',
                'recall',
                'f1'
                ]
    save_df = pd.DataFrame(columns=columns)

    exec_time = []

    with open(cfg) as json_file:
        param = json.load(json_file)

    device = 'cuda'

    # creating variables for 3d grid for diff SDF renderer
    threshold = param['threshold']
    grid_density = param['grid_density']
    precision = torch.float32

    # define encoder
    if param['encoder'] == 'big':
        encoder = EncoderBig(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'small_pool':
        encoder = EncoderPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'erfnet':
        encoder = ERFNetEncoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'pool':
        encoder = EncoderBigPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'double':
        encoder = DoubleEncoder(out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'point_cloud':
        encoder = PointCloudEncoder(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'point_cloud_large':
        encoder = PointCloudEncoderLarge(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'foldnet':
        encoder = FoldNetEncoder(in_channels=3, out_channels=latent_size).to(device)
    else:
        encoder = Encoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)

    ckpt = os.path.join(param['checkpoint_dir'], param['checkpoint_file'])
    # import ipdb;ipdb.set_trace()
    encoder.load_state_dict(torch.load(ckpt)['encoder_state_dict'])
    decoder.load_state_dict(torch.load(ckpt)['decoder_state_dict'])

    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)

    # transformations
    tfs = [Pad(size=param["input_size"])]
    tf = Compose(tfs)

    if param['encoder'] in ['point_cloud', 'point_cloud_large', 'foldnet']:
        cl_dataset = PointCloudDataset(
            data_source=param["data_dir"],
            pad_size=param["input_size"],
            pretrain=pretrain,
            use_partial=False,
            supervised_3d=True
        )
    else:
        cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                            tf=tf, 
                                            color_tf = None,
                                            pretrain=pretrain,
                                            pad_size=param["input_size"],
                                            detection_input=param["detection_input"],
                                            normalize_depth=param["normalize_depth"],
                                            depth_min=param["depth_min"],
                                            depth_max=param["depth_max"],
                                            supervised_3d=True,
                                            sdf_loss=param["3D_loss"],
                                            grid_density=param["grid_density"],
                                            split='test',
                                            overfit=False,
                                            species=param["species"]
                                            )    
    dataset = DataLoader(cl_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():

        for n_iter, item in enumerate(tqdm(iter(dataset))):
            volume, chamfer_distance, prec, rec, f1 = 0, 0, 0, 0, 0
            try:
                box = tensor_dict_2_float_dict(item['bbox'])
            except:
                box = {
                    'xmin': -1.0,
                    'xmax': 1.0,
                    'ymin': -1.0,
                    'ymax': 1.0,
                    'zmin': -1.0,
                    'zmax': 1.0
                }

            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
            gt = o3d.geometry.PointCloud()
            gt.points = o3d.utility.Vector3dVector(item['target_pcd'][0].numpy())

            start = time.time()

            # unpacking inputs
            if param['encoder'] != 'point_cloud' and param['encoder'] != 'point_cloud_large' and param['encoder'] != 'foldnet':
                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
            else: 
                encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device) ## be aware: the current partial pcd is not registered to the target pcd!

            latent = encoder(encoder_input)

            # save the latent vector for further inspection
            latent_save = latent.detach().to('cpu').squeeze()
            save_path = os.path.join(os.path.dirname(pretrain), "encoder")
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            torch.save(latent_save, os.path.join(save_path, item['frame_id'][0] + ".pth"))

            grid_3d = Grid3D(grid_density, device, precision, bbox=box)
            deepsdf_input = torch.cat([latent.expand(grid_3d.points.size(0), -1),
                                        grid_3d.points], dim=1).to(latent.device, latent.dtype)
            
            # 采用分块推理避免显存溢出 (OOM)
            # grid_density=128 会产生约 200 万个点，一次性送入 Decoder 需要近 8GB 显存
            chunk_size = 65536  # 每次处理的特征数量 (约 256^2)
            pred_sdf_list = []
            for start_idx in range(0, deepsdf_input.size(0), chunk_size):
                end_idx = min(start_idx + chunk_size, deepsdf_input.size(0))
                chunk_input = deepsdf_input[start_idx:end_idx]
                with torch.no_grad():
                    chunk_pred = decoder(chunk_input)
                pred_sdf_list.append(chunk_pred)
            pred_sdf = torch.cat(pred_sdf_list, dim=0)

            try:
                # 使用 Marching Cubes 替代凸包，获得高质量等值面
                N = grid_density
                sdf_values = pred_sdf.detach().cpu().numpy().reshape(N, N, N)
                
                # bbox 范围
                x_min, x_max = box.get('xmin', -1.0), box.get('xmax', 1.0)
                y_min, y_max = box.get('ymin', -1.0), box.get('ymax', 1.0)
                z_min, z_max = box.get('zmin', -1.0), box.get('zmax', 1.0)
                spacing = ((x_max - x_min) / (N - 1),
                           (y_max - y_min) / (N - 1),
                           (z_max - z_min) / (N - 1))
                
                verts, faces, normals_mc, _ = skimage.measure.marching_cubes(
                    sdf_values, level=0.0, spacing=spacing
                )
                # 将顶点坐标从网格空间转到实际空间
                verts[:, 0] += x_min
                verts[:, 1] += y_min
                verts[:, 2] += z_min
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                
                # 连通域过滤：只保留最大的连通组件（草莓本体），移除远场噪声碎片
                triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                if len(cluster_n_triangles) > 0:
                    largest_cluster = cluster_n_triangles.argmax()
                    triangles_to_remove = triangle_clusters != largest_cluster
                    mesh.remove_triangles_by_mask(triangles_to_remove)
                    mesh.remove_unreferenced_vertices()
                
                # 清理退化几何元素
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                # Taubin 平滑：比 Laplacian 更好，不会缩小网格体积
                mesh = mesh.filter_smooth_taubin(number_of_iterations=30)
                
                # Save reconstructed mesh
                # 清除法线后保存，让可视化器自行计算正确的法线方向
                mesh_save = o3d.geometry.TriangleMesh(mesh)
                mesh_save.vertex_normals = o3d.utility.Vector3dVector()
                mesh_save_path = os.path.join(os.path.dirname(pretrain), "meshes")
                if not os.path.exists(mesh_save_path):
                    os.makedirs(mesh_save_path)
                o3d.io.write_triangle_mesh(os.path.join(mesh_save_path, item['frame_id'][0] + ".ply"), mesh_save)

                # 体积计算：使用 scipy ConvexHull（100% 可靠，无水密性要求）
                try:
                    hull = ConvexHull(np.asarray(mesh.vertices))
                    volume = hull.volume
                except:
                    volume = 0
            except Exception as e:
                print(f"  [Mesh Error] {item['frame_id'][0]}: {e}")

            inference_time = time.time() - start

            if n_iter > 0:
                exec_time.append(inference_time)

            cd.reset()
            cd.update(gt, mesh)
            chamfer_distance = cd.compute(print_output=False)

            pr.reset()
            pr.update(gt, mesh)
            prec, rec, f1, _ = pr.compute_at_threshold(0.005, print_output=False)

            # Retrieve info dynamically if available, otherwise mock it.
            # Removed pd read_csv for 3DPotatoTwinDemo explicitly
            cur_data = {
                'fruit_id': item['fruit_id'][0],
                'frame_id': item['frame_id'][0],
                'mesh_volume_ml': round(volume * 1e6, 1) if volume > 0 else 0,
                'chamfer_distance': round(chamfer_distance, 6),
                'precision': round(prec, 1),
                'recall': round(rec, 1),
                'f1': round(f1, 1)
                }
                
            save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
            save_df.to_csv("shape_completion_results.csv", mode='w+', index=False)


        print(f"Average time for 3D shape completion, including postprocessing: {np.mean(exec_time)*1e3:.1f} ms")
        print("Results saved in: " + os.getcwd() + "/shape_completion_results.csv")

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="shape completion main file, assume a pretrained deepsdf model")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--cfg",
        "-c",
        dest="cfg",
        required=True,
        help="Config file for the outer network.",
    )
    arg_parser.add_argument(
        "--checkpoint_decoder",
        dest="checkpoint",
        default="500",
        help="The checkpoint weights to use. This should be a number indicated an epoch",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    # loading deepsdf model
    specs = ws.load_experiment_specifications(args.experiment_directory)
    latent_size = specs["CodeLength"]
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    path = os.path.join(args.experiment_directory, 'ModelParameters', args.checkpoint) + '.pth'
    model_state = net_utils.load_without_parallel(torch.load(path))
    decoder.load_state_dict(model_state)
    decoder = net_utils.set_require_grad(decoder, False)

    # 根据可用的 latent codes 设置 pretrain_path
    pretrain_path_partial = os.path.join(args.experiment_directory, 'Reconstructions', args.checkpoint, 'Codes', 'partial')
    pretrain_path_complete = os.path.join(args.experiment_directory, 'Reconstructions', args.checkpoint, 'Codes', 'complete')
    
    if os.path.exists(pretrain_path_partial):
        pretrain_path = pretrain_path_partial
    elif os.path.exists(pretrain_path_complete):
        pretrain_path = pretrain_path_complete
    else:
        pretrain_path = pretrain_path_complete # fallback

    main_function(decoder=decoder,
                  pretrain=pretrain_path,
                  cfg=args.cfg,
                  latent_size=latent_size)