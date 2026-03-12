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

class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, pad_size, norm_scale=45.54):
        self.data_source = data_source
        self.pad_size = pad_size
        self.norm_scale = norm_scale
        self.files = []
        
        # Support a single file or a directory
        if os.path.isfile(data_source) and data_source.endswith('.ply'):
            self.files.append(data_source)
        elif os.path.isdir(data_source):
            for root, _, files in os.walk(data_source):
                for f in sorted(files):
                    if f.endswith('.ply'):
                        self.files.append(os.path.join(root, f))
        else:
            print(f"[Error] Custom test data_source {data_source} is invalid.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Use filename without extension as fruit_id
        fruit_id = os.path.splitext(os.path.basename(file_path))[0]
        
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        num_points = points.shape[0]
        
        if num_points >= self.pad_size:
            random_list = np.random.choice(num_points, size=self.pad_size, replace=False)
            sampled_points = points[random_list, :]
        else:
            # 如果点数不够，则有放回地随机采样进行补全
            if num_points == 0:
                sampled_points = np.zeros((self.pad_size, 3))
            else:
                random_list = np.random.choice(num_points, size=self.pad_size, replace=True)
                sampled_points = points[random_list, :]

        # --- Global Physical Normalization ---
        # 对于自然数据，直接应用统一测绘好的全局缩放常数(norm_scale)。
        # 不使用局部点云自身包围盒，避免抹去不同草莓的大小尺度差异
        scale = self.norm_scale
        center = np.zeros(3)

        sampled_points = sampled_points / scale
        
        item = {
            'fruit_id': fruit_id,
            'frame_id': fruit_id,
            'target_pcd': torch.Tensor(sampled_points).float(), # Only for output tracking shape
            'partial_pcd': torch.Tensor(sampled_points).float(),
            'center': torch.Tensor(center).float(),
            'scale': torch.tensor(scale).float(),
            'bbox': {
                'min': torch.Tensor([-1.0, -1.0, -1.0]),
                'max': torch.Tensor([1.0, 1.0, 1.0])
            }
        }
        
        # padding for other potential accesses
        item['rgb'] = torch.zeros((3, self.pad_size, self.pad_size))
        item['depth'] = torch.zeros((1, self.pad_size, self.pad_size))
        item['mask'] = torch.zeros((1, self.pad_size, self.pad_size))

        return item

from networks.models import Encoder, EncoderBig, ERFNetEncoder, EncoderBigPooled, EncoderPooled, DoubleEncoder, PointCloudEncoder, PointCloudEncoderLarge, FoldNetEncoder
from networks.pointnext import PointNeXtEncoder, build_pointnext_encoder
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


def main_function(decoder, pretrain, cfg, latent_size, test_data_dir=None):
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
    elif param['encoder'] == 'pointnext':
        encoder = build_pointnext_encoder(out_channels=latent_size, cfg=param).to(device)
    else:
        encoder = Encoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)

    ckpt = os.path.join(param['checkpoint_dir'], param['checkpoint_file'])
    ckpt_data = torch.load(ckpt)
    if 'encoder_state_dict' in ckpt_data:
        encoder.load_state_dict(ckpt_data['encoder_state_dict'])
    else:
        encoder.load_state_dict(ckpt_data)
    if 'decoder_state_dict' in ckpt_data:
        decoder.load_state_dict(ckpt_data['decoder_state_dict'])
    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)

    # transformations
    tfs = [Pad(size=param["input_size"])]
    tf = Compose(tfs)

    if test_data_dir is not None:
        norm_scale = param.get("normalization_scale", 45.54)
        cl_dataset = CustomTestDataset(data_source=test_data_dir, pad_size=param["input_size"], norm_scale=norm_scale)
        print(f"Testing on custom dataset directory: {test_data_dir}")
    else:
        if param['encoder'] in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
            cl_dataset = PointCloudDataset(
                data_source=param["data_dir"],
                pad_size=param["input_size"],
                pretrain=pretrain,
                split='test',
                use_partial=False,
                supervised_3d=False
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
                                                supervised_3d=False,
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
            if param['encoder'] not in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
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

            try:
                import deepsdf.deep_sdf.mesh
                mesh_save_path = "/home/tianqi/corepp2/logs/strawberry/output"
                if not os.path.exists(mesh_save_path):
                    os.makedirs(mesh_save_path)
                    
                mesh_filename = os.path.join(mesh_save_path, item['frame_id'][0]) # no .ply ext
                
                scale_val = 1.0
                center_val = np.array([0., 0., 0.])
                if 'scale' in item and 'center' in item:
                    scale_val = item['scale'].item()
                    center_val = item['center'][0].cpu().numpy()
                
                deepsdf.deep_sdf.mesh.create_mesh(
                    decoder, 
                    latent, 
                    mesh_filename, 
                    start=time.time(), 
                    N=grid_density, 
                    max_batch=int(2 ** 18),
                    offset=-center_val,
                    scale=1.0 / scale_val
                )
                
                # Load the generated mesh back using open3d for the subsequent metric computations
                mesh_ply_file = mesh_filename + ".ply"
                mesh = o3d.io.read_triangle_mesh(mesh_ply_file)
                mesh.compute_vertex_normals()

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

            # 因为此前在 create_mesh 时已经传入了动态算出的 offset 和 scale，使得输出的 .ply 文件以及 read_triangle_mesh 生成的 mesh 坐标刚好被复原为了等比例真实的相机系物理大小！
            # 凸包求得的无量纲体积即是真实的立体体积 (mm³) -> /1000 = 真实物理体积
            if volume > 0:
                physical_volume_ml = volume / 1000.0
            else:
                physical_volume_ml = 0

            cur_data = {
                'fruit_id': item['fruit_id'][0],
                'frame_id': item['frame_id'][0],
                'mesh_volume_ml': round(physical_volume_ml, 2),
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

    arg_parser.add_argument(
        "--test_data_dir",
        "-t",
        dest="test_data_dir",
        default=None,
        help="Optional: a specific custom directory containing point clouds (.ply) to test on instead of using the original split.json from training data.",
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
                  latent_size=latent_size,
                  test_data_dir=args.test_data_dir)