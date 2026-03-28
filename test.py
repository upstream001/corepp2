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

        # --- Local Physical Normalization ---
        # 改为局部归一化：将当前部分点云平移至质心并缩放到单位球内，保存缩放因子
        center = np.mean(sampled_points, axis=0)
        norm_points = sampled_points - center
        scale = np.max(np.linalg.norm(norm_points, axis=1))
        if scale == 0:
            scale = 1.0

        sampled_points = norm_points / scale
        
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
import csv

from utils import sdf2mesh_cuda
import skimage.measure
from scipy.spatial import ConvexHull
from metrics_3d import chamfer_distance, precision_recall

cd = chamfer_distance.ChamferDistance()
pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

torch.autograd.set_detect_anomaly(True)
import pandas as pd
from sklearn.metrics import mean_squared_error


def _map_vertices_from_canonical_cube(vertices, bbox_min, bbox_max, cube_min=-3.0, cube_max=3.0):
    if vertices.size == 0:
        return vertices
    cube_size = cube_max - cube_min
    if cube_size <= 0:
        return vertices
    vertices_01 = (vertices - cube_min) / cube_size
    return bbox_min + vertices_01 * (bbox_max - bbox_min)


def _compute_volume_ml(mesh, unit="cm"):
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0.0

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if unit == "mm":
        factor = 1.0 / 1000.0
    elif unit == "m":
        factor = 1_000_000.0
    else:
        factor = 1.0  # cm -> mL

    try:
        if mesh.is_watertight():
            return abs(float(mesh.get_volume())) * factor
    except Exception:
        pass

    try:
        return float(ConvexHull(np.asarray(mesh.vertices)).volume) * factor
    except Exception:
        return 0.0


def _threshold_tag(t):
    s = f"{float(t):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _load_ground_truth_volumes(gt_csv_path):
    gt_volumes = {}
    if not os.path.exists(gt_csv_path):
        return gt_volumes

    with open(gt_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filename = (row.get("filename") or "").strip()
            volume_ml = (row.get("volume_ml") or "").strip()
            if not filename or not volume_ml:
                continue
            try:
                gt_volumes[filename] = float(volume_ml)
            except ValueError:
                continue

    return gt_volumes


def _load_complete_volume_lookup(data_source, gt_csv_path):
    volume_lookup = {}
    if not data_source:
        return volume_lookup

    mapping_path = os.path.join(data_source, "mapping.json")
    if not os.path.exists(mapping_path):
        return volume_lookup

    gt_volumes = _load_ground_truth_volumes(gt_csv_path)
    if not gt_volumes:
        return volume_lookup

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for partial_name, complete_name in mapping.items():
        if complete_name not in gt_volumes:
            continue
        volume_ml = gt_volumes[complete_name]
        volume_lookup[partial_name] = volume_ml
        volume_lookup[os.path.splitext(partial_name)[0]] = volume_ml

    return volume_lookup


def _write_aligned_csv(dataframe, output_path):
    if dataframe.empty:
        dataframe.to_csv(output_path, index=False)
        return

    str_df = dataframe.copy()
    str_df = str_df.replace({np.nan: ""})
    for col in str_df.columns:
        str_df[col] = str_df[col].map(lambda v: str(v))

    widths = {}
    for col in str_df.columns:
        cell_width = str_df[col].map(len).max() if len(str_df[col]) > 0 else 0
        widths[col] = max(len(col), cell_width)

    lines = []
    header = ", ".join(col.ljust(widths[col]) for col in str_df.columns)
    lines.append(header)

    for _, row in str_df.iterrows():
        line = ", ".join(row[col].ljust(widths[col]) for col in str_df.columns)
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main_function(decoder, pretrain, cfg, latent_size, test_data_dir=None):
    torch.manual_seed(133)
    np.random.seed(133)
    
    df = pd.DataFrame()
    columns = ['fruit_id',
                'frame_id',
                'complete_volume_ml',
                'pred_volume_head_raw_ml',
                'pred_volume_head_ml',
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
    metric_threshold = float(param.get("metric_threshold", 0.005))
    metric_thresholds = param.get("metric_thresholds", [0.005, 0.01, 0.02, 0.03, 0.05])
    metric_thresholds = sorted({float(t) for t in metric_thresholds})
    if metric_threshold not in metric_thresholds:
        metric_thresholds.append(metric_threshold)
        metric_thresholds = sorted(metric_thresholds)

    multi_columns = list(columns)
    for t in metric_thresholds:
        tag = _threshold_tag(t)
        multi_columns.extend([f"precision_t{tag}", f"recall_t{tag}", f"f1_t{tag}"])
    save_df_multi = pd.DataFrame(columns=multi_columns)

    volume_unit = str(param.get("volume_unit", "cm")).lower()
    volume_scale_factor = float(param.get("volume_scale_factor", 1.0))
    remap_mesh_to_gt_bbox = bool(param.get("remap_mesh_to_gt_bbox", False))
    repo_root = os.path.dirname(os.path.abspath(__file__))
    gt_csv_path = param.get("ground_truth_csv", os.path.join(repo_root, "ground_truth.csv"))
    mapping_data_source = test_data_dir if test_data_dir is not None else param.get("data_dir")
    complete_volume_lookup = _load_complete_volume_lookup(mapping_data_source, gt_csv_path)
    pr_max_t = max(0.01, max(metric_thresholds))
    pr_num = max(10, int(round((pr_max_t - 0.001) / 0.001)) + 1)
    cd_metric = chamfer_distance.ChamferDistance()
    pr_metric = precision_recall.PrecisionRecall(0.001, pr_max_t, pr_num)

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

    volume_head = nn.Sequential(
        nn.Linear(latent_size, latent_size),
        nn.ReLU(inplace=True),
        nn.Linear(latent_size, 1),
    ).to(device)

    ckpt = os.path.join(param['checkpoint_dir'], param['checkpoint_file'])
    ckpt_data = torch.load(ckpt)
    if 'encoder_state_dict' in ckpt_data:
        encoder.load_state_dict(ckpt_data['encoder_state_dict'])
    else:
        encoder.load_state_dict(ckpt_data)
    if 'decoder_state_dict' in ckpt_data:
        decoder.load_state_dict(ckpt_data['decoder_state_dict'])
    volume_head_enabled = 'volume_head_state_dict' in ckpt_data
    if volume_head_enabled:
        volume_head.load_state_dict(ckpt_data['volume_head_state_dict'])
    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)
    volume_head.to(device)
    volume_head.eval()

    volume_head_calibration_coeffs = np.array([1.0, 0.0], dtype=np.float64)
    use_volume_head_calibration = bool(param.get('calibrate_volume_head_on_val', False))

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

    if volume_head_enabled and use_volume_head_calibration and test_data_dir is None and param['encoder'] in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
        cal_dataset = PointCloudDataset(
            data_source=param["data_dir"],
            pad_size=param["input_size"],
            pretrain=pretrain,
            split='val',
            use_partial=False,
            supervised_3d=False,
            sdf_loss=False,
            grid_density=param["grid_density"],
        )
        cal_loader = DataLoader(cal_dataset, batch_size=1, shuffle=False)
        cal_preds = []
        cal_targets = []
        with torch.no_grad():
            for item in cal_loader:
                if 'volume_ml' not in item:
                    continue
                encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device)
                latent = encoder(encoder_input)
                raw_pred = torch.expm1(volume_head(latent)).item()
                cal_preds.append(raw_pred)
                cal_targets.append(float(item['volume_ml'].item()))
        if len(cal_preds) >= 2:
            x = np.asarray(cal_preds, dtype=np.float64)
            y = np.asarray(cal_targets, dtype=np.float64)
            A = np.stack([x, np.ones(len(x), dtype=np.float64)], axis=1)
            volume_head_calibration_coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
            print(f"[Info] Volume-head calibration fitted on val split: linear coeffs={volume_head_calibration_coeffs.tolist()}")
        else:
            print("[Warning] Not enough val samples with volume labels to calibrate volume head.")

    with torch.no_grad():

        for n_iter, item in enumerate(tqdm(iter(dataset))):
            volume, chamfer_dist_value, prec, rec, f1 = 0, 0, 0, 0, 0
            frame_id = item['frame_id'][0]
            complete_volume_ml = complete_volume_lookup.get(frame_id)

            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
            gt = o3d.geometry.PointCloud()
            gt.points = o3d.utility.Vector3dVector(item['target_pcd'][0].numpy())
            gt_pts = np.asarray(gt.points)
            # Use tight bbox from GT points to avoid 10% padding inflation from dataloader bbox.
            bbox_min = gt_pts.min(axis=0)
            bbox_max = gt_pts.max(axis=0)

            start = time.time()
            mesh = None

            # unpacking inputs
            if param['encoder'] not in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
            else: 
                encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device) ## be aware: the current partial pcd is not registered to the target pcd!

            latent = encoder(encoder_input)
            if volume_head_enabled:
                pred_volume_head_raw_ml = torch.expm1(volume_head(latent)).item()
                if use_volume_head_calibration:
                    pred_volume_head_ml = max(0.0, float(pred_volume_head_raw_ml * volume_head_calibration_coeffs[0] + volume_head_calibration_coeffs[1]))
                else:
                    pred_volume_head_ml = pred_volume_head_raw_ml
            else:
                pred_volume_head_raw_ml = float('nan')
                pred_volume_head_ml = float('nan')

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
                    max_batch=int(2 ** 18)
                )
                
                # Load the generated mesh back using open3d for the subsequent metric computations
                mesh_ply_file = mesh_filename + ".ply"
                mesh = o3d.io.read_triangle_mesh(mesh_ply_file)
                mesh.compute_vertex_normals()

                # Optional remap from canonical cube to GT bbox.
                # Keep disabled by default because some datasets are already in physical coordinates.
                if remap_mesh_to_gt_bbox:
                    mesh_pts = np.asarray(mesh.vertices)
                    mesh_pts = _map_vertices_from_canonical_cube(mesh_pts, bbox_min, bbox_max)
                    mesh.vertices = o3d.utility.Vector3dVector(mesh_pts)

                volume_ml = _compute_volume_ml(mesh, unit=volume_unit)
                volume_ml *= volume_scale_factor
            except Exception as e:
                print(f"  [Mesh Error] {item['frame_id'][0]}: {e}")
                volume_ml = 0.0

            inference_time = time.time() - start

            if n_iter > 0:
                exec_time.append(inference_time)

            if mesh is None:
                cur_data = {
                    'fruit_id': item['fruit_id'][0],
                    'frame_id': frame_id,
                    'complete_volume_ml': round(complete_volume_ml, 6) if complete_volume_ml is not None else np.nan,
                    'pred_volume_head_ml': round(pred_volume_head_ml, 6),
                    'pred_volume_head_raw_ml': round(pred_volume_head_raw_ml, 6) if np.isfinite(pred_volume_head_raw_ml) else np.nan,
                    'mesh_volume_ml': round(volume_ml, 6),
                    'chamfer_distance': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
                save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
                save_df.to_csv("shape_completion_results.csv", mode='w+', index=False)

                cur_multi = dict(cur_data)
                for t in metric_thresholds:
                    tag = _threshold_tag(t)
                    cur_multi[f"precision_t{tag}"] = 0.0
                    cur_multi[f"recall_t{tag}"] = 0.0
                    cur_multi[f"f1_t{tag}"] = 0.0
                save_df_multi = pd.concat([save_df_multi, pd.DataFrame([cur_multi])], ignore_index=True)
                _write_aligned_csv(save_df_multi, "shape_completion_results_multi_threshold.csv")
                continue

            # --- Metric Normalization: 在计算衡量指标前临时缩放到单位球内，以对齐固定的阈值判定范围 ---
            mesh_pts = np.asarray(mesh.vertices)
            
            # 找到以质心为原点的缩放上限
            local_center = np.mean(gt_pts, axis=0)
            shifted_gt = gt_pts - local_center
            local_scale = np.max(np.linalg.norm(shifted_gt, axis=1)) 
            if local_scale == 0: local_scale = 1.0

            # 缩放供检测的独立对象副本
            eval_gt = o3d.geometry.PointCloud()
            eval_gt.points = o3d.utility.Vector3dVector(shifted_gt / local_scale)
            
            eval_mesh = o3d.geometry.TriangleMesh()
            eval_mesh.vertices = o3d.utility.Vector3dVector((mesh_pts - local_center) / local_scale)
            eval_mesh.triangles = mesh.triangles

            cd_metric.reset()
            cd_metric.update(eval_gt, eval_mesh)
            chamfer_dist_value = cd_metric.compute(print_output=False)

            pr_metric.reset()
            pr_metric.update(eval_gt, eval_mesh)
            metrics_by_t = {}
            for t in metric_thresholds:
                p_t, r_t, f_t, _ = pr_metric.compute_at_threshold(t, print_output=False)
                metrics_by_t[t] = (p_t, r_t, f_t)

            prec, rec, f1 = metrics_by_t[metric_threshold]

            cur_data = {
                'fruit_id': item['fruit_id'][0],
                'frame_id': frame_id,
                'complete_volume_ml': round(complete_volume_ml, 6) if complete_volume_ml is not None else np.nan,
                'pred_volume_head_ml': round(pred_volume_head_ml, 6),
                'pred_volume_head_raw_ml': round(pred_volume_head_raw_ml, 6) if np.isfinite(pred_volume_head_raw_ml) else np.nan,
                'mesh_volume_ml': round(volume_ml, 6),
                'chamfer_distance': round(chamfer_dist_value, 6),
                'precision': round(prec, 1),
                'recall': round(rec, 1),
                'f1': round(f1, 1)
                }
                
            save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
            save_df.to_csv("shape_completion_results.csv", mode='w+', index=False)

            cur_multi = dict(cur_data)
            for t in metric_thresholds:
                p_t, r_t, f_t = metrics_by_t[t]
                tag = _threshold_tag(t)
                cur_multi[f"precision_t{tag}"] = round(p_t, 1)
                cur_multi[f"recall_t{tag}"] = round(r_t, 1)
                cur_multi[f"f1_t{tag}"] = round(f_t, 1)
            save_df_multi = pd.concat([save_df_multi, pd.DataFrame([cur_multi])], ignore_index=True)
            _write_aligned_csv(save_df_multi, "shape_completion_results_multi_threshold.csv")


        print(f"Average time for 3D shape completion, including postprocessing: {np.mean(exec_time)*1e3:.1f} ms")
        print("Results saved in: " + os.getcwd() + "/shape_completion_results.csv")
        print("Multi-threshold results saved in: " + os.getcwd() + "/shape_completion_results_multi_threshold.csv")

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
