#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import ConvexHull
from torch.utils.data import DataLoader
from tqdm import tqdm

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
from networks.models import (
    DoubleEncoder,
    Encoder,
    EncoderBig,
    EncoderBigPooled,
    EncoderPooled,
    ERFNetEncoder,
    FoldNetEncoder,
    PointCloudEncoder,
    PointCloudEncoderLarge,
)
from networks.pointnext import build_pointnext_encoder


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class UnseenPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, pad_size, unit_scale_to_cm=100.0):
        self.data_source = data_source
        self.pad_size = pad_size
        self.unit_scale_to_cm = unit_scale_to_cm
        self.files = []

        if os.path.isfile(data_source) and data_source.endswith('.ply'):
            self.files.append(data_source)
        elif os.path.isdir(data_source):
            for root, _, files in os.walk(data_source):
                for name in sorted(files):
                    if name.endswith('.ply'):
                        self.files.append(os.path.join(root, name))
        else:
            raise FileNotFoundError(f'Invalid unseen data path: {data_source}')

        if not self.files:
            raise FileNotFoundError(f'No .ply files found in: {data_source}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        frame_id = Path(file_path).stem

        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32) * float(self.unit_scale_to_cm)
        num_points = points.shape[0]

        if num_points == 0:
            sampled_points = np.zeros((self.pad_size, 3), dtype=np.float32)
            center = np.zeros(3, dtype=np.float32)
            scale = 1.0
        else:
            replace = num_points < self.pad_size
            choice = np.random.choice(num_points, size=self.pad_size, replace=replace)
            sampled_points = points[choice, :].astype(np.float32)
            center = points.mean(axis=0).astype(np.float32)
            sampled_points = sampled_points - center
            scale = 1.0

        item = {
            'fruit_id': frame_id,
            'frame_id': frame_id,
            'partial_pcd': torch.from_numpy(sampled_points).float(),
            'target_pcd': torch.from_numpy(sampled_points).float(),
            'center': torch.from_numpy(np.asarray(center)).float(),
            'scale': torch.tensor(scale).float(),
            'source_path': file_path,
        }
        return item


def _restore_mesh_to_physical_scale(mesh, center_cm, scale_cm):
    if len(mesh.vertices) == 0:
        return mesh
    vertices = np.asarray(mesh.vertices)
    vertices = vertices * float(scale_cm) + np.asarray(center_cm, dtype=np.float32)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def _compute_volume_ml(mesh, unit='cm'):
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0.0

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if unit == 'mm':
        factor = 1.0 / 1000.0
    elif unit == 'm':
        factor = 1_000_000.0
    else:
        factor = 1.0

    try:
        return float(ConvexHull(np.asarray(mesh.vertices)).volume) * factor
    except Exception:
        return 0.0


def _build_encoder(param, latent_size, device):
    name = param['encoder']
    if name == 'big':
        return EncoderBig(in_channels=4, out_channels=latent_size, size=param['input_size']).to(device)
    if name == 'small_pool':
        return EncoderPooled(in_channels=4, out_channels=latent_size, size=param['input_size']).to(device)
    if name == 'erfnet':
        return ERFNetEncoder(in_channels=4, out_channels=latent_size, size=param['input_size']).to(device)
    if name == 'pool':
        return EncoderBigPooled(in_channels=4, out_channels=latent_size, size=param['input_size']).to(device)
    if name == 'double':
        return DoubleEncoder(out_channels=latent_size, size=param['input_size']).to(device)
    if name == 'point_cloud':
        return PointCloudEncoder(in_channels=3, out_channels=latent_size).to(device)
    if name == 'point_cloud_large':
        return PointCloudEncoderLarge(in_channels=3, out_channels=latent_size).to(device)
    if name == 'foldnet':
        return FoldNetEncoder(in_channels=3, out_channels=latent_size).to(device)
    if name == 'pointnext':
        return build_pointnext_encoder(out_channels=latent_size, cfg=param).to(device)
    return Encoder(in_channels=4, out_channels=latent_size, size=param['input_size']).to(device)


def main():
    parser = argparse.ArgumentParser(description='Run reconstruction on unseen point clouds.')
    parser.add_argument('--experiment', '-e', required=True, help='DeepSDF experiment directory.')
    parser.add_argument('--cfg', '-c', required=True, help='Encoder config JSON.')
    parser.add_argument('--checkpoint_decoder', dest='checkpoint', default='500', help='DeepSDF checkpoint id.')
    parser.add_argument('--input_dir', '-i', default='/home/tianqi/corepp2/data/D405_data', help='Directory with unseen .ply point clouds.')
    parser.add_argument('--output_dir', '-o', default='/home/tianqi/corepp2/unseen_output', help='Directory to save reconstructed meshes and CSV.')
    parser.add_argument('--input_unit', choices=['m', 'cm', 'mm'], default='m', help='Unit of unseen input point clouds.')
    deep_sdf.add_common_args(parser)
    args = parser.parse_args()

    deep_sdf.configure_logging(args)

    with open(args.cfg, 'r', encoding='utf-8') as f:
        param = json.load(f)

    specs = ws.load_experiment_specifications(args.experiment)
    latent_size = specs['CodeLength']
    arch = __import__('deepsdf.networks.' + specs['NetworkArch'], fromlist=['Decoder'])
    decoder = arch.Decoder(latent_size, **specs['NetworkSpecs']).cuda()

    device = 'cuda'
    encoder = _build_encoder(param, latent_size, device)
    volume_head = nn.Sequential(
        nn.Linear(latent_size, latent_size),
        nn.ReLU(inplace=True),
        nn.Linear(latent_size, 1),
    ).to(device)

    ckpt_path = os.path.join(param['checkpoint_dir'], param['checkpoint_file'])
    ckpt_data = torch.load(ckpt_path)
    if 'encoder_state_dict' in ckpt_data:
        encoder.load_state_dict(ckpt_data['encoder_state_dict'])
    else:
        encoder.load_state_dict(ckpt_data)
    if 'decoder_state_dict' in ckpt_data:
        decoder.load_state_dict(ckpt_data['decoder_state_dict'])
    volume_head_enabled = 'volume_head_state_dict' in ckpt_data
    if volume_head_enabled:
        volume_head.load_state_dict(ckpt_data['volume_head_state_dict'])

    encoder.eval()
    decoder.eval()
    volume_head.eval()

    unit_scale_to_cm = {'m': 100.0, 'cm': 1.0, 'mm': 0.1}[args.input_unit]
    dataset = UnseenPointCloudDataset(args.input_dir, pad_size=param['input_size'], unit_scale_to_cm=unit_scale_to_cm)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_dir = Path(args.output_dir)
    mesh_dir = output_dir / 'meshes'
    latent_dir = output_dir / 'latents'
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    import deepsdf.deep_sdf.mesh

    rows = []
    grid_density = param['grid_density']
    volume_unit = str(param.get('volume_unit', 'cm')).lower()
    volume_scale_factor = float(param.get('volume_scale_factor', 1.0))

    with torch.no_grad():
        for item in tqdm(loader, desc='Testing unseen point clouds'):
            frame_id = item['frame_id'][0]
            encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device)
            latent = encoder(encoder_input)

            pred_volume_ml = float('nan')
            if volume_head_enabled:
                pred_volume_ml = torch.expm1(volume_head(latent)).item()

            latent_path = latent_dir / f'{frame_id}.pth'
            torch.save(latent.detach().cpu().squeeze(), latent_path)

            mesh_prefix = mesh_dir / frame_id
            mesh_volume_ml = 0.0
            mesh_path = None
            start = time.time()
            try:
                deepsdf.deep_sdf.mesh.create_mesh(
                    decoder,
                    latent,
                    str(mesh_prefix),
                    start=time.time(),
                    N=grid_density,
                    max_batch=int(2 ** 18),
                )
                mesh_path = mesh_prefix.with_suffix('.ply')
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                mesh.compute_vertex_normals()
                # Keep the mesh in the same centered physical coordinate frame used by
                # training/test.py. Translation is unnecessary for volume and scale is 1.
                o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)
                mesh_volume_ml = _compute_volume_ml(mesh, unit=volume_unit) * volume_scale_factor
            except Exception as exc:
                print(f'[Mesh Error] {frame_id}: {exc}')

            rows.append({
                'frame_id': frame_id,
                'source_path': item['source_path'][0],
                'pred_volume_ml': round(pred_volume_ml, 6) if np.isfinite(pred_volume_ml) else np.nan,
                'mesh_volume_ml': round(mesh_volume_ml, 6),
                'latent_path': str(latent_path),
                'mesh_path': str(mesh_path) if mesh_path is not None and mesh_path.exists() else '',
                'inference_time_ms': round((time.time() - start) * 1000.0, 3),
            })

    result_df = pd.DataFrame(rows)
    csv_path = output_dir / 'unseen_results.csv'
    result_df.to_csv(csv_path, index=False)
    print(f'Saved {len(result_df)} results to {csv_path}')
    print(f'Reconstructed meshes saved in {mesh_dir}')


if __name__ == '__main__':
    main()
