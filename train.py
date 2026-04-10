#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import json
from tqdm import tqdm

import os
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws

from sdfrenderer.grid import Grid3D

from dataloaders.cameralaser_w_masks import MaskedCameraLaserData
from dataloaders.pointcloud_dataset import PointCloudDataset
from dataloaders.transforms import Pad, Rotate, RandomHorizontalFlip, RandomVerticalFlip

from networks.models import Encoder, EncoderBig, ERFNetEncoder, EncoderBigPooled, EncoderPooled, DoubleEncoder, PointCloudEncoder, PointCloudEncoderLarge, FoldNetEncoder
import networks.utils as net_utils
from networks.pointnext import PointNeXtEncoder, build_pointnext_encoder
from loss import KLDivLoss, SuperLoss, SDFLoss, SDFLoss_new, RegLatentLoss, AttRepLoss, LatentSpreadLoss, VolumeLoss
from utils import sdf2mesh_cuda, save_model, tensor_dict_2_float_dict

DEBUG = True

torch.autograd.set_detect_anomaly(True)

from metrics_3d import chamfer_distance
cd = chamfer_distance.ChamferDistance()
from sklearn.metrics import mean_squared_error


def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder


def decode_sdf_in_chunks(decoder, latent_batch, grid_batch, latent_size, chunk_size):
    pred_sdf = []
    for latent, grid in zip(latent_batch, grid_batch):
        latent = latent.unsqueeze(0)
        sample_preds = []
        num_points = grid.points.size(0)
        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)
            points_chunk = grid.points[start_idx:end_idx]
            decoder_input = torch.cat([latent.expand(points_chunk.size(0), -1), points_chunk], dim=1)
            sample_preds.append(decoder(decoder_input))
        pred_sdf.append(torch.cat(sample_preds, dim=0))
    return torch.stack(pred_sdf, dim=0)


def _mesh_volume_ml(mesh, unit="cm"):
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return float("nan")

    try:
        volume = abs(float(mesh.get_volume()))
    except Exception:
        return float("nan")

    if unit == "mm":
        return volume / 1000.0
    if unit == "m":
        return volume * 1_000_000.0
    return volume


def _rmse(values_a, values_b):
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    return float(np.sqrt(np.mean((arr_a - arr_b) ** 2)))


def _compute_decoder_mesh_volume_ml(decoder, latent, mesh_filename, grid_density, unit, volume_scale_factor):
    import deepsdf.deep_sdf.mesh

    with torch.no_grad():
        deepsdf.deep_sdf.mesh.create_mesh(
            decoder,
            latent,
            mesh_filename,
            start=0.0,
            N=grid_density,
            max_batch=int(2 ** 18),
        )

    import open3d as o3d
    from scipy.spatial import ConvexHull

    mesh = o3d.io.read_triangle_mesh(mesh_filename + ".ply")
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return float("nan")

    vertices = np.asarray(mesh.vertices)
    try:
        volume = float(ConvexHull(vertices).volume)
    except Exception:
        return float("nan")

    if unit == "mm":
        volume /= 1000.0
    elif unit == "m":
        volume *= 1_000_000.0

    return volume * volume_scale_factor


def resolve_supervision_path(experiment_directory, checkpoint, split, allow_matrix):
    matrix_path = os.path.join(experiment_directory, ws.latent_codes_subdir, checkpoint + ".pth")
    codes_root = os.path.join(experiment_directory, "Reconstructions", checkpoint, "Codes")

    if allow_matrix and os.path.exists(matrix_path):
        return matrix_path

    candidate_dirs = [
        os.path.join(codes_root, split),
        os.path.join(codes_root, "complete"),
        os.path.join(codes_root, "partial"),
    ]

    for candidate in candidate_dirs:
        if os.path.isdir(candidate):
            return candidate

    return None


def main_function(decoder, train_pretrain, val_pretrain, cfg, latent_size, trunc_val, overfit, update_decoder):

    if DEBUG:
        torch.manual_seed(133)
        random.seed(133)
        np.random.seed(133)

    cfg_fname = cfg.split('/')[-1].replace('.json', '')  # getting filename

    with open(cfg) as json_file:
        param = json.load(json_file)

    check_direxcist(param["checkpoint_dir"])
    device = 'cuda'
    lambda_super = float(param.get("lambda_super", 0.3))
    lambda_latent_spread = float(param.get("lambda_latent_spread", 1.0))
    lambda_volume = float(param.get("lambda_volume", 0.5))
    volume_loss_relative_weight = float(param.get("volume_loss_relative_weight", 0.5))
    train_volume_head = lambda_volume > 0
    validate_mesh_volume = bool(param.get("validate_mesh_volume", True))
    volume_unit = str(param.get("volume_unit", "cm")).lower()
    volume_scale_factor = float(param.get("volume_scale_factor", 1.0))
    validation_mesh_dir = os.path.join(param["checkpoint_dir"], "..", "val_output")
    shuffle = True
    last_val_score = np.inf

    # creating variables for 3d grid for diff SDF renderer
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

    #############################
    # TRAINING LOOP STARTS HERE #
    #############################

    writer = SummaryWriter(filename_suffix='__'+cfg_fname, log_dir=param["log_dir"])
    decoder.to(device)

    # transformations
    geo_tfs = v2.RandomChoice([Rotate(angle=45), RandomHorizontalFlip(), RandomVerticalFlip()])
    color_tfs = [Pad(size=param["input_size"]), v2.ColorJitter(brightness=0.5, hue=(-0.1, 0.1), saturation=0.5), geo_tfs]
    color_tf = v2.Compose(color_tfs)
    default_tfs = [Pad(size=param["input_size"]), geo_tfs]
    default_tf = v2.Compose(default_tfs)

    sdf_chunk_size = int(param.get("sdf_chunk_size", 65536))

    if param['encoder'] in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
        cl_dataset = PointCloudDataset(
            data_source=param["data_dir"],
            pad_size=param["input_size"],
            pretrain=train_pretrain,
            split='train',
            use_partial=False,
            supervised_3d=param["supervised_3d"],
            sdf_loss=param["3D_loss"],
            grid_density=param["grid_density"],
        )
    else:
        cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                            tf=default_tf,
                                            color_tf = color_tf, 
                                            pretrain=train_pretrain,
                                            pad_size=param["input_size"],
                                            detection_input=param["detection_input"],
                                            normalize_depth=param["normalize_depth"],
                                            depth_min=param["depth_min"],
                                            depth_max=param["depth_max"],
                                            supervised_3d=param["supervised_3d"],
                                            sdf_loss=param["3D_loss"],
                                            grid_density=param["grid_density"],
                                            split='train',
                                            overfit=overfit,
                                            species=param["species"]
                                            )
    dataset = DataLoader(cl_dataset, batch_size=param["batch_size"], shuffle=shuffle, drop_last=True)

    if update_decoder:
        params = list(encoder.parameters()) + list(decoder.parameters())
    else:
        params = list(encoder.parameters())
    if train_volume_head:
        params += list(volume_head.parameters())
    
    optim = torch.optim.Adam(params, lr=param["lr"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.97)

    print('\ncfg: ', json.dumps(param, indent=4), '\n')
    print(encoder)
    print(volume_head)
    print(decoder)

    # import ipdb; ipdb.set_trace()
    n_iter = 0  # used for tensorboard
    last_epoch = -1
    last_loss = torch.tensor(float("nan"), device=device)
    for e in range(param["epoch"]):
        last_epoch = e
        for idx, item in enumerate(iter(dataset)):

            # import ipdb;ipdb.set_trace()
            n_iter += 1  # for tensorboard
            logging_string = 'epoch: {}/{} -- iteration {}/{}'.format(e+1, param["epoch"], idx, len(dataset))

            optim.zero_grad()
            loss = 0

            # unpacking inputs
            if param['encoder'] not in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
            else:
                encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device) ## be aware: the current partial pcd is not registered to the target pcd!

            # encoding
            latent_batch_unnormd = encoder(encoder_input)
            norms_batch = torch.linalg.norm(latent_batch_unnormd, dim=1)

            latent_batch = latent_batch_unnormd #/ norms_batch.unsqueeze(dim=1)
            if train_volume_head:
                pred_volume = volume_head(latent_batch)

            writer.add_scalar('Debug/Train/LatentNormMean', norms_batch.mean(), n_iter)
            writer.add_scalar('Debug/Train/LatentNormStd', norms_batch.std(unbiased=False), n_iter)
            if 'latent' in item:
                target_norms = torch.linalg.norm(item['latent'].to(device), dim=1)
                writer.add_scalar('Debug/Train/TargetLatentNormMean', target_norms.mean(), n_iter)
                writer.add_scalar('Debug/Train/TargetLatentNormStd', target_norms.std(unbiased=False), n_iter)

            if param["contrastive"]:
                fruit_ids = [list(dataset.dataset.Ks.keys()).index(fid) for fid in item['fruit_id']]
                fruit_ids = torch.Tensor(fruit_ids)

                att_loss = AttRepLoss(latent_batch, fruit_ids, device)
                loss += param['lambda_attraction']*att_loss

                # logging
                writer.add_scalar('Loss/Train/Att', param['lambda_attraction']*att_loss, n_iter)
                logging_string += ' -- loss att: {}'.format(param['lambda_attraction']*att_loss.item())

            if param["kl_divergence"]:

                loss_kl, determinant = KLDivLoss(latent_batch, cl_dataset, device)
                loss += param['lambda_kl']*loss_kl

                # logging
                writer.add_scalar('Loss/Train/KLDiv', param['lambda_kl']*loss_kl, n_iter)
                logging_string += ' -- loss kl: {}'.format(param['lambda_kl']*loss_kl.item())
                logging_string += ' -- det: {}'.format(determinant.item())

                writer.add_scalar('Debug/Train/BatchCovDet', determinant, n_iter)

            if param['supervised_3d']:
                loss_super = SuperLoss(latent_batch, item['latent'])
                loss += lambda_super * loss_super

                # logging
                writer.add_scalar('Loss/Train/SuperLoss', lambda_super * loss_super, n_iter)
                logging_string += ' -- loss super: {}'.format((lambda_super * loss_super).item())

                if lambda_latent_spread > 0 and latent_batch.shape[0] > 1:
                    loss_spread = LatentSpreadLoss(latent_batch, item['latent'].to(device))
                    loss += lambda_latent_spread * loss_spread
                    writer.add_scalar('Loss/Train/LatentSpreadLoss', lambda_latent_spread * loss_spread, n_iter)
                    logging_string += ' -- loss spread: {}'.format((lambda_latent_spread * loss_spread).item())

            if train_volume_head and 'volume_ml' in item:
                loss_volume = VolumeLoss(pred_volume, item['volume_ml'].to(device).view(-1, 1), relative_weight=volume_loss_relative_weight)
                loss += lambda_volume * loss_volume
                writer.add_scalar('Loss/Train/VolumeLoss', lambda_volume * loss_volume, n_iter)
                logging_string += ' -- loss volume: {}'.format((lambda_volume * loss_volume).item())
            
            if param['reg_latent']:
                loss_reg = RegLatentLoss(latent_batch, param["lambda_reg_latent"], e)
                loss += loss_reg

                # logging
                writer.add_scalar('Loss/Train/RegLoss',loss_reg, n_iter)
                logging_string += ' -- loss reg: {}'.format(loss_reg.item())

            if param["3D_loss"]:
                # creating a Grid3D for each latent in the batch
                current_batch_size = encoder_input.shape[0]

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

                grid_batch = []
                for _ in range(current_batch_size):
                    grid_batch.append(Grid3D(grid_density, device, precision, bbox=box))

                pred_sdf = decode_sdf_in_chunks(
                    decoder,
                    latent_batch,
                    grid_batch,
                    latent_size,
                    sdf_chunk_size,
                )

                if param.get('loss_type', 'original') == 'weighted':
                    loss_sdf = SDFLoss_new(pred_sdf, item['target_sdf'].to(device), item['target_sdf_weights'].to(device), sdf_trunc=cl_dataset.sdf_trunc, points=grid_batch, alpha=param.get('sdf_alpha', 15.0))
                else:
                    loss_sdf = SDFLoss(pred_sdf, item['target_sdf'].to(device), item['target_sdf_weights'].to(device), sdf_trunc=cl_dataset.sdf_trunc, points=grid_batch)
                loss += param['lambda_sdf']*loss_sdf

                # logging
                writer.add_scalar('Loss/Train/SDFLoss', param['lambda_sdf']* loss_sdf, n_iter)
                logging_string += ' -- loss sdf: {}'.format( param['lambda_sdf']*loss_sdf.item())

            loss.backward()
            optim.step()
            last_loss = loss.detach()

            # tensorboard logging
            writer.add_scalar('LRate', scheduler.get_last_lr()[0], n_iter)
            writer.add_scalar('Loss/Train/Total', loss, n_iter)
            logging_string += ' -- loss: {}'.format(loss.item())
            logging_string += ' -- lr: {}'.format(scheduler.get_last_lr()[0])
            print(logging_string)

        scheduler.step()

        # validation step
        if (e+1) % param["validation_frequency"] == 0:
            with torch.no_grad():
                val_tfs = [Pad(size=param["input_size"])]
                val_tf = v2.Compose(val_tfs)
                val_supervised = param["supervised_3d"] and val_pretrain is not None

                if param["supervised_3d"] and not val_supervised:
                    print(
                        "\n[Warning] Validation latent supervision is unavailable for split 'val'. "
                        "Skipping validation MSE because no val latent-code source was found."
                    )

                if param['encoder'] in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
                    val_cl_dataset = PointCloudDataset(
                        data_source=param["data_dir"],
                        pad_size=param["input_size"],
                        pretrain=val_pretrain,
                        split='val',
                        use_partial=False,
                        supervised_3d=val_supervised,
                        sdf_loss=False,
                        grid_density=param["grid_density"],
                    )
                else:
                    val_cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                                            tf=val_tf,
                                                            color_tf = None, 
                                                            pretrain=val_pretrain,
                                                            pad_size=param["input_size"],
                                                            detection_input=param["detection_input"],
                                                            normalize_depth=param["normalize_depth"],
                                                            depth_min=param["depth_min"],
                                                            depth_max=param["depth_max"],
                                                            supervised_3d=val_supervised,
                                                            sdf_loss=False,
                                                            grid_density=param["grid_density"],
                                                            split='val',
                                                            overfit=overfit,
                                                            species=param["species"]
                                                            )

                val_dataset = DataLoader(val_cl_dataset, batch_size=1, shuffle=False)

                val_losses = []
                val_volume_losses = []
                gt_mesh_volumes = []
                pred_mesh_volumes = []
                print('\nvalidation...')
                if val_supervised:
                    for sample_idx, item in enumerate(tqdm(iter(val_dataset))):
                        try:
                            if param['encoder'] not in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
                                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
                            else:
                                encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device)

                            latent_val = encoder(encoder_input)

                            target_latent = item['latent'].to(device)
                            mse = torch.nn.functional.mse_loss(latent_val.squeeze(), target_latent.squeeze())
                            val_losses.append(mse.item())

                            if train_volume_head and 'volume_ml' in item:
                                pred_volume_val = volume_head(latent_val)
                                vol_loss = VolumeLoss(pred_volume_val, item['volume_ml'].to(device).view(-1, 1), relative_weight=volume_loss_relative_weight)
                                val_volume_losses.append(vol_loss.item())

                            if validate_mesh_volume and 'volume_ml' in item:
                                os.makedirs(validation_mesh_dir, exist_ok=True)
                                mesh_filename = os.path.join(validation_mesh_dir, item['fruit_id'][0])
                                pred_mesh_volume = _compute_decoder_mesh_volume_ml(
                                    decoder,
                                    latent_val,
                                    mesh_filename,
                                    grid_density,
                                    volume_unit,
                                    volume_scale_factor,
                                )
                                if np.isfinite(pred_mesh_volume):
                                    pred_mesh_volumes.append(pred_mesh_volume)
                                    gt_mesh_volumes.append(float(item['volume_ml'].item()))

                            if args.overfit:
                                break
                        except Exception as e:
                            sample_name = item.get('fruit_id', ['unknown'])
                            if isinstance(sample_name, (list, tuple)):
                                sample_name = sample_name[0]
                            print(f"[Warning] Validation sample {sample_idx} ({sample_name}) failed: {e}")

                if len(val_losses) > 0:
                    rmse_volume = sum(val_losses) / len(val_losses)
                else:
                    rmse_volume = float('nan')
                    if val_supervised:
                        print("[Warning] No validation samples produced a valid latent MSE. Recording NaN.")
                    else:
                        print("[Info] Validation Latent MSE skipped for this run.")

                if len(val_volume_losses) > 0:
                    val_volume_loss = sum(val_volume_losses) / len(val_volume_losses)
                else:
                    val_volume_loss = float('nan')

                if len(pred_mesh_volumes) > 0:
                    rmse_mesh_volume = _rmse(gt_mesh_volumes, pred_mesh_volumes)
                else:
                    rmse_mesh_volume = float('nan')

                if np.isfinite(rmse_volume):
                    print('Mean Validation Latent MSE: ', round(rmse_volume, 5))
                else:
                    print('Mean Validation Latent MSE: NaN')

                if np.isfinite(val_volume_loss):
                    print('Mean Validation Volume Loss: ', round(val_volume_loss, 5))
                else:
                    print('Mean Validation Volume Loss: NaN')

                if np.isfinite(rmse_mesh_volume):
                    print('Validation Mesh Volume RMSE: ', round(rmse_mesh_volume, 5))
                else:
                    print('Validation Mesh Volume RMSE: NaN')

                if np.isfinite(rmse_mesh_volume):
                    val_score = rmse_mesh_volume
                elif np.isfinite(rmse_volume) and np.isfinite(val_volume_loss):
                    val_score = rmse_volume + lambda_volume * val_volume_loss
                else:
                    val_score = rmse_volume

            # logging
            writer.add_scalar('Val/rmse_volume', rmse_volume, n_iter)
            if np.isfinite(val_volume_loss):
                writer.add_scalar('Val/volume_loss', val_volume_loss, n_iter)
            if np.isfinite(rmse_mesh_volume):
                writer.add_scalar('Val/mesh_volume_rmse', rmse_mesh_volume, n_iter)
            if np.isfinite(val_score):
                writer.add_scalar('Val/score', val_score, n_iter)
            # saving best model
            if np.isfinite(val_score) and val_score < last_val_score:
                last_val_score = val_score
                save_model(
                    encoder,
                    decoder,
                    e,
                    optim,
                    loss,
                    param["checkpoint_dir"]+'_'+cfg_fname+'_best_model.pt',
                    volume_head=volume_head if train_volume_head else None,
                )
                print('saving best model')
            print()

        # saving checkpoints
        if (e+1) % param["checkpoint_frequency"] == 0:
            save_model(
                encoder,
                decoder,
                e,
                optim,
                loss,
                param["checkpoint_dir"]+'_'+cfg_fname+'_checkpoint.pt',
                volume_head=volume_head if train_volume_head else None,
            )

    # saving last model
    if n_iter == 0:
        raise RuntimeError(
            "No training iterations were run. Check that `epoch` is > 0, the train split is non-empty, "
            "and DataLoader is not dropping all samples with drop_last=True."
        )

    save_model(
        encoder,
        decoder,
        last_epoch,
        optim,
        last_loss,
        param["checkpoint_dir"]+'_'+cfg_fname+'_final_model.pt',
        volume_head=volume_head if train_volume_head else None,
    )

    return


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
        "--overfit",
        dest="overfit",
	    action='store_true',
        help="Overfit the network.",
    )

    arg_parser.add_argument(
        "--checkpoint_decoder",
        dest="checkpoint",
        default="500",
        help="The checkpoint weights to use. This should be a number indicated an epoch",
    )
    arg_parser.add_argument(
        "--decoder",
        dest="decoder",
	    action='store_true',
        help="Update decoder network.",
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
    decoder = net_utils.set_require_grad(decoder, True)

    train_pretrain_path = resolve_supervision_path(
        args.experiment_directory, args.checkpoint, split="train", allow_matrix=True
    )
    val_pretrain_path = resolve_supervision_path(
        args.experiment_directory, args.checkpoint, split="val", allow_matrix=False
    )

    main_function(decoder=decoder,
                  train_pretrain=train_pretrain_path,
                  val_pretrain=val_pretrain_path,
                  cfg=args.cfg,
                  latent_size=latent_size,
                  trunc_val=specs['ClampingDistance'],
		          overfit=args.overfit,
                  update_decoder=args.decoder)
