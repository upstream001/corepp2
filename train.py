#!/usr/bin/env python3
import torch
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
from loss import KLDivLoss, SuperLoss, SDFLoss, SDFLoss_new, RegLatentLoss, AttRepLoss
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


def main_function(decoder, pretrain, cfg, latent_size, trunc_val, overfit, update_decoder):

    if DEBUG:
        torch.manual_seed(133)
        random.seed(133)
        np.random.seed(133)

    cfg_fname = cfg.split('/')[-1].replace('.json', '')  # getting filename

    with open(cfg) as json_file:
        param = json.load(json_file)

    check_direxcist(param["checkpoint_dir"])
    device = 'cuda'
    shuffle = True
    last_rmse = np.inf

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

    if param['encoder'] in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
        cl_dataset = PointCloudDataset(
            data_source=param["data_dir"],
            pad_size=param["input_size"],
            pretrain=pretrain,
            use_partial=False,
            supervised_3d=param["supervised_3d"]
        )
    else:
        cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                            tf=default_tf,
                                            color_tf = color_tf, 
                                            pretrain=pretrain,
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
        params = list(encoder.parameters()) #+ list(decoder.parameters())
    
    optim = torch.optim.Adam(params, lr=param["lr"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.97)

    print('\ncfg: ', json.dumps(param, indent=4), '\n')
    print(encoder)
    print(decoder)

    # import ipdb; ipdb.set_trace()
    n_iter = 0  # used for tensorboard
    for e in range(param["epoch"]):
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
                loss += loss_super

                # logging
                writer.add_scalar('Loss/Train/SuperLoss', loss_super, n_iter)
                logging_string += ' -- loss super: {}'.format(loss_super.item())
            
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

                deepsdf_input = torch.zeros((current_batch_size, grid_density**3, latent_size+3))
                for batch_idx, (latent, grid) in enumerate(zip(latent_batch, grid_batch)):
                    deepsdf_input[batch_idx] = torch.cat([latent.expand(grid.points.size(0), -1), grid.points], dim=1)

                deepsdf_input = deepsdf_input.to(device, latent.dtype)

                # decoding
                pred_sdf = decoder(deepsdf_input)

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

                if param['encoder'] in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
                    val_cl_dataset = PointCloudDataset(
                        data_source=param["data_dir"],
                        pad_size=param["input_size"],
                        pretrain=pretrain,
                        use_partial=False,
                        supervised_3d=param["supervised_3d"]
                    )
                else:
                    val_cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                                            tf=val_tf,
                                                            color_tf = None, 
                                                            pretrain=pretrain,
                                                            pad_size=param["input_size"],
                                                            detection_input=param["detection_input"],
                                                            normalize_depth=param["normalize_depth"],
                                                            depth_min=param["depth_min"],
                                                            depth_max=param["depth_max"],
                                                            supervised_3d=param["supervised_3d"],
                                                            sdf_loss=param["3D_loss"],
                                                            grid_density=param["grid_density"],
                                                            split='val',
                                                            overfit=overfit,
                                                            species=param["species"]
                                                            )

                val_dataset = DataLoader(val_cl_dataset, batch_size=1, shuffle=False)

                val_losses = []
                print('\nvalidation...')
                for _, item in enumerate(tqdm(iter(val_dataset))):
                    try:
                        if param['encoder'] not in ['point_cloud', 'point_cloud_large', 'foldnet', 'pointnext']:
                            encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
                        else:
                            encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device)

                        # encoding
                        latent_val = encoder(encoder_input)
                        
                        target_latent = item['latent'].to(device)
                        mse = torch.nn.functional.mse_loss(latent_val.squeeze(), target_latent.squeeze())
                        val_losses.append(mse.item())

                        if args.overfit: break                        
                    except Exception as e:
                        pass

                if len(val_losses) > 0:
                    rmse_volume = sum(val_losses) / len(val_losses)
                else:
                    rmse_volume = float('inf')
                print('Mean Validation Latent MSE: ', round(rmse_volume, 5))

            # logging
            writer.add_scalar('Val/rmse_volume', rmse_volume, n_iter)
            # saving best model
            if rmse_volume < last_rmse:
                last_rmse = rmse_volume
                save_model(encoder, decoder, e, optim, loss, param["checkpoint_dir"]+'_'+cfg_fname+'_best_model.pt')
                print('saving best model')
            print()

        # saving checkpoints
        if (e+1) % param["checkpoint_frequency"] == 0:
            save_model(encoder, decoder, e, optim, loss,  param["checkpoint_dir"]+'_'+cfg_fname+'_checkpoint.pt')

    # saving last model
    save_model(encoder, decoder, e, optim, loss,  param["checkpoint_dir"]+'_'+cfg_fname+'_final_model.pt')

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

    # 优先加载 DeepSDF 各 Epoch 训练时同步生成的全局 Latent 矩阵真值 (跳过单独解构环节)
    pretrain_matrix = os.path.join(args.experiment_directory, ws.latent_codes_subdir, args.checkpoint + ".pth")
    # 向下兼容使用 reconstruct 脚本抽取的独立部分结果
    pretrain_path_partial = os.path.join(args.experiment_directory, 'Reconstructions', args.checkpoint, 'Codes', 'partial')
    pretrain_path_complete = os.path.join(args.experiment_directory, 'Reconstructions', args.checkpoint, 'Codes', 'complete')
    
    if os.path.exists(pretrain_matrix):
        pretrain_path = pretrain_matrix
    elif os.path.exists(pretrain_path_partial):
        pretrain_path = pretrain_path_partial
    elif os.path.exists(pretrain_path_complete):
        pretrain_path = pretrain_path_complete
    else:
        pretrain_path = pretrain_matrix # fallback to matrix

    main_function(decoder=decoder,
                  pretrain=pretrain_path,
                  cfg=args.cfg,
                  latent_size=latent_size,
                  trunc_val=specs['ClampingDistance'],
		          overfit=args.overfit,
                  update_decoder=args.decoder)