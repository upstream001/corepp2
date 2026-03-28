import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


def KLDivLoss(latents, dataset, device):
    target_distrib = MultivariateNormal(loc=dataset.target_mean.to(device),
                                        covariance_matrix=dataset.target_cov.to(device))

    batch_mean = torch.mean(latents, dim=0)

    # estimating covariance
    samples = latents - batch_mean

    n = latents.shape[0]
    batch_cov = (1/n) * samples.T @ samples

    with torch.no_grad():
        det = torch.det(batch_cov)

    batch_cov += 0.001*torch.eye(12).to(device)

    batch_distrib = MultivariateNormal(loc=batch_mean, covariance_matrix=batch_cov)

    loss = torch.distributions.kl_divergence(batch_distrib, target_distrib).mean()
    return loss, det


def AttRepLoss(latents, fruit_ids, device, delta_rep=.5):
    h_loss = 0
    hinged_loss = torch.nn.HingeEmbeddingLoss(margin=delta_rep, reduction='none')

    torch.set_printoptions(linewidth=500, sci_mode=False)

    for f_id, c_lat in zip(fruit_ids, latents):

        dist = torch.linalg.norm(c_lat - latents, dim=1)
        mask = (fruit_ids == f_id)*2 - 1

        c_loss = hinged_loss(dist,mask.cuda())
        h_loss += c_loss.sum()

    return h_loss


def SuperLoss(pred, gt):
    return nn.MSELoss()(pred.cuda(), gt.cuda())


def SDFLoss(pred, target, target_weights, sdf_trunc, points):

    pred = torch.clamp(pred, min=-sdf_trunc, max=sdf_trunc)

    # import ipdb;ipdb.set_trace()
    # targets = []

    # import utils
    # for i in  range(target.shape[0]):
    #     pnt = points[i].points
    #     tar = target[i]
    #     con = tar.abs() <= 1
    #     # import ipdb;ipdb.set_trace()
    #     data = torch.cat((pnt[con[:,0]], tar[con].unsqueeze(dim=1)), dim=1)
    #     pcd = utils.visualize_sdf(data.detach().cpu().numpy())
    #     targets.append(pcd)

    # import open3d as o3d
    # o3d.visualization.draw_geometries(targets, window_name='all')

    target = target[target_weights !=0 ]
    pred = pred[target_weights !=0 ]

    narrow_band = target.abs() < 1
    target = target[narrow_band]
    pred = pred[narrow_band]

    pred /= sdf_trunc # normalized sdf predd
    pred = log_transform(pred)
    target = log_transform(target)
    
    return nn.L1Loss()(pred, target)


def log_transform(sdf):
        return sdf.sign() * (sdf.abs() + 1.0).log()

def SDFLoss_new(pred, target, target_weights, sdf_trunc, points, alpha=15.0):

    pred = torch.clamp(pred, min=-sdf_trunc, max=sdf_trunc)

    target = target[target_weights !=0 ]
    pred = pred[target_weights !=0 ]

    narrow_band = target.abs() < 1
    target = target[narrow_band]
    pred = pred[narrow_band]

    pred /= sdf_trunc # normalized sdf predd
    
    pred_sdf_log = log_transform(pred)
    target_sdf_log = log_transform(target)
    
    weights = torch.exp(-alpha * torch.abs(target))
    weights = weights / (weights.sum() + 1e-8) * len(weights)  # 归一化保持量纲
    
    loss = (weights * torch.abs(pred_sdf_log - target_sdf_log)).mean()
    return loss

def RegLatentLoss(batch_vecs, code_reg_lambda, epoch):
    loss = torch.abs(1 - torch.norm(batch_vecs, dim=1)).mean()
    loss *= code_reg_lambda
    return loss


def LatentSpreadLoss(pred, gt, eps=1e-6):
    pred_std = torch.sqrt(torch.var(pred, dim=0, unbiased=False) + eps)
    gt_std = torch.sqrt(torch.var(gt.detach(), dim=0, unbiased=False) + eps)
    return nn.MSELoss()(pred_std, gt_std)


def VolumeLoss(pred_volume, target_volume, log_target=True, relative_weight=0.5, eps=1e-6):
    target = target_volume.float().view_as(pred_volume)
    if log_target:
        target_log = torch.log1p(target)
    else:
        target_log = target

    base_loss = nn.SmoothL1Loss()(pred_volume, target_log)

    pred_volume_ml = torch.expm1(pred_volume)
    relative_error = torch.abs(pred_volume_ml - target) / (target.abs() + eps)
    relative_loss = relative_error.mean()

    return (1.0 - relative_weight) * base_loss + relative_weight * relative_loss
