#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deepsdf.deep_sdf.utils


def create_mesh(
    decoder, latent_vec, filename, start, N=256, max_batch=32 ** 3, offset=None, scale=None,
):
    ply_filename = filename

    decoder.eval()

    # 扩大体素网格范围，防止草莓底部被截断
    # 草莓 SDF 数据 Y 轴最小值可达 -0.59，原始 [-0.5,0.5] 会裁掉底部
    voxel_origin = [-0.7, -0.7, -0.7]
    voxel_size = 1.4 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            deepsdf.deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )
    return end - start


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    print(numpy_3d_sdf_tensor.min(), numpy_3d_sdf_tensor.max())

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # ========== 使用 Open3D 进行连通域过滤和 Laplacian 平滑 ==========
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 过滤连通域，清除高轮数训练时外推产生的伪SDF零等值面囊泡
    try:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        if len(cluster_n_triangles) > 1:
            valid_clusters = []
            vertices_np = np.asarray(mesh.vertices)
            
            for cluster_idx in range(len(cluster_n_triangles)):
                mask = triangle_clusters == cluster_idx
                cluster_triangles = np.asarray(mesh.triangles)[mask]
                unique_vertex_indices = np.unique(cluster_triangles)
                cluster_verts = vertices_np[unique_vertex_indices]
                
                min_bound = cluster_verts.min(axis=0)
                max_bound = cluster_verts.max(axis=0)
                ranges = max_bound - min_bound
                
                # 外围伪壳会撑满整个 [-0.7, 0.7] 盒子（约 1.4 大小）
                is_boundary_artifact = (ranges[0] > 1.33 and ranges[1] > 1.33 and ranges[2] > 1.33)
                
                if not is_boundary_artifact:
                    valid_clusters.append((cluster_idx, cluster_n_triangles[cluster_idx]))
            
            if len(valid_clusters) > 0:
                valid_clusters.sort(key=lambda x: x[1], reverse=True)
                best_cluster_idx = valid_clusters[0][0]
            else:
                best_cluster_idx = cluster_n_triangles.argmax()
                print("[Warning] All components span the full bounding box, falling back to largest cluster.")
                
            triangles_to_remove = triangle_clusters != best_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            
    except Exception as e:
        print("Connected component filtering failed, ignored: ", str(e))

    # Laplacian 平滑: 消除高轮数训练导致的表面波纹
    try:
        mesh.compute_vertex_normals()
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=10, lambda_filter=0.5)
    except Exception as e:
        print("Laplacian smoothing failed, ignored: ", str(e))

    # ========== 使用 plyfile 保存（不写入法线，让可视化器自行计算，避免渲染异常） ==========
    final_verts = np.asarray(mesh.vertices)
    final_faces = np.asarray(mesh.triangles)

    num_verts = final_verts.shape[0]
    num_faces = final_faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(num_verts):
        verts_tuple[i] = tuple(final_verts[i, :])

    faces_building = []
    for i in range(num_faces):
        faces_building.append(((final_faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
