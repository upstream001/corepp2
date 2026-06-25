#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import math
import re
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm


GROUPED_NAME_PATTERN = re.compile(r"^straw(\d+)_(\d+)$")


def build_rotation_matrix(x_deg: float, y_deg: float) -> np.ndarray:
    x_rad = math.radians(x_deg)
    y_rad = math.radians(y_deg)
    return o3d.geometry.get_rotation_matrix_from_xyz((x_rad, y_rad, 0.0))


def transform_geometry(
    path: Path,
    output_path: Path,
    x_deg: float,
    y_deg: float,
    scale_factor: float,
) -> None:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_vertices():
        center = mesh.get_center()
        mesh.rotate(build_rotation_matrix(x_deg, y_deg), center=center)
        mesh.scale(scale_factor, center=center)
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False)
        return

    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise ValueError(f"无法读取有效点云或网格: {path}")

    center = pcd.get_center()
    pcd.rotate(build_rotation_matrix(x_deg, y_deg), center=center)
    pcd.scale(scale_factor, center=center)
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)


def parse_ply_name(path: Path) -> tuple[str, int, int]:
    stem = path.stem
    if stem.isdigit():
        index = int(stem)
        return ("numeric", index, 0)

    match = GROUPED_NAME_PATTERN.match(stem)
    if match:
        group_id, variant_id = match.groups()
        return ("grouped", int(group_id), int(variant_id))

    raise ValueError(
        f"不支持的文件名格式: {path.name}，仅支持纯数字或 straw<group>_<variant>.ply"
    )


def build_output_name(name_mode: str, primary_id: int, secondary_id: int) -> str:
    if name_mode == "numeric":
        return f"{primary_id}.ply"
    if name_mode == "grouped":
        return f"straw{primary_id}_{secondary_id}.ply"
    raise ValueError(f"未知命名模式: {name_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将输入目录中的 .ply 文件复制到新目录，并为每个文件额外生成一个随机旋转缩放后的增强版本。"
    )
    parser.add_argument(
        "--input_dir",
        default="/home/tianqi/corepp2/data/55_straw",
        help="输入点云目录，目录内应为纯数字命名的 .ply 文件",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/tianqi/corepp2/data/55_straw_rotated",
        help="输出目录，脚本会保留原文件并追加同数量的增强文件",
    )
    parser.add_argument(
        "--min_angle",
        type=float,
        default=15.0,
        help="x/y 轴随机旋转最小角度（度）",
    )
    parser.add_argument(
        "--max_angle",
        type=float,
        default=30.0,
        help="x/y 轴随机旋转最大角度（度）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42；修改后可得到不同旋转结果",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="输出文件起始编号，默认 1",
    )
    parser.add_argument(
        "--min_scale",
        type=float,
        default=0.9,
        help="随机缩放最小倍数，默认 0.9",
    )
    parser.add_argument(
        "--max_scale",
        type=float,
        default=1.1,
        help="随机缩放最大倍数，默认 1.1",
    )
    parser.add_argument(
        "--num_augments",
        type=int,
        default=1,
        help="每个输入文件额外生成多少个增强版本，默认 1",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if args.min_angle > args.max_angle:
        raise ValueError("--min_angle 不能大于 --max_angle")
    if args.min_scale > args.max_scale:
        raise ValueError("--min_scale 不能大于 --max_scale")
    if args.num_augments < 1:
        raise ValueError("--num_augments 至少为 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_files = sorted(input_dir.glob("*.ply"), key=parse_ply_name)
    if not ply_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 .ply 文件")
    if any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请先清空或换一个目录: {output_dir}")

    rng = np.random.default_rng(args.seed)
    num_files = len(ply_files)
    first_mode, _, _ = parse_ply_name(ply_files[0])
    if any(parse_ply_name(path)[0] != first_mode for path in ply_files):
        raise ValueError("输入目录中的文件名格式不一致，不能混合数字命名和分组命名")

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"旋转范围: x/y 轴 [{args.min_angle}, {args.max_angle}] 度")
    print(f"缩放范围: [{args.min_scale}, {args.max_scale}] 倍")
    print(f"随机种子: {args.seed}")
    print(f"每个输入文件新增增强版本数: {args.num_augments}")

    if first_mode == "numeric":
        print(
            f"命名规则: 原文件 {args.start_index}-{args.start_index + num_files - 1}.ply, "
            f"增强文件总数 {num_files * args.num_augments}"
        )
        for idx, ply_path in enumerate(tqdm(ply_files, desc="Rotating")):
            original_out = output_dir / build_output_name(
                "numeric", args.start_index + idx, 0
            )
            transform_geometry(ply_path, original_out, 0.0, 0.0, 1.0)

            generated_names = []
            for aug_idx in range(args.num_augments):
                rotated_index = args.start_index + num_files * (aug_idx + 1) + idx
                rotated_out = output_dir / build_output_name("numeric", rotated_index, 0)
                x_deg = float(rng.uniform(args.min_angle, args.max_angle))
                y_deg = float(rng.uniform(args.min_angle, args.max_angle))
                scale_factor = float(rng.uniform(args.min_scale, args.max_scale))
                transform_geometry(ply_path, rotated_out, x_deg, y_deg, scale_factor)
                generated_names.append(
                    f"{rotated_out.name}(x={x_deg:.3f}, y={y_deg:.3f}, scale={scale_factor:.4f})"
                )

            print(f"{ply_path.name}: original -> {original_out.name}, augments -> {', '.join(generated_names)}")
    else:
        grouped_files: dict[int, list[tuple[int, Path]]] = {}
        for ply_path in ply_files:
            _, group_id, variant_id = parse_ply_name(ply_path)
            grouped_files.setdefault(group_id, []).append((variant_id, ply_path))

        for group_id, entries in sorted(grouped_files.items()):
            entries.sort(key=lambda item: item[0])
            next_variant = 1
            for variant_id, ply_path in entries:
                original_out = output_dir / build_output_name("grouped", group_id, variant_id)
                transform_geometry(ply_path, original_out, 0.0, 0.0, 1.0)
                next_variant = max(next_variant, variant_id + 1)

            for variant_id, ply_path in tqdm(entries, desc=f"Rotating straw{group_id}", leave=False):
                generated_names = []
                for _ in range(args.num_augments):
                    rotated_out = output_dir / build_output_name("grouped", group_id, next_variant)
                    x_deg = float(rng.uniform(args.min_angle, args.max_angle))
                    y_deg = float(rng.uniform(args.min_angle, args.max_angle))
                    scale_factor = float(rng.uniform(args.min_scale, args.max_scale))
                    transform_geometry(ply_path, rotated_out, x_deg, y_deg, scale_factor)
                    generated_names.append(
                        f"{rotated_out.name}(x={x_deg:.3f}, y={y_deg:.3f}, scale={scale_factor:.4f})"
                    )
                    next_variant += 1

                print(
                    f"{ply_path.name}: original kept, augments -> {', '.join(generated_names)}"
                )

    total_outputs = num_files * (args.num_augments + 1)
    print(f"\n处理完成，共输出 {total_outputs} 个文件到: {output_dir}")


if __name__ == "__main__":
    main()
