#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd


def natural_key(text):
    parts = []
    token = ""
    is_digit = None
    for ch in str(text):
        ch_is_digit = ch.isdigit()
        if is_digit is None or ch_is_digit == is_digit:
            token += ch
        else:
            parts.append(int(token) if is_digit else token.lower())
            token = ch
        is_digit = ch_is_digit
    if token:
        parts.append(int(token) if is_digit else token.lower())
    return parts


def load_pairs_from_csv(results_csv):
    df = pd.read_csv(results_csv)
    pairs = []
    for _, row in df.iterrows():
        source_path = str(row.get("source_path", "")).strip()
        aligned_mesh_path = str(row.get("aligned_mesh_path", "")).strip()
        mesh_path = aligned_mesh_path or str(row.get("mesh_path", "")).strip()
        frame_id = str(row.get("frame_id", "")).strip()
        if not source_path or not mesh_path:
            continue
        if not Path(source_path).exists() or not Path(mesh_path).exists():
            continue
        pairs.append(
            {
                "frame_id": frame_id or Path(source_path).stem,
                "source_path": source_path,
                "mesh_path": mesh_path,
            }
        )
    return sorted(pairs, key=lambda item: natural_key(item["frame_id"]))


def load_pairs_from_dirs(input_dir, mesh_dir):
    input_dir = Path(input_dir)
    mesh_dir = Path(mesh_dir)
    mesh_map = {p.stem: p for p in mesh_dir.glob("*.ply")}
    pairs = []
    for p in sorted(input_dir.glob("*.ply"), key=lambda path: natural_key(path.stem)):
        mesh_path = mesh_map.get(p.stem)
        if mesh_path is None:
            continue
        pairs.append(
            {
                "frame_id": p.stem,
                "source_path": str(p),
                "mesh_path": str(mesh_path),
            }
        )
    return pairs


def shift_geometries(partial, mesh, gap_scale=1.6):
    partial_bbox = partial.get_axis_aligned_bounding_box()
    mesh_bbox = mesh.get_axis_aligned_bounding_box()
    partial_extent = partial_bbox.get_extent()
    mesh_extent = mesh_bbox.get_extent()
    max_extent_x = max(float(partial_extent[0]), float(mesh_extent[0]), 1.0)
    offset = np.array([max_extent_x * gap_scale, 0.0, 0.0], dtype=np.float64)

    partial = o3d.geometry.PointCloud(partial)
    mesh = o3d.geometry.TriangleMesh(mesh)
    partial.translate(-offset / 2.0)
    mesh.translate(offset / 2.0)
    return partial, mesh


class UnseenReconstructionViewer:
    def __init__(self, pairs):
        self.pairs = pairs
        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.partial = None
        self.mesh = None

    def _load_current(self):
        pair = self.pairs[self.index]
        partial = o3d.io.read_point_cloud(pair["source_path"])
        mesh = o3d.io.read_triangle_mesh(pair["mesh_path"])
        mesh.compute_vertex_normals()

        partial.paint_uniform_color([0.88, 0.25, 0.20])
        # Match test.py/Open3D default appearance: do not override mesh color here.
        # If the mesh file ever contains per-vertex colors, Open3D will render them directly.
        partial, mesh = shift_geometries(partial, mesh)
        return pair, partial, mesh

    def _render_current(self, reset_view=False):
        if self.partial is not None:
            self.vis.remove_geometry(self.partial, reset_bounding_box=False)
        if self.mesh is not None:
            self.vis.remove_geometry(self.mesh, reset_bounding_box=False)

        pair, self.partial, self.mesh = self._load_current()
        self.vis.add_geometry(self.partial, reset_bounding_box=reset_view)
        self.vis.add_geometry(self.mesh, reset_bounding_box=False)

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = 4.0
        opt.mesh_show_back_face = False

        print(
            f"[{self.index + 1}/{len(self.pairs)}] {pair['frame_id']} | "
            f"left=partial right=mesh"
        )

    def next_callback(self, vis):
        self.index = (self.index + 1) % len(self.pairs)
        self._render_current(reset_view=False)
        return False

    def prev_callback(self, vis):
        self.index = (self.index - 1) % len(self.pairs)
        self._render_current(reset_view=False)
        return False

    def run(self):
        self.vis.create_window(
            window_name="Unseen Reconstruction Viewer (N: next, P: prev, Q: quit)",
            width=1440,
            height=840,
        )
        self.vis.add_geometry(self.axis, reset_bounding_box=True)
        self.vis.register_key_callback(ord("N"), self.next_callback)
        self.vis.register_key_callback(ord("P"), self.prev_callback)
        self.vis.register_key_callback(ord("Q"), lambda vis: vis.destroy_window())
        self._render_current(reset_view=True)
        self.vis.run()
        self.vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize unseen partial point clouds and reconstructed meshes side by side."
    )
    parser.add_argument(
        "--results-csv",
        default="/home/tianqi/corepp2/unseen_output2/unseen_results.csv",
        help="CSV produced by test_unseen_data.py.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Optional directory of input partial point clouds. Used only if results CSV is unavailable.",
    )
    parser.add_argument(
        "--mesh-dir",
        default=None,
        help="Optional directory of reconstructed meshes. Used only if results CSV is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_csv = Path(args.results_csv)

    if results_csv.exists():
        pairs = load_pairs_from_csv(results_csv)
    else:
        if args.input_dir is None or args.mesh_dir is None:
            raise FileNotFoundError(
                f"Results CSV not found: {results_csv}. "
                "Please provide --input-dir and --mesh-dir."
            )
        pairs = load_pairs_from_dirs(args.input_dir, args.mesh_dir)

    if not pairs:
        raise RuntimeError("No valid partial/mesh pairs found for visualization.")

    viewer = UnseenReconstructionViewer(pairs)
    viewer.run()


if __name__ == "__main__":
    main()
