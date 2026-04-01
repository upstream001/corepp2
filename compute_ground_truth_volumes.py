#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np

try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None

try:
    from scipy.spatial import ConvexHull  # type: ignore
except Exception:
    ConvexHull = None


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def volume_to_ml(volume: float, length_unit: str) -> float:
    # Convert cubic units to milliliters (1 mL = 1 cm^3).
    if length_unit == "cm":
        return volume
    if length_unit == "mm":
        return volume / 1000.0
    if length_unit == "m":
        return volume * 1_000_000.0
    raise ValueError(f"Unsupported length unit: {length_unit}")


def mesh_volume_if_possible(mesh) -> float:
    # Open3D computes volume only for watertight meshes.
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0.0
    try:
        return float(mesh.get_volume())
    except Exception:
        return -1.0


def convex_hull_volume_from_vertices(vertices: np.ndarray) -> float:
    if ConvexHull is None:
        return -1.0
    if vertices.shape[0] < 4:
        return 0.0
    try:
        return float(ConvexHull(vertices).volume)
    except Exception:
        return 0.0


def compute_volume_for_ply(path: Path) -> float:
    if trimesh is not None:
        try:
            mesh = trimesh.load(str(path), force="mesh", process=False)
            if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
                return 0.0
            try:
                if mesh.is_volume:
                    return float(abs(mesh.volume))
            except Exception:
                pass
            try:
                return float(abs(mesh.convex_hull.volume))
            except Exception:
                pass
        except Exception as exc:
            print(f"[Warning] trimesh failed on {path.name}: {exc}. Falling back to Open3D.")

    if o3d is None:
        raise RuntimeError(
            "No mesh backend available. Install one of: "
            "`pip install trimesh` or `pip install open3d scipy`."
        )

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.is_empty():
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        v = mesh_volume_if_possible(mesh)
        if v >= 0:
            return v

        vertices = np.asarray(mesh.vertices)
        v_hull = convex_hull_volume_from_vertices(vertices)
        if v_hull >= 0:
            return v_hull

    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.is_empty():
        vertices = np.asarray(pcd.points)
        v_hull = convex_hull_volume_from_vertices(vertices)
        if v_hull >= 0:
            return v_hull

    raise RuntimeError(
        "Unable to compute hull volume without scipy. Install with `pip install scipy`."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute volume for each .ply in a folder and export to CSV."
    )
    parser.add_argument(
        "--input-dir",
        default="/home/tianqi/corepp2/data/45_straw",
        help="Directory containing .ply files.",
    )
    parser.add_argument(
        "--output-csv",
        default="ground_truth.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--length-unit",
        choices=["mm", "cm", "m"],
        default="cm",
        help="Coordinate length unit in .ply files (used to convert volume to mL).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ply_files = sorted(input_dir.glob("*.ply"), key=natural_sort_key)
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {input_dir}")

    rows = []
    for ply_path in ply_files:
        volume_raw = compute_volume_for_ply(ply_path)
        volume_ml = volume_to_ml(volume_raw, args.length_unit)
        rows.append({"filename": ply_path.name, "volume_ml": round(volume_ml, 6)})

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "volume_ml"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv.resolve()}")


if __name__ == "__main__":
    main()
