import os
import json
import argparse
import random
import re


def natural_sort_key(name):
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def resolve_dataset_name(complete_dir):
    normalized_dir = os.path.normpath(complete_dir)
    base_name = os.path.basename(normalized_dir)
    if base_name == "complete":
        return os.path.basename(os.path.dirname(normalized_dir))
    return base_name


def resolve_dataset_root(complete_dir):
    normalized_dir = os.path.normpath(complete_dir)
    if os.path.basename(normalized_dir) == "complete":
        return os.path.dirname(normalized_dir)
    return normalized_dir


def load_shape_groups(dataset_root, complete_dir):
    mapping_path = os.path.join(dataset_root, "mapping.json")
    if not os.path.exists(mapping_path):
        all_files = []
        for fname in os.listdir(complete_dir):
            if fname.endswith(".ply"):
                all_files.append(fname[:-4])
        all_files.sort(key=natural_sort_key)
        return {fid: [fid] for fid in all_files}, "file_stem"

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    shape_groups = {}
    for sample_name, complete_name in mapping.items():
        sample_id = os.path.splitext(sample_name)[0]
        complete_id = os.path.splitext(complete_name)[0]
        shape_groups.setdefault(complete_id, []).append(sample_id)

    for sample_ids in shape_groups.values():
        sample_ids.sort(key=natural_sort_key)

    ordered_groups = dict(sorted(shape_groups.items(), key=lambda item: natural_sort_key(item[0])))
    return ordered_groups, "mapping"


def create_split_file(complete_dir, train_ratio=0.8, val_ratio=0.1, output_file=None, seed=42):
    """
    Split complete point clouds at shape level using the complete .ply files
    as the only source of truth.
    """
    if not os.path.isdir(complete_dir):
        raise FileNotFoundError(f"Complete point cloud directory not found: {complete_dir}")

    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1) or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios: require 0 < train < 1, 0 <= val < 1, and train + val < 1.")

    dataset_root = resolve_dataset_root(complete_dir)
    shape_groups, group_source = load_shape_groups(dataset_root, complete_dir)
    shape_ids = list(shape_groups.keys())

    rng = random.Random(seed)
    rng.shuffle(shape_ids)

    num_shapes = len(shape_ids)
    train_end = int(num_shapes * train_ratio)
    val_end = train_end + int(num_shapes * val_ratio)

    train_shapes = shape_ids[:train_end]
    val_shapes = shape_ids[train_end:val_end]
    test_shapes = shape_ids[val_end:]

    train_split = [sample_id for shape_id in train_shapes for sample_id in shape_groups[shape_id]]
    val_split = [sample_id for shape_id in val_shapes for sample_id in shape_groups[shape_id]]
    test_split = [sample_id for shape_id in test_shapes for sample_id in shape_groups[shape_id]]

    print(f"Complete point cloud source: {complete_dir}")
    print(f"Shape grouping source: {group_source}")
    print(f"Total shapes: {num_shapes}")
    print(
        f"Train shapes: {len(train_shapes)}, Val shapes: {len(val_shapes)}, "
        f"Test shapes: {len(test_shapes)}"
    )
    print(f"Expanded samples -> Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
    print(f"Random seed: {seed}")

    # Standard dataloader split (split.json)
    split_dict = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    if output_file is None:
        output_file = os.path.join(resolve_dataset_root(complete_dir), "split.json")

    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)
        
    print(f"Created split file: {output_file}")


def create_deepsdf_splits(dataset_name, split_json_path, output_dir):
    """
    DeepSDF needs separate JSON files for train and test splits, and it expects
    them in a specific format representing the class name and instance names.
    Since we only have one class ('strawberry' or '20260301_dataset'), we'll use a dummy 'fruit'.
    """
    with open(split_json_path, 'r') as f:
        splits = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        if split_name not in splits: continue
        
        # DeepSDF structure: {'DatasetName': {'ClassName': ['instance1', 'instance2', ... ]}}
        # In our case we just pass the file names as instance names.
        dsdf_split = {
            dataset_name: {
                "fruit": splits[split_name] 
            }
        }
        
        out_file = os.path.join(output_dir, f"{dataset_name}_{split_name}.json")
        with open(out_file, 'w') as f:
            json.dump(dsdf_split, f, indent=4)
        print(f"Created DeepSDF split: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complete_dir",
        "-d",
        default="./data/strawberry/complete",
        help="Directory containing the complete shape-level .ply files only",
    )
    parser.add_argument("--deepsdf_splits_dir", default="./deepsdf/experiments/splits", help="Where to output DeepSDF split json files")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to shuffle samples before splitting")
    
    args = parser.parse_args()
    
    create_split_file(args.complete_dir, args.train, args.val, seed=args.seed)
    
    split_file_path = os.path.join(resolve_dataset_root(args.complete_dir), "split.json")
    dataset_name = resolve_dataset_name(args.complete_dir)
    create_deepsdf_splits(dataset_name, split_file_path, args.deepsdf_splits_dir)
