import argparse
import json
import os
import random
import re
from collections import defaultdict


PRIMARY_GROUP_PATTERN = re.compile(r"^straw(\d+)_(\d+)$")


def natural_sort_key(name):
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def resolve_dataset_name(dataset_root):
    return os.path.basename(os.path.normpath(dataset_root))


def load_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not mapping:
        raise ValueError(f"mapping.json 为空: {mapping_path}")
    return mapping


def build_shape_groups(mapping):
    shape_groups = defaultdict(list)
    for partial_name, complete_name in mapping.items():
        partial_id = os.path.splitext(partial_name)[0]
        complete_id = os.path.splitext(complete_name)[0]
        shape_groups[complete_id].append(partial_id)

    ordered = {}
    for complete_id in sorted(shape_groups.keys(), key=natural_sort_key):
        ordered[complete_id] = sorted(shape_groups[complete_id], key=natural_sort_key)
    return ordered


def validate_group_sizes(shape_groups, expected_partials_per_shape):
    counts = {shape_id: len(sample_ids) for shape_id, sample_ids in shape_groups.items()}
    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        raise ValueError(f"每个完整草莓对应的残缺点云数量不一致: {unique_counts}")

    actual = unique_counts[0]
    if expected_partials_per_shape is not None and actual != expected_partials_per_shape:
        raise ValueError(
            f"期望每个完整草莓对应 {expected_partials_per_shape} 个残缺点云，实际为 {actual}"
        )
    return actual


def build_primary_groups(shape_ids):
    primary_groups = defaultdict(list)
    for shape_id in shape_ids:
        match = PRIMARY_GROUP_PATTERN.match(shape_id)
        if match is None:
            raise ValueError(
                f"完整草莓命名不符合 straw<group>_<variant> 格式: {shape_id}"
            )
        primary_id = int(match.group(1))
        primary_groups[primary_id].append(shape_id)

    ordered = {}
    for primary_id in sorted(primary_groups.keys()):
        ordered[primary_id] = sorted(primary_groups[primary_id], key=natural_sort_key)
    return ordered


def split_primary_groups(primary_groups, train_count, test_count, val_count, seed):
    total = sum(len(shape_ids) for shape_ids in primary_groups.values())
    if train_count + test_count + val_count != total:
        raise ValueError(
            f"切分数量之和必须等于完整草莓总数: "
            f"{train_count} + {test_count} + {val_count} != {total}"
        )

    unique_group_sizes = sorted({len(shape_ids) for shape_ids in primary_groups.values()})
    if len(unique_group_sizes) != 1:
        raise ValueError(f"原始草莓组大小不一致: {unique_group_sizes}")
    group_size = unique_group_sizes[0]
    if train_count % group_size != 0 or test_count % group_size != 0 or val_count % group_size != 0:
        raise ValueError(
            f"train/test/val 数量必须都是原始草莓组大小 {group_size} 的整数倍，"
            f"当前为 {train_count}/{test_count}/{val_count}"
        )

    group_ids = list(primary_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    train_group_count = train_count // group_size
    test_group_count = test_count // group_size
    val_group_count = val_count // group_size

    train_group_ids = group_ids[:train_group_count]
    test_group_ids = group_ids[train_group_count:train_group_count + test_group_count]
    val_group_ids = group_ids[train_group_count + test_group_count:]

    if len(val_group_ids) != val_group_count:
        raise ValueError("分组后得到的验证集组数不正确")

    train_shapes = flatten_grouped_shapes(primary_groups, train_group_ids)
    test_shapes = flatten_grouped_shapes(primary_groups, test_group_ids)
    val_shapes = flatten_grouped_shapes(primary_groups, val_group_ids)
    return train_shapes, test_shapes, val_shapes, train_group_ids, test_group_ids, val_group_ids, group_size


def flatten_grouped_shapes(primary_groups, group_ids):
    shapes = []
    for group_id in group_ids:
        shapes.extend(primary_groups[group_id])
    return shapes


def flatten_samples(shape_groups, shape_ids):
    samples = []
    for shape_id in shape_ids:
        samples.extend(shape_groups[shape_id])
    return samples


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def create_split_json(dataset_root, train_shapes, test_shapes, val_shapes, shape_groups):
    split_payload = {
        "train": flatten_samples(shape_groups, train_shapes),
        "test": flatten_samples(shape_groups, test_shapes),
        "val": flatten_samples(shape_groups, val_shapes),
    }
    output_path = os.path.join(dataset_root, "split.json")
    write_json(output_path, split_payload)
    return output_path, split_payload


def create_deepsdf_splits(dataset_name, output_dir, split_payload):
    os.makedirs(output_dir, exist_ok=True)
    split_sets = {
        "train": split_payload["train"],
        "test": split_payload["test"],
        "val": split_payload["val"],
    }

    created = []
    for split_name, shape_ids in split_sets.items():
        payload = {
            dataset_name: {
                "fruit": list(shape_ids),
            }
        }
        output_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.json")
        write_json(output_path, payload)
        created.append(output_path)
    return created


def main():
    parser = argparse.ArgumentParser(
        description="按完整草莓级别切分数据集，并基于 mapping.json 展开到残缺点云 split。"
    )
    parser.add_argument(
        "--dataset_root",
        default="/home/tianqi/corepp2/data/20260622_dataset",
        help="包含 complete/、partial/ 和 mapping.json 的数据集根目录",
    )
    parser.add_argument(
        "--train_count",
        type=int,
        default=128,
        help="训练集完整草莓数量",
    )
    parser.add_argument(
        "--test_count",
        type=int,
        default=64,
        help="测试集完整草莓数量",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=48,
        help="验证集完整草莓数量",
    )
    parser.add_argument(
        "--partials_per_shape",
        type=int,
        default=6,
        help="每个完整草莓对应的残缺点云数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机打乱完整草莓顺序时使用的种子",
    )
    parser.add_argument(
        "--deepsdf_splits_dir",
        default="/home/tianqi/corepp2/deepsdf/experiments/splits",
        help="DeepSDF split json 输出目录",
    )
    args = parser.parse_args()

    mapping_path = os.path.join(args.dataset_root, "mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"找不到 mapping.json: {mapping_path}")

    mapping = load_mapping(mapping_path)
    shape_groups = build_shape_groups(mapping)
    partials_per_shape = validate_group_sizes(shape_groups, args.partials_per_shape)
    shape_ids = list(shape_groups.keys())
    primary_groups = build_primary_groups(shape_ids)

    (
        train_shapes,
        test_shapes,
        val_shapes,
        train_group_ids,
        test_group_ids,
        val_group_ids,
        shapes_per_primary_group,
    ) = split_primary_groups(
        primary_groups,
        args.train_count,
        args.test_count,
        args.val_count,
        args.seed,
    )

    split_json_path, split_payload = create_split_json(
        args.dataset_root,
        train_shapes,
        test_shapes,
        val_shapes,
        shape_groups,
    )

    dataset_name = resolve_dataset_name(args.dataset_root)
    deepsdf_files = create_deepsdf_splits(dataset_name, args.deepsdf_splits_dir, split_payload)

    print(f"数据集: {args.dataset_root}")
    print(f"完整草莓总数: {len(shape_ids)}")
    print(f"原始草莓组总数: {len(primary_groups)}")
    print(f"每个原始草莓组包含完整草莓数: {shapes_per_primary_group}")
    print(f"每个完整草莓对应残缺点云数: {partials_per_shape}")
    print(
        f"完整草莓切分 -> train: {len(train_shapes)}, "
        f"test: {len(test_shapes)}, val: {len(val_shapes)}"
    )
    print(
        f"原始草莓组切分 -> train: {len(train_group_ids)}, "
        f"test: {len(test_group_ids)}, val: {len(val_group_ids)}"
    )
    print(
        f"展开后残缺点云 -> train: {len(split_payload['train'])}, "
        f"test: {len(split_payload['test'])}, val: {len(split_payload['val'])}"
    )
    print(f"随机种子: {args.seed}")
    print(f"split.json: {split_json_path}")
    for path in deepsdf_files:
        print(f"DeepSDF split: {path}")


if __name__ == "__main__":
    main()
