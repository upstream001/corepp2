import os
import json
import argparse
import random

def create_split_file(data_dir, train_ratio=0.8, val_ratio=0.1, output_file=None):
    """
    Reads the complete directory of the point cloud dataset and splits it
    into train, val, and test sets.
    """
    complete_dir = os.path.join(data_dir, "complete")
    if not os.path.exists(complete_dir):
        print(f"Error: Could not find {complete_dir}")
        return

    # List all point cloud files
    all_files = []
    for fname in os.listdir(complete_dir):
        if fname.endswith(".ply"):
            fid = fname[:-4]
            all_files.append(fid)

    all_files.sort()
    # random.shuffle(all_files)  # Shuffle if needed before splitting

    num_files = len(all_files)
    train_end = int(num_files * train_ratio)
    val_end = train_end + int(num_files * val_ratio)

    train_split = all_files[:train_end]
    val_split = all_files[train_end:val_end]
    test_split = all_files[val_end:]

    print(f"Total models: {num_files}")
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")

    # Standard dataloader split (split.json)
    split_dict = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    if output_file is None:
        output_file = os.path.join(data_dir, "split.json")

    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)
        
    print(f"Created split file: {output_file}")


def create_deepsdf_splits(data_dir, split_json_path, output_dir):
    """
    DeepSDF needs separate JSON files for train and test splits, and it expects
    them in a specific format representing the class name and instance names.
    Since we only have one class ('strawberry' or '20260301_dataset'), we'll use a dummy 'fruit'.
    """
    with open(split_json_path, 'r') as f:
        splits = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(data_dir))

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
    parser.add_argument("--data_dir", "-d", default="./data/strawberry", help="Dataset directory containing complete/ and partial/ folders")
    parser.add_argument("--deepsdf_splits_dir", default="./deepsdf/experiments/splits", help="Where to output DeepSDF split json files")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    
    args = parser.parse_args()
    
    create_split_file(args.data_dir, args.train, args.val)
    
    split_file_path = os.path.join(args.data_dir, "split.json")
    create_deepsdf_splits(args.data_dir, split_file_path, args.deepsdf_splits_dir)

