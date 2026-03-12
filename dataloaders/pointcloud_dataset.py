import os
import torch
import numpy as np
import open3d as o3d
import random

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, pad_size, pretrain=None, split=None, use_partial=False, supervised_3d=True):
        """
        专用点云数据集加载类，可用于加载草莓数据集、20260301_dataset 等任意仅包含 complete/partial 结构的数据集。
        
        Args:
            data_source: 数据集根目录, 下方应包含 `complete` 目录 (或 `partial` 目录)
            pad_size: 采样后的点集大小（即喂入 Encoder 的点数）
            pretrain: 包含由 DeepSDF 预先计算好的 Latent Code (.pth) 的文件夹路径
            split: (用于兼容参数) 当前可以不使用，因为我们直接读取文件夹下的所有 ply
            use_partial: 如果为 True, 则读取 `partial` 文件夹而不是 `complete` 文件夹
            supervised_3d: 如果为 True, 会读取 pretrain 下的 latent code 作为 ground truth
        """
        self.data_source = data_source
        self.pad_size = pad_size
        self.split = split
        self.use_partial = use_partial
        self.supervised_3d = supervised_3d
        
        if self.supervised_3d and pretrain is not None:
            self.latents_dict = self.get_latents_dict(pretrain)
        else:
            self.latents_dict = {}
            
        self.files = self.get_instance_filenames()

    def get_latents_dict(self, path):
        latent_dictionary = {}
        if not os.path.exists(path):
            print(f"[Warning] Latent codes path {path} does not exist.")
            return latent_dictionary

        # Support Unified Latent Matrix (.pth file instead of directory)
        if os.path.isfile(path) and path.endswith('.pth'):
            try:
                latents_file = torch.load(path)
                if 'latent_codes' in latents_file and 'weight' in latents_file['latent_codes']:
                    latents_matrix = latents_file['latent_codes']['weight']
                    import json
                    # 尝试从 DeepSDF 对应的划分表中寻找名称以对齐索引 (例如 deepsdf/experiments/splits/xxx_train.json)
                    base_exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(path))) # 回退至 experiments 级别
                    ds_name = os.path.basename(os.path.dirname(os.path.dirname(path))) # 例如 20260301_dataset_aug
                    split_type = self.split if self.split else 'train'
                    split_file = os.path.join(base_exp_dir, 'splits', f"{ds_name}_{split_type}.json")
                    
                    if os.path.exists(split_file):
                        with open(split_file, 'r') as f:
                            splits_data = json.load(f)
                            
                        # 解析 DeepSDF 预设的结构 {"DatasetName": {"ClassName": [list of instances]}}
                        instance_names = []
                        if ds_name in splits_data:
                            class_dict = splits_data[ds_name]
                            if len(class_dict) == 1:
                                class_key = list(class_dict.keys())[0]
                                instance_names = class_dict[class_key]
                            elif 'fruit' in class_dict:
                                instance_names = class_dict['fruit']
                                
                        if len(instance_names) == latents_matrix.shape[0]:
                            for i, inst in enumerate(instance_names):
                                latent_dictionary[inst] = latents_matrix[i].squeeze()
                            print(f"[Info] Successfully mapped {len(instance_names)} latent codes from matrix {path} via {split_type} split.")
                        else:
                            print(f"[Warning] Matrix rows ({latents_matrix.shape[0]}) mismatch DeepSDF split size ({len(instance_names)}).")
                    else:
                        print(f"[Warning] Cannot find DeepSDF split definition at {split_file} to map the latent matrix.")
                return latent_dictionary
            except Exception as e:
                print(f"[Error] Failed to load unified latent file {path}: {e}")
                return latent_dictionary

        # Support Directory with individual .pth files (Legacy reconstruct optimization)
        for fname in os.listdir(path):
            if fname.endswith('.pth'):
                latent = torch.load(os.path.join(path, fname))
                key = fname[:-4]
                latent_dictionary[key] = latent
        return latent_dictionary

    def get_instance_filenames(self):
        import json
        subfolder = 'partial' if self.use_partial else 'complete'
        pcd_dir = os.path.join(self.data_source, subfolder)
        
        allowed_keys = None
        if self.split is not None:
            split_file = os.path.join(self.data_source, 'split.json')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    splits = json.load(f)
                    if self.split in splits:
                        allowed_keys = set(splits[self.split])
                    else:
                        print(f"[Warning] Split '{self.split}' not found in {split_file}")
            else:
                print(f"[Warning] Split file {split_file} not found, using all files.")

        files = []
        if os.path.exists(pcd_dir):
            for fname in sorted(os.listdir(pcd_dir)):
                if not fname.endswith('.ply'): 
                    continue
                key = fname[:-4]
                
                # 如果启用了 split，过滤掉不属于该 split 的文件
                if allowed_keys is not None and key not in allowed_keys:
                    continue
                
                # 如果开启了 supervised_3d，必须确保该点云有对应的 latent code 存在
                if self.supervised_3d:
                    if key in self.latents_dict:
                        files.append(os.path.join(pcd_dir, fname))
                    else:
                        print(f"[Warning] Found {fname} but no corresponding latent code in {key}.pth")
                else:
                    files.append(os.path.join(pcd_dir, fname))
        else:
            print(f"[Error] Data directory {pcd_dir} does not exist!")
            
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        fruit_id = os.path.basename(file_path)[:-4]
        
        # Load Point Cloud
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        num_points = points.shape[0]
        
        # 降采样或随机采样，以匹配 pad_size
        if num_points >= self.pad_size:
            random_list = np.random.choice(num_points, size=self.pad_size, replace=False)
            sampled_points = points[random_list, :]
        else:
            # 如果点数不够，则有放回地随机采样进行补全
            random_list = np.random.choice(num_points, size=self.pad_size, replace=True)
            sampled_points = points[random_list, :]

        # 构造返回字典
        item = {
            'fruit_id': fruit_id,
            'target_pcd': torch.Tensor(sampled_points).float(), # 同样返回切分后的长度，避免 collate_fn 报错
            'partial_pcd': torch.Tensor(sampled_points).float(),
            'bbox': {
                'min': torch.Tensor([0.0, 0.0, 0.0]),
                'max': torch.Tensor([1.0, 1.0, 1.0])
            }
        }

        # 填充一些 dummy data 避免依赖 MaskedCameraLaserData 返回格式的地方报错 (按需使用)
        item['rgb'] = torch.zeros((3, self.pad_size, self.pad_size))
        item['depth'] = torch.zeros((1, self.pad_size, self.pad_size))
        item['mask'] = torch.zeros((1, self.pad_size, self.pad_size))

        if self.supervised_3d:
            trained_latent = self.latents_dict[fruit_id]
            item['latent'] = trained_latent.squeeze().float()
        item['frame_id'] = str(fruit_id)
        return item

if __name__ == '__main__':
    # Test script
    ds = PointCloudDataset("/home/tianqi/corepp2/data/strawberry", pad_size=2048, supervised_3d=False)
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        item = ds[0]
        print(f"Sample item partial_pcd shape: {item['partial_pcd'].shape}")
