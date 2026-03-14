import os
import shutil
import json
from tqdm import tqdm
from pathlib import Path

def main():
    # 输入目录
    complete_pc_dir = '/home/tianqi/corepp2/data/scanned_straw_meshed_resize_ml'
    partial_dirs_base = [
        '/home/tianqi/corepp2/data/render_output_perspective',
        '/home/tianqi/corepp2/data/render_output'
    ]
    
    # 输出目录
    output_base_dir = '/home/tianqi/corepp2/data/20260312_dataset'
    partial_out_dir = os.path.join(output_base_dir, 'partial')
    complete_out_dir = os.path.join(output_base_dir, 'complete')
    
    # 创建目录
    os.makedirs(partial_out_dir, exist_ok=True)
    os.makedirs(complete_out_dir, exist_ok=True)

    # 获取所有完整点云文件
    complete_files = sorted([f for f in os.listdir(complete_pc_dir) if f.endswith('.ply')])
    
    global_count = 0
    mapping = {}
    
    print(f"找到 {len(complete_files)} 个完整点云模型。")
    
    for complete_file in complete_files:
        model_name = Path(complete_file).stem
        complete_path = os.path.join(complete_pc_dir, complete_file)
        
        print(f"\n正在处理模型: {model_name}")
        
        # 对于每个完整模型，去各个 partial 里面找同名文件夹
        for base_dir in partial_dirs_base:
            folder_path = os.path.join(base_dir, model_name)
            
            if not os.path.exists(folder_path):
                print(f"  跳过不存在的文件夹: {folder_path}")
                continue
                
            # 获取该文件夹下的所有 partial 点云
            partial_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.ply')])
            print(f"  从 {os.path.basename(base_dir)} 中找到 {len(partial_files)} 个分量...")
            
            for part_file in tqdm(partial_files, leave=False):
                src_partial = os.path.join(folder_path, part_file)
                
                # 生成新的统一文件名
                new_filename = f"{global_count:05d}.ply"
                
                dst_partial = os.path.join(partial_out_dir, new_filename)
                dst_complete = os.path.join(complete_out_dir, new_filename)
                
                # 复制文件
                shutil.copy(src_partial, dst_partial)
                shutil.copy(complete_path, dst_complete)
                
                mapping[new_filename] = complete_file
                
                global_count += 1

    json_path = os.path.join(output_base_dir, 'mapping.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    print(f"\n数据集构造完成，总计生成 {global_count} 对数据。")
    print(f"映射文件已保存至: {json_path}")
    print(f"输出路径: {output_base_dir}")

if __name__ == "__main__":
    main()
