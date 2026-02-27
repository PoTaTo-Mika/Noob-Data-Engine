# 现在我们有多线程生成好的带有GPU id的内容
# {"file_name": "st_sketch_g0_000000001", "task": "style_transfer", "reference_artist": "poco_(asahi_age)", "converted_pid": 41, "gpu_id": 0}
# 这个脚本主要就是俩任务，一个是修复json文件，一个是修复原始图片。

import json
import os
import argparse
import sys

def process_dataset(directory):
    jsonl_path = os.path.join(directory, "metadata.jsonl")
    
    # 检查 metadata 文件是否存在
    if not os.path.exists(jsonl_path):
        print(f"错误: 找不到文件 {jsonl_path}")
        return

    print(f"正在读取元数据: {jsonl_path} ...")
    
    data_list = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except Exception as e:
        print(f"读取 JSONL 出错: {e}")
        return

    total_files = len(data_list)
    print(f"共读取到 {total_files} 条数据。")

    # ---------------------------------------------------------
    # 1. 排序 (Sorting)
    # ---------------------------------------------------------
    # 这里的逻辑是：先按 gpu_id 从小到大排，如果 gpu_id 相同，则按原始 file_name 字母顺序排
    # 这样可以保证原本在 g0 里的 001 还是排在 002 前面，且 g0 的所有图都在 g1 之前

    print("正在对数据进行排序...")
    data_list.sort(key=lambda x: (int(x.get('gpu_id', 0)), x.get('file_name', '')))

    # ---------------------------------------------------------
    # 2. 生成新文件名并准备重命名 (Mapping)
    # ---------------------------------------------------------
    print("正在生成新文件名索引...")
    
    rename_ops = []  # 存储 (旧路径, 新路径) 的列表
    new_metadata = [] # 存储修复后的 json 对象
    
    NEW_PREFIX = "st_sketch_" 
    
    for index, item in enumerate(data_list, 1):
        old_filename_base = item['file_name']
        old_path = os.path.join(directory, old_filename_base + ".png")  
        # 生成新名字：sk_sketch_000000001 (9位数字，不够补0)
        new_filename_base = f"{NEW_PREFIX}{index:09d}"
        new_path = os.path.join(directory, new_filename_base + ".png")
        
        # 记录文件重命名操作
        rename_ops.append((old_path, new_path))
        item['file_name'] = new_filename_base
        new_metadata.append(item)

    # ---------------------------------------------------------
    # 3. 执行重命名 (Execution)
    # ---------------------------------------------------------
    print("开始重命名图片文件...")
    success_count = 0
    error_count = 0
    
    if rename_ops:
        print(f"示例映射: {os.path.basename(rename_ops[0][0])} -> {os.path.basename(rename_ops[0][1])}")
    
    for old_p, new_p in rename_ops:
        try:
            # 只有当旧文件存在时才重命名
            if os.path.exists(old_p):
                # 防止覆盖已存在的新文件名（虽然按逻辑不太可能，但为了安全）
                if os.path.exists(new_p) and old_p != new_p:
                   print(f"警告: 目标文件已存在，跳过: {new_p}")
                   error_count += 1
                   continue
                
                os.rename(old_p, new_p)
                success_count += 1
            else:
                # 只有当旧文件不存在，且新文件也不存在时才报错
                # 如果新文件已经存在，说明可能之前运行过脚本，这里就不报错了
                if not os.path.exists(new_p):
                    print(f"找不到源文件: {old_p}")
                    error_count += 1
        except OSError as e:
            print(f"重命名失败: {old_p} -> {new_p}, 错误: {e}")
            error_count += 1
            
        # 简单的进度显示
        if success_count % 5000 == 0 and success_count > 0:
            print(f"已处理 {success_count}/{total_files} 个文件...")

    print(f"文件重命名完成。成功: {success_count}, 失败/跳过: {error_count}")

    # ---------------------------------------------------------
    # 4. 写入新的 metadata (Saving)
    # ---------------------------------------------------------
    output_jsonl = os.path.join(directory, "metadata_final.jsonl")
    print(f"正在写入新的元数据文件: {output_jsonl}")
    
    try:
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for item in new_metadata:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("元数据写入完成！")
        print(f"请检查目录下的 metadata_fixed.jsonl 和重命名后的图片。确认无误后可删除旧的 metadata.jsonl。")
        
    except Exception as e:
        print(f"写入 metadata_fixed.jsonl 失败: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="修复多GPU生成的数据集文件名和元数据")
    parser.add_argument("dir", help="包含图片和metadata.jsonl的目录路径")
    
    args = parser.parse_args()
    
    target_dir = args.dir
    if os.path.isdir(target_dir):
        confirm = input(f"即将处理目录: {target_dir}\n这将重命名该目录下的所有PNG文件。确定吗？(y/n): ")
        if confirm.lower() == 'y':
            process_dataset(target_dir)
        else:
            print("操作已取消。")
    else:
        print("提供的路径不是一个有效的目录。")