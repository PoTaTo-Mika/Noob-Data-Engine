import os
import json
import argparse
import webdataset as wds

def create_dataset_tar(input_dir, output_tar):
    """
    收集目录下的 webp 文件，为其生成对应的 pid JSON，并打包为新的 tar 文件。
    """
    webp_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".webp")]
    
    if not webp_files:
        print(f"警告：在目录 {input_dir} 中未找到任何 webp 文件。")
        return

    print(f"开始处理，共发现 {len(webp_files)} 个文件。")

    # 使用 webdataset.TarWriter 进行打包
    with wds.TarWriter(output_tar) as sink:
        for i, filename in enumerate(webp_files):
            pid = os.path.splitext(filename)[0]
            file_path = os.path.join(input_dir, filename)
            
            # 直接读取和写入，减少 try-except
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            # 构造 JSON 元数据: {"pid": "xxxx"}
            metadata = {"pid": pid}
            
            # 写入样本
            sink.write({
                "__key__": pid,
                "json": metadata,
                "webp": image_data
            })
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(webp_files)} 个文件...")

    print(f"数据处理并打包完成：{output_tar}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 WebP 图片集合生成 JSON 元数据并打包成 WebDataset 格式的 Tar 包")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入目录，包含 webp 文件")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 tar 文件路径")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    create_dataset_tar(args.input, args.output)
