import os
import tarfile
from pathlib import Path

def pack_images_to_tar(input_folder, output_folder, prefix, batch_size=10000):

    # 将路径转换为 Path 对象以便跨平台处理
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 如果目标输出目录不存在，则自动创建
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"[INFO] 目标目录不存在，已创建: {output_path}")

    # 1. 搜集所有的 webp 文件并排序，确保文件的组织是有序的
    print(f"[INFO] 正在扫描: {input_path}")
    # 使用 glob 匹配所有 webp 后缀的文件
    all_files = sorted(list(input_path.glob("*.webp")))
    total_files = len(all_files)
    print(f"[INFO] 扫描完成，共找到 {total_files} 个 webp 文件")

    if total_files == 0:
        print("[WARN] 未找到 webp 文件，请检查路径。")
        return

    # 2. 按 batch_size 步长进行切片处理
    # i 是当前批次的起始索引
    for i in range(0, total_files, batch_size):
        # 计算当前批次的序号 (1, 2, 3...)
        batch_num = (i // batch_size) + 1
        
        # 构造输出文件名，格式为 prefix_01.tar.gz, prefix_02.tar.gz ...
        # :02d 表示至少两位数字，不足补零
        output_name = f"{prefix}_{batch_num:02d}.tar.gz"
        target_tar_path = output_path / output_name
        
        # 获取当前这一组的文件切片
        current_batch_files = all_files[i : i + batch_size]
        
        print(f"[PROCESS] 正在打包第 {batch_num} 组: {len(current_batch_files)} 张图片 -> {output_name}")
        
        # 3. 开始执行压缩打包
        # mode='w:gz' 指定写入并启用 gzip 压缩
        # compresslevel=9 是最高级别的压缩，会消耗更多 CPU 但产生更小的文件
        with tarfile.open(target_tar_path, mode="w:gz", compresslevel=9) as tar:
            for file_path in current_batch_files:
                # arcname=file_path.name 确保压缩包内只存文件名，不带长路径
                tar.add(file_path, arcname=file_path.name)
        
        print(f"[SUCCESS] 已生成: {target_tar_path}")

    print("\n[DONE] 所有文件已成功打包完毕。")

if __name__ == "__main__":
    
    # 存放 webp 图片的源目录
    INPUT_DIR = r"./" 
    
    # 压缩包存放的目标目录
    OUTPUT_DIR = r"./"
    
    # 输出文件名的前缀
    OUTPUT_PREFIX = "lineart"
    
    # 每包多少张
    PER_BATCH_SIZE = 10000

    # 调用主函数
    pack_images_to_tar(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        prefix=OUTPUT_PREFIX,
        batch_size=PER_BATCH_SIZE
    )
