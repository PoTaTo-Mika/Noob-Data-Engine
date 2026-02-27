import os
import argparse
import time
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm # 进度条库，如果没有安装，请注释掉相关代码

def process_single_image(args):
    """
    单个图片处理函数，用于多进程调用
    """
    png_file, delete_original = args
    result = {
        "success": False,
        "filename": png_file.name,
        "saved_bytes": 0,
        "msg": ""
    }

    try:
        webp_file = png_file.with_suffix('.webp')
        
        with Image.open(png_file) as img:
            # 依然保持 method=6, quality=100, lossless=True
            img.save(webp_file, 'webp', lossless=True, quality=100, method=6)
        
        original_size = os.path.getsize(png_file)
        new_size = os.path.getsize(webp_file)
        saved = original_size - new_size
        
        result["success"] = True
        result["saved_bytes"] = saved
        
        # 删除逻辑
        if delete_original and webp_file.exists():
            os.remove(png_file)
            result["msg"] = "已转换并删除原图"
        else:
            result["msg"] = "已转换"

    except Exception as e:
        result["msg"] = f"错误: {str(e)}"
    
    return result

def convert_to_webp_multicore(target_dir, delete_original=False):
    target_path = Path(target_dir)
    
    if not target_path.exists():
        print(f"错误: 目录 '{target_dir}' 不存在。")
        return

    png_files = list(target_path.glob('*.png'))
    total_files = len(png_files)
    
    if not png_files:
        print(f"在 '{target_dir}' 下没有找到 PNG 图片。")
        return

    # 获取 CPU 核心数，留一个核给系统，避免电脑卡死
    max_workers = max(1, multiprocessing.cpu_count() - 8)
    
    print(f"找到 {total_files} 张图片。")
    print(f"正在使用 {max_workers} 个 CPU 核心进行并行加速 (method=6)...")
    print("-" * 30)

    # 准备任务列表
    tasks = [(f, delete_original) for f in png_files]
    
    success_count = 0
    total_saved_bytes = 0
    start_time = time.time()

    # 开始多进程处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        results = list(tqdm(executor.map(process_single_image, tasks), total=total_files, unit="img"))
        
        for res in results:
            if res["success"]:
                success_count += 1
                total_saved_bytes += res["saved_bytes"]
            else:
                print(f"[失败] {res['filename']}: {res['msg']}")

    end_time = time.time()
    duration = end_time - start_time

    print("-" * 30)
    print(f"处理完成！耗时: {duration:.2f} 秒")
    print(f"成功转换: {success_count}/{total_files}")
    print(f"共计节省空间: {total_saved_bytes / 1024 / 1024:.2f} MB")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="多核加速：将目录下的PNG图片转换为无损WebP")
    parser.add_argument("directory", help="包含PNG图片的目录路径")
    parser.add_argument("--keep", action="store_true", help="保留原PNG文件 (默认删除，加上此参数则保留)")
    
    args = parser.parse_args()
    delete_original = not args.keep
    
    convert_to_webp_multicore(args.directory, delete_original)