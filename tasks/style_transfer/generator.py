import torch
import diffusers
import os
import webdataset as wds
import json
import time

import sys
from pathlib import Path

# 自动定位项目根目录并加入 sys.path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.data_io.read_tar import read_sample_from_tar
from sketch_extract import get_abs_path

# setup config
config_path = current_file_path.parent / "configs" / "generate_config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_CHECKPOINTS_PATH = get_abs_path(config["model_checkpoints_path"])
LORA_PATH = get_abs_path(config["distill_lora_path"])
SKETCH_DATA_PATH = get_abs_path(config["sketch_data_path"])
REF_DATA_PATH = get_abs_path(config["ref_data_path"])
SAVE_PATH = os.path.join(get_abs_path(config["save_path"]), config['task_name']) + "/"
PROMPT = config['generate_prompt']
IF_RESIZE = config.get('if_resize', 'True') == 'True'

import diffsynth
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler
from PIL import Image
from sketch_extract import setup_attention_kernel, get_vram_gb, build_pipeline, LOW_VRAM_THRESHOLD_GB

# 读取两组文件,一组是sketch,一组是ref,然后进行风格迁移
# 然后sketch里面会给一些标准信息，最后要构建：
# {
#     "file_name":"st_000000001",
#     "task":"style_transfer",
#     "reference_pid":"12345",
#     "reference_artist":"ocean_cat",
#     "converted_pid":"30001",
# }

def get_data(folder, resize):
    # ref和sketch每个都用一次get_data构建迭代器
    # 查找目录下所有的 tar 文件
    tar_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tar")]
    # 使用 read_sample_from_tar 迭代读取
    for metadata, picture in read_sample_from_tar(tar_files):
        if resize:
            img = Image.open(picture).convert("RGB")
            picture = img.resize((1024, 1024), Image.LANCZOS)
        yield metadata, picture # 这边读metadata是为了后面写入json

def process_one_picture(pipe, ref_meta, converted_meta, ref_pic, converted_pic):
    # 秉持着简单易懂的原则，这里多写几行代码
    if isinstance(ref_pic, Image.Image):
        ref_img = ref_pic
    else:
        ref_img = Image.open(ref_pic).convert("RGB")
    
    if isinstance(converted_pic, Image.Image):
        converted_img = converted_pic
    else:
        converted_img = Image.open(converted_pic).convert("RGB")
    
    # 记录尺寸，使用处理后的图像对象
    width, height = converted_img.size
    
    # 构建编辑图片列表 [参考图, 待处理图]
    # 注意：Qwen-Image-Edit 的输入顺序和 prompt 对应关系通常是列表顺序
    edit_image = [ref_img, converted_img]
    prompt = PROMPT

    image = pipe(
        prompt,
        edit_image=edit_image,
        seed=1,
        num_inference_steps=4,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=True,
        cfg_scale=1.0,
    )

    return image, ref_meta, converted_meta

def main():
    
    setup_attention_kernel()
    pipe = build_pipeline()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Created directory: {SAVE_PATH}")

    jsonl_path = os.path.join(SAVE_PATH, "metadata.jsonl")
    
    count = 1

    print(f"Start processing. Sketch path: {SKETCH_DATA_PATH}, Ref path: {REF_DATA_PATH}")
    
    with open(jsonl_path, 'w', encoding='utf-8') as f_json:
        # 准备迭代器
        ref_iter = get_data(REF_DATA_PATH, resize=IF_RESIZE)
        sketch_iter = get_data(SKETCH_DATA_PATH, resize=IF_RESIZE)
        
        # 同时遍历两组数据
        for (ref_meta, ref_pic), (sketch_meta, sketch_pic) in zip(ref_iter, sketch_iter):
            t0 = time.time()
            
            # 获取必要的 PID 信息
            ref_pid = ref_meta.get("pid")
            converted_pid = sketch_meta.get("pid")

            # 调用生成函数
            generated_result, _, _ = process_one_picture(pipe, ref_meta, sketch_meta, ref_pic, sketch_pic)

            # 获取生成的图像内容
            if isinstance(generated_result, list):
                final_image = generated_result[0]
            elif hasattr(generated_result, 'images'):
                final_image = generated_result.images[0]
            else:
                final_image = generated_result

            # 保存生成的图像
            file_name_no_ext = f"st_{count:09d}"
            image_filename = f"{file_name_no_ext}.png"
            save_image_path = os.path.join(SAVE_PATH, image_filename)
            final_image.save(save_image_path)

            # 5. 生成元数据 (file_name, task, reference_pid, converted_pid 是必填项)
            meta_data = {
                "file_name": file_name_no_ext,
                "task": "style_transfer",
                "reference_pid": str(ref_pid) if ref_pid else None,
                "converted_pid": str(converted_pid) if converted_pid else None,
            }
            
            # 处理可选的 artist 字段
            if "artist" in ref_meta:
                meta_data["reference_artist"] = ref_meta["artist"]
            
            # 写入记录
            f_json.write(json.dumps(meta_data, ensure_ascii=False) + "\n")
            f_json.flush()

            # 打印进度日志
            process_time = time.time() - t0
            print(f"[{count}] Processed: Ref({ref_pid}) + Sketch({converted_pid}) -> {image_filename} ({process_time:.2f}s)")
            
            count += 1

    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
