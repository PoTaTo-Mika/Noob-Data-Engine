import torch
import diffusers
import os
import webdataset as wds
import json
import time

from utils.data_io.read_tar import read_sample_from_tar

# setup config
with open("./configs/generate_config.json", "r") as f:
    config = json.load(f)

MODEL_CHECKPOINTS_PATH = config["model_checkpoints_path"]
LORA_PATH = config["distill_lora_path"]
CONVERTED_DATASET_PATH = config["converted_dataset_path"]
SAVE_PATH = config["save_path"] + config['task_name'] + "/"
PROMPT = config['generate_prompt']

import diffsynth
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id=MODEL_CHECKPOINTS_PATH, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id=MODEL_CHECKPOINTS_PATH, origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id=MODEL_CHECKPOINTS_PATH, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

lora = ModelConfig(
    model_id=LORA_PATH,
    origin_file_pattern="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"
)

pipe.load_lora(pipe.dit, lora, alpha=8/64)
pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")

def get_data(folder):
    # 查找目录下所有的 tar 文件
    # 因为我们做线稿的话是只需要它原本的pid的
    tar_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tar")]
    # 使用 read_sample_from_tar 迭代读取
    for metadata, picture in read_sample_from_tar(tar_files):
        pid = metadata.get("pid")
        yield pid, picture

def process_one_picture(pipe, pid, picture):
    # 提取线稿只需要一张图+prompt即可
    img = Image.open(picture)
    width, height = img.size
    edit_image = [img]
    prompt = PROMPT

    image = pipe(
        prompt,
        edit_image=edit_image,
        seed=1,
        num_inference_steps=4,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=True, # This is a special parameter introduced by Qwen-Image-Edit-2511
        cfg_scale=1.0,
    )

    return image, pid

def main():

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Created directory: {SAVE_PATH}")

    jsonl_path = os.path.join(SAVE_PATH, "metadata.jsonl")
    
    # 计数器，用于生成文件名 (从1开始)
    count = 1

    print(f"Start processing from: {CONVERTED_DATASET_PATH}")
    
    with open(jsonl_path, 'w', encoding='utf-8') as f_json:
        
        # 4. 遍历数据 (get_data 内部调用了 read_sample_from_tar)
        for pid, picture_stream in get_data(CONVERTED_DATASET_PATH):
            try:
                # 记录开始时间
                t0 = time.time()

                # 调用处理函数
                # 注意：process_one_picture 返回的是 pipeline 的输出对象和 pid
                generated_result, _ = process_one_picture(pipe, pid, picture_stream)

                if isinstance(generated_result, list):
                     final_image = generated_result[0]
                elif hasattr(generated_result, 'images'):
                     final_image = generated_result.images[0]
                else:
                     final_image = generated_result # 假设直接返回了 PIL Image

                file_name_no_ext = f"st_sketch_{count:09d}"
                image_filename = f"{file_name_no_ext}.png"
                save_image_path = os.path.join(SAVE_PATH, image_filename)

                final_image.save(save_image_path)

                meta_data = {
                    "file_name": file_name_no_ext, 
                    "task": "style_transfer",
                    "converted_pid": pid
                }
                
                f_json.write(json.dumps(meta_data, ensure_ascii=False) + "\n")
                
                # 强制刷新缓冲区，防止程序意外中断导致数据丢失
                f_json.flush()

                # 打印进度
                process_time = time.time() - t0
                print(f"[{count}] Processed PID: {pid} -> {image_filename} ({process_time:.2f}s)")
                
                # 计数器自增
                count += 1

            except Exception as e:
                print(f"Error processing PID {pid}: {e}")
                import traceback
                traceback.print_exc()
                # 遇到错误跳过当前图片，继续处理下一张
                continue

    print("Processing completed!")

if __name__ == "__main__":
    main()