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

# setup config
config_path = current_file_path.parent / "configs" / "sketch.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

def get_abs_path(p):
    # 处理配置文件中的相对路径，使其相对于项目根目录
    if p.startswith("../../"):
        return str(project_root / p.replace("../../", ""))
    return p

MODEL_CHECKPOINTS_PATH = get_abs_path(config["model_checkpoints_path"])
LORA_PATH = get_abs_path(config["distill_lora_path"])
CONVERTED_DATASET_PATH = get_abs_path(config["converted_dataset_path"])
SAVE_PATH = os.path.join(get_abs_path(config["save_path"]), config['task_name']) + "/"
PROMPT = config['generate_prompt']
IF_RESIZE = config['if_resize']

import diffsynth
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler
from PIL import Image
import torch

# VRAM 检测阈值（单位：GB），低于此值启用低显存低精度卸载方案
LOW_VRAM_THRESHOLD_GB = 48

def setup_attention_kernel():
    from diffsynth.core.attention.attention import attention_forward, ATTENTION_IMPLEMENTATION
    from einops import rearrange
    import diffsynth.models.qwen_image_dit as qwen_image_dit
    import diffsynth.models.qwen_image_text_encoder as text_encoder

    def apply_flash_attn(q, k, v, num_heads, attention_mask=None, enable_fp8_attention=False, **kwargs):
        # q, k, v: [batch, heads, seq_len, head_dim] (b n s d)
        x = attention_forward(
            q, k, v,
            q_pattern="b n s d",
            k_pattern="b n s d",
            v_pattern="b n s d",
            out_pattern="b s n d",
            attn_mask=attention_mask 
        )
        # x: [batch, seq_len, heads, head_dim] (b s n d)
        return rearrange(x, "b s n d -> b s (n d)")

    # 替换注意力内核
    qwen_image_dit.qwen_image_flash_attention = apply_flash_attn

    if "flash_attention" in ATTENTION_IMPLEMENTATION:
        print("[Attention Kernel] Applied Flash Attention to DiT.")
    else:
        print("[Attention Kernel] Applied original SDPA Kernel to DiT.")

""" 
    # 因为bf16的问题还没搞定,暂且注释这个加速
    # 替换qwen-vl注意力内核
    from transformers import Qwen2_5_VLConfig
    text_encoder_init = text_encoder.QwenImageTextEncoder.__init__
    config_init = Qwen2_5_VLConfig.__init__
    
    def vlm_attn_init(self, *args, **kwargs):
        kwargs["attn_implementation"] = "flash_attention_2"
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config_init(self, *args, **kwargs)
    
    Qwen2_5_VLConfig.__init__ = vlm_attn_init
    
    if "flash_attention" in ATTENTION_IMPLEMENTATION:
        print("[Attention Kernel] Applied Flash Attention to VLM.")
    else:
        print("[Attention Kernel] Applied original SDPA Kernel to VLM.")
"""

def get_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return total_bytes / (1024 ** 3)

def build_pipeline() -> QwenImagePipeline:
    vram_gb = get_vram_gb()
    print(f"[Pipeline] Detected VRAM: {vram_gb:.1f} GB (threshold: {LOW_VRAM_THRESHOLD_GB} GB)")

    if vram_gb < LOW_VRAM_THRESHOLD_GB:
        # https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511-Lightning.py
        print("[Pipeline] Using LOW VRAM mode (FP8 + disk offload)")
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    else:
        print("[Pipeline] Using FULL VRAM mode (BF16)")
        vram_config = {}

    pipeline = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                local_model_path=MODEL_CHECKPOINTS_PATH,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                skip_download=True,
                **vram_config
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                local_model_path=MODEL_CHECKPOINTS_PATH,
                origin_file_pattern="text_encoder/model*.safetensors",
                skip_download=True,
                **vram_config
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                local_model_path=MODEL_CHECKPOINTS_PATH,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                skip_download=True,
                **vram_config
            ),
        ],
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit-2511",
            local_model_path=MODEL_CHECKPOINTS_PATH,
            origin_file_pattern="processor/",
            skip_download=True,
        ),
        vram_limit=get_vram_gb()-0.5 if vram_gb < LOW_VRAM_THRESHOLD_GB else LOW_VRAM_THRESHOLD_GB,
    )

    lora = ModelConfig(
        model_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        local_model_path=LORA_PATH,
        origin_file_pattern="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        skip_download=True,
    )
    pipeline.load_lora(pipeline.dit, lora, alpha=8/64)
    pipeline.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")

    return pipeline

def get_data(folder, resize):
    # 查找目录下所有的 tar 文件
    # 因为我们做线稿的话是只需要它原本的pid的
    tar_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tar")]
    # 使用 read_sample_from_tar 迭代读取
    for metadata, picture in read_sample_from_tar(tar_files):
        pid = metadata.get("pid")
        if resize:
            img = Image.open(picture).convert("RGB")
            picture = img.resize((1024, 1024), Image.LANCZOS)
        yield pid, picture

def process_one_picture(pipe, pid, picture):
    # 提取线稿只需要一张图+prompt即可
    if isinstance(picture, Image.Image):
        img = picture
    else:
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

    setup_attention_kernel()
    pipe = build_pipeline()
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Created directory: {SAVE_PATH}")

    jsonl_path = os.path.join(SAVE_PATH, "metadata.jsonl")
    
    # 计数器，用于生成文件名 (从1开始)
    count = 1

    print(f"Start processing from: {CONVERTED_DATASET_PATH}")
    
    with open(jsonl_path, 'w', encoding='utf-8') as f_json:
        
        # 4. 遍历数据 (get_data 内部调用了 read_sample_from_tar)
        for pid, picture_stream in get_data(CONVERTED_DATASET_PATH, resize=IF_RESIZE):
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

