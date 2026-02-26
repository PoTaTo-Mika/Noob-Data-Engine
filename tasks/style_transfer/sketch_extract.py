import torch
import diffusers
import os
import json
import time
import sys
import glob
import math
import shutil
from pathlib import Path
import torch.multiprocessing as mp

# 自动定位项目根目录并加入 sys.path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.data_io.read_tar import read_sample_from_tar
import diffsynth
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler
from PIL import Image

# setup config
config_path = current_file_path.parent / "configs" / "sketch.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

def get_abs_path(p):
    if p.startswith("../../"):
        return str(project_root / p.replace("../../", ""))
    return p

MODEL_CHECKPOINTS_PATH = get_abs_path(config["model_checkpoints_path"])
LORA_PATH = get_abs_path(config["distill_lora_path"])
CONVERTED_DATASET_PATH = get_abs_path(config["converted_dataset_path"])
SAVE_PATH = os.path.join(get_abs_path(config["save_path"]), config['task_name']) + "/"
PROMPT = config['generate_prompt']
IF_RESIZE = config['if_resize']
NUM_GPUS = config.get("num_gpus", torch.cuda.device_count())

# VRAM 检测阈值（单位：GB）
LOW_VRAM_THRESHOLD_GB = 48

# --------------------------------------------------------------------------
# Pipeline构建与模型相关函数 (这些函数将在子进程中运行)
# --------------------------------------------------------------------------

def setup_attention_kernel():
    from diffsynth.core.attention.attention import attention_forward, ATTENTION_IMPLEMENTATION
    from einops import rearrange
    import diffsynth.models.qwen_image_dit as qwen_image_dit
    
    def apply_flash_attn(q, k, v, num_heads, attention_mask=None, enable_fp8_attention=False, **kwargs):
        x = attention_forward(
            q, k, v,
            q_pattern="b n s d",
            k_pattern="b n s d",
            v_pattern="b n s d",
            out_pattern="b s n d",
            attn_mask=attention_mask 
        )
        return rearrange(x, "b s n d -> b s (n d)")

    qwen_image_dit.qwen_image_flash_attention = apply_flash_attn

    if "flash_attention" in ATTENTION_IMPLEMENTATION:
        print("[Attention Kernel] Applied Flash Attention to DiT.")
    else:
        print("[Attention Kernel] Applied original SDPA Kernel to DiT.")

def get_vram_gb(device_id=0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    # 获取指定设备的显存
    total_bytes = torch.cuda.get_device_properties(device_id).total_memory
    return total_bytes / (1024 ** 3)

def build_pipeline(device_id) -> QwenImagePipeline:

    target_device = f"cuda:{device_id}"    
    # 传入 device_id 查询对应显卡显存
    vram_gb = get_vram_gb(device_id)
    print(f"[Pipeline] GPU {device_id} Detected VRAM: {vram_gb:.1f} GB")

    if vram_gb < LOW_VRAM_THRESHOLD_GB:
        # H200 一般不会进这里，但为了代码严谨，把 device 也改掉
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": target_device, 
            "computation_dtype": torch.bfloat16,
            "computation_device": target_device, 
        }
    else:
        vram_config = {}

    pipeline = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=target_device,
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
        vram_limit=get_vram_gb(device_id)-0.5 if vram_gb < LOW_VRAM_THRESHOLD_GB else LOW_VRAM_THRESHOLD_GB,
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

# --------------------------------------------------------------------------
# 数据处理逻辑
# --------------------------------------------------------------------------

def get_data_generator(tar_files_shard, resize):
    for metadata, picture in read_sample_from_tar(tar_files_shard):
        pid = metadata.get("pid")
        artist = metadata.get("artist")
        if resize:
            img = Image.open(picture).convert("RGB")
            picture = img.resize((1024, 1024), Image.LANCZOS)
        yield pid, artist, picture

def process_one_picture(pipe, pid, picture):
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
        zero_cond_t=True,
        cfg_scale=1.0,
    )
    return image, pid

# --------------------------------------------------------------------------
# Worker Process (每个GPU执行的函数)
# --------------------------------------------------------------------------

def gpu_worker(gpu_id, tar_files_shard):

    torch.cuda.set_device(gpu_id)
    
    # 打印进程信息
    print(f"[GPU-{gpu_id}] Initializing... Assigned {len(tar_files_shard)} tar files.")

    # 初始化模型（在进程内部初始化，避免上下文冲突）
    setup_attention_kernel()
    try:
        pipe = build_pipeline(gpu_id)
    except Exception as e:
        print(f"[GPU-{gpu_id}] Failed to build pipeline: {e}")
        return

    # 独立的临时 jsonl 文件，避免写锁冲突
    temp_jsonl_path = os.path.join(SAVE_PATH, f"metadata_part_{gpu_id}.jsonl")
    
    count = 1
    
    with open(temp_jsonl_path, 'w', encoding='utf-8') as f_json:
        # 获取数据生成器
        data_gen = get_data_generator(tar_files_shard, resize=IF_RESIZE)
        
        for pid, artist, picture_stream in data_gen:
            try:
                t0 = time.time()
                
                generated_result, _ = process_one_picture(pipe, pid, picture_stream)

                if isinstance(generated_result, list):
                     final_image = generated_result[0]
                elif hasattr(generated_result, 'images'):
                     final_image = generated_result.images[0]
                else:
                     final_image = generated_result

                # 引入了多个GPU之后就要注意防止冲突了。
                file_name_no_ext = f"st_sketch_g{gpu_id}_{count:09d}"
                image_filename = f"{file_name_no_ext}.png"
                save_image_path = os.path.join(SAVE_PATH, image_filename)

                final_image.save(save_image_path)

                meta_data = {
                    "file_name": file_name_no_ext, 
                    "task": "style_transfer",
                    "reference_artist": artist,
                    "converted_pid": pid,
                    "gpu_id": gpu_id  # 可选：记录是哪个卡跑的
                }
                
                f_json.write(json.dumps(meta_data, ensure_ascii=False) + "\n")
                f_json.flush()

                process_time = time.time() - t0
                # 减少打印频率，避免8个卡同时刷屏太乱
                if count % 10 == 0:
                    print(f"[GPU-{gpu_id}] Processed {count} imgs. Last PID: {pid} ({process_time:.2f}s)")
                
                count += 1

            except Exception as e:
                print(f"[GPU-{gpu_id}] Error processing PID {pid}: {e}")
                # import traceback
                # traceback.print_exc()
                continue
    
    print(f"[GPU-{gpu_id}] Finished! Total processed: {count-1}")

# --------------------------------------------------------------------------
# Main Coordinator
# --------------------------------------------------------------------------

def main():
    # 确保保存目录存在
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Created directory: {SAVE_PATH}")

    all_tar_files = sorted([os.path.join(CONVERTED_DATASET_PATH, f) for f in os.listdir(CONVERTED_DATASET_PATH) if f.endswith(".tar")])
    total_files = len(all_tar_files)
    
    if total_files == 0:
        print("No tar files found!")
        return

    print(f"Total tar files found: {total_files}")
    print(f"Distributing across {NUM_GPUS} GPUs...")

    # 计算每个GPU分多少个文件
    chunk_size = math.ceil(total_files / NUM_GPUS)
    gpu_tasks = []
    
    for i in range(NUM_GPUS):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        shard = all_tar_files[start_idx:end_idx]
        if shard:
            gpu_tasks.append(shard)
        else:
            # 如果文件不够分，后面的GPU可能分不到
            print(f"Warning: GPU {i} has no files assigned.")

    # ！！使用 'spawn' 启动方式，这对CUDA多进程是必须的
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for gpu_id, task_shard in enumerate(gpu_tasks):
        p = mp.Process(target=gpu_worker, args=(gpu_id, task_shard))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All GPU tasks finished. Merging metadata...")

    final_jsonl_path = os.path.join(SAVE_PATH, "metadata.jsonl")
    with open(final_jsonl_path, 'w', encoding='utf-8') as outfile:
        for gpu_id in range(len(gpu_tasks)):
            part_file = os.path.join(SAVE_PATH, f"metadata_part_{gpu_id}.jsonl")
            if os.path.exists(part_file):
                with open(part_file, 'r', encoding='utf-8') as infile:
                    shutil.copyfileobj(infile, outfile)
                os.remove(part_file)
    
    print(f"Metadata merged into {final_jsonl_path}")
    print("Job Done!")

if __name__ == "__main__":
    main()