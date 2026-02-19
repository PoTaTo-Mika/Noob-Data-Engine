import torch
import diffusers
import os
import webdataset as wds
import json

from utils.data_io.read_tar import read_sample_from_tar

# setup config
with open("./configs/generate_config.json", "r") as f:
    config = json.load(f)

MODEL_CHECKPOINTS_PATH = config["model_checkpoints_path"]
LORA_PATH = config["distill_lora_path"]
REFERENCE_DATASET_PATH = config["reference_dataset_path"]
CONVERTED_DATASET_PATH = config["converted_dataset_path"]
