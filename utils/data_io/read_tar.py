import webdataset as wds
import numpy as np
import io

# 工具函数只负责读取给定目录下所有的tar，配合一个外部的测试工具去检测数据内容

def read_sample_from_tar(tar_file):
    """
    tar_file: 可以是单个tar也可以是一个列表，列表里内容是tar的索引
    """
    inside_data = wds.WebDataset(tar_file).decode()
    for sample in inside_data:
        metadata = sample["json"]
        picture_data = sample["webp"]
        
        # 如果是 bytes，包装成 BytesIO 以便 PIL.Image.open 正确识别
        if isinstance(picture_data, bytes):
            picture = io.BytesIO(picture_data)
        else:
            picture = picture_data
            
        yield metadata, picture