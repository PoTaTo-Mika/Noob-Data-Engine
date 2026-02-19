import json

def make_json(metadata, output_path):
    
    with open(output_path, "w") as f:
        json.dump(metadata, f)
    
    # 通过metadata里面的任务类别去创建相关json

