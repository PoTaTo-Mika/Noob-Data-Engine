我们的数据以两种方法保存，一种是纯粹的metadata和图片，另一种是经过VAE处理和tokenizer编码的序列parquet。

## Danbooru数据处理须知
整体数据处理流程使用webdataset的方式，读取tar包进行处理，避免解压大量小文件。

我们约定metadata的保存方式为json+webp格式的图片。其中json的元数据预览如下所示：

```json
{
    "pid":114514,
    "tags":"here are the tags crawl from danbooru, they are discrete.",
    "description":"here is the description of this picture, with vlm labelled or human labelled natural language.",
    "extra":{
        "diff":{
            "difference":[114515,114516],
            "diff_description":{
                "compared_with":114515,
                "description":"here is the natural language description of the difference between this picture and the target picture."
            }
        }
    }
}
```

我们使用danbooru pid作为沟通各个管线之间的桥梁，因此pid信息需要被精确保存下来。

而管线中相关任务所需要的特征则被保存在extra下。

## 风格迁移数据保存方案

```json
{
    "file_name":"st_000000001",
    "task":"style_transfer",
    "reference_pid":"12345",
    "reference_artist":"ocean_cat",
    "converted_pid":"30001",
}
```

风格迁移任务造出的数据，回头在处理序列的时候只需要读取图片，计算VAE，把token嵌入到整体序列当中即可。