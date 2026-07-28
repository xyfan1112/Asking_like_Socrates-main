# LLaMA-Factory数据目录契约

## 核心规则

训练配置写：

```yaml
dataset_dir: /path/to/data
dataset: my_train
```

LLaMA-Factory会读取：

```text
/path/to/data/dataset_info.json
```

然后读取其中：

```json
{
  "my_train": {
    "file_name": "train.json"
  }
}
```

最终数据文件是：

```text
/path/to/data/train.json
```

因此以下结构是错误的：

```text
dataset_dir=/external/data
registry=/LLaMA-Factory/data/dataset_info.json
```

v4.3.1采用外部自包含数据目录，避免修改官方仓库成为训练必需条件。
