# v4.3.1：从零运行 B1 Direct OBB

## 一、这次报错的根因

LLaMA-Factory 的规则是：

```text
训练 YAML 中的 dataset_dir = X
→ 固定读取 X/dataset_info.json
→ dataset_info.json 中的 file_name 再相对 X 解析
```

v4.3 错误地做成了：

```text
YAML dataset_dir:
/home/nhl/fxy/datasets/dota128-llamafactory

实际注册位置：
/home/nhl/Asking_like_Socrates/LLaMA-Factory/data/dataset_info.json
```

因此训练必然寻找：

```text
/home/nhl/fxy/datasets/dota128-llamafactory/dataset_info.json
```

而该文件没有生成。

v4.3.1 改为自包含目录：

```text
/home/nhl/fxy/datasets/dota128-llamafactory/
├── dataset_info.json
├── dota128_direct_train_official.json
└── dota128_direct_val_official.json
```

训练 YAML 的 `dataset_dir` 与上述目录完全一致。

---

## 二、B1 Direct 不需要哪些步骤

B1 只训练 Direct OBB，不依赖：

- 三智能体；
- Official Parquet；
- raw/strict/merge Socratic 轨迹；
- `build-official-parquet`；
- `official-generation`；
- `postproc`。

B1 只需要：

```text
DOTA原始标签
→ data-full
→ train_direct.json / val_direct.json
→ Direct LLaMA-Factory JSON
→ dataset_info.json
→ B1 YAML
→ LoRA训练
```

---

## 三、推荐：一条命令从头运行

先部署新包并迁移 settings，然后：

```bash
conda activate als_sft
cd /home/nhl/Asking_like_Socrates/re_scripts

python main_layer/run.py b1-from-scratch \
  --settings settings.json \
  fresh
```

这条命令依次完成：

1. 重新运行数据层；
2. 生成 Direct SFT 文件；
3. 在 Direct 数据目录生成权威 `dataset_info.json`；
4. 校验190条训练样本与12条验证样本；
5. 生成 B1 YAML；
6. 用当前 LLaMA-Factory 的真实 parser 校验数据契约；
7. 自动停止8001/8002/8003三智能体；
8. 归档旧B1输出；
9. 双A6000启动LoRA训练。

`fresh`表示归档旧的 `b1_direct_lora`，从头训练。

---

## 四、分步运行与每一步的作用

### 0. 环境检查

```bash
python main_layer/run.py runtime-check --settings settings.json
```

必须满足：

```text
data found=true
sft found=true
qwen25/rs_eot compatible=true
GPU count=2
```

Qwen3环境和RL环境不存在不影响B1。

### 1. 重新生成数据层

```bash
python main_layer/run.py data-full --settings settings.json
```

作用：

- 审计并丢弃越界OBB；
- 生成有效Direct数据；
- 生成Ref与agent inputs；
- 最终检查坐标范围。

必须看到：

```text
[DATA LAYER GATE] PASS
invalid_grounding_gt=0
invalid_direct=0
```

关键文件：

```text
/home/nhl/fxy/results/dota128_pipeline/agent_inputs/train_direct.json
/home/nhl/fxy/results/dota128_pipeline/agent_inputs/val_direct.json
```

### 2. 生成Direct SFT和数据注册文件

```bash
python main_layer/run.py prepare-direct-sft \
  --settings settings.json \
  --register
```

作用：

- 复制Direct训练/验证JSON；
- 生成权威外部 `dataset_info.json`；
- 可选镜像注册到官方 `LLaMA-Factory/data/dataset_info.json`，供WebUI查看。

必须生成：

```text
/home/nhl/fxy/datasets/dota128-llamafactory/dataset_info.json
/home/nhl/fxy/datasets/dota128-llamafactory/dota128_direct_train_official.json
/home/nhl/fxy/datasets/dota128-llamafactory/dota128_direct_val_official.json
```

注意：训练依赖第一个外部 `dataset_info.json`，不依赖官方目录中的镜像。

### 3. 校验Direct数据本身

```bash
python main_layer/run.py validate-train \
  --settings settings.json \
  --target b1
```

必须看到：

```text
[TRAIN DATA VALIDATE] PASS target=b1
```

检查：

- 图片路径；
- messages/images结构；
- Direct检测标记；
- 样本数量；
- 预计优化步数。

### 4. 生成训练YAML

```bash
python main_layer/run.py generate-train-config \
  --settings settings.json
```

查看：

```bash
grep -E 'dataset_dir|^dataset:|eval_dataset' \
  /home/nhl/fxy/results/dota128_training/configs/b1_direct_lora.yaml
```

必须是：

```text
dataset_dir: /home/nhl/fxy/datasets/dota128-llamafactory
dataset: dota128_direct_train_official
eval_dataset: dota128_direct_val_official
```

### 5. 用真实LLaMA-Factory parser做启动前检查

```bash
python main_layer/run.py validate-lf-contract \
  --settings settings.json \
  --target b1
```

必须看到：

```text
[LLAMAFACTORY CONTRACT] PASS target=b1
parser_check: PASS
```

该步骤会在占用GPU之前检查：

- `dataset_dir/dataset_info.json`是否存在；
- dataset名称是否注册；
- `file_name`能否解析；
-JSON是否可读取；
- ShareGPT/OpenAI messages格式；
- `<image>`数量与images数量；
- 每张图片是否存在；
- YAML与registry是否完全一致；
- 当前LLaMA-Factory parser能否真实解析。

### 6. 停止三智能体

```bash
python main_layer/run.py stop-agents --settings settings.json
```

v4.3.1不只依赖旧PID文件，还会通过端口和vLLM命令查找真实监听进程。

必须看到：

```text
[FREE] reasoner port=8001
[FREE] perceiver port=8002
[FREE] verifier port=8003
[AGENTS STOP] PASS
```

### 7. 训练B1

```bash
python main_layer/run.py train-b1 --settings settings.json
```

`train-b1`内部会再次执行：

```text
prepare-direct-sft
validate-train
生成YAML
validate-lf-contract
停止智能体检查
```

所以即使跳过前面某一步，也不会直接进入torchrun。

训练输出：

```text
/home/nhl/fxy/results/dota128_training/b1_direct_lora
```

---

## 五、哪些日志不是致命错误

下面的NCCL信息通常只是警告，不是本次失败原因：

```text
using GPU 0/1 to perform barrier as devices used ... unknown
```

真正失败原因要看其后的第一条Python异常。本次第一条异常是：

```text
Cannot open .../dota128-llamafactory/dataset_info.json
```

v4.3.1会在torchrun前拦截该问题，因此不会再先初始化双卡后才失败。

---

## 六、训练完成后

合并LoRA：

```bash
python main_layer/run.py merge --settings settings.json b1
```

预期：

```text
/home/nhl/fxy/models/RS-EoT-7B-DOTA128-Direct-Merged
```

之后启动测试服务并评估B1。B2 Socratic应等新的Grounding轨迹通过严格门控后再训练。
