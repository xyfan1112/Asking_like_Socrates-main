# v1.2.2：2A6000 Direct/JSON＋中文修复＋Qwen3-Omni AWQ-No-TTS TP4

## 1. 版本定位

本包是累计补丁：

```text
2A6000稳定分支
+ v1.2.0 中文ROI、坐标意图、任务识别、中文重复检测、有限fallback
+ v1.2.1 Direct强制基线、四种Direct QA、中文独立输出目录
+ v1.2.2 指定Qwen3-Omni AWQ-No-TTS、单服务四卡TP、服务管理和性能测试
```

目标checkpoint：

```text
tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS
```

## 2. 四张A6000如何使用

默认物理拓扑：

```text
GPU 0 ─┐
GPU 1 ─┼─ 一份Qwen3-Omni权重，Tensor Parallel=4
GPU 2 ─┼─ 每次模型前向都跨四张卡
GPU 3 ─┘
               ↓
      一个OpenAI兼容端点:8091
               ↓
Reasoner / Perceiver / Verifier三个逻辑角色
```

这不是三份30B模型，也不是每个Agent固定一张卡。三个Agent通过不同System Prompt共用同一物理模型。

TP=4的主要收益：

- 四卡显存合并用于模型权重、非量化多模态模块和KV Cache；
- 单次请求由四卡共同计算；
- Full阶段可通过并发请求提升吞吐；
- 不需要重复加载三份权重。

限制：四卡TP不保证单请求延迟缩短四倍。A6000之间若主要通过PCIe通信，all-reduce开销可能限制加速。必须以本包的`omni-benchmark`实测为准。

## 3. 模型容量与兼容性

四张A6000共有约192GB标称显存。官方Qwen3-Omni BF16 Instruct长视频参考显存低于该总量；No-TTS量化checkpoint通常更轻。因此从容量角度四卡足够。

但“能放下”和“当前vLLM能够加载该第三方AWQ格式”是两个问题。v1.2.2执行四级门控：

```text
config/weight index静态检查
→ vLLM启动
→ /v1/models＋文本＋真实图像smoke
→ fresh Debug40
```

出现`weight_packed`、shape mismatch、AWQ kernel、空图像响应时立即停止，不把它解释成显存不足。

## 4. 三种服务性能档

在`commands/user_config.sh`设置：

```bash
export OMNI_PROFILE="safe"
export OMNI_PROFILE="balanced"
export OMNI_PROFILE="fast"
```

含义：

|档位|用途|默认行为|
|---|---|---|
|safe|首次排错|eager、较低显存利用率、较低并发|
|balanced|正式推荐|CUDA Graph可用、`max_num_seqs=4`、显存利用率0.92|
|fast|吞吐实验|`max_num_seqs=8`、显存利用率0.94，必须先benchmark|

首次运行建议`balanced`。若OOM改`safe`；只有balanced的smoke、benchmark、Debug都稳定后再试`fast`。

## 5. 关键配置

```bash
export OMNI_VISIBLE_GPUS="0,1,2,3"
export OMNI_TP_SIZE="4"
export OMNI_MAX_MODEL_LEN="12288"
export OMNI_MAX_NUM_SEQS="4"
export OMNI_GPU_MEMORY_UTILIZATION="0.92"
export OMNI_FULL_CONCURRENCY="4"
```

Debug仍由原正式脚本强制`concurrency=1`，便于复现和审计。Full使用`OMNI_FULL_CONCURRENCY`。

## 6. 中文与英文目录

英文：

```text
继续使用BASE_SETTINGS和原输出目录，不改动
```

中文：

```text
全部写入ZH_OUTPUT_ROOT
```

默认中文目录：

```text
/home/yk/fxy/results_ssh/yw_v1.2.2_zh
```

中文与英文不混写，不允许跨语言resume。

## 7. Direct基线保留

四种模式仍可选择：

```text
legacy_class_roi_obb
all_image_json_obb
all_image_json_hbb
single_ref_json_obb
```

正式B0/B1同协议比较优先使用`legacy_class_roi_obb`。JSON模式可以训练，但继续原test-matrix属于跨协议迁移评测，必须显式设置`ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX=1`。

不进行Socratic时可直接：

```bash
bash commands/run_v1.2.2.sh safe-direct
```

## 8. Omni轨迹运行顺序

```bash
bash commands/run_v1.2.2.sh prepare
bash commands/run_v1.2.2.sh datafull
bash commands/run_v1.2.2.sh omni-preflight
bash commands/run_v1.2.2.sh omni-start
bash commands/run_v1.2.2.sh omni-smoke
bash commands/run_v1.2.2.sh omni-benchmark
bash commands/run_v1.2.2.sh omni-shared-debug
```

Debug PASS后才执行：

```bash
bash commands/run_v1.2.2.sh omni-shared-full
```

最后：

```bash
bash commands/run_v1.2.2.sh omni-stop
```

## 9. 不改变的科学门槛

v1.2.2继续固定：

```text
question_similarity_threshold = 0.95
max_loop = 基础稳定settings原值
Reasoner/Perceiver/Verifier max_tokens = 基础稳定settings原值
Debug严格率与重复问题门槛 = 基础稳定settings原值
Full必须先通过Debug
```

脚本不会自动降低门槛、增加轮数或增加Token。

## 10. 训练边界

该AWQ-No-TTS checkpoint在v1.2.2中的默认用途是推理和轨迹生成。

不建议直接把未知第三方AWQ当作正式全参数微调基座。更稳的权重链是：

```text
官方未量化Qwen3-Omni基座
→ LoRA/QLoRA Adapter
→ 保存Adapter
→ 可选合并BF16/FP16
→ 重新AWQ量化
→ 独立评测
```

本包不自动安装Omni训练环境、不自动微调Omni、不自动重量化。

## 11. 新增文件

```text
commands/manage_omni_tp4.sh
files/tools/check_qwen3_omni_4gpu.py
files/tools/check_shared_omni_endpoint.py
files/tools/benchmark_shared_omni_endpoint.py
files/tools/materialize_shared_omni_agent_settings.py
```

此外，包内保留v1.2.0和v1.2.1的全部累计修改文件。
