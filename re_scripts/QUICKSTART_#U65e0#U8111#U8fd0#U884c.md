# v1.2.2 无脑运行

## 1. 覆盖代码

```bash
cd /home/yk/v1.2.2_2A6000_direct_json_shared_omni_tp4_patch
cp -a files/. /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts/
```

或者执行只负责复制和语法检查的命令：

```bash
bash commands/copy_files_to_re_scripts.sh
```

## 2. 只修改一个配置文件

编辑：

```text
/home/yk/v1.2.2_2A6000_direct_json_shared_omni_tp4_patch/commands/user_config.sh
```

至少确认：

```bash
export REPO_ROOT="/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main"
export BASE_SETTINGS="/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts/settings.scene_disjoint.json"
export CLASSES_FILE="/home/vieo/vieo/fxy_workspace/fxy/datasets/yw/classes.txt"
export ZH_OUTPUT_ROOT="/home/vieo/vieo/fxy_workspace/fxy/results_ssh/yw_v1.2.2_zh"
export DIRECT_QA_MODE="legacy_class_roi_obb"

export OMNI_MODEL_PATH="/home/vieo/vieo/fxy_workspace/fxy/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS"
export OMNI_PYTHON="/home/vieo/anaconda3/envs/omni_vllm/bin/python"
export OMNI_PROFILE="balanced"
```

## 3. 只做B0＋Direct B1

```bash
cd /home/yk/v1.2.2_2A6000_direct_json_shared_omni_tp4_patch
bash commands/run_v1.2.2.sh safe-direct
```

顺序：

```text
prepare → data-full → B0 → Direct门控 → Direct训练/合并 → B1 Direct test-matrix
```

中间任何一步失败立即停止，不运行Socratic。

## 4. 下载指定Omni权重（本地已有则跳过）

```bash
bash commands/run_v1.2.2.sh omni-download
```

目标ModelScope ID：

```text
tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS
```

## 5. 一个模型使用四张A6000

先检查：

```bash
bash commands/run_v1.2.2.sh prepare
bash commands/run_v1.2.2.sh datafull
bash commands/run_v1.2.2.sh omni-preflight
```

启动一个TP=4服务：

```bash
bash commands/run_v1.2.2.sh omni-start
```

这个拓扑是：

```text
一份模型权重
→ Tensor Parallel = 4
→ 每次前向计算都跨GPU 0/1/2/3
→ Reasoner、Perceiver、Verifier共用同一端点
```

不要对共享Omni配置运行普通`start-agents`。

检查服务：

```bash
bash commands/run_v1.2.2.sh omni-status
bash commands/run_v1.2.2.sh omni-smoke
bash commands/run_v1.2.2.sh omni-benchmark
```

运行fresh Debug40：

```bash
bash commands/run_v1.2.2.sh omni-shared-debug
```

只有Debug基础设施和正式质量门全部PASS后：

```bash
bash commands/run_v1.2.2.sh omni-shared-full
```

停止服务：

```bash
bash commands/run_v1.2.2.sh omni-stop
```

## 6. 一条命令运行到Debug40

```bash
bash commands/run_v1.2.2.sh omni-safe-debug
```

它执行：

```text
prepare → data-full → preflight → TP4启动 → smoke → benchmark → fresh Debug40 → stop
```

## 7. 启动失败时

```bash
bash commands/run_v1.2.2.sh omni-log 200
bash commands/run_v1.2.2.sh omni-status
```

常见阻断错误：

```text
weight_packed / shape mismatch
AWQ kernel不兼容
config或weight index缺失
四卡不可见
模型端点没有图像响应
finish_reason=length
```

不通过增加`max_loop`、降低质量门或修改`question_similarity_threshold=0.95`掩盖错误。
