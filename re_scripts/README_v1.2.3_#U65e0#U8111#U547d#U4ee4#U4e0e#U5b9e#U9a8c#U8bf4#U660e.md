# re_scripts v1.2.3：OBB Direct / Socratic / B0-B1-B2 无脑运行手册

## 1. 版本定位

本包以用户上传的 `re_scripts1.2.2(2).zip` 为唯一代码基线，内部数据流水线仍保持 `4.3.3`，稳定基础仍为 `ssh1.2.2.2`。本次补丁版本为 **v1.2.3**。

v1.2.3 不主动修改下列科学条件：

- `question_similarity_threshold`；
- Socratic `max_loop`；
- Reasoner、Perceiver、Verifier 的 token 上限；
- Debug 质量阈值；
- LoRA rank、alpha、dropout、epoch、learning rate；
- Direct 默认协议 `legacy_class_roi_obb`；
- 视觉塔冻结策略。

改变的是运行拓扑、语言/路径隔离、环境检查、四卡并行、产物审计与故障打包。

## 2. 先修改一个文件

只修改：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts
vim commands/user_config.sh
```

最常改的变量：

```bash
export RUN_LANG="en"                    # 默认英文；中文改 zh
export DATASET_VARIANT="scene_disjoint" # 正式实验默认无场景泄漏数据
export DIRECT_QA_MODE="legacy_class_roi_obb"
export SOCRATIC_TOPOLOGY="shared"       # shared 或 independent
```

### 数据集配置

v1.2.3 不再要求仓库必须存在 `settings.scene_disjoint.json`。它始终以真实的 `settings.json` 为基础，只在生成运行配置时覆盖：

```bash
RAW_DATASET_ROOT=/home/vieo/vieo/fxy_workspace/fxy/datasets/yw128
SCENE_DISJOINT_DATASET_ROOT=/home/vieo/vieo/fxy_workspace/fxy/datasets/yw128_scene_disjoint
DATASET_VARIANT=scene_disjoint
```

因此 `settings.json` 与所谓 `settings.scene_disjoint.json` 其余字段完全相同时，不需要维护两份易漂移的配置。

### classes.txt

无需把 16 类手工发给代码。默认自动读取：

```text
/home/vieo/vieo/fxy_workspace/fxy/datasets/yw128/classes.txt
```

`class_id` 仍由行号确定，标注 txt 中的数字和 OBB 坐标不会被改动。

英文第一次执行 `prepare` 时，代码会根据真实 classes.txt 自动生成：

```text
<EN_OUTPUT_ROOT>/config/class_map.zh_en.json
<EN_OUTPUT_ROOT>/config/class_translation_todo.txt
```

对已经是英文的类别会自动原样填写；中文车辆类别的英文名称不会被模型擅自猜测。把 `classes[].en` 补全后再次执行 `prepare`。这一步只确认名称，不改变类别 ID。

## 3. B0、B1、B2 的定义

- **B0**：RS-EoT 基础模型，不训练，直接测试矩阵评估。
- **B1**：同一 RS-EoT 基础模型，只用 Direct OBB 数据做 LoRA SFT，合并后评估。
- **B2**：同一 RS-EoT 基础模型，用 Direct 加严格门控后的 Socratic 轨迹做 LoRA SFT，合并后评估。

B2 不从 B1 权重继续训练。B1 与 B2 都从相同基础模型开始，避免把串行训练误当成 Socratic 增益。

## 4. `safe-direct` 不需要三个智能体

```bash
bash commands/run_v1.2.3.sh safe-direct
```

执行：

```text
prepare
→ data
→ B0 四卡评估
→ B1 Direct 四卡 LoRA
→ B1 合并
→ B1 四卡评估
```

它不生成 Socratic 轨迹，不需要 Reasoner、Perceiver、Verifier，也不需要 Omni。

旧日志中的：

```text
[FREE] Reasoner port=8001
[FREE] Perceiver port=8002
[FREE] Verifier port=8003
```

仅表示端口空闲，不是一种模型模式，也不代表服务已启动。

## 5. 为什么 v1.2.2 的 `fast` 没让 B0/B1 四卡工作

`OMNI_PROFILE=fast` 只控制共享 30B Omni 服务。旧 B0/B1 代码明确执行：

```bash
CUDA_VISIBLE_DEVICES=1
```

所以只有 GPU 1 接近满载是旧代码的预期行为，并非 `fast` 失效。

v1.2.3 的 7B 评估改为：

```text
GPU0 / 8010 / shard0
GPU1 / 8011 / shard1
GPU2 / 8012 / shard2
GPU3 / 8013 / shard3
```

每个 GPU 放一份 7B 模型，每个协议用 SHA256 稳定分片同时跑四份，最后检查数量、去重并按 ID 合并。对能单卡放下的 7B 模型，这比 TP4 更适合吞吐量。

共享 30B Omni 仍是一个模型端点、TP=4，每个请求由四张卡共同完成。`fast` 默认并发为 4、显存利用率为 0.92，不会暗中跳到 8 并发；确认真实模型压测有余量后才手动提高。

## 6. 四卡 LoRA 训练

基础 v1.2.2：

```text
2 GPU × per_device 1 × gradient_accumulation 4 = 有效全局 batch 8
```

v1.2.3：

```text
4 GPU × per_device 1 × gradient_accumulation 2 = 有效全局 batch 8
```

因此只改变 DDP 并行度，不改变有效全局 batch。

当前训练合同：

```text
finetuning_type = lora
lora_target = all
freeze_vision_tower = true
freeze_multi_modal_projector = false
```

解释：视觉编码塔被冻结；多模态 projector 没有被冻结；LoRA 插入由本机 LLaMA-Factory 的 `lora_target=all` 规则决定。训练前会生成合同报告，训练日志后会提取 LLaMA-Factory 实际打印的 trainable/all params；训练成功后还会直接读取 LoRA adapter 的 safetensors 键，生成 `adapter_trainable_scope_*.json`。若要求冻结视觉塔却出现明显的 `visual.blocks`/`vision_tower` LoRA 张量，流程会停止。Projector/merger 与语言模型张量会分别统计，未知命名不会被擅自归类。

## 7. 基础无脑命令

```bash
# 查看配置、端口、GPU 和已有产物
bash commands/run_v1.2.3.sh status

# 只创建/检查当前语言配置；不生成QA、不启动模型、不训练
bash commands/run_v1.2.3.sh prepare

# 生成并审计 OBB Canonical、Direct、Ref、Agent inputs
bash commands/run_v1.2.3.sh data

# 单独得到 B0
bash commands/run_v1.2.3.sh b0

# Direct LoRA、合并和评估，得到 B1
bash commands/run_v1.2.3.sh b1

# 使用已经生成并通过门控的 Socratic 数据训练、合并和评估，得到 B2
bash commands/run_v1.2.3.sh b2

# 汇总 B0/B1/B2
bash commands/run_v1.2.3.sh compare
```

## 8. Socratic 两种拓扑

一般只需要在 `commands/user_config.sh` 改一项：

```bash
export SOCRATIC_TOPOLOGY="shared"       # shared | independent
```

然后使用不带拓扑名称的统一命令：

```bash
bash commands/run_v1.2.3.sh socratic-debug
bash commands/run_v1.2.3.sh socratic-full
```

### 8.1 默认：共享 Omni

显式命令仍保留，便于固定实验拓扑：

```bash
bash commands/run_v1.2.3.sh socratic-one-debug
bash commands/run_v1.2.3.sh socratic-one-full
```

物理结构：

```text
Reasoner  ─┐
Perceiver ─┼→ 8091 / 一个 Omni 权重副本 / TP4
Verifier  ─┘
```

共享的是模型权重、GPU 和服务进程。三个角色的 system prompt、messages、采样参数和解析器分别独立；不同样本之间也不共享对话历史。不存在跨样本“长期上下文”。

### 8.2 原三服务

```bash
bash commands/run_v1.2.3.sh socratic-three-debug
bash commands/run_v1.2.3.sh socratic-three-full
```

```text
Reasoner  → 8001
Perceiver → 8002
Verifier  → 8003
```

在 `user_config.sh` 中可以分别设置三个模型路径。默认三个服务使用同一个 Qwen3-VL-8B-Instruct，但仍是三个独立模型进程。

## 9. 完整安全流程

```bash
# 数据→B0→B1；不运行Socratic
bash commands/run_v1.2.3.sh safe-direct

# 根据 SOCRATIC_TOPOLOGY 自动选择共享 Omni 或原三服务
bash commands/run_v1.2.3.sh safe-b2
bash commands/run_v1.2.3.sh safe-all

# 需要把拓扑固定写进实验命令时，仍可显式使用：
bash commands/run_v1.2.3.sh safe-b2-one
bash commands/run_v1.2.3.sh safe-b2-three
bash commands/run_v1.2.3.sh safe-all-one
bash commands/run_v1.2.3.sh safe-all-three
```

Full 必须先通过 Debug gate。失败时立即停止，不会自动放宽阈值或继续 Full。Full 默认使用 `official_socratic.current_task_samples=999999999`，即处理当前输入中的全部任务；`Debug40` 才只抽 40 条。

## 10. 安装/覆盖代码包

先备份旧版，仅替换 `re_scripts`，不要覆盖数据、模型和 Conda 环境：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main
STAMP=$(date +%Y%m%d_%H%M%S)
mv re_scripts re_scripts_v1.2.2_backup_${STAMP}
unzip /你的传输目录/re_scripts_v1.2.3.zip -d /tmp/re_scripts_v1.2.3_unpack
mv /tmp/re_scripts_v1.2.3_unpack/re_scripts ./re_scripts
cd re_scripts
chmod +x commands/*.sh test_layer/*.sh test_layer/protocols/*.sh train_layer/*.sh tools/*.sh
```

先编辑 `commands/user_config.sh`，再执行：

```bash
bash commands/run_v1.2.3.sh status
bash commands/run_v1.2.3.sh prepare
```

英文首次 `prepare` 因缺少 16 类的确认英文名而以退出码 3 停止，是设计行为，不是报错。补完 `class_map.zh_en.json` 后重跑。

## 11. 当前图 1 的 B1 状态

图中已通过：

```text
B1 DIRECT CONFIG PASS
B1 DIRECT CONTRACT PASS
```

随后在真正进入训练前失败：

```text
ModuleNotFoundError: No module named 'llamafactory'
[FAIL] SFT environment cannot import llamafactory/torch/transformers
```

因此当前状态是：

- Direct 数据和训练 YAML 已准备；
- 训练合同已通过；
- **B1 LoRA 尚未开始或至少未成功完成**；
- 没有可靠的 B1 adapter；
- 没有可靠的 B1 merged model；
- 没有 B1 测试结果。

激活提示 `(als_sft)` 不能证明该环境完整。图中 `bin/pip: cannot execute: required file not found` 很像环境从其他绝对路径迁移后，`pip` shebang 仍指向旧路径；同时该环境的 Python 确实没有导入到 `llamafactory`。

迁移/修复后先执行：

```bash
/home/vieo/anaconda3/envs/als_sft/bin/python -m pip --version
/home/vieo/anaconda3/envs/als_sft/bin/python -c \
'import sys,torch,transformers,llamafactory; print(sys.executable, torch.__version__, transformers.__version__, llamafactory.__file__)'

bash tools/check_offline_env.sh \
  /home/vieo/anaconda3/envs/als_sft/bin/python \
  /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/LLaMA-Factory
```

不要直接用损坏的 `pip` 可执行文件；优先使用 `python -m pip`。

## 12. 出错时一条命令打包

```bash
bash commands/run_v1.2.3.sh collect error
bash commands/run_v1.2.3.sh collect data
bash commands/run_v1.2.3.sh collect socratic
bash commands/run_v1.2.3.sh collect train-b1
bash commands/run_v1.2.3.sh collect train-b2
bash commands/run_v1.2.3.sh collect eval-b0
bash commands/run_v1.2.3.sh collect eval-b1
bash commands/run_v1.2.3.sh collect eval-b2
bash commands/run_v1.2.3.sh collect all
```

诊断包包含配置、版本、环境/GPU 信息、报告、日志首尾、少量 JSONL 样本和指标；自动排除模型权重、LoRA 权重、图片和常见密钥字段。

典型对应关系：

| 问题 | 命令 |
|---|---|
| 路径、classes、数据数量、scene split | `collect data` |
| `Let's look at the im...`、`MAX_ROUNDS_EXCEEDED`、重复问题、Verifier 拒绝 | `collect socratic` |
| `llamafactory`、OOM、NCCL、loss/step 异常 | `collect train-b1` 或 `collect train-b2` |
| 只用一张卡、预测为空、IoU/mAP 为 0 | 对应 `collect eval-*` |
| 不清楚阶段 | `collect error` |

## 13. 不要混淆两个“B2/Stage 2”

本工程中的 B2 指“Direct + Socratic 轨迹的 SFT 模型”。论文的 Stage 2 指 RL-VQA。二者不是同一阶段。本 v1.2.3 重点完成 OBB Direct、Socratic SFT 与 B0/B1/B2 训练评估，不自动启动论文的两阶段 RL。
