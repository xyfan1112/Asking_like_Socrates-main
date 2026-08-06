# v1.2.3-r3 vieo 临时机迁移与运行说明

## 固定目录

本包已经按下面的绝对路径配置：

```text
/home/vieo/vieo/fxy_workspace/
├── Asking_like_Socrates-main/
└── fxy/
    ├── datasets/
    ├── models/
    ├── results_ssh/
    ├── offlinetransfer/   # 推荐放 conda-pack tar.gz
    └── offline/           # 也会自动搜索
```

Conda 环境安装到：

```text
/home/vieo/anaconda3/envs/als_sft
/home/vieo/anaconda3/envs/als_vllm
```

## GPU 默认分配

目标机器只有物理 GPU `0,1,3` 可用。本包默认采用：

```text
共享 Qwen3-Omni Agent：物理 GPU 0,1；TP=2；端口 8091
SFT：物理 GPU 0,1；world_size=2；保持有效全局 batch=8
评估：物理 GPU 3；single；端口 8010
```

在启动前必须先用 `nvidia-smi` 确认这些卡没有其他用户的任务。本包不会主动杀死其他用户进程。

## 重要实验边界

本包完成的是 **当前 v1.2.3-r3 工程的 vieo 路径与三卡硬件适配**：

- 数据/Direct/Ref/Agent inputs 可以继续运行；
- Qwen3-Omni No-TTS 可以作为共享 Socratic Agent 服务；
- 原有 RS-EoT-7B B0/B1/B2 训练与评估路径仍保留。

本包 **没有把 B0/B1/B2 的学生模型改成 Qwen3-Omni**。当前配置中：

```text
OMNI_*：只控制共享 Socratic Agent 服务
models.rs_eot / rs_eot_b1_direct / rs_eot_b2_socratic：仍是原学生模型链路
```

因此，不要把 `safe-direct`、`b1`、`b2` 的结果解释为 Omni 微调结果。要正式实现：

```text
B0 = 原始 Qwen3-Omni No-TTS
B1 = Qwen3-Omni + Direct LoRA
B2 = Qwen3-Omni + Direct + Socratic LoRA
```

还需要单独修改 Qwen3-Omni 的 LLaMA-Factory 模板、AWQ/QLoRA、LoRA target、adapter 加载与评估协议。

## 第 1 步：备份并解压代码

假设压缩包已放到：

```text
/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main_vieo.zip
```

执行：

```bash
cd /home/vieo/vieo/fxy_workspace

if [[ -d Asking_like_Socrates-main ]]; then
  mv Asking_like_Socrates-main \
     Asking_like_Socrates-main_backup_$(date +%Y%m%d_%H%M%S)
fi

unzip Asking_like_Socrates-main_vieo.zip \
  -d /home/vieo/vieo/fxy_workspace
```

解压后必须是：

```text
/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main
```

检查：

```bash
test -f \
  /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts/settings.json \
  && echo '[PASS] 代码目录正确' \
  || echo '[FAIL] 解压目录不正确'
```

## 第 2 步：离线部署 als_sft 和 als_vllm

先确认 `als_sft*.tar.gz` 和 `als_vllm*.tar.gz` 已放在下面任一目录：

```text
/home/vieo/vieo/fxy_workspace/fxy/offlinetransfer
/home/vieo/vieo/fxy_workspace/fxy/offline
/home/vieo/vieo/fxy_workspace/fxy
```

然后执行：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main

bash re_scripts/commands/setup_vieo_offline_envs.sh
```

脚本会：

1. 自动寻找唯一的 `als_sft` 和 `als_vllm` conda-pack 压缩包；
2. 校验同目录 SHA256（存在时）；
3. 解压到 `/home/vieo/anaconda3/envs/`；
4. 执行各自的 `conda-unpack`；
5. 将项目内 `LLaMA-Factory/src` 注册到 `als_sft`；
6. 检查 Python、PyTorch、vLLM、Transformers、PEFT 和 LLaMA-Factory 导入。

如果目标环境目录已经存在，默认只复用，不覆盖。确认必须重装时：

```bash
bash re_scripts/commands/setup_vieo_offline_envs.sh --replace
```

`--replace` 会先把旧环境改名备份，不直接删除。

如果压缩包不在默认搜索目录：

```bash
bash re_scripts/commands/setup_vieo_offline_envs.sh \
  --archive-dir /你的实际压缩包目录
```

## 第 3 步：迁移 scene manifest 和生成型配置

你复制的 `yw128_scene_disjoint/scene_split_manifest.json` 可能仍记录 `/home/yk/...`。复制的 `runtime_settings` 和活动 settings 也可能保存旧绝对路径。

执行：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main

bash re_scripts/commands/prepare_vieo_workspace.sh
```

该脚本会：

- 备份并重写 scene manifest 的 `source_root`、`output_root`；
- 保留 `train=128 / val=20 / overlap=0` 等原统计；
- 在 manifest 中追加 relocation 记录；
- 补充 `yw128_scene_disjoint/classes.txt`（缺失时）；
- 删除复制来的生成型 `runtime_settings`；
- 备份旧活动 settings，让 `prepare` 在 vieo 路径重新生成；
- 修复脚本执行权限。

它不会删除数据、模型、轨迹、adapter 或正式结果。

## 第 4 步：只读检查

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main

bash re_scripts/commands/verify_vieo_machine.sh
```

随后：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts

bash commands/run_v1.2.3.sh doctor
bash commands/run_v1.2.3.sh eval-route-check
bash commands/run_v1.2.3.sh prepare
```

`eval-route-check` 用于确认评估清理不会再把 `rs_eot` 错改成 `rs_eot_b2_socratic`。

## 第 5 步：检查 GPU 是否被占用

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
```

查看所有 GPU 进程及所有者：

```bash
for PID in $(
  nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits |
  sort -u
); do
  ps -o user,pid,ppid,etime,cmd -p "$PID"
done
```

不要执行 `pkill python`、`killall python` 或无条件 `fuser -k`。

## 第 6 步：验证共享 Omni TP2

先只做静态检查：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts

bash commands/run_v1.2.3.sh omni-preflight
```

本包对 vLLM 0.11.0 使用：

```text
/home/vieo/anaconda3/envs/als_vllm/bin/python
  -m vllm.entrypoints.cli.main serve --help=all
```

不会依赖可能受外部 PATH 影响的 `bin/vllm`。

启动：

```bash
bash commands/run_v1.2.3.sh omni-start
```

状态与日志：

```bash
bash commands/run_v1.2.3.sh omni-status
bash commands/run_v1.2.3.sh omni-log 200
```

停止：

```bash
bash commands/run_v1.2.3.sh omni-stop
```

默认参数是保守 smoke 配置：

```text
CUDA_VISIBLE_DEVICES=0,1
TP=2
max_model_len=4096
max_num_seqs=1
gpu_memory_utilization=0.80
enforce_eager=true（safe profile）
```

## 第 7 步：数据和 Socratic Debug

数据层：

```bash
bash commands/run_v1.2.3.sh data
```

共享 Omni Debug40：

```bash
bash commands/run_v1.2.3.sh socratic-one-debug
```

Debug 通过后再运行 Full：

```bash
bash commands/run_v1.2.3.sh socratic-one-full
```

不要第一次直接运行完整 `safe-b2-one`，应先分阶段确认环境、Omni 加载、图像请求和 Debug40。

## 目标机默认配置位置

```text
/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/
  re_scripts/commands/user_config.sh
```

当前关键默认值：

```bash
RUN_LANG="zh"
DATASET_VARIANT="scene_disjoint"

OMNI_VISIBLE_GPUS="0,1"
OMNI_TP_SIZE="2"
OMNI_PROFILE="safe"

SFT_VISIBLE_GPUS="0,1"
SFT_WORLD_SIZE="2"

EVAL_TOPOLOGY="single"
EVAL_GPUS="3"
```

## 收集故障包

```bash
bash commands/run_v1.2.3.sh collect error
bash commands/run_v1.2.3.sh collect socratic
bash commands/run_v1.2.3.sh collect all
```

诊断包默认不包含模型权重和全量图片。
