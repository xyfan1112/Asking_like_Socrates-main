# 无网络迁移 `als_sft` / `als_vllm` 环境操作手册

## 1. 为什么不能直接复制 Miniconda 环境目录

Conda 环境中常见以下绝对路径：

- `bin/pip`、`bin/llamafactory-cli` 等脚本首行 shebang；
- `.pth` 文件；
- editable install 指向的源码目录；
- 某些动态库 RPATH、符号链接和包缓存路径。

从 `/home/nhl/...` 直接复制到 `/home/yk/...` 后，环境名字可能仍显示为 `(als_sft)`，Python 也可能部分可用，但 `pip` 会报：

```text
cannot execute: required file not found
```

这通常不是“pip 包缺失”，而是脚本第一行仍指向源机器不存在的解释器路径。

跨不同绝对路径迁移，推荐使用 `conda-pack`，目标端解压后执行 `conda-unpack`。不要把环境文件夹直接拖进目标 `envs/` 后就开始训练。

## 2. 源机器准备（环境正常的机器）

以下命令在源机器执行。示例源用户为 `/home/nhl`。

```bash
source /home/nhl/miniconda3/etc/profile.d/conda.sh
mkdir -p /home/nhl/fxy/offline_transfer

# 先验证源环境真的正常
/home/nhl/miniconda3/envs/als_sft/bin/python -c \
'import torch,transformers,llamafactory; print(torch.__version__, transformers.__version__, llamafactory.__file__)'

/home/nhl/miniconda3/envs/als_vllm/bin/python -c \
'import torch,transformers,vllm; print(torch.__version__, transformers.__version__, vllm.__version__)'

# 检查 conda-pack
/home/nhl/miniconda3/bin/python -c 'import conda_pack; print(conda_pack.__file__)'
```

若源机器尚可联网且没有 `conda-pack`，只在 base 环境安装一次。不要在稳定的 `als_sft`/`als_vllm` 内升级其他包。

```bash
/home/nhl/miniconda3/bin/python -m pip install conda-pack
```

打包：

```bash
/home/nhl/miniconda3/bin/conda-pack \
  -p /home/nhl/miniconda3/envs/als_sft \
  -o /home/nhl/fxy/offline_transfer/als_sft.tar.gz \
  --force

/home/nhl/miniconda3/bin/conda-pack \
  -p /home/nhl/miniconda3/envs/als_vllm \
  -o /home/nhl/fxy/offline_transfer/als_vllm.tar.gz \
  --force
```

项目源码单独打包。不要把模型权重和大数据集误放进去：

```bash
cd /home/nhl

tar --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='outputs' \
    --exclude='checkpoints' \
    -czf /home/nhl/fxy/offline_transfer/Asking_like_Socrates_code.tar.gz \
    Asking_like_Socrates-main
```

保存校验：

```bash
cd /home/nhl/fxy/offline_transfer
sha256sum als_sft.tar.gz als_vllm.tar.gz Asking_like_Socrates_code.tar.gz \
  > SHA256SUMS
```

把以下四个文件传到目标机器：

```text
als_sft.tar.gz
als_vllm.tar.gz
Asking_like_Socrates_code.tar.gz
SHA256SUMS
```

## 3. 目标机器解压（无网络）

以下命令在 `/home/yk` 目标机器执行。

```bash
mkdir -p /home/vieo/vieo/fxy_workspace/fxy/offline_transfer
cd /home/vieo/vieo/fxy_workspace/fxy/offline_transfer
sha256sum -c SHA256SUMS
```

先备份损坏环境：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)

[ ! -d /home/vieo/anaconda3/envs/als_sft ] || \
  mv /home/vieo/anaconda3/envs/als_sft \
     /home/vieo/anaconda3/envs/als_sft.broken_${STAMP}

[ ! -d /home/vieo/anaconda3/envs/als_vllm ] || \
  mv /home/vieo/anaconda3/envs/als_vllm \
     /home/vieo/anaconda3/envs/als_vllm.broken_${STAMP}
```

解压与重定位：

```bash
mkdir -p /home/vieo/anaconda3/envs/als_sft
mkdir -p /home/vieo/anaconda3/envs/als_vllm

tar -xzf /home/vieo/vieo/fxy_workspace/fxy/offline_transfer/als_sft.tar.gz \
  -C /home/vieo/anaconda3/envs/als_sft

tar -xzf /home/vieo/vieo/fxy_workspace/fxy/offline_transfer/als_vllm.tar.gz \
  -C /home/vieo/anaconda3/envs/als_vllm

/home/vieo/anaconda3/envs/als_sft/bin/conda-unpack
/home/vieo/anaconda3/envs/als_vllm/bin/conda-unpack
```

项目源码：

```bash
cd /home/yk
[ ! -d Asking_like_Socrates-main ] || \
  mv Asking_like_Socrates-main Asking_like_Socrates-main.backup_${STAMP}

tar -xzf /home/vieo/vieo/fxy_workspace/fxy/offline_transfer/Asking_like_Socrates_code.tar.gz \
  -C /home/yk
```

## 4. 修复 LLaMA-Factory 的本地源码引用

环境中若原先是 editable install，迁移后 `.pth` 可能仍指向 `/home/nhl/...`。在目标机器使用本地源码重新注册，不访问网络：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/LLaMA-Factory

/home/vieo/anaconda3/envs/als_sft/bin/python -m pip install \
  -e . --no-deps --no-build-isolation
```

注意：必须使用 `python -m pip`，不要先调用可能仍有旧 shebang 的 `bin/pip`。

若 vLLM 环境也依赖仓库内的可编辑源码，按相同方式在对应源码目录执行：

```bash
/home/vieo/anaconda3/envs/als_vllm/bin/python -m pip install \
  -e /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/<本地源码目录> \
  --no-deps --no-build-isolation
```

不要对 PyTorch、CUDA、transformers、vLLM 做在线升级。

## 5. 目标端强制验证

```bash
/home/vieo/anaconda3/envs/als_sft/bin/python -m pip --version
/home/vieo/anaconda3/envs/als_sft/bin/python -c \
'import sys,torch,transformers,llamafactory,peft,datasets; print(sys.executable); print(torch.__version__, transformers.__version__, llamafactory.__file__)'

/home/vieo/anaconda3/envs/als_vllm/bin/python -m pip --version
/home/vieo/anaconda3/envs/als_vllm/bin/python -c \
'import sys,torch,transformers,vllm; print(sys.executable); print(torch.__version__, transformers.__version__, vllm.__version__)'
```

检查脚本 shebang：

```bash
head -n 1 /home/vieo/anaconda3/envs/als_sft/bin/pip
head -n 1 /home/vieo/anaconda3/envs/als_sft/bin/llamafactory-cli
head -n 1 /home/vieo/anaconda3/envs/als_vllm/bin/vllm
```

这些行中不应再出现 `/home/nhl/`。

使用 v1.2.3 自检：

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts

bash tools/check_offline_env.sh \
  /home/vieo/anaconda3/envs/als_sft/bin/python \
  /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/LLaMA-Factory

bash tools/check_offline_env.sh \
  /home/vieo/anaconda3/envs/als_vllm/bin/python
```

## 6. 将准确解释器写入 v1.2.3

优先在 `settings.json` 的 workload 中写绝对路径，或者在 `commands/user_config.sh` 覆盖：

```bash
export SFT_PYTHON_OVERRIDE="/home/vieo/anaconda3/envs/als_sft/bin/python"
export VLLM_PYTHON_OVERRIDE="/home/vieo/anaconda3/envs/als_vllm/bin/python"
```

然后重新生成运行配置：

```bash
bash commands/run_v1.2.3.sh prepare
bash commands/run_v1.2.3.sh status
```

## 7. 重新开始 B1

环境验证全部通过后：

```bash
bash commands/run_v1.2.3.sh b1
```

不要把图 1 中只完成的 `B1 DIRECT CONFIG PASS`、`B1 DIRECT CONTRACT PASS` 当成训练完成。真正成功至少应看到：

```text
B1 Direct training completed
B1 Direct merged
B1 test_matrix_status.json passed=true
```

## 8. 没有 conda-pack 时

在源路径 `/home/nhl` 与目标路径 `/home/yk` 不同的情况下，直接 tar 环境目录不是可靠方案。若源端完全无网络且未安装 `conda-pack`，安全做法是先在另一台兼容机器离线准备 `conda-pack` wheel/conda 包，再安装到源 base 环境后打包；不要通过批量 `sed` 修改整个环境中的二进制文件。
