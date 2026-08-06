# re_scripts_omni：干净的 Qwen3-Omni Direct 路由

本包基于用户上传的 `re_scripts_VIEO.zip`，先恢复失败补丁前的 6 个文件，再做独立模型路由。

## 核心保证

- 原 `models.rs_eot`、`models.rs_eot_b1_direct`、`models.rs_eot_b2_socratic` 保留，不再伪装成 Omni。
- 新增 `qwen3_omni_b0`、`qwen3_omni_b1_direct`、`qwen3_omni_b2_socratic`。
- `safe-direct` 从 `model_routing` 读取 B0/B1 键；三卡评估使用通用 `replica`，不再把三卡误叫 replica4。
- Qwen3-Omni B1 保存 Adapter，不合并到 AWQ 基座。
- 修复 materializer 中 `base_path` 被字符串覆盖后触发的 `str.open` 错误。
- 只有 `commands/user_config.sh` 是用户配置入口。
- Qwen3-Omni B2 在本 Direct-only 包中明确阻断，避免误走旧 RS-EoT merge 流程。

## 安装

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main
mv re_scripts re_scripts_backup_$(date +%Y%m%d_%H%M%S)
unzip /实际位置/re_scripts_omni.zip -d /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main
cd re_scripts
```

## 第一次只做配置验证

```bash
rm -rf /home/vieo/vieo/fxy_workspace/fxy/results_ssh/yw_v1.2.2_zh/v1.2.3/config
bash commands/run_v1.2.3.sh prepare
bash commands/run_v1.2.3.sh eval-route-check
bash commands/run_v1.2.3.sh status
```

检查最终配置：

```bash
CFG=/home/vieo/vieo/fxy_workspace/fxy/results_ssh/yw_v1.2.2_zh/v1.2.3/config/settings.zh.v1.2.3.json
python3 - "$CFG" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(json.dumps(s['model_routing'],ensure_ascii=False,indent=2))
for key in ('rs_eot','qwen3_omni_b0','qwen3_omni_b1_direct'):
    print(key, s['models'][key])
PY
```

## 运行顺序

先 B0：

```bash
bash commands/run_v1.2.3.sh b0
```

训练环境和本地 LLaMA-Factory 通过前置检查后，再 B1：

```bash
bash commands/run_v1.2.3.sh b1-train
bash commands/run_v1.2.3.sh b1-eval
```

最后才运行：

```bash
bash commands/run_v1.2.3.sh safe-direct
```

如果本地 LLaMA-Factory 没有 `qwen3_omni` 支持，`b1-train` 会在训练前明确停止，不会继续错误训练。
