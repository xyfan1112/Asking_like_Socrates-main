# re_scripts_ssh1.2.2 完整修正版

本目录基于用户上传的 `re_scripts_ssh1.2.2.zip` 整合，保留原有模型路径、GPU编号、端口、数据目录和 `results4.3.3` 输出目录。

## 核心输入流

- Reasoner：纯文本，不接收原图或ROI图。
- Classification Perceiver：接收完整原图 + 放大ROI图。
- Grounding语义轮 Perceiver：接收完整原图 + 放大ROI图。
- Grounding坐标轮 Perceiver：只接收放大ROI图，输出ROI局部norm1000坐标；adapter映射回原图norm1000。
- Verifier：结构化文本验证，不接收图片。

该结构保持 Asking like Socrates 的 Reasoner/Perceiver职责分离，同时针对DOTA小目标增加Perceiver侧ROI增强。

## 关键参数

```text
Reasoner:  max_model_len=12288, max_tokens=896
Perceiver: max_model_len=8192,  max_tokens=512
Verifier:  max_model_len=4096,  max_tokens=256

max_repair_attempts=3
question_similarity_threshold=0.90
focus_crop_long_side=1024
focus_crop_padding_ratio=0.02
classification_use_full_and_crop=true
classification_crop_only_max_area=0.0
grounding_coordinate_crop_only=true
reject_unsolicited_coordinates=true
perceiver_fail_closed=true
reasoner_fail_closed=true
max_crop_coordinate_coverage=0.94
```

质量门槛没有降低：

```text
grounding_strict_rate >= 0.30
classification_strict_rate >= 0.70
duplicate_question_trajectories = 0
classification_contradiction_trajectories = 0
classification_leading_question_trajectories = 0
```

## 与上传版本相比的修复

1. `settings.json` 中 Perceiver `max_tokens` 从256提高到512。
2. Reasoner `max_tokens` 从768提高到896；Verifier保持256。
3. `02_run_official_generation.sh` 现在会读取并导出全部ROI、坐标和fail-closed配置，避免settings写了但运行时不生效。
4. Classification恢复为完整图+ROI，不再默认只看ROI。
5. Grounding坐标轮仍保持ROI单图，避免两个坐标系混淆。
6. Perceiver最多执行3次有界repair，不再固定只修一次。
7. 非坐标问题中主动输出OBB/bbox会被拒绝。
8. 覆盖整个ROI的坐标框、重复点、共线点、自交框会被拒绝。
9. repair耗尽后使用安全fallback，不让已知错误坐标进入轨迹。
10. 审计区分 `unrepaired_fail_closed` 和 `unrepaired_unsafe`；只有后者是硬失败。

## 安装检查

进入目录后执行：

```bash
python tools/check_ssh122_install.py settings.json

python -m py_compile \
  data_layer/official_socratic/local_api_adapter/utils.py \
  data_layer/official_socratic/02b_audit_debug_generation.py \
  data_layer/official_socratic/02c_audit_full_generation.py

bash -n data_layer/official_socratic/02_run_official_generation.sh

pytest -q \
  tests/test_v433_quality_gates.py \
  tests/test_v432_regressions.py \
  tests/test_v4_contracts.py \
  tests/test_ssh122_fixed.py
```

## 启动与Debug40

```bash
cd /root/Asking_like_Socrates-main/re_scripts_ssh1.2.2
CFG=$PWD/settings.json

python main_layer/run.py stop-agents --settings "$CFG" || true
python main_layer/run.py start-agents --settings "$CFG"

python main_layer/run.py official-socratic-full \
  --settings "$CFG" \
  train debug 40 fresh
```

运行日志必须出现：

```text
classification_full_and_crop=true
grounding_coordinate_crop_only=true
reject_unsolicited=true
perceiver_fail_closed=true
reasoner_fail_closed=true
```

Perceiver启动日志应含：

```text
--limit-mm-per-prompt {"image":2}
```

该参数表示每次请求最多允许两张图。Grounding坐标轮实际只发送一张ROI图。
