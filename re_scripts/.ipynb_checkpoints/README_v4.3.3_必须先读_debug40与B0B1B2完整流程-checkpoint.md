# re_scripts v4.3.3：先修复 Socratic 轨迹，再跑 debug40 和 B0/B1/B2

这是 v4.3.3 的权威操作入口。所有命令一次只执行一条，不要把两条
`python main_layer/run.py ...` 粘在同一行。

## 1. 对你这批 v4.3.2 结果的结论

这批数据不能继续用于正式全量生成或 B1/B2 训练。旧审计显示
`official_success=6/20`，但用本版严格门控重新复核后结果如下：

| 项目 | 旧 debug20 的复核结果 | 要求 |
|---|---:|---:|
| Perceiver 调用 | 80 | — |
| 丢失原始目标上下文 | 80/80 | 0 |
| 丢失 focus ROI | 80/80 | 0 |
| 重复或高度相似问题轨迹 | 9/20 | 0 |
| 分类视觉证据矛盾 | 8/10 | 0 |
| 通过直接类别问题取答案 | 10/10 | 0 |
| grounding 同时通过 IoU 与 center | 1/10 | 至少 30% |
| classification strict | 0/10 | 至少 70% |
| 新门控状态 | `FAIL`，退出码 2 | `PASS`，退出码 0 |

旧 full 394 条用新门控复核只剩 18 条：classification 8/197、
grounding 10/197；其中有 120 条重复问题轨迹、79 条分类证据矛盾、
178 条类别引导问题。旧 full 自身还存在 7 次长度截断和 2 次未修复格式错误。
因此不要复用这批 raw，也不要在它上面继续 postproc、SFT 或 B1/B2 比较。

## 2. 本版具体修改了什么

| 必改项 | 实现位置 | 现在如何拦截 |
|---|---|---|
| Perceiver 保留原始目标与 ROI | `data_layer/official_socratic/local_api_adapter/utils.py`、`02_build_dota128_ref.py` | 每个线程保存不可变 User Query/Image Metadata；每次 Perceiver 请求附带原上下文和自动生成的 coarse focus ROI；缺任一项立即失败 |
| 真正的语义唯一性 | `data_layer/ref_semantics.py`、`02_build_dota128_ref.py`、`03_validate_dota128_ref.py` | 将类别、区域、几何、全图极值、唯一锚点及方向作为结构约束，对同图所有有效目标重新求匹配；结果必须恰好为当前 object index |
| 拒绝重复问题 | adapter 与 `trajectory_gates.py` | 归一化 token 后同时计算序列相似度和 Jaccard；阈值默认 0.82；生成时先修复，仍重复则记为 unrepaired，审计再次独立检查 |
| 分类证据一致性 | `trajectory_gates.py` | 拒绝目标缺失、强错误类别判断、多类别冲突、场景区域替代单目标、目标集合替代单目标 |
| 不准直接问类别 | adapter、prompt、审计 | 分类的中间问题不能询问 class/category/label，也不能包含 DOTA 类别及常见别名 |
| grounding 双几何门 | `trajectory_gates.py` | `HBB IoU >= 0.10` 且预测中心落入扩张 GT 区域；`pass_if_iou_or_center=false` |
| 提高可见性 | `settings*.json` | 短边至少 20/1000，面积比至少 0.0002 才进入 Socratic 子集 |
| WARN 必须阻断 | `02b_audit_debug_generation.py`、`run_official_socratic_full.sh` | PASS=0、FAIL=2、WARN=3；只有 PASS 才打印 `[DEBUG GATE PASS]` |
| 防止混用旧 raw | `00_manage_run_state.py` | manifest 签名包含版本、Parquet、prompt、adapter、Ref 语义解析器、轨迹门控和关键配置；full 使用 `fresh` 会先归档旧 canonical raw |

Reasoner 的默认生成上限也从 768 提高到 1024；Perceiver 的
`max_model_len` 提高到 8192。你的旧 full 日志中 Reasoner 和 Perceiver
都发生过长度截断，而附加目标上下文后 prompt 会更长，这两项必须同步调整。

## 3. 覆盖安装 v4.3.3

先确定实际仓库目录。你最新日志使用的是
`/home/nhl/Asking_like_Socrates-main`；如果没有 `-main`，下面会自动选择旧路径。

```bash
if [[ -d /home/nhl/Asking_like_Socrates-main/re_scripts ]]; then
  REPO=/home/nhl/Asking_like_Socrates-main
else
  REPO=/home/nhl/Asking_like_Socrates
fi

cd "$REPO"
STAMP=$(date +%Y%m%d_%H%M%S)
cp -a re_scripts "re_scripts_backup_$STAMP"
```

把 `re_scripts4.3.3_socratic_quality_fixed.zip` 放到 `$REPO` 后执行：

```bash
unzip -q -o re_scripts4.3.3_socratic_quality_fixed.zip -d "$REPO"
cd "$REPO/re_scripts"
conda activate als_sft
```

压缩包内顶层就是 `re_scripts/`。覆盖不会删除你的
`settings.scene_disjoint.json`，并且前一步已经完整备份旧脚本。

## 4. 升级你的 scene-disjoint 配置

后续始终使用同一个配置：

```bash
CFG=/root/Asking_like_Socrates-main/re_scripts/settings.json
test -f "$CFG" || { echo "缺少 $CFG"; exit 2; }
```

先只查看将要修改的字段，不落盘：

```bash
python tools/upgrade_v433_socratic_quality.py \
  --settings "$CFG" \
  --bump-output-version
```

确认输出后再写入。工具会先产生带时间戳的配置备份：

```bash
python tools/upgrade_v433_socratic_quality.py \
  --settings "$CFG" \
  --bump-output-version \
  --in-place
```

它会把配置中已有的 `4.3.2` 产物路径改为 `4.3.3`，所以不会复用旧 raw。
它不会修改原始 DOTA 图像/标签路径，也不会删除旧结果。

检查关键配置：

```bash
python - "$CFG" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
c, t, a = s["data_conversion"], s["trajectory"], s["agents"]
print("short_side =", c["socratic_min_short_side_norm1000"])
print("area_ratio =", c["socratic_min_area_ratio"])
print("semantic_unique =", c["require_unique_reference"])
print("perceiver_context =", t["require_perceiver_original_context"])
print("question_similarity =", t["question_similarity_threshold"])
print("geometry =", t["geometry_gate"])
print("debug rates =", t["debug_min_grounding_strict_rate"],
      t["debug_min_classification_strict_rate"])
print("reasoner max_tokens =", a["reasoner"]["max_tokens"])
print("perceiver max_model_len =", a["perceiver"]["max_model_len"])
print("raw_output_dir =", s["official_socratic"]["raw_output_dir"])
PY
```

必须看到：

```text
short_side = 20.0
area_ratio = 0.0002
semantic_unique = True
perceiver_context = True
question_similarity = 0.82
pass_if_iou_or_center = False
debug rates = 0.3 0.7
reasoner max_tokens >= 1024
perceiver max_model_len >= 8192
```

## 5. 先运行本地代码测试

```bash
bash tests/run_smoke_test.sh
```

应连续出现：

```text
SMOKE TEST PASS
V4.3.3 CONTRACT TESTS PASS
V4.3.3 REGRESSION TESTS PASS
V4.3.3 QUALITY GATE TESTS PASS
```

只要其中一个不是 PASS，就不要启动智能体。

## 6. 重新构建数据层

先停掉可能残留的三个 agent 服务，然后从原始 scene-disjoint DOTA 数据重新构建：

```bash
python main_layer/run.py stop-agents --settings "$CFG"
```

```bash
python main_layer/run.py data-full --settings "$CFG"
```

不要复制旧版的 `train.jsonl`、`train_socratic.jsonl`、agent input、Parquet 或 raw。
v4.3.3 的 Ref ID、query schema 与语义验证规则都已改变。

查看最终数据门：

```bash
python - "$CFG" <<'PY'
import json, sys
from pathlib import Path
s = json.load(open(sys.argv[1], encoding="utf-8"))
p = Path(s["paths"]["pipeline_work_root"]) / "reports/data_layer_final_report.json"
r = json.load(open(p, encoding="utf-8"))
print(json.dumps(r, ensure_ascii=False, indent=2))
print("REPORT =", p)
raise SystemExit(0 if r.get("passed") else 2)
PY
```

此外，`dota128_ref_root/validation_report.json` 中以下计数必须为 0：

- `scene_leakage`；
- `semantic_reference_not_unique`；
- `semantic_resolution_mismatch`；
- `classification_alias_leak`；
- `grounding_question_focus_missing_or_mismatched`；
- `classification_question_focus_missing_or_mismatched`；
- `duplicated_area_word`；
- 坐标越界与无效 polygon。

可视化抽查 `pipeline_work_root/ref_previews/`，至少检查密集车辆、船、圆形目标和
边界目标。规则验证只能证明“指代约束唯一”，不能代替视觉可读性抽查。

## 7. 启动三个本地 agent

```bash
python main_layer/run.py start-agents --settings "$CFG"
```

必须等到 Reasoner、Perceiver、Verifier 都完成真实 chat completion 预检。
仅看到 `/models` 列表不算服务可用。

如 GPU 显存不足，先停止其他 eval 服务：

```bash
python main_layer/run.py stop-eval --settings "$CFG" rs_eot 8010
```

然后重新执行 `start-agents`。B1/B2 训练前还要再次停止 agents。

## 8. 重新跑 debug40

严格按一条完整命令执行：

```bash
python main_layer/run.py official-socratic-full \
  --settings "$CFG" \
  train debug 40 fresh
```
python main_layer/run.py stop-agents --settings "$CFG"

紧接着查看 shell 退出码：

```bash
echo "debug_exit=$?"
```

含义：

- `0`：PASS，可以继续 full；
- `2`：硬错误/配置错误，禁止继续；
- `3`：质量 WARN，仍然禁止继续。

只有成功时末尾才会出现：

```text
[DEBUG GATE PASS] debug audit met every v4.3.3 quality threshold.
[NEXT] Only now may you run official-socratic-full in full/fresh mode.
```

再用独立检查器复核最新 audit：

```bash
python tools/check_latest_debug_audit.py --settings "$CFG"
```

debug40 的硬性通过标准是：

| 字段 | 标准 |
|---|---:|
| `status` | `PASS` |
| `api.api_errors` | 0 |
| `api.length_truncation` | 0 |
| `api.unrepaired` | 0 |
| `coordinate_rewrite_mutations` | 0 |
| `api.perceiver_context_missing` | 0 |
| `api.perceiver_focus_roi_missing` | 0 |
| `duplicate_question_trajectories` | 0 |
| `classification_contradiction_trajectories` | 0 |
| `classification_leading_question_trajectories` | 0 |
| `strict_rate_by_task.ref_classification` | ≥ 0.70 |
| `strict_rate_by_task.ref_grounding_obb` | ≥ 0.30 |
| grounding strict | 每条同时通过 `iou_gate` 和 `center_gate` |

旧版的 `official_success` 不是正式通过标准。Verifier 仍作为参考记录，最终是否进入
SFT 由确定性的格式、语义与几何门控共同决定。

## 9. debug40 不通过时怎么处理

不要先降低阈值，也不要直接跑 full。按 audit 字段定位：

| 失败字段 | 首先检查 |
|---|---|
| `perceiver_context_missing > 0` | 实际运行的是否为本包 `utils.py`；启动日志中的 `PYTHONPATH` 是否指向当前 `re_scripts/data_layer/official_socratic/local_api_adapter` |
| `perceiver_focus_roi_missing > 0` | 是否重新执行 `data-full`；grounding/classification query 是否都含 `coarse focus region [...]`；不能复用旧 Parquet |
| `length_truncation > 0` | 配置是否已将 Reasoner `max_tokens>=1024`、Perceiver `max_model_len>=8192`；确认实际 materialized settings |
| `duplicate_question_trajectories > 0` | 查看 audit 中 `duplicate_question_pairs`；不要只改审计，应检查 adapter 是否启用了 0.82 的生成时拒绝 |
| `classification_leading... > 0` | Reasoner 中间问题仍在问类别或直接出现 DOTA 类别名；检查 prompt profile 是否为 `obb_grounding_v3` |
| `classification_contradiction... > 0` | Perceiver 把周边场景、多个目标或其他类别当成 target；检查该条原始 ROI 可视化，必要时提高可见性门槛，不要 teacher-force 掩盖 |
| grounding IoU 过、center 不过 | 粗框落在错误的相邻实例上；检查 reference/ROI 和密集目标消歧 |
| center 过、IoU 不过 | 只猜了一个很小点/框；不能用旧 OR 规则放行，应改 Perceiver 提问以取得边界证据 |
| classification < 70% | 先按类别统计；若集中在 storage tank/roundabout 或 small/large vehicle，检查目标像素大小、masked reference 与视觉证据，不要直接给 agent 类别答案 |

每次修改 prompt、adapter、Ref 构造或门控后都重新执行：

```bash
python main_layer/run.py data-full --settings "$CFG"
```

```bash
python main_layer/run.py official-socratic-full \
  --settings "$CFG" \
  train debug 40 fresh
```

## 10. 只有 debug40 PASS 后才跑 full

```bash
python main_layer/run.py official-socratic-full \
  --settings "$CFG" \
  train full 40 fresh
```

`fresh` 会归档同一 v4.3.3 路径下已有的 canonical full raw。完成后脚本会依次：

1. 审计完整 API 日志；
2. 运行 strict 语义/几何门；
3. 检查 grounding 与 classification 各至少 20 条；
4. 运行官方 postproc；
5. 构造 matched B1 final-only 与 B2 Socratic；
6. 验证两组问题、图像、GT、pair ID 和行数完全一致；
7. 生成 LLaMA-Factory 训练配置。

只要中间任一步失败，命令就会非零退出，不应手动跳过。

查看 strict 报告：

```bash
python - "$CFG" <<'PY'
import json, sys
from pathlib import Path
s = json.load(open(sys.argv[1], encoding="utf-8"))
p = Path(s["official_socratic"]["raw_output_dir"]) / \
    "dota128_train_official_strict.report.json"
print(json.dumps(json.load(open(p, encoding="utf-8")),
                 ensure_ascii=False, indent=2))
print("REPORT =", p)
PY
```

## 11. 训练匹配的 B1 与 B2
bash /root/Asking_like_Socrates-main/re_scripts/main_layer/train_matched_b1_b2.sh

B1/B2 都由 full 的同一批 strict pair 自动产生。不要再单独执行
`prepare-direct-sft` 作为主实验，也不要把它和 `train-b1` 粘在同一行。

先停止三个 agent：

```bash
python main_layer/run.py stop-agents --settings "$CFG"
```

验证训练数据：

```bash
python main_layer/run.py validate-train --settings "$CFG"
```

训练 B1：

```bash
python main_layer/run.py train-b1 --settings "$CFG"
```

训练 B2：

```bash
python main_layer/run.py train-b2 --settings "$CFG"
```

合并两个 LoRA：

```bash
python main_layer/run.py merge --settings "$CFG" both
```

如果提示 `adapter unavailable`，说明前面的训练没有成功产出
`adapter_config.json` 与 adapter 权重；不能把 `[SKIP]` 当成成功。

你的双 A6000 默认使用 2 卡、每卡 batch 1、gradient accumulation 4、LoRA rank 16、
冻结视觉塔、训练投影层、`cutoff_len=8192`、`image_max_pixels=802816`。B1 和 B2
必须用完全相同的训练超参数。若 OOM，先把两者的
`image_max_pixels` 同时调到 602112；不要只调整其中一个实验。

## 12. 测试 B0、B1、B2

一次只启动一个模型，并等待 `[READY]` 后再运行 test matrix。

B0：

```bash
python main_layer/run.py start-eval --settings "$CFG" rs_eot 1 8010
```

```bash
python main_layer/run.py test-matrix \
  --settings "$CFG" \
  rs_eot http://127.0.0.1:8010/v1 fresh
```

```bash
python main_layer/run.py stop-eval --settings "$CFG" rs_eot 8010
```

B1：

```bash
python main_layer/run.py start-eval --settings "$CFG" rs_eot_b1_direct 1 8010
```

```bash
python main_layer/run.py test-matrix \
  --settings "$CFG" \
  rs_eot_b1_direct http://127.0.0.1:8010/v1 fresh
```

```bash
python main_layer/run.py stop-eval --settings "$CFG" rs_eot_b1_direct 8010
```

B2：

```bash
python main_layer/run.py start-eval --settings "$CFG" rs_eot_b2_socratic 1 8010
```

```bash
python main_layer/run.py test-matrix \
  --settings "$CFG" \
  rs_eot_b2_socratic http://127.0.0.1:8010/v1 fresh
```

```bash
python main_layer/run.py stop-eval --settings "$CFG" rs_eot_b2_socratic 8010
```

汇总：

```bash
python main_layer/run.py compare-b0-b1-b2 --settings "$CFG"
```

测试矩阵会尝试：

- DOTA 类别条件 OBB 检测；
- DOTA-Ref 单目标 OBB 定位 K=1；
- DOTA-Ref 分类；
- DOTA-Ref 定位稳定性 K=5。

主指标应报告：

- 定位：mean IoU、Localization Acc@0.5/0.7、Joint class+IoU Acc；
- 检测：precision/recall/F1@IoU 0.5/0.7、每 GT mean IoU、负查询准确率；
- 分类：accuracy、macro-F1、每类 support 和 confusion matrix；
- 协议：格式成功率、截断率、缺行率。

mAP 可以作为类别条件检测的辅助指标，但 Ref 单目标定位的核心应是 IoU 与
Acc@IoU。状态为 `INVALID` 时不要解释 mAP。

## 13. 如何解释三个实验

- B1 > B0：自动 OBB→QA/Direct 监督有效；
- B2 > B1：在相同问题、图像、最终答案、pair ID 与样本数下，Socratic 轨迹有效；
- B2 只提高格式率：轨迹主要教会输出协议，没有改善视觉定位；
- 分类提高但 IoU 不提高：需要更强边界/尺度证据或视觉侧适配；
- 小目标/密集目标仍差：先按目标短边、面积和实例密度分桶分析。

当前 val 只有 8 张切片且类别分布不均，不能仅凭一个总分得出稳定结论。
正式报告至少给出每类 support、seen/unseen class，并在条件允许时做 scene-level
group K-fold 或多个固定 scene split。focus ROI 完全由现有 OBB 自动计算，不增加
人工标注成本，但它属于 GT-derived object pointer，论文和实验说明中必须披露。

## 14. 参考实现边界

本包不修改官方 `SocraticAgent/generation.py`、`postproc.py` 或
LLaMA-Factory；通过本地 adapter、自动 Ref 构造、确定性审计和转换层接入。
Asking Like Socrates 官方仓库：
<https://github.com/GeoX-Lab/Asking_like_Socrates>。

