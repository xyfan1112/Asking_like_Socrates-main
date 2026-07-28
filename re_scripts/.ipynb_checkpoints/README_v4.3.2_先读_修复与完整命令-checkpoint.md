# re_scripts v4.3.2：DOTA128 OBB→视觉问答→Socratic→匹配 SFT→IoU/分类评测

这是本包的权威入口。旧的 v4.3.1 文档只用于追溯，不再定义 B1/B2 和评测协议。

## 1. 结论与任务边界

目标没有变成新的语义任务。原始监督始终是同一份 DOTA128 OBB：

- 分类：目标属于哪个 DOTA 类别；
- 定位：预测该目标或该类别目标的 OBB；
- 自动 QA 化：把现有类别、OBB、相对位置自动写成问题和答案；
- Socratic 化：为一部分可靠 QA 自动生成多轮视觉证据轨迹；
- SFT：用 LLaMA-Factory 训练同一个基础视觉语言模型；
- 评测：以 OBB IoU 和分类正确率为主，mAP 只保留为检测辅助指标。

Reasoner、Perceiver、Verifier 是生成训练轨迹的三个角色，不是 B0/B1/B2 三个被测模型。主实验中应固定这三个角色及其配置。

## 2. 对现有 v4.3.1 产物的审计结论

| 问题 | 附件中的证据 | 影响 | v4.3.2 修复 |
|---|---:|---|---|
| B0 输出协议失败 | 120 行均收到响应，0 API 错误；6 行截断；严格重解析只有 7 行是合法空框，0 个合法预测框 | 原来的 10 个预测来自宽松解析器抓取推理文本中的前 8 个数字，不可作为框 | 正式评测启用严格解析；格式错误一律 0 预测、IoU=0 |
| 原始与有效 GT 混用 | val 原始 133 个框，其中 17 个越界；训练/评测可用 GT 应为 116 | 原 mAP 的分母和 SFT 使用的框集合不一致 | 检测评测直接读取 `val_all.jsonl` 的规范化有效 GT |
| Ref 覆盖较低 | train 1937 个有效框→342 Ref；val 116→19，约 16%–18% | 不能把 Ref 数量当成完整检测框数量 | `*_all.jsonl` 保存所有有效框；Ref QA 池与 Socratic 子集分开 |
| 指代表达语法错误 | 342/342 train Ref、19/19 val Ref 含 `area area` | 问句质量差 | 修复区域短语拼接并增加验证门控 |
| Direct 分块语义矛盾 | train 190 条 Direct；152 条是重复问题的额外分块；同一问题可重复 19 次但答案不同 | 模型面对同一输入学到互相冲突的标签 | 对密集类别递归划分互斥 ROI，问题明确“中心落在 ROI 内” |
| 没有负样本 | train/val Direct 负样本均为 0 | 模型只学会输出，未学会正确返回空检测 | 每图自动抽 2 个缺失类别作为空块监督 |
| 场景泄漏 | 原场景 `P1571` 同时进入 train 与 val | 切片外观泄漏，指标偏高 | 新增按原场景重划分工具；验证器默认硬拦截 |
| Socratic 全部被拒 | 52/52 被拒：missing thinking 28、少于要求轮数 16、Verifier 连接错误 16、格式错误 5、超轮数 3；原因有重叠 | 主要不是“模型没学过 DOTA 知识” | 三服务执行真实 chat 预检；标签规范化；修复常见 XML 变体；调整过严门控 |
| B1/B2 不可归因 | 旧 B1=Direct，旧 B2=Direct+额外 Ref/Socratic，行数与问题不同 | B2 的变化可能来自更多 QA，而非 Socratic 轨迹 | 由同一批通过门控的 pair ID 同时构造匹配 B1/B2 |
| 图像分辨率过低 | 原 `image_max_pixels=262144`，2048 图像约缩到 512 边；很多目标短边只有约 9–17 原始像素 | 微小目标进一步缩成 2–4 像素 | 默认提高为 802816，并把 `cutoff_len` 提到 8192 |
| 测试矩阵中途退出 | DOTA 指标返回非零后 `set -e` 直接结束 | Ref 定位/分类没有执行 | 四个阶段都会尝试，最后统一报告失败阶段 |

额外限制：原 train 有效框中没有 `basketball court`，而 val 有该类。验证器会将其标为 `val_only_classes`；它在当前划分中属于零样本类别，不能把该类的差表现归因于 Socratic。

## 3. v4.3.2 的 DOTA128-Ref 规则

### 3.1 三种产物

1. `<split>_all.jsonl`
   - 每个合法 OBB 一行；
   - 用于 Direct 检测、规范 GT、血缘审计；
   - 不要求能自动生成唯一自然语言指代。

2. `<split>.jsonl`
   - 完整 Ref QA 池；
   - 必须有自动唯一、非序数的 grounding reference；
   - classification 使用由 GT 自动扩张得到的粗 focus HBB 指向目标，但不泄露类别；
   - 用于 Ref 定位/分类评测。

3. `<split>_socratic.jsonl`
   - Ref QA 池的保守子集；
   - 只用于生成冷启动长轨迹，不用于缩小评测集。

### 3.2 Socratic 子集门控

默认必须同时满足：

- OBB 四点有限、非退化、位于图内；
- 短边至少为图像短边的 `8/1000`；
- 面积比例至少 `0.00005`；
- 指代表达不仅是弱区域描述，还应满足以下之一：
  - 图中该类只有一个实例；
  - 有稳定的 leftmost/rightmost/topmost/bottommost 特征；
  - 有稳定的不同类别锚点。

小目标或弱指代不会从 Ref QA 池消失，只是不进入 Socratic 轨迹生成。

### 3.3 分类 QA

问题包含自动生成的粗 focus region，例如：

```text
The target is centered inside the coarse focus region [x1,y1,x2,y2] in normalized [0,1000] coordinates ...
Which canonical DOTA category is it?
```

focus region 是对象指针，不是分类答案，也不是人工新标注。它解决了“road-side target”之类描述在密集小目标场景中无法唯一指定对象的问题。

## 4. 可归因的 B0/B1/B2

| 实验 | 训练数据 | 作用 |
|---|---|---|
| B0 | 原 RS-EoT，不做 DOTA SFT | 零样本起点 |
| B1 | 相同 Direct ROI 数据 + 通过严格门控的 Ref pair 的 final-only 版本 | 测试“自动 QA/OBB 监督”本身 |
| B2 | 与 B1 完全相同的 Direct 数据 + 同一批 Ref pair 的 Socratic 轨迹版本 | 在相同问题、图像、最终 GT、pair ID、样本行数下测试 Socratic 轨迹 |

`train_layer/01_validate_training_data.py` 会验证：

- B1/B2 总行数相同；
- Direct 行逐行相同；
- Ref pair ID 集合相同；
- 每一对的图像、用户问题、最终 GT 相同；
- 唯一允许变化的是 assistant 中 final 前的推理轨迹。

配置中的模型 key `rs_eot_b1_direct` 和 `rs_eot_b2_socratic` 为兼容旧目录保留；v4.3.2 中 B1 的准确含义是“Direct + matched final-only Ref”。

## 5. 指标

### 5.1 类别条件 OBB 检测

主指标：

- `mean_iou_per_gt`；
- `precision_iou_0_5`、`recall_iou_0_5`、`f1_iou_0_5`；
- 对应 IoU 0.7 指标；
- `negative_query_accuracy`；
- `output_protocol_success_rate`。

辅助指标：

- `map50`、`map50_95`。

密集类别会被划成最多 12 个对象的互斥 ROI。每个答案仍使用原图全局坐标。

### 5.2 Ref OBB 定位

- `mean_iou_all_runs`；
- `localization_acc_iou_0_5`、`localization_acc_iou_0_7`；
- `joint_class_iou_acc_0_5/0_7`；
- `class_accuracy`；
- unique/nonunique 分组；
- K=5 时的 Avg/Conv/Pass@K 稳定性。

### 5.3 Ref 分类

- `accuracy`；
- `macro_f1_present_classes`；
- 每类 precision/recall/F1；
- confusion matrix；
- unique/nonunique 分组。

状态含义：

- `VALID`：行数完整，无 API/截断/格式失败；
- `DEGRADED`：行数完整，但部分行失败；指标仍按失败行=错误计算；
- `INVALID`：应有结果行缺失，不能比较。

## 6. 从零运行命令

以下命令假定仓库在 `/home/nhl/Asking_like_Socrates`，数据和结果仍位于 `/home/nhl/fxy`。

### 6.1 安装包并检查配置

先备份旧脚本，再解压本包，使目录为：

```text
/home/nhl/Asking_like_Socrates/re_scripts
```

然后：

```bash
cd /home/nhl/Asking_like_Socrates/re_scripts
conda activate als_sft
python main_layer/run.py preflight --settings settings.json
```

确认以下模型目录真实存在：

```text
/home/nhl/fxy/models/Qwen2.5-7B-Instruct-AWQ
/home/nhl/fxy/models/Qwen2.5-VL-7B-Instruct-AWQ
/home/nhl/fxy/models/Qwen2.5-3B-Instruct-AWQ
/home/nhl/fxy/models/RS-EoT-7B
```

### 6.2 生成无场景泄漏的数据副本

```bash
python main_layer/run.py split-scenes \
  --settings settings.json \
  --output-root /home/nhl/fxy/datasets/dota128_scene_disjoint \
  --settings-output /home/nhl/Asking_like_Socrates-main/re_scripts/settings.scene_disjoint.json
```

此命令优先创建硬链接，失败才复制；不会删除或修改原始 DOTA128。

后续都使用：

```bash
CFG=/home/nhl/Asking_like_Socrates-main/re_scripts/settings.scene_disjoint.json
```

### 6.3 完整数据层

```bash
python main_layer/run.py data-full --settings "$CFG"
```

必须为 PASS：

```text
/home/nhl/fxy/datasets/dota128-Ref/validation_report.json
/home/nhl/fxy/results/dota128_pipeline/reports/lineage_audit.json
/home/nhl/fxy/results/dota128_pipeline/reports/data_layer_final_report.json
```

重点检查：

- `scene_leakage` 为 0；
- `direct_missing_objects` 为 0；
- `direct_duplicate_objects` 为 0；
- `agent_max_obb_iou_error` 接近 0；
- `train_socratic.jsonl` 非空；
- `val_only_classes` 只作为已知限制出现。

### 6.4 启动三个本地智能体

```bash
python main_layer/run.py start-agents --settings "$CFG"
```

新版本不仅检查 `/models`，还会对三个角色分别发送真实 chat completion。任一角色失败都不得继续生成。

### 6.5 先跑 20 条 debug

```bash
python main_layer/run.py official-socratic-full \
  --settings "$CFG" \
  train debug 20 fresh
```

继续全量前要求：

- `api_errors=0`；
- `length_truncation=0`；
- `unrepaired=0`；
- 两类任务都有 strict candidate；
- grounding strict candidate 比例最好不低于 30%；
- classification 最好不低于 50%。

debug 报告位于：

```text
/home/nhl/fxy/results/dota128_pipeline/official_socratic/raw/*debug*.audit.json
```

### 6.6 全量 fresh 生成并构造匹配 B1/B2

```bash
python main_layer/run.py official-socratic-full \
  --settings "$CFG" \
  train full 20 fresh
```

必须检查：

```text
/home/nhl/fxy/results/dota128_pipeline/official_socratic/raw/dota128_train_official_strict.report.json
/home/nhl/fxy/datasets/dota128-llamafactory/dota128_train_official_llamafactory_report.json
/home/nhl/fxy/results/dota128_training/validation_report_all.json
```

关键字段：

- grounding/classification 接受数各不少于配置的 20；
- `matched_row_count_equal=true`；
- `matched_comparison.issues=[]`；
- `sft_matched_pair_issues=0`。

旧 raw 结果不能 resume；本次必须使用 `fresh`，因为 Ref schema、ID、prompt 和门控都已改变。

### 6.7 训练 B1 与 B2

```bash
python main_layer/run.py stop-agents --settings "$CFG"
python main_layer/run.py train-b1 --settings "$CFG"
python main_layer/run.py train-b2 --settings "$CFG"
python main_layer/run.py merge --settings "$CFG" both
```

若旧 adapter 输出目录已存在，不要删除；先移动为带时间戳的备份，再训练。

默认适配双 A6000：

- 2 卡；
- 每卡 batch 1；
- gradient accumulation 4；
- LoRA rank 16；
- 视觉塔冻结、投影层可训练；
- 802816 image pixels；
- cutoff 8192。

如果 OOM，按顺序调整：

1. `image_max_pixels: 602112`；
2. `cutoff_len: 6144`；
3. 保持 batch=1，不先降低图像到原来的 262144；
4. 记录变更，并让 B1/B2 使用完全相同配置。

### 6.8 分别测试 B0、B1、B2

B0：

```bash
python main_layer/run.py start-eval --settings "$CFG" rs_eot 1 8010
python main_layer/run.py test-matrix --settings "$CFG" rs_eot http://127.0.0.1:8010/v1 fresh
python main_layer/run.py stop-eval --settings "$CFG" rs_eot 8010
```

B1：

```bash
python main_layer/run.py start-eval --settings "$CFG" rs_eot_b1_direct 1 8010
python main_layer/run.py test-matrix --settings "$CFG" rs_eot_b1_direct http://127.0.0.1:8010/v1 fresh
python main_layer/run.py stop-eval --settings "$CFG" rs_eot_b1_direct 8010
```

B2：

```bash
python main_layer/run.py start-eval --settings "$CFG" rs_eot_b2_socratic 1 8010
python main_layer/run.py test-matrix --settings "$CFG" rs_eot_b2_socratic http://127.0.0.1:8010/v1 fresh
python main_layer/run.py stop-eval --settings "$CFG" rs_eot_b2_socratic 8010
```

每次必须先停掉旧服务，再启动下一个模型；同一端口不能同时服务两个模型。

### 6.9 汇总三组结果

```bash
python main_layer/run.py compare-b0-b1-b2 --settings "$CFG"
```

输出：

```text
/home/nhl/fxy/results/dota128_testing/comparison/b0_b1_b2/comparison.json
/home/nhl/fxy/results/dota128_testing/comparison/b0_b1_b2/comparison.csv
/home/nhl/fxy/results/dota128_testing/comparison/b0_b1_b2/comparison.md
```

## 7. 如何解释实验

- B1 > B0：证明自动 OBB→QA/Direct 的监督有效；
- B2 > B1：在相同 QA、最终答案和样本数下，证明 Socratic 轨迹带来增益；
- B2 仅格式率提高、IoU 不提高：轨迹主要教会输出协议，没有改善几何定位；
- 分类提高、定位不提高：需要更多高分辨率定位监督或视觉侧适配；
- nonunique 明显低于 unique：指代表达/密集目标消歧仍是瓶颈；
- 小类波动大：DOTA128 的 val 很小，应报告每类 support，不能只报一个总均值。

不要以“训练 loss 降低”代替任务指标，也不要把 Socratic 接受率当成模型最终定位性能。

## 8. 仍然存在的科学限制

- DOTA128 规模很小，val 仅约 8 张切片；单次划分方差很大；
- 某些类别只在 val 出现，属于零样本评测；
- 粗 focus box 来源于 GT，适合研究自动 QA 冷启动，但必须在论文中明确；
- OBB 坐标离散到 0–1000，微小目标会有量化误差；
- 当前 Ref 表达是规则自动生成，不能替代人工语言多样性评估；
- 若要发表更稳定结论，建议在 scene-disjoint 前提下增加 3–5 个固定种子的场景级划分或交叉验证。

## 9. 对照的官方实现

- Asking Like Socrates：<https://github.com/GeoX-Lab/Asking_like_Socrates>
- VRSBench：<https://github.com/lx709/VRSBench>
- LLaMA-Factory 多模态数据契约：<https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md>
- DOTA devkit：<https://github.com/CAPTAIN-WHU/DOTA_devkit>

本包不修改官方 `SocraticAgent/generation.py` 和 `postproc.py`，只通过本地 adapter、数据门控和 LLaMA-Factory 转换层接入。
