# v4.3 运行手册：Direct B1 与 Socratic B2

## 两条路线互不阻塞

- **B1 Direct**：直接使用所有有效 DOTA OBB，先得到可用定位基线；不依赖三智能体。
- **B2 Socratic**：重建自然 Ref，重新生成干净的多智能体轨迹，再与 Direct 混合训练。

建议先完成 B1，再修正并生成 B2。

---

## A. Direct B1：立即可执行

### A1. 准备并注册 Direct 数据

```bash
conda activate als_sft
cd /home/yk/Asking_like_Socrates/re_scripts
python main_layer/run.py prepare-direct-sft --settings settings.json --register
```

作用：把数据层已生成的 `train_direct.json/val_direct.json` 复制成 LLaMA-Factory 数据，并写入 `dataset_info.json`。不读取 raw/strict/merge。

### A2. 验证 B1 数据

```bash
python main_layer/run.py validate-train --settings settings.json --target b1
```

必须显示 `PASS`。它只检查 Direct，不会因为 B2 Socratic 不存在而失败。

### A3. 生成训练配置

```bash
python main_layer/run.py generate-train-config --settings settings.json
```

检查报告里的：

```text
b1_dataset = dota128_direct_train_official
eval_dataset = dota128_direct_val_official
```

### A4. 关闭三智能体释放双卡

```bash
python main_layer/run.py stop-agents --settings settings.json
nvidia-smi
```

作用：B1 使用两张 A6000；8001/8002/8003 不关闭会占用约 50GB 显存，训练脚本现在会主动拒绝启动。

### A5. 训练 B1

```bash
python main_layer/run.py train-b1 --settings settings.json
```

### A6. 合并 B1 LoRA

```bash
python main_layer/run.py merge --settings settings.json b1
```

---

## B. Socratic B2：必须重新生成

### B1. 重建 v4.3 Ref 和 agent inputs

```bash
conda activate als_sft
python main_layer/run.py data-full --settings settings.json
python main_layer/run.py build-official-parquet --settings settings.json --split train
```

作用：使用新的自然 Query 和新的样本 ID。旧 v4.2 raw 不能复用。

### B2. 人工抽查 Ref 预览

查看：

```text
/home/yk/fxy/results/dota128_pipeline/ref_previews
```

重点：

- 不应出现 `of its class`；
- storage tank/roundabout/harbor 不应带方向词；
- harbor 应描述整个港区，不是单根 pier；
- classification reference 不应直接出现 canonical label；
- 同一 reference 必须唯一。

### B3. 启动智能体

```bash
conda activate als_vllm
python main_layer/run.py start-agents --settings settings.json
python data_layer/agents/check_three_agents.py --settings settings.json
```

### B4. 运行干净的 debug 20

```bash
bash data_layer/official_socratic/02_run_official_generation.sh \
  settings.json train debug 20 fresh
```

Debug 自动使用 `concurrency=1`，便于阅读，不会与 canonical full raw 混合。

通过标准：

- coordinate mutation = 0；
- token length truncation = 0；
- API error = 0；
- unrepaired format = 0；
- Grounding strict candidates 建议 >= 30%；
- Classification strict candidates 建议 >= 50%。

### B5. 全量 fresh 生成

```bash
bash data_layer/official_socratic/02_run_official_generation.sh \
  settings.json train full 0 fresh
```

`fresh` 会归档旧 canonical raw、strict、merge 和旧 Socratic/mixed 数据，但保留 Parquet 和 Direct 数据。

以后同一签名中断后续跑可使用：

```bash
bash data_layer/official_socratic/02_run_official_generation.sh \
  settings.json train full 0 auto
```

只有签名完全相同才允许 resume。

### B6. 严格过滤和官方 postproc

```bash
bash data_layer/official_socratic/03_run_official_postproc.sh settings.json train
```

默认要求 strict 中至少：

- Grounding 20 条；
- Classification 20 条。

不足时停止，不会再生成“只有17条分类”的伪 B2 数据。

### B7. 转换并注册 B2 数据

```bash
python data_layer/official_socratic/04_convert_official_to_llamafactory.py \
  --settings settings.json --split train --register
```

### B8. 验证和训练 B2

```bash
python main_layer/run.py validate-train --settings settings.json --target b2
python main_layer/run.py generate-train-config --settings settings.json
python main_layer/run.py stop-agents --settings settings.json
python main_layer/run.py train-b2 --settings settings.json
python main_layer/run.py merge --settings settings.json b2
```

---

## C. 审计已有 raw / strict / merge

```bash
python tools/audit_socratic_artifacts.py \
  --raw /home/yk/fxy/results/dota128_pipeline/official_socratic/raw/dota128_train_official.jsonl \
  --strict /home/yk/fxy/results/dota128_pipeline/official_socratic/raw/dota128_train_official_strict.jsonl \
  --merge /home/yk/fxy/results/dota128_pipeline/official_socratic/postproc/dota128_train_official_merge.json
```

作用：统计 Grounding/Classification 成功率、坐标制是否被重写、旧式 Query 数量、strict/merge 的任务构成，以及明显矛盾的分类轨迹。只有 Grounding 和 Classification 都达到门槛时才允许作为 B2 数据。

## D. 常见状态解释

- `generation command completed`：只表示程序完成尝试并写出结果，不代表轨迹质量通过。
- `success`：官方循环/Verifier 的结果；v4.3 后官方精确坐标判断仅作为参考，最终由语义门控和 Python 粗几何门控决定。
- `strict`：真正允许进入 teacher forcing 和 postproc 的轨迹。
- `merge`：官方 postproc 产物；如果全部是 Classification，就不是定位增强数据。
- Debug 与 full 使用独立 manifest；Debug 不会刷新 full 的 resume 签名。

## E. 从旧版本迁移 settings

不要直接用新模板覆盖模型路径和 GPU 配置。部署新包后执行：

```bash
python tools/migrate_settings_to_v43.py \
  --old /tmp/settings_old.json \
  --template settings.example.json \
  --output settings.json
```

它保留旧路径、模型和智能体配置，同时补齐 v4.3 新字段，并强制 `obb_grounding_v3`、`max_loop>=8`、格式修复和 B2 最小样本门槛。
