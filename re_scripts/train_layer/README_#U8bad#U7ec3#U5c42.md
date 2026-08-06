# 训练层

> v4.3.3 主实验以根目录
> `README_v4.3.3_必须先读_debug40与B0B1B2完整流程.md` 为准。B1 现为
> Direct + matched final-only Ref，不再是旧的 Direct-only 主基线。

## B1 Direct

不依赖多智能体轨迹。使用全部有效 OBB，回答格式包含 `FINAL_DETECTIONS ... END_DETECTIONS`。

```bash
python main_layer/run.py prepare-direct-sft --settings settings.json --register
python main_layer/run.py validate-train --settings settings.json --target b1
python main_layer/run.py generate-train-config --settings settings.json
python main_layer/run.py stop-agents --settings settings.json
python main_layer/run.py train-b1 --settings settings.json
python main_layer/run.py merge --settings settings.json b1
```

## B2 Socratic

只有 strict 中 Grounding 和 Classification 都达到门槛后才允许转换、训练。

```bash
python main_layer/run.py validate-train --settings settings.json --target b2
python main_layer/run.py train-b2 --settings settings.json
python main_layer/run.py merge --settings settings.json b2
```

## 注意

- B1/B2 默认都用标准 causal LM loss；不要在首轮实验同时改数据、轨迹和 loss。
- 双 A6000 训练前必须停止 8001/8002/8003 三智能体服务。
- 训练脚本会解析真实 `als_sft` Python，不依赖当前 shell 激活状态。
