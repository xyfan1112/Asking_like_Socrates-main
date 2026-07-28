# 数据层

> v4.3.3 权威规则见根目录
> `README_v4.3.3_必须先读_debug40与B0B1B2完整流程.md`；Ref QA 池与 Socratic 子集已分离。

## 目标

把原始 DOTA OBB 转成三类可独立使用的数据：

1. `*_all.jsonl`：全部有效 OBB，供 Direct SFT 和检测评估；
2. `train.jsonl/val.jsonl`：可被自然语言唯一指代的 DOTA-Ref；
3. `agent_inputs/*.jsonl`：分类和 OBB Grounding 的多智能体输入。

## 主流程

```bash
python main_layer/run.py data-full --settings settings.json
```

它依次完成：原始标签审计、无效框 drop、Ref 构建、严格坐标验证、可视化、Agent Inputs、RL 数据和最终门控。

## 必须通过的报告

- `dota128-Ref/validation_report.json`: `passed=true`
- `dota128_pipeline/reports/data_layer_final_report.json`: `passed=true`
- `invalid_grounding_gt=0`
- `invalid_direct=0`

## v4.3 Query 规则

- 禁止 `of its class`；
- storage tank、roundabout、harbor 不使用方向词；
- harbor 描述整个港区，而不是单根 pier；
- 不使用不稳定的同类 nearest anchor；
- classification reference 隐藏 canonical label；
- Grounding 坐标统一为整数 `[0,1000]`。
