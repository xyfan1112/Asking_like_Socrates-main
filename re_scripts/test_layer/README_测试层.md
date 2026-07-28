# 测试层

> v4.3.3 正式评测与操作顺序见根目录
> `README_v4.3.3_必须先读_debug40与B0B1B2完整流程.md`；旧的宽松解析结果不可比较。

## DOTA Direct 检测

主协议是“图片 + 指定类别 → 该类别所有 OBB”，与 Direct 训练一致。报告 parse success、class-aware mAP50、mAP50:95 和每类 AP。

## Ref Grounding

主结果使用低温 K=1，报告 class accuracy、mIoU、IoU@0.5、IoU@0.7。K=5 仅作为稳定性扩展，报告 Avg/Conv/Pass@5。

## VRSBench VQA

继续使用 Avg/Conv/Pass@5，但不能与 Grounding IoU 混成同一指标。

## 无效运行

API 中断、输出截断或实际运行数不足时必须标记 `INVALID/INCOMPLETE`，不能当成模型 mAP=0。
