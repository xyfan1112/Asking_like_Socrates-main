# 实验命名

## 主实验 B

- `B00`: RS-EoT 原模型 zero-shot
- `B0`: Qwen2.5-VL zero-shot
- `B1`: Direct OBB SFT
- `B2`: Direct + 通用 Socratic SFT
- `B3`: Direct + OBB 定向 Socratic SFT

核心比较：`B1 vs B3` 判断 OBB 思考轨迹是否有效；`B2 vs B3` 判断专用提示词是否优于通用提示词。

## 消融 A

A1 单次观察；A2 无 teacher forcing；A3 无 DOTA alias；A4 解冻视觉塔；A5 无几何门控；A6 不问 bbox；A7 仅 Grounding；A8 原图像素；A9 `[0,100]`。

所有消融只改变一个因素，其他训练 token、step、基础模型和测试 prompt 保持一致。
