# Qwen2.5 与 Qwen3 环境说明

不要为了 Qwen3 直接升级已经跑通的 Qwen2.5 环境。推荐独立配置：

- `als_sft` / `als_vllm`: Qwen2.5-VL 与 RS-EoT；
- `als_qwen3_sft` / `als_qwen3_vllm`: Qwen3-VL。

`runtime-check` 会分别检查模型架构、Transformers、vLLM 和 LLaMA-Factory 路径。缺少 Qwen3 环境不会阻塞 Qwen2.5 的数据、SFT和测试。
