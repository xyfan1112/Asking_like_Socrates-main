# 外部依据（2026-08-04核对）

本包设计依据以下主要资料：

1. ModelScope：`tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS`模型页。
2. Qwen官方`Qwen3-Omni-30B-A3B-Instruct`模型卡：
   - vLLM当前以Thinker文本服务方式在线提供；
   - 官方给出四GPU `-tp 4`启动示例；
   - BF16长视频显存参考；
   - 不需要音频输出时禁用Talker/只返回文本可减少显存并加快响应。
3. vLLM官方Qwen3-Omni示例：Thinker-only推理和Tensor Parallel。
4. vLLM-Omni量化说明：预量化checkpoint从`quantization_config`识别，量化通常只覆盖Thinker语言模型，其他多模态模块可保持高精度。
5. 2A6000分支和ssh1.2.2.2/v4.3.3正式门控流程。

边界：ModelScope模型页和静态config不能证明当前服务器上的vLLM/AWQ kernel兼容。最终必须以实际加载、真实图像smoke和Debug40为准。
