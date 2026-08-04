# Qwen3-Omni AWQ-No-TTS：部署与微调边界

## 轨迹生成

目标checkpoint：

```text
tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS
```

推荐部署：一个普通vLLM Thinker/text服务，`tensor_parallel_size=4`。Reasoner、Perceiver、Verifier只是三种Prompt角色，共用一个端点，不加载三份权重。

No-TTS适合当前任务，因为轨迹只需要图像/文本输入和文本输出，不需要Talker、code2wav或音频生成。

## AWQ兼容性

checkpoint名称带AWQ不等于当前vLLM一定能加载。兼容性取决于：

```text
config.json中的quantization_config
权重打包格式
vLLM/compressed-tensors版本
A6000可用AWQ kernel
模型是否完整删除TTS权重
```

必须通过：

```text
preflight → 服务启动 → /v1/models → 文本smoke → 图像smoke → Debug40
```

## 四卡速度

TP=4使每个请求都跨四张卡，并增加可用KV Cache和批处理空间。它不保证单请求线性四倍加速；PCIe拓扑上的跨卡通信可能成为瓶颈。

正式选择`safe/balanced/fast`前，必须运行：

```bash
bash commands/run_v1.2.2.sh omni-benchmark
```

## 微调

本包把该AWQ-No-TTS权重作为推理和轨迹教师，不自动对它训练。

推荐：

```text
官方未量化Qwen3-Omni基座
→ LoRA/QLoRA
→ 保存Adapter
→ 可选合并BF16/FP16
→ 重新AWQ量化
→ 独立测试
```

不建议直接把未知第三方AWQ当作全参数训练基座，也不建议未经验证地把Adapter直接合并进该AWQ目录。
