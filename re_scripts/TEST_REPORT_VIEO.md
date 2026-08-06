# Asking_like_Socrates-main vieo 部署版测试报告

## 构建基线

- 输入归档：`Asking_like_Socrates-mainv1.2.3r3.zip`
- 基础版本：v1.2.3-r3
- 目标项目路径：`/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main`
- 目标数据/模型/结果根：`/home/vieo/vieo/fxy_workspace/fxy`
- 目标环境根：`/home/vieo/anaconda3/envs`

## 适配内容

- 活动配置中的 `/home/yk/...` 路径改为双 `vieo` 工作区路径；
- `als_sft`、`als_vllm` 均使用 `/home/vieo/anaconda3/envs/.../bin/python`；
- 共享 Omni 默认物理 GPU 0、1，TP=2；
- SFT 默认物理 GPU 0、1，world_size=2，gradient accumulation=4，有效全局 batch=8；
- 评估默认物理 GPU 3，single，端口 8010；
- vLLM 0.11.0 参数检查使用 `python -m vllm.entrypoints.cli.main serve --help=all`；
- shared Omni 启动器不再强制 TP4；
- 修复 Bash 动态作用域导致评估模型 key 从 B0/B1 被改写成 B2 的问题；
- 增加 conda-pack 解压/conda-unpack、scene manifest 迁移和目标机只读检查脚本。

## 已执行测试

1. 全项目 Python `compileall`：PASS。
2. 87 个 Shell 脚本 `bash -n`：PASS。
3. `verify_v123r3_install.py` 源码指纹检查：PASS。
4. `materialize_language_settings.py` 临时数据测试：PASS。
   - `eval_topology=single`
   - `eval_gpus=3`
   - `sft_world_size=2`
   - `gradient_accumulation_steps=4`
   - `effective_global_batch=8`
5. `report_training_contract.py` 双卡合同测试：PASS。

## 未在构建沙箱执行

- 目标机 `/home/vieo/...` 真实路径检查；
- `conda-unpack` 真实归档解压；
- 物理 GPU 0、1、3 占用情况；
- Qwen3-Omni AWQ-No-TTS TP2 权重真实加载；
- 图像请求 smoke；
- 实际 SFT 和评估。

这些必须在目标机器按 `README_VIEO_迁移与运行.md` 分阶段验证。

## 科学边界

本部署版没有实现 Qwen3-Omni 作为 B0/B1/B2 微调目标。原 B0/B1/B2 模型注册仍为 RS-EoT 学生模型链路；Omni 仅用于共享 Socratic Agent 服务。

## 回归测试补充

- `pytest -q tests/test_v123_r3_taxonomy.py tests/test_v123_release.py`：8 passed。
- 其余测试（排除需要沙箱未安装 `openai` 的 `test_ssh122_fixed.py`）：19 passed，1 个继承自原包的 `test_v433_quality_gates.py::test_semantic_and_geometry_gates` 断言不一致。
- 该继承不一致未为本次路径/GPU迁移而修改，避免擅自改变 Socratic 语义过滤定义。
