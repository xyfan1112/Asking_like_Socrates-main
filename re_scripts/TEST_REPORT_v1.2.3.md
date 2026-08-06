# re_scripts v1.2.3 测试报告

测试日期：2026-08-04

## 已通过

1. 全部 106 个 Python 源文件完成 `compileall` 语法编译。
2. 全部 44 个 Shell 脚本通过 `bash -n`。
3. v1.2.3 新增回归测试：`4 passed`。
   - 英文首次生成类别映射并 fail-closed；补全映射后正确生成配置。
   - 2 卡 `accum=4` 自动换算为 4 卡 `accum=2`，有效全局 batch 保持 8。
   - 四卡分片结果确定性合并、数量检查和排序通过。
   - B2 完整保留 B1 Direct 前缀的审计通过。
   - v1.2.3 prompt profile 别名存在。
4. 共享 Omni 配置合成测试通过：一个物理端点、TP4、三个逻辑角色。
5. 原三服务配置合成测试通过：8001/8002/8003 和三独立模型路径。
6. Adapter scope 合成测试通过：语言/merger adapter 接受，视觉编码器 adapter 在冻结要求下被拒绝。
7. 原有测试离线执行结果：21 项通过，1 项失败。

## 原有 1 项失败的性质

失败项是 `tests/test_v433_quality_gates.py::test_semantic_and_geometry_gates`，其测试文本为：

```text
It seems to fit the description of a storage tank.
```

而现有 v1.2.2/4.3.3 生产正则只把 `is/appears to be/looks like/seems to be/identified as/classified as` 识别为强类别断言，因此该句没有触发测试预期。经 SHA256 对比，生产文件 `data_layer/trajectory_gates.py` 和该测试文件在用户上传的 v1.2.2 ZIP 与 v1.2.3 中完全相同；这是继承的测试—实现不一致，不是本次四卡、双语或训练改造引入的回归。

本次没有为了“让测试变绿”而改动 Socratic 语义门，因为那会改变轨迹过滤数量和实验变量。后续若要修复，应单独作为数据门版本升级并重新跑 Debug/Full 对照。

## 未能在当前沙箱完成

- 未加载用户服务器上的 RS-EoT、Qwen3-VL 或 Qwen3-Omni 权重。
- 未在 4×RTX A6000 上执行真实 vLLM、LoRA、NCCL、B0/B1/B2 全流程。
- 未访问用户真实的 16 类 `classes.txt` 和 `yw128_scene_disjoint` 数据。
- 未验证用户目标机 `als_sft`/`als_vllm` 环境，因为图 1 显示 `als_sft` 当前缺少 `llamafactory`。

因此本报告证明的是静态正确性、配置契约和合成小样本行为；服务器真实模型运行仍必须依次通过 `prepare`、环境自检、B0 小规模评估和 Debug40 gate。
