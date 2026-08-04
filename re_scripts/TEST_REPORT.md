# v1.2.0 静态与合成测试报告

测试日期：2026-08-03

测试方法：将 `files/` 覆盖到完整 `re_scripts_ssh1.2.2.2` 副本后进行静态检查和最小合成数据测试。

| 测试 | 结果 |
|---|---|
| 19个补丁Python文件 `py_compile` | PASS |
| 全部Shell脚本 `bash -n` | PASS |
| `main_layer/run.py --help` 包含新命令 | PASS |
| 中文“粗略搜索区域”ROI提取 | PASS |
| 中文OBB坐标意图识别 | PASS |
| `[TASK=ref_classification]`任务识别 | PASS |
| 完全相同中文问题相似度=1.0 | PASS |
| 中文单角点问题拒绝 | PASS |
| Classification候选类别对Perceiver隐藏 | PASS |
| 6个Grounding恢复问题互不重复且用尽后返回None | PASS |
| 旧固定generic fallback字符串已删除 | PASS |
| `Let’s look at the im`不完整响应识别 | PASS |
| “细长”替换为完整二维长短轴描述 | PASS |
| 中文settings保持输入数据和基础模型路径不变 | PASS |
| 中文生成目录全部进入用户指定根目录 | PASS |
| `question_similarity_threshold=0.95`保持 | PASS |
| `max_loop`与三个agent `max_tokens`保持原配置 | PASS |
| Official Parquet携带`lang=zh`、`qa_language=zh` | PASS |
| D1 Direct-only数据准备、验证和YAML生成 | PASS |
| 使用20260803100148失败audit验证Full阻断 | PASS |

未测试：

- YK服务器真实数据全量 `data-full`；
- 三个Qwen3-VL-8B vLLM进程；
- 真实Debug40与Full；
- 双GPU D1训练和模型合并；
- B0/D1真实评测。

上述项目只能在YK服务器现有环境和数据上完成。补丁不会声称这些在线步骤已经通过。
