# v1.2.2 静态与合成测试报告

## 1. 测试范围

本报告验证补丁本身的可复制性、Python/Shell语法、累计中文修复、Direct四种模式、中文独立settings、共享Qwen3-Omni三逻辑角色配置、单服务TP=4约束、No-TTS静态检查和端点测试脚本。

本地容器没有YK服务器的四张A6000、目标AWQ checkpoint、Omni专用vLLM环境、真实数据和RS-EoT训练环境，因此没有声称完成真实模型加载、在线Debug40、Full、B0/B1训练或test-matrix。

## 2. 已通过检查

| 检查 | 结果 |
|---|---|
| 补丁内全部Python文件`py_compile` | PASS |
| 补丁内全部Shell脚本`bash -n` | PASS |
| 补丁覆盖到干净`re_scripts`副本后Python语法 | PASS |
| `question_similarity_threshold`固定为0.95 | PASS |
| 不自动增加`max_loop` | PASS |
| 不自动增加三个角色`max_tokens` | PASS |
| 中文粗略搜索区域ROI提取 | PASS |
| 中文OBB/角点坐标意图识别 | PASS |
| 完全相同中文问题相似度=1.0 | PASS |
| 固定generic boundary无限fallback已删除 | PASS |
| 中文独立输出settings生成 | PASS |
| 英文原settings和路径不改写 | PASS |
| `legacy_class_roi_obb` | PASS（合成构建） |
| `all_image_json_obb` | PASS（合成构建） |
| `all_image_json_hbb` | PASS（合成构建） |
| `single_ref_json_obb` | PASS（合成构建） |
| B1-Direct-Standalone训练前合约脚本语法 | PASS |
| 三个逻辑Agent指向同一Omni端点 | PASS |
| 物理模型副本数约束为1 | PASS |
| Tensor Parallel约束为4 | PASS |
| Full并发写入共享Omni settings | PASS |
| TP4服务start/stop/status/log脚本语法 | PASS |
| ModelScope checkpoint ID写入配置 | PASS |
| 伪造No-TTS权重索引不含Talker时通过 | PASS |
| 伪造权重索引出现Talker键时可检测 | PASS |
| 文本＋真实图片＋Verifier端点smoke脚本语法 | PASS |
| 并发1/2/4与四卡利用率benchmark脚本语法 | PASS |
| Full仍依赖fresh Debug门控 | PASS |

## 3. 中文关键回归结果

使用Adapter工具函数进行离线回归：

```text
输入：粗略搜索区域 [458,469,538,600]
输出ROI：(458.0, 469.0, 538.0, 600.0)

输入：请提供精确目标的四个顺时针OBB角点坐标。
坐标类型：obb

相同中文问题重复检测：
duplicate=True, similarity=1.0
```

测试导入时使用了本地`openai`接口占位模块，因为当前容器安装的`openai`版本与YK目标环境不一致；这不替代YK环境中的真实API smoke。

## 4. YK服务器必须完成的真实门控

依次执行：

```bash
bash commands/run_v1.2.2.sh omni-preflight
bash commands/run_v1.2.2.sh omni-start
bash commands/run_v1.2.2.sh omni-smoke
bash commands/run_v1.2.2.sh omni-benchmark
bash commands/run_v1.2.2.sh omni-shared-debug
```

只有以下条件全部满足，才能认定该第三方AWQ-No-TTS checkpoint适用于正式轨迹生成：

```text
四张A6000均可见
TP=4服务启动成功
/models返回指定served name
Reasoner文本请求成功
Perceiver真实图片请求成功
Verifier文本请求成功
finish_reason不是length
四卡均有显存占用且benchmark期间参与计算
中文ROI/crop基础设施门通过
fresh Debug40正式质量门通过
```

## 5. 未验证事项

- 未在四张A6000 48GB上实装该checkpoint；
- 未验证目标AWQ格式与YK实际vLLM版本的kernel/shape兼容性；
- 未证明TP=4一定比TP=2或单卡具有更低单请求延迟；
- 未完成真实并发吞吐测试；
- 未运行真实Socratic Debug40或Full；
- 未运行B0、B1-Direct训练、合并和test-matrix；
- 未对该第三方AWQ checkpoint进行微调或权重合并。

这些结果必须由YK服务器运行日志和审计报告确认，不能由静态检查替代。
