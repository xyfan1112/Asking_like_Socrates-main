# re_scripts v1.2.3 修改清单

## 基线与边界

- 唯一代码基线：用户上传的 `re_scripts1.2.2(2).zip`。
- 稳定整合基础：`ssh1.2.2.2`。
- 内部数据流水线：`4.3.3`。
- 未用远程 `yk` 分支覆盖本地 ZIP。
- 未修改 `question_similarity_threshold`、Socratic `max_loop`、三个角色 token 上限、Debug 阈值、LoRA rank/alpha/dropout/epoch/learning rate、Direct 默认协议和视觉塔冻结策略。

## 配置与双语

- 新增 `commands/run_v1.2.3.sh`，统一 B0/B1/B2、Socratic、服务和诊断命令。
- 重写 `commands/user_config.sh`：默认 `RUN_LANG=en`，一个开关联动数据、类别、prompt、轨迹、训练、评估和报告路径。
- 不再依赖缺失的 `settings.scene_disjoint.json`；以 `settings.json` 为基线，通过 `DATASET_VARIANT` 选择 `yw128` 或 `yw128_scene_disjoint`。
- 自动读取真实 `classes.txt`，不写死 16 类。
- 英文首次 `prepare` 自动创建 `class_map.zh_en.json` 与待翻译清单；类别 ID、标注数字和 OBB 不变。中文军事类别名称不自动猜测。
- 增加 `custom_obb_bilingual_v1_2_3` prompt profile 别名；实际科学 prompt 内容沿用 v1.2.2，避免无意改变实验变量。

## 四卡评估

- `01_raw_dota_obb_infer.py`、`03_ref_grounding_infer.py`、`10_ref_classification_infer.py` 增加 SHA256 稳定分片参数。
- 新增 3 个 `*_replica4.sh` 协议脚本：GPU 0–3 分别启动单卡 7B 副本，端口 8010–8013。
- 新增 `merge_jsonl_shards.py`：检查分片数量、重复 ID、期望行数，按 `(query_id/id, run_id)` 确定性合并。
- 新增 `run_test_matrix_replica4.sh`，B0/B1/B2 的每个协议均四卡并行，不把可单卡容纳的 7B 模型强行切成 TP4。

## 四卡 LoRA

- 将 SFT 从 2 卡 DDP 改为 4 卡 DDP。
- 自动保持有效全局 batch：`2×1×4=8` 改为 `4×1×2=8`。
- 训练前输出 LoRA/冻结/批量合同报告。
- 训练日志输出到固定目录并提取实际 trainable/all params、OOM、NCCL 和完成标志。
- 训练后直接检查 adapter safetensors 键；要求冻结视觉塔时，若发现明显视觉编码器 LoRA 张量则停止。
- LLaMA-Factory 优先通过准确解释器执行 `python -m llamafactory.cli`，避免迁移后 `llamafactory-cli` shebang 仍指向旧机器路径。

## Socratic 拓扑

- 默认 `SOCRATIC_TOPOLOGY=shared`：一个 8091 Qwen3-Omni 服务、TP4、一个权重副本、三个逻辑角色和独立 messages。
- 保留 `independent`：Reasoner/Perceiver/Verifier 分别为 8001/8002/8003，可指定三个不同模型。
- 新增统一命令 `socratic-debug`、`socratic-full`、`safe-b2`、`safe-all`，由 `SOCRATIC_TOPOLOGY` 自动选择。
- `fast` 默认仍限制为并发 4、显存利用率 0.92；不暗中提高到并发 8。
- Full 处理全部输入，Debug 才抽 40 条；Full 前继续强制通过 Debug gate。

## 实验完整性与分析产物

- 新增 B1/B2 Direct 前缀审计：B2 必须完整、同序包含 B1 使用的每一条 Direct 数据，并且必须追加至少一条严格通过的 Socratic 轨迹。
- 新增 `pipeline_summary_v1_2_3.json`，汇总 Direct、Agent input、B1/B2、Socratic funnel 和拒绝原因；缺失阶段记为 `null`，不伪装成 0。
- 新增 B2 模型注册检查。
- 新增 `status` 与按阶段 `collect` 诊断包；默认不包含模型/LoRA 权重和图片，限制日志大小并脱敏常见密钥。

## 环境迁移

- 增加 `无网络迁移_als_sft_als_vllm_操作手册.md`。
- 增加 `tools/check_offline_env.sh`。
- 明确使用 `conda-pack`/`conda-unpack`，并用 `python -m pip` 修复本地 editable LLaMA-Factory；禁止直接复制环境目录后视为完成。
