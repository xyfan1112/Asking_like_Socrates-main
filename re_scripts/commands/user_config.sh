#!/usr/bin/env bash
# v1.2.3 用户配置：通常只修改本文件。
# shellcheck shell=bash

# -----------------------------------------------------------------------------
# 0. 学生模型路由：独立 Qwen3-Omni 键，不覆盖原 RS-EoT 注册
# -----------------------------------------------------------------------------
# qwen3_omni：safe-direct 使用 qwen3_omni_b0 / qwen3_omni_b1_direct。
# rs_eot：恢复使用原 rs_eot / rs_eot_b1_direct 流程。
export STUDENT_MODEL_FAMILY="qwen3_omni"       # qwen3_omni | rs_eot
export STUDENT_BASE_MODEL_PATH="/home/vieo/vieo/fxy_workspace/fxy/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS"
export STUDENT_MODEL_TEMPLATE="qwen3_omni"
export STUDENT_B1_ADAPTER_OUTPUT="/home/vieo/vieo/fxy_workspace/fxy/models/adapters/qwen3_omni_b1_direct"
export STUDENT_B2_ADAPTER_OUTPUT="/home/vieo/vieo/fxy_workspace/fxy/models/adapters/qwen3_omni_b2_socratic"
export STUDENT_ADAPTER_ONLY="1"
export STUDENT_FREEZE_VISION_TOWER="1"
export STUDENT_FREEZE_PROJECTOR="1"
# LLaMA-Factory 的稳定通用值；冻结开关负责排除视觉塔和 projector。
export STUDENT_LORA_TARGET="all"

# -----------------------------------------------------------------------------
# 1. 工程与语言
# -----------------------------------------------------------------------------
export REPO_ROOT="/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main"
export BASE_SETTINGS="${REPO_ROOT}/re_scripts/settings.json"

# 默认英文。改成 zh 后，数据、Direct、Socratic、训练、评估和报告会整体切换。
export RUN_LANG="zh"                    # en | zh

# 中文沿用原中文结果根；v1.2.3 在其下使用独立版本子目录，避免覆盖 v1.2.2。
export ZH_OUTPUT_ROOT="/home/vieo/vieo/fxy_workspace/fxy/results_ssh/yw_v1.2.2_zh/v1.2.3"
# 英文是新生成/翻译后的独立结果。
export EN_OUTPUT_ROOT="/home/vieo/vieo/fxy_workspace/fxy/results_ssh/yw_v1.2.3_en"

# -----------------------------------------------------------------------------
# 2. 数据集与 classes.txt
# -----------------------------------------------------------------------------
export RAW_DATASET_ROOT="/home/vieo/vieo/fxy_workspace/fxy/datasets/yw128"
export SCENE_DISJOINT_DATASET_ROOT="/home/vieo/vieo/fxy_workspace/fxy/datasets/yw128_scene_disjoint"
# 正式实验默认使用无场景泄漏版本。仅调试原始数据时改 raw。
export DATASET_VARIANT="scene_disjoint" # scene_disjoint | raw
# 已复制无泄漏数据时默认只校验；缺失时允许从yw128重新生成。
export AUTO_BUILD_SCENE_DISJOINT="1"

# 留空时自动读取 RAW_DATASET_ROOT/classes.txt；不需要手工抄写 16 类。
export CLASSES_FILE=""
# 英文类别映射。首次英文 prepare 会按真实 classes.txt 自动生成模板；
# 对中文类别的英文名称必须人工确认，脚本不会擅自猜测车辆类别。
export CLASS_MAP_FILE="${EN_OUTPUT_ROOT}/config/class_map.zh_en.json"

# -----------------------------------------------------------------------------
# 3. Direct QA
# -----------------------------------------------------------------------------
# legacy_class_roi_obb  正式 B0/B1/B2 对比默认；与现有 test-matrix 协议一致。
# all_image_json_obb    全图 JSON OBB；实验模式。
# all_image_json_hbb    全图 HBB，会丢失旋转信息；实验模式。
# single_ref_json_obb   单目标 Ref OBB；实验模式。
export DIRECT_QA_MODE="legacy_class_roi_obb"
export ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX="0"

# -----------------------------------------------------------------------------
# 4. vieo评估：默认物理GPU 3单卡；避免占用0、1上的共享Omni/训练
# -----------------------------------------------------------------------------
export EVAL_TOPOLOGY="replica"           # 按 EVAL_GPUS 数量启动通用多副本评估
# export EVAL_TOPOLOGY="replica4"
export EVAL_GPUS="1,2,3"
export EVAL_BASE_PORT="8010"            # single模式只使用8010
export EVAL_GPU_MEMORY_UTILIZATION="0.80"
export EVAL_MAX_MODEL_LEN="4096"
export EVAL_SERVER_WAIT_SECONDS="420"

# -----------------------------------------------------------------------------
# 5. vieo双卡LoRA SFT：物理GPU 0、1，保持有效全局batch=8
# -----------------------------------------------------------------------------
export SFT_VISIBLE_GPUS="1,2,3"
export SFT_WORLD_SIZE="3"
# auto 会保持基础配置的有效全局batch；当前2卡会保持accum=4。
export SFT_GRADIENT_ACCUMULATION="3"
# 默认必须冻结视觉塔；误解冻时训练前硬停止。
export REQUIRE_FROZEN_VISION_TOWER="1"

# 可直接覆盖损坏/迁移后的环境。留空则使用 settings.json 的 workload 解析。
export DATA_PYTHON_OVERRIDE="/home/vieo/anaconda3/envs/als_sft/bin/python"
export SFT_PYTHON_OVERRIDE="/home/vieo/anaconda3/envs/als_sft/bin/python"
export VLLM_PYTHON_OVERRIDE="/home/vieo/anaconda3/envs/omni_vllm/bin/python"

# -----------------------------------------------------------------------------
# 6. Socratic 拓扑
# -----------------------------------------------------------------------------
# shared：默认。一个 8091 Omni 物理端点，Reasoner/Perceiver/Verifier 三个独立消息上下文。
# independent：原三服务 8001/8002/8003，可分别指定不同模型。
export SOCRATIC_TOPOLOGY="shared"       # shared | independent
export SOCRATIC_DEBUG_SAMPLES="40"
export SOCRATIC_FULL_TASK_SAMPLES="999999999"  # Full 实际按 settings.current_task_samples 跑全部输入

# 原三服务模型；默认三个角色使用同一个 Qwen3-VL-8B-Instruct。
export REASONER_MODEL_PATH="/home/vieo/vieo/fxy_workspace/fxy/models/Qwen3-VL-8B-Instruct"
export PERCEIVER_MODEL_PATH="/home/vieo/vieo/fxy_workspace/fxy/models/Qwen3-VL-8B-Instruct"
export VERIFIER_MODEL_PATH="/home/vieo/vieo/fxy_workspace/fxy/models/Qwen3-VL-8B-Instruct"

# -----------------------------------------------------------------------------
# 7. 共享 Qwen3-Omni 服务：vieo默认物理GPU 0、1，TP2
# -----------------------------------------------------------------------------
export OMNI_MODELSCOPE_ID="tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS"
export OMNI_MODEL_PATH="/home/vieo/vieo/fxy_workspace/fxy/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS"
# 留空时自动由模型 ID 推导。
export OMNI_SERVED_NAME=""
export OMNI_PYTHON="/home/vieo/anaconda3/envs/als_vllm/bin/python"
export OMNI_VISIBLE_GPUS="1,2"
export OMNI_TP_SIZE="2"
export OMNI_HOST="127.0.0.1"
export OMNI_PORT="8091"
export OMNI_BASE_URL="http://${OMNI_HOST}:${OMNI_PORT}/v1"
export OMNI_PROFILE="safe"              # vieo临时机先以TP2保守启动；稳定后再改balanced/fast
export OMNI_MAX_MODEL_LEN="4096"
export OMNI_GPU_MEMORY_UTILIZATION="0.80"
export OMNI_MAX_NUM_SEQS="1"
# 临时机默认单并发。真实加载和图像smoke通过后再手动提高。
export OMNI_FAST_MAX_NUM_SEQS="1"
export OMNI_FAST_GPU_MEMORY_UTILIZATION="0.80"
export OMNI_FULL_CONCURRENCY="1"
export OMNI_MM_LIMITS='{"image":2,"video":0,"audio":0}'
export OMNI_ALLOWED_LOCAL_MEDIA_PATH="/home/vieo/vieo/fxy_workspace/fxy"
export OMNI_QUANTIZATION="auto"
export OMNI_TRUST_REMOTE_CODE="1"
export OMNI_REQUIRE_NO_TTS="1"

# -----------------------------------------------------------------------------
# 8. 诊断包
# -----------------------------------------------------------------------------
export SUPPORT_BUNDLE_MAX_LOG_MB="20"
export SUPPORT_BUNDLE_INCLUDE_SAMPLE_ROWS="20"
