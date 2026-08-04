#!/usr/bin/env bash
# v1.2.2 用户配置。只修改这里，不在代码中写死任何示例类别。

# ===== 2A6000 / v1.2.2.2 稳定工程 =====
export REPO_ROOT="/home/yk/Asking_like_Socrates-main"
export BASE_SETTINGS="/home/yk/Asking_like_Socrates-main/re_scripts/settings.scene_disjoint.json"
export CLASSES_FILE="/home/yk/fxy/datasets/yw/classes.txt"

# 中文结果全部进入新目录；英文继续使用 BASE_SETTINGS 里的原目录。
export ZH_OUTPUT_ROOT="/home/yk/fxy/results_ssh/yw_v1.2.2_zh"

# ===== Direct QA 模式 =====
# legacy_class_roi_obb  正式 B0/B1 基线推荐，和原 test-matrix 协议最一致
# all_image_json_obb    全图全部目标，JSON objects 列表，保留旋转 OBB
# all_image_json_hbb    bbox_2d + categories，丢失旋转信息
# single_ref_json_obb   单个指代目标，一次输出完整 OBB
export DIRECT_QA_MODE="legacy_class_roi_obb"

# 非 legacy Direct 模型若继续跑原 test-matrix，属于跨协议迁移评测。
# 明确接受后才改为 1。
export ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX="0"

# ===== Qwen3-Omni AWQ-No-TTS 四卡共享服务 =====
# ModelScope 模型：tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS
export OMNI_MODELSCOPE_ID="tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS"
export OMNI_MODEL_PATH="/home/yk/fxy/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS"
export OMNI_SERVED_NAME="qwen3-omni-30b-a3b-instruct-awq-no-tts"

# 推荐单独环境。脚本不会自动升级稳定的 als_sft / als_vllm 环境。
export OMNI_PYTHON="/home/yk/miniconda3/envs/omni_vllm/bin/python"

# 一个服务进程、一个权重副本、四张卡共同完成每次前向计算。
export OMNI_VISIBLE_GPUS="0,1,2,3"
export OMNI_TP_SIZE="4"
export OMNI_HOST="127.0.0.1"
export OMNI_PORT="8091"
export OMNI_BASE_URL="http://${OMNI_HOST}:${OMNI_PORT}/v1"

# safe: 最稳，eager，低并发；balanced: 推荐；fast: 更高并发，先通过 benchmark 再用。
export OMNI_PROFILE="balanced"
export OMNI_MAX_MODEL_LEN="12288"
export OMNI_GPU_MEMORY_UTILIZATION="0.92"
export OMNI_MAX_NUM_SEQS="4"

# Full 轨迹并发。Debug 在原流程中仍强制 concurrency=1。
export OMNI_FULL_CONCURRENCY="4"

# 只允许图像输入；本任务不需要视频、音频和 TTS。
export OMNI_MM_LIMITS='{"image":2,"video":0,"audio":0}'
export OMNI_ALLOWED_LOCAL_MEDIA_PATH="/home/yk/fxy"

# 0=不强制传 --quantization，交给 config 自动识别；通常最稳。
# 若当前 vLLM 明确要求，可改成 awq。
export OMNI_QUANTIZATION="auto"

# 1=允许 trust_remote_code；本地第三方 checkpoint 通常建议保留。
export OMNI_TRUST_REMOTE_CODE="1"

# 1=要求 preflight 证明 checkpoint 不含 talker/code2wav；无法证明时阻断。
# 若模型 index 本身不记录这些 key，但你已人工确认，可改为 0，仅保留警告。
export OMNI_REQUIRE_NO_TTS="1"

# 服务 PID、日志和 benchmark 报告目录。
export OMNI_RUNTIME_DIR="${ZH_OUTPUT_ROOT}/omni_tp4_service"
