#!/usr/bin/env bash
# 只修改下面路径/模式。classes.txt 中有多少行就有多少类，代码不写死示例类别。
export REPO_ROOT="/home/yk/Asking_like_Socrates-main"
export BASE_SETTINGS="/home/yk/Asking_like_Socrates-main/re_scripts/settings.scene_disjoint.json"
export CLASSES_FILE="/home/yk/fxy/datasets/yw/classes.txt"
export ZH_OUTPUT_ROOT="/home/yk/fxy/results_ssh/yw_v1.2.1_zh"

# Direct QA模式：
#   legacy_class_roi_obb  推荐，和原test-matrix协议最一致
#   all_image_json_obb    全图全部目标，JSON对象列表，保留OBB
#   all_image_json_hbb    用户示例形式，bbox_2d+categories，丢失旋转信息
#   single_ref_json_obb   单个指代目标，一次返回完整OBB
export DIRECT_QA_MODE="legacy_class_roi_obb"

# 非legacy Direct模型仍可训练；若要继续用原test-matrix评测这种模型，必须显式设为1。
# 该评测衡量迁移能力，但不能当作和legacy Direct完全同协议的严格消融。
export ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX="0"

# Qwen3-Omni共享服务配置。Direct-only路线不使用这些变量。
# 模型可以是官方BF16路径，也可以是第三方AWQ路径；AWQ必须先通过preflight+smoke。
export OMNI_MODEL_PATH=""
export OMNI_SERVED_NAME="qwen3-omni-30b-a3b-instruct"
export OMNI_BASE_URL="http://127.0.0.1:8091/v1"
# vllm_thinker：普通vLLM，Thinker文本输出，推荐用于图像+文本轨迹，TP=4。
# vllm_omni：vLLM-Omni多阶段服务，需要请求modalities=["text"]。
export OMNI_SERVER_MODE="vllm_thinker"
# 单独Omni环境的Python；不存在时仅preflight退回当前python，不会自动安装包。
export OMNI_PYTHON="/home/yk/miniconda3/envs/omni_vllm/bin/python"
