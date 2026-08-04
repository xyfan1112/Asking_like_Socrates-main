# v1.2.0：2A6000 自定义类别中文独立运行修复包

基线：`https://github.com/xyfan1112/Asking_like_Socrates-main/tree/2A6000`

用途：在现有 `re_scripts` 上直接覆盖，继续从任意行数的 `classes.txt` 动态读取中文、英文或混合类别；保留 Canonical OBB、Direct QA、Ref Grounding、Ref Classification、Socratic 轨迹及 GT teacher forcing。

本版本专门修复 2026-08-03 Debug40 中暴露的问题：中文问题被记录为 `lang=en`、256/256 Perceiver 调用缺少真实 focus ROI、`focus_crop_calls=0`、中文坐标问题识别失败、固定 generic fallback 重复到 `MAX_ROUNDS_EXCEEDED`。该次审计中 `length_truncation=0`、`api_errors=0`，因此本版不通过扩大 token 或轮数掩盖解析和路由错误。

## 一、不会改变的参数

本版本明确保持：

```text
question_similarity_threshold = 0.95
max_loop                     = 原稳定配置值（当前为 8）
Reasoner max_tokens          = 原配置值
Perceiver max_tokens         = 原配置值
Verifier max_tokens          = 原配置值
Debug正式门槛               = 不降低
```

只有 API 日志真实出现 `finish_reason=length`，才允许单独评估发生截断的角色，不得一次性放大全部 token。

## 二、为什么 start-agents / stop-agents 不需要 --lang

`start_three_agents.sh` 只读取：

- 模型路径；
- served name；
- host/port；
- GPU；
- `gpu_memory_utilization`；
- `max_model_len`；
- `max_num_seqs`；
- 是否为多模态模型。

它启动的是三个通用 vLLM OpenAI API 服务。模型服务本身没有“中文服务”和“英文服务”之分。

中文或英文由 `official-socratic-full` 发出的每次 API 请求中的 Prompt/Profile 决定。因此：

```bash
python main_layer/run.py start-agents --settings "$ZH_CFG"
python main_layer/run.py stop-agents  --settings "$ZH_CFG"
```

不传 `--lang`。使用中文 settings 的意义只是让 agent 日志和 PID 位于中文 `pipeline_work_root`，不是改变模型语言。

`data-full` 和 `official-socratic-full` 会实际生成自然语言 QA/轨迹，所以它们继续明确传：

```bash
--classes-file "$CLASSES_FILE" --lang zh
```

## 三、目录隔离策略

英文完全不变：

```text
原 settings
原输入路径
原英文输出目录
```

中文新建一个用户指定根目录，例如：

```text
/home/yk/fxy/results_ssh/yw_v1.2.0_zh/
├── config/settings.zh.v1.2.0.json
├── datasets/
│   ├── ref/
│   ├── llamafactory/
│   └── rl/
├── pipeline/
│   ├── agent_inputs/
│   ├── official_socratic/input/
│   ├── official_socratic/raw/
│   └── official_socratic/postproc/
├── training/
├── models/
├── testing/
└── rl/
```

中文 settings 只重定向生成物。以下内容保持原配置：

- `paths.dota128_root` 输入图像和 OBB 标签；
- `project`；
- `runtime`/Conda；
- 原始模型路径；
- GPU与agent配置；
- max loop；
- max tokens；
-训练超参数。

## 四、补丁修改内容

### 1. 动态类别

实际类别只来自：

```text
/home/yk/fxy/datasets/yw/classes.txt
```

文件可以有任意多行。行号从0开始对应标签中的 `class_id`。代码不内置“长安汽车、福特汽车、波音731”等示例，不推断层级，不合并类别，不翻译标准标签。

### 2. 中文 ROI

识别：

```text
粗略搜索区域
搜索区域
目标区域
候选区域
坐标范围
区域
范围
```

中文 Debug 必须达到：

```text
perceiver_focus_roi_missing = 0
classification_focus_roi_missing = 0
focus_crop_missing = 0
focus_crop_calls >= perceiver_calls > 0
```

### 3. 中文坐标意图

识别“坐标、角点、四个角点、顺时针、旋转框、外接矩形、定位框”等。Grounding 禁止逐个询问左上角或右上角，必须一次询问完整 OBB，并要求：

```text
obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]
```

### 4. 中文任务识别

优先识别：

```text
[TASK=ref_classification]
[TASK=ref_grounding_obb]
```

并兼容中文自然语言标记。

### 5. 中文重复问题

使用 Unicode 中文字符和 n-gram，完全相同的中文问题直接得到相似度 `1.0`。阈值仍为 `0.95`。

### 6. 删除无限固定 fallback

不再无限使用：

```text
哪一个直接可见的边界或结构特征最能把精确目标与周围分开？
```

分类和定位分别使用有限、互不重复的问题库。全部用尽后 fail-closed，不再注入同一句问题。

### 7. “细长”改为可计算几何表达

原语义阈值不变，只修改文案：

```text
长轴明显长于短轴且具有完整二维目标轮廓
长轴远长于短轴且具有完整二维目标轮廓
```

Prompt明确说明道路标线、细杆、阴影、裁剪边缘不是车辆目标。

### 8. Classification候选类别对Perceiver隐藏

Reasoner负责规划，Verifier负责检查，最终类别仍由GT控制。Classification Perceiver只看完整图、ROI和类别中性的目标指代，不再看到完整候选类别列表，降低“看见类别词后虚构雷达罩、发射管、车轮”等标签诱导。

## 五、安装

下载ZIP并解压。你可以不运行安装器，直接把：

```text
v1.2.0_2A6000_custom_zh_stable_patch/files/
```

中的全部内容复制到：

```text
/home/yk/Asking_like_Socrates-main/re_scripts/
```

覆盖同名文件并保留新增文件。

终端复制命令：

```bash
cd /ZIP解压后的/v1.2.0_2A6000_custom_zh_stable_patch
cp -a files/. /home/yk/Asking_like_Socrates-main/re_scripts/
```

或：

```bash
bash commands/copy_files_to_re_scripts.sh
```

已追踪文件可在VS Code中 `Discard Changes`。补丁新增的未追踪文件不会被普通Discard删除，回退时需先查看：

```bash
cd /home/yk/Asking_like_Socrates-main
git status
```

## 六、只改四行

编辑：

```text
commands/user_config.sh
```

默认内容：

```bash
export REPO_ROOT="/home/yk/Asking_like_Socrates-main"
export BASE_SETTINGS="/home/yk/Asking_like_Socrates-main/re_scripts/settings.scene_disjoint.json"
export CLASSES_FILE="/home/yk/fxy/datasets/yw/classes.txt"
export ZH_OUTPUT_ROOT="/home/yk/fxy/results_ssh/yw_v1.2.0_zh"
```

`BASE_SETTINGS`不存在时，运行器会尝试 `re_scripts/settings.json`，但最好填写你实际稳定版正式使用的 settings。

## 七、最简单的运行顺序

进入解压后的补丁目录：

```bash
cd /ZIP解压后的/v1.2.0_2A6000_custom_zh_stable_patch
```

### 第一步：复制补丁

```bash
bash commands/copy_files_to_re_scripts.sh
```

### 第二步：中文运行到Debug40并自动停止

```bash
bash commands/run_v1.2.0.sh safe-until-debug
```

顺序是：

```text
创建中文独立settings
→ 检查classes和标签ID
→ 中文data-full
→ 启动三智能体（不传--lang）
→ 中文fresh Debug40
→ 检查真实finish_reason
→ 检查ROI/crop基础设施
→ 检查正式Debug质量门
→ 停止，不自动跑Full
```

任何一步失败都会停止。

### 第三步：同时得到未训练B0指标

```bash
bash commands/run_v1.2.0.sh b0
```

### 第四步：训练Direct基线

```bash
bash commands/run_v1.2.0.sh train-baseline
```

逻辑：

```text
先验证matched B1
├── PASS：训练并合并matched B1
└── FAIL：构建、验证、训练并合并D1 Direct-only
```

D1是独立基线，不得在论文中冒充与B2样本严格匹配的B1。

评测D1：

```bash
bash commands/run_v1.2.0.sh eval-d1
```

### 第五步：只有Debug通过后运行Full

```bash
bash commands/run_v1.2.0.sh full
```

该命令会再次执行两个检查。只要Debug状态、ROI crop、重复问题或正式质量门不合格，就会退出，绝不降低门槛。

## 八、分步命令

```bash
bash commands/run_v1.2.0.sh prepare
bash commands/run_v1.2.0.sh datafull
bash commands/run_v1.2.0.sh agents
bash commands/run_v1.2.0.sh debug40
bash commands/run_v1.2.0.sh inspect-tokens
bash commands/run_v1.2.0.sh check
```

查看英文原流程：

```bash
bash commands/run_v1.2.0.sh en-info
```

英文继续使用原 settings 和原目录，不建立新的英文输出根目录。

## 九、Debug验收顺序

先看基础设施：

```text
settings language = zh
api_errors = 0
perceiver_context_missing = 0
perceiver_focus_roi_missing = 0
classification_focus_roi_missing = 0
focus_crop_missing = 0
focus_crop_calls >= perceiver_calls > 0
```

再看生成质量：

```text
length_truncation
reasoner_format_unrepaired
perceiver_response_unrepaired
max_rounds_exceeded
duplicate_question_trajectories
grounding strict rate
classification strict rate
```

`length_truncation=0`时不修改max_tokens。Debug未PASS时不运行Full。

## 十、边界说明

本补丁在本地完成静态语法、覆盖结构和合成函数测试，但本地没有你的YK数据、Conda环境、GPU进程和Qwen3-VL服务，因此不能声称已在YK上完成真实Debug40或Full。真实验收以你的新中文独立目录中生成的audit和API日志为准。
