# 主控层

主控层只负责：

- 合并 settings 与实验 profile；
- 解析 data/SFT/vLLM/RL 的 Python 环境；
- 调度数据、训练、测试脚本；
- 保存 materialized settings。

科学逻辑不放在主控层。常用入口：

```bash
python main_layer/run.py runtime-check --settings settings.json
python main_layer/run.py data-full --settings settings.json
python main_layer/run.py prepare-direct-sft --settings settings.json --register
python main_layer/run.py train-b1 --settings settings.json
```
