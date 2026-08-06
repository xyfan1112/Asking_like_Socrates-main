# v1.2.3-r3 taxonomy 路径根修复

## 现象

报错路径中出现了两行内容：

```text
/home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts/[PASS] scene-disjoint manifest: train=128 val=20 overlap=0
/home/vieo/vieo/fxy_workspace/fxy/results_ssh/.../config/classes.zh.txt
```

这不是 `classes.zh.txt` 缺失。真正的问题是旧 runner 在：

```bash
classes=$(active_classes)
```

中调用了会输出 scene-disjoint 日志的函数。Bash 把日志和路径一起放进变量，Python 随后把整个多行字符串当成文件名。

## 为什么删除 runtime_settings、复制或改名无效

污染在每次执行 runner 时重新发生，早于真正的数据处理。因此删除缓存或把 `classes.zh.txt` 改成 `classes_zh.txt` 都不会消除污染源。

## r3 的根修复

1. data 与 Socratic 阶段不再通过 Bash 捕获类别路径。
2. `main_layer/run.py` 直接从活动 settings 读取 `taxonomy.classes_file`。
3. Python taxonomy 层增加旧污染值恢复：多行值中只有一个真实存在文件时使用该文件并打印警告；歧义时硬停止。
4. prepare 和每个正式阶段运行严格 taxonomy preflight。
5. runner 检查自身目录与 `REPO_ROOT/re_scripts` 必须是同一个目录，避免“新脚本调用旧代码树”。
6. 新增 `doctor` 命令，显示实际运行脚本、代码根、源码指纹和活动类别路径。

## 安装后执行顺序

```bash
cd /home/vieo/vieo/fxy_workspace/Asking_like_Socrates-main/re_scripts
bash commands/run_v1.2.3.sh doctor
bash commands/run_v1.2.3.sh prepare
bash commands/run_v1.2.3.sh data
```

`doctor` 必须显示：

```text
[RUNNER] release=1.2.3-r3
[VERIFY INSTALL][1.2.3-r3] PASS
[TAXONOMY PREFLIGHT][1.2.3-r3] PASS
```

规范文件名保持：

```text
中文：config/classes.zh.txt
英文：config/classes.en.txt
```
