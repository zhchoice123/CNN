# CNN 项目上下文（给人和 AI 用）

本仓库是一套**离线深度学习 / 计算机视觉学习资料 + 双语学习站 + 本机 PyTorch 训练实验室**。根目录 `README.md` 是书单目录；日常开发先读本文件。

工作目录默认是仓库根：`/Users/zhcho/Desktop/zhcho/CNN`。不要移动 `datasets/`。

---

## 先打开什么

| 目的 | 命令 | 地址 |
|---|---|---|
| 日常推荐：学习站 + 实验室（同一端口） | `bash lab/start.sh`（Windows：`lab\start_windows.bat`） | http://localhost:8766 |
| 只看静态学习站 | `python3 -m http.server 8765 --directory study_site` | http://localhost:8765 |
| 首次装 PyTorch | `python3 lab/setup.py`（Windows NVIDIA：`lab\setup_windows.bat`） | — |

Claude Desktop 可用 `.claude/launch.json`：`cnn-lab` → 8766，`study-site` → 8765。

环境变量：`LAB_PORT`（默认 8766）、`LAB_DATA`（默认 `<仓库根>/datasets`）。

机器：日常是 macOS Apple 芯片（MPS）；另一台目标机是 Windows RTX 3060（CUDA）。Windows 实机验收还没勾完，见 `lab/ACCEPTANCE.md`。

---

## 目录：改哪里 / 别改哪里

```
CNN/
├── AGENTS.md              ← 本文件
├── README.md              书单目录 + 重命名对照
├── books/                 30+ PDF，编号 = 阅读顺序 = tts 稿编号
├── book_code/             教材配套代码（代码库索引来源）
├── audiobooks/            tts_scripts/ 口播稿 + mp3
├── notes/                 学习笔记
├── datasets/              离线数据（体积大，不入库；lab 硬依赖此路径）
├── study_site/            学习站（静态 SPA，hash 路由）
└── lab/                   本机训练 + 练习评测服务器
```

### 学习站 `study_site/`

| 改这些源文件 | 不要手改（构建产物） |
|---|---|
| `content/chXX.py` 章节正文 | `assets/js/course-data.js` |
| `code/chXX_*.py` 验证脚本 | `code/.outputs.json`（运行缓存） |
| `exercises/exNN_*.py` 练习答案 | `assets/js/exercises-data.js` |
| `assets/js/resources.js` 路线 / 先修 / 外部资源（改完**不用**重建） | `assets/js/codelib-data.js` |
| `assets/js/app.js` `lab.js` `codelib.js` `practice.js` `style.css` `index.html` | `audio/` 下已有 mp3（增量生成会跳过） |

构建 Python：文档里写的是 `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`（numpy + edge-tts）。

```bash
cd study_site
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
$PY tools/build.py              # 全量：跑代码 + 校验题目 + 缺的 mp3
$PY tools/build.py --no-audio   # 只改文字
$PY tools/build.py --only ch08  # 单章
python3 tools/build_exercises.py
python3 tools/build_codelib.py  # 扫 book_code/ + 实验室模板 + 练习答案
python3 tools/gen_audio.py      # 语音：en-US-AriaNeural / zh-CN-XiaoxiaoNeural
```

改完 JS/CSS 若页面没变：`Cmd+Shift+R`，或改 `index.html` 里 `?v=`。`file://` 能看课，代码「在浏览器运行」要联网加载 Pyodide。

正文轻量标记：空行分段、`- ` / `1. ` 列表、`> ` 提示、`|` 表格、`**粗体**`、`` `代码` ``、`$` / `$$` 公式。

### 实验室 `lab/`

| 文件 | 作用 |
|---|---|
| `server.py` | 标准库 HTTP：静态站 + `/api`；训练在 `.venv` 子进程 |
| `train.py` | 统一训练（数据常驻设备、逐轮写指标） |
| `lab_schema.py` | **超参数唯一来源**；网页表单由 `/api/schema` 生成 |
| `lab_models.py` | `mlp` / `lenet` / `simple_cnn` / `vgg_mini` / `resnet_mini` / `resnet18` |
| `datasets_manager.py` | 检查 / 下载 / 离线加载 |
| `labkit.py` | 自定义脚本：`DATA_ROOT` `get_device` `load_tensors` `log_epoch` |
| `templates/` | 可读完整训练循环 |
| `platform_utils.py` | venv 路径、进程组停止、UTF-8、硬件信息 |
| `exercise_core.py` / `ex_runner.py` | 练习标记解析 + 临时目录评测 |
| `verify_exercises.py` | 答案全过 + 错误样例能检出 + 单次 ≤ 30s |
| `runs/` `.venv/` | 不入库 |

安全：只听 `127.0.0.1`；校验 Host / Origin；写操作必须带 `X-Lab`。自定义脚本等于在本机跑任意 Python。

macOS 官方 Python 缺根证书时用 `certifi`，**不要关 HTTPS 校验**。

---

## 页面路由（hash）

| 路由 | 功能 |
|---|---|
| `#/` | 首页：今日任务、路线进度 |
| `#/start` | 新手自测 → 4 条路线 |
| `#/ch/<id>/{learn,examples,quiz,vocab,cards,res}` | 章内标签 |
| `#/practice` `#/practice/<exId>` | 代码练习 L1–L4 |
| `#/code?q=` | 代码库搜索 |
| `#/lab` | 训练实验室 |
| `#/tools` | 字号、番茄钟、Anki/笔记导出、进度备份 |
| `#/review` `#/mistakes` `#/search/...` | 闪卡到期、错题本、搜索 |

进度、笔记、闪卡在 **localStorage**。学习路线等元数据在 `study_site/assets/js/resources.js`。

章 01–16：线代 → 概率信息论 → ML 评估 → 线性模型 → 树/SVM/聚类 → 神经网络 → 训练技巧 → **CNN（核心）** → 图像处理 → 检测分割 → 多视图几何 → 序列/Transformer → 生成模型 → RL/RLHF → 概率机器人 → 工程。

---

## 实验室 API

静态资源：`/` → `study_site/`。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/health` | 健康检查、硬件、PyTorch |
| GET | `/api/schema` | 超参数字段 |
| GET | `/api/templates` | 模板列表 |
| GET | `/api/datasets` | 本地数据集状态 |
| GET | `/api/runs` `/api/runs/<id>` | 运行列表 / 详情 |
| GET | `/api/runs/export` `/api/compare` | 导出、对比 |
| POST | `/api/runs` | 开训（内置 `train.py`） |
| POST | `/api/script` | 跑自定义脚本 |
| POST | `/api/runs/import` | 导入另一台机器的 json |
| POST | `/api/runs/<id>/stop` | 停止 |
| POST | `/api/datasets/<name>/download` | 下载 |
| POST | `/api/exercise/check` | L3/L4 评测 |
| DELETE | `/api/runs/<id>` | 删除 |

同时只跑一个训练任务（共用 GPU，并行会让计时不准）。效率：稳态吞吐排除第 1 轮；「达到共同目标准确率所需时间」才是公平收敛比较；计时不含测试评估。

`labkit` 最小用法：

```python
from labkit import DATA_ROOT, get_device, load_tensors, log_epoch
device = get_device()
xtr, ytr, xte, yte = load_tensors("cifar10", device)
log_epoch(epoch, train_loss=..., train_acc=..., test_loss=..., test_acc=..., epoch_time=...)
```

教材里 `download=True` 的 torchvision 代码：代码库页「复制（改用本地数据集）」会改成 `DATA_ROOT` + `download=False`。D2L 多数依赖 `d2l` 库；Keras/TF 本机未装，多数只能读。

数据集：预装 MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100；KMNIST / SVHN 按需。

```bash
lab/.venv/bin/python -I lab/datasets_manager.py status
lab/.venv/bin/python -I lab/datasets_manager.py download svhn
```

---

## 代码练习

6 篇：`ex01` MNIST MLP → `ex02` 卷积形状 → `ex03` LeNet → `ex04` autograd → `ex05` CIFAR CNN 技巧 → `ex06` ResNet 检查点。

写法：`study_site/exercises/README.md`。源文件是**能直接跑的完整答案** + 注释标记；构建器生成 L1 选择 / L2 填空 / L3 写整块 / L4 默写。

硬规则：

1. `# @step` 必须顶格；类/循环不能跨步骤（评测按步骤标题切开再注入 `@check` / `@fast`）。
2. `@fast` 后不能再有代码。
3. 检查写行为（形状、损失下降、准确率阈值），不要比对字符串。阈值给 MPS/CUDA 数值差留余量。

```bash
python3 lab/verify_exercises.py
```

---

## 书与代码索引

- PDF：`books/`，编号 01–30（另有 31《PyTorch 深度学习实战》）。原名对照见根 `README.md`。
- 口播稿：`audiobooks/tts_scripts/`（纯文本、无 Markdown；编号与书一致；**不含 31**）。约定见该目录 README。
- 配套代码：`book_code/d2l_动手学深度学习`、`raschka_machine-learning-book`、`elgendy_deep-learning-for-vision-systems`、`_archives/d2l-en.zip`。

---

## 现状与未做

已完成：16 章课程（64 节 / 48 例题 / 256 测验题 / 224 术语 / 288 闪卡 / 73 段验证代码 / 688 mp3）；新手路线与工具箱；代码库约 700+ 片段；实验室；6 篇练习；Mac M3 Pro 验收通过。

未做 / 勿主动开干除非被要求：

- 站内交互演示（卷积核滑动、梯度下降轨迹）
- 测验错题关联回对应小节
- 资源链接定期自动检查
- Keras/TF 可运行；d2l 库兼容模式
- 迁移学习预训练权重；训完模型保存与预测演示
- Windows RTX 3060 按 `lab/ACCEPTANCE.md` 实机勾选

改学习站 UI 时：走真实点击流程，不要只看一帧截图；进度存在 localStorage，改读写逻辑要核对首页 / 错题本 / 闪卡 / 工具箱备份。

---

## 更细的文档

| 文件 | 内容 |
|---|---|
| `README.md` | 书单树 + 重命名对照 |
| `study_site/README.md` | 学习站功能与构建 |
| `study_site/PLAN.md` | 分阶段计划与数量统计 |
| `study_site/exercises/README.md` | 练习标记语法 |
| `lab/README.md` | 实验室用法 |
| `lab/ACCEPTANCE.md` | 两台机器验收清单 |
| `datasets/README.md` | torchvision 目录结构 |
| `audiobooks/tts_scripts/README.md` | TTS 稿约定 |
