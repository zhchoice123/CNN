# 深度学习与计算机视觉 · 双语学习站

把本目录下 30 本 PDF 教材按主题合并成 16 章，做成一个**离线可用**的 HTML + CSS + JS 学习站。

## 打开方式

- **推荐**：在 `CNN/` 目录运行本地服务器，然后打开 <http://localhost:8765>
  ```bash
  python3 -m http.server 8765 --directory study_site
  ```
- 也可以直接双击 `index.html`。课程数据以 JS 文件内嵌，`file://` 下同样能用；只有代码区的“在浏览器运行”需要联网加载 Pyodide。

## 内容

| 模块 | 每章内容 | 全站合计 |
|---|---|---|
| 学习 Learn | 4 节：讲解（含公式）→ 要点细化 → 代码验证（真实运行输出）→ 出处 | 64 节 |
| 例题 Examples | 3 道，分步展开解析，大多附验证代码 | 48 道 |
| 测验 Quiz | 选择 6、判断 5、填空 4、英文阅读理解 1 篇（3 题，带朗读和中文译文） | 256 题 |
| 词汇 Vocabulary | 14 个核心术语：英文、中文、英文释义、中文释义、例句，配英文/中文/例句三段 mp3；遮挡自测；中英互译选择练习 | 224 个 |
| 闪卡 Flashcards | 词汇卡 + 4 张概念卡；英→中或中→英；Leitner 五盒间隔复习 | 288 张 |

另有：全站搜索、错题本、到期闪卡复习、学习进度（存在浏览器 localStorage）、亮/暗主题、手机布局。

### 新手友好功能

| 功能 | 说明 |
|---|---|
| 🧭 新手入门（`#/start`） | 5 个勾选题自测 → 推荐路线；4 条路线（零基础 / 会编程直奔 CNN / 视觉进阶 / 完整），选定后侧边栏标出步骤序号 |
| 📅 今日任务（首页） | 一键“继续学习”（按所选路线找到下一个未掌握小节）、到期闪卡、错题数、连续学习天数和 14 天热力图 |
| 章节元信息 | 每章显示难度、预计用时、建议先修章节和“学完能做到什么” |
| 入门必读 / 进阶选读 | 每节有标记；勾选“只看入门必读”后进阶小节折叠，从搜索或链接跳转时自动展开 |
| 术语悬停释义 | 正文中的 224 个术语带虚线，悬停/点击显示中英释义、发音和词条链接（每节只标第一次出现） |
| 🆘 求助面板 | 每节：4 种 AI 提示词（大白话讲一遍 / 出题考我 / 检查我的理解 / 写最小示例），一键在 Claude、ChatGPT 打开或复制；另有 YouTube / B 站 / Google / 维基搜索 |
| ✍️ 我的笔记 | 每节一个笔记框（费曼学习法），自动保存；“检查我的理解”会带上你的笔记 |
| 资源页 | 每章 4–7 条精选外部资源：视频、交互演示、文章教材、动手实践（3Blue1Brown、TensorFlow Playground、CNN Explainer、Karpathy、D2L、CS231n 等） |
| 🧰 工具箱（`#/tools`） | 字号调节、番茄钟（右下角悬浮，跨页面不中断）、**导出 Anki**、导出笔记 Markdown、备份/导入进度 JSON、学习数据统计 |

路线、先修关系、必读小节、预计用时和外部资源都写在 `assets/js/resources.js` 里，**直接编辑即可，不需要重新构建**。链接指向外部网站，需要联网。

### 代码库与训练实验室（第三阶段）

| 功能 | 说明 |
|---|---|
| 🔎 代码库（`#/code`） | 把项目里的 D2L、Raschka、Elgendy、NumPy 验证代码和实验室模板切成 700 多个片段并打上中文标签。用中文/英文提问（如“用 PyTorch 在 CIFAR-10 上训练 ResNet，学习率0.1”）会解析出数据集/模型/框架/参数，给出排序结果；可复制、**改用本地数据集后复制**、载入实验室运行；还会预填一个可直接训练的配置 |
| 🧪 训练实验室（`#/lab`） | 在本机用 PyTorch（Apple GPU/MPS）训练。内置 6 种模型、4+2 个数据集、完整超参数表单；**参数对比**模式只改一个参数依次训练；对比曲线 + 效率表（吞吐、达到共同目标准确率所需时间）+ 自动读图要点；也能运行任意脚本并一起对比 |
| 📦 预下载数据集 | 项目根目录 `datasets/`：MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100（torchvision 原生结构，任何代码都可 `download=False` 直接用） |

实验室需要启动本地服务（`bash lab/start.sh`，打开 <http://localhost:8766/#/lab>），首次使用先 `bash lab/setup.sh`。详见 [`../lab/README.md`](../lab/README.md)。代码库本身是静态数据，不启动服务也能搜索；教材代码更新后重建索引：

```bash
python3 tools/build_codelib.py
```

### 代码练习（第四阶段）

`#/practice`：6 篇练习把训练代码拆成固定环节，四级难度——L1 选择、L2 填空（浏览器内判分，离线可用）、L3 写整块、L4 默写完整脚本（在本机运行评测，按步骤告诉你哪里错、第几行、为什么）。答错的题进错题本；章节页底部有对应练习入口；L4 通过后可一键放进实验室完整训练。

| 篇 | 内容 | 数据 |
|---|---|---|
| 01 | 第一个完整训练循环：设备、Dataset、DataLoader、模型、损失与优化器、评估、五步训练、保存 | MNIST |
| 02 | 卷积层与池化：NCHW、通道/填充/步长、输出尺寸公式、参数量、拉平长度 | 随机张量 |
| 03 | LeNet-5：卷积网络结构 + 五步训练再写一遍 | MNIST |
| 04 | 拆开黑盒：requires_grad、backward、手写 step 与 zero_grad，并与 torch.optim.SGD 对照 | 自造数据 |
| 05 | 数据增强、BatchNorm、Dropout、权重衰减、余弦学习率、为什么评估要 model.eval() | CIFAR-10 |
| 06 | 残差块与小 ResNet、检查点保存与断点恢复 | CIFAR-10 |

练习源文件在 `exercises/`（写法见 `exercises/README.md`），构建：`python3 tools/build_exercises.py`；验收：`python3 ../lab/verify_exercises.py`。

### 章节与素材

| 章 | 主题 | 主要素材 |
|---|---|---|
| 01 | 线性代数 | Strang、MML |
| 02 | 概率、统计与信息论 | 《统计学概要》、MacKay、MML、Bishop |
| 03 | 机器学习基础：泛化与评估 | ISLR、Flach、Géron |
| 04 | 线性模型、正则化与梯度下降 | ISLR、Bishop、MML、Géron |
| 05 | 树、集成、SVM、聚类与降维 | ISLR、Géron、Flach、MML |
| 06 | 神经网络与反向传播 | Nielsen、Kelleher、邱锡鹏、Buduma |
| 07 | 训练深度网络 | Buduma、邱锡鹏、Géron、Nielsen |
| 08 | 卷积神经网络与经典架构 | Elgendy、Chollet、邱锡鹏 |
| 09 | 数字图像处理 | Gonzalez & Woods、Torralba |
| 10 | 迁移学习、检测与分割 | Elgendy、《PyTorch 现代计算机视觉》、Pointer |
| 11 | 几何视觉与三维重建 | Hartley & Zisserman、Szeliski、Forsyth & Ponce、Solem |
| 12 | 序列模型与 Transformer | Chollet、Buduma、邱锡鹏、Géron |
| 13 | 生成模型：VAE、GAN、扩散 | Foster、《PyTorch 现代计算机视觉》 |
| 14 | 强化学习与 RLHF | Lambert、Buduma、Géron |
| 15 | 概率机器人 | Thrun, Burgard & Fox |
| 16 | 工程基础：Python、PyTorch、OS、部署 | Lutz、Pointer、Tanenbaum、Zhou 等 |

## 目录结构

```
study_site/
├── index.html               单页应用入口
├── assets/css/style.css     样式（亮/暗主题、响应式）
├── assets/js/app.js         路由、渲染、测验判分、闪卡、进度、音频、路线、术语提示、工具箱
├── assets/js/resources.js   ★ 手工维护：学习路线、各章元信息、入门必读小节、外部资源链接
├── assets/js/lab.js         训练实验室页面（按需加载）
├── assets/js/codelib.js     代码库搜索页面（按需加载）
├── assets/js/practice.js    代码练习页面（按需加载）
├── assets/js/exercises-data.js 构建产物：练习数据（tools/build_exercises.py 生成）
├── exercises/exNN_*.py      ★ 代码练习源文件（带标记的完整答案脚本）
├── assets/js/codelib-data.js 构建产物：代码库索引（tools/build_codelib.py 生成）
├── assets/js/course-data.js 构建产物（不要手改）
├── assets/vendor/           KaTeX（公式）、highlight.js（代码高亮），已本地化
├── content/chXX.py          ★ 章节内容源文件（编辑这里）
├── code/chXX_*.py           ★ 验证代码（构建时真实运行，输出写进页面）
├── audio/                   edge-tts 生成的 mp3（vocab/<章>/、reading/）
└── tools/
    ├── build.py             构建：汇总内容 → 运行代码 → 校验题目 → 生成音频 → 写 course-data.js
    └── gen_audio.py         调用 edge-tts 生成 mp3（与 ~/Documents/EdgeTtsProject 同一方案）
```

## 修改后页面没变化？

用 `python3 -m http.server` 时浏览器会缓存脚本。改完 `app.js` / `style.css` / `resources.js` 后，要么按 `Cmd + Shift + R` 强制刷新，要么把 `index.html` 里 `?v=` 后面的版本号改一下。

## 修改与重新构建

需要带 numpy 和 edge-tts 的 Python（本机为 `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`）：

```bash
cd study_site
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
$PY tools/build.py              # 全量构建（新增词条会自动生成 mp3，已有的跳过）
$PY tools/build.py --no-audio   # 只改了文字时更快
$PY tools/build.py --only ch08  # 只重建某一章
```

构建会：
- 校验题目格式，例如选择题答案索引越界、填空题缺少 `____`、词条重复；
- 执行每一段引用到的代码，任何一段报错都会中止构建。结果按源码哈希缓存在 `code/.outputs.json`；
- 对选择题选项做**确定性洗牌**，避免正确答案总在同一个位置。

`content/chXX.py` 中的正文支持轻量标记：空行分段、`- ` 列表、`1. ` 编号、`> ` 提示框、`|` 表格、`**粗体**`、`` `代码` ``、`$行内公式$`、`$$独立公式$$`。

语音：英文 `en-US-AriaNeural`，中文 `zh-CN-XiaoxiaoNeural`，在 `tools/gen_audio.py` 中修改。
