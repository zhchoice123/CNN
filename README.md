# CNN 学习资料目录

给人和 AI 的项目地图（怎么启动、改哪、别改哪、路由、API、未做项）：[`AGENTS.md`](AGENTS.md)。

```
CNN/
├── books/                 教材 PDF，按类别分组；编号 = 推荐阅读顺序，与 audiobooks/tts_scripts 稿子编号一致
│   ├── 1_数学基础/         01–04
│   ├── 2_经典机器学习/     05–09
│   ├── 3_深度学习原理/     10–16, 31
│   ├── 4_计算机视觉/       17–25
│   ├── 5_生成与对齐/       26–27
│   └── 6_机器人与工程/     28–30
├── book_code/             教材配套代码（study_site/tools/build_codelib.py 从这里建代码库索引）
│   ├── d2l_动手学深度学习/
│   ├── raschka_machine-learning-book/
│   ├── elgendy_deep-learning-for-vision-systems/          git 克隆版（含运行输出）
│   └── _archives/d2l-en.zip                               D2L 完整压缩包（含 TensorFlow 等版本）
├── audiobooks/            有声导读：tts_scripts/ 口播稿 + 生成的 mp3
├── notes/                 学习笔记（deepseek_pytorch学习笔记/）
├── study_site/            学习网站
├── lab/                   本机训练实验室
└── datasets/              离线数据集（lab 代码依赖此路径，勿移动）
```

## 重命名对照（原名 → 新位置）

| 原名 | 新位置 |
|---|---|
| `linear-algebra-author-gilbert-strang.pdf` | `books/1_数学基础/01_线性代数导论_Strang.pdf` |
| `mml-book.pdf` | `books/1_数学基础/02_机器学习的数学_MML.pdf` |
| `OReilly.Statistics.in.a.Nutshell.A.Desktop.Quick.Reference.Aug.2008.pdf` | `books/1_数学基础/03_统计学概要_Statistics_in_a_Nutshell.pdf` |
| `Information Theory.pdf` | `books/1_数学基础/04_信息论推断与学习算法_MacKay.pdf` |
| `ISLR_First_Printing.pdf` | `books/2_经典机器学习/05_统计学习导论_ISLR.pdf` |
| `machine-learning-the-art-and-science-of-algorithms-that-make-sense-of-data.pdf` | `books/2_经典机器学习/06_机器学习_算法的艺术与科学_Flach.pdf` |
| `Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` | `books/2_经典机器学习/07_模式识别与机器学习_Bishop_PRML.pdf` |
| `Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf` | `books/2_经典机器学习/08_机器学习实战_Hands-On_ML_Geron.pdf` |
| `DeepLearning_Bonus.pdf` | `books/2_经典机器学习/09_深度学习图解附加章_Scikit-Learn与Keras_Glassner.pdf` |
| `Deep_Learning.pdf` | `books/3_深度学习原理/10_深度学习_MIT精要系列_Kelleher.pdf` |
| `neuralnetworksanddeeplearning.pdf` | `books/3_深度学习原理/11_神经网络与深度学习_Nielsen.pdf` |
| `神经网络与深度学习.pdf` | `books/3_深度学习原理/12_神经网络与深度学习_邱锡鹏.pdf` |
| `Fundamentals of Deep Learning.pdf` | `books/3_深度学习原理/13_深度学习基础_Fundamentals_of_DL_Buduma.pdf` |
| `Deep Learning with Python.pdf` | `books/3_深度学习原理/14_Python深度学习_Chollet.pdf` |
| `Deep Learning A Practitioner’s Approach.pdf` | `books/3_深度学习原理/15_深度学习实践者之道_Patterson_Gibson.pdf` |
| `programming-pytorch-for-deep-learning-creating-and-deploying-deep-learning-applications.pdf` | `books/3_深度学习原理/16_PyTorch深度学习编程_Pointer.pdf` |
| `deepSeekLearn/Deep_Learning_with_PyTorch.pdf` | `books/3_深度学习原理/31_PyTorch深度学习实战_Stevens_Antiga.pdf` |
| `Gonzales,Woods-Digital.Image.Processing.4th.Edition.pdf` | `books/4_计算机视觉/17_数字图像处理_Gonzalez_Woods.pdf` |
| `Computer Vision - A Modern Approach - D. Forsyth, J. Ponce.pdf` | `books/4_计算机视觉/18_计算机视觉_一种现代方法_Forsyth_Ponce.pdf` |
| `Computer Vision_ Algorithms and Applications - Szeliski.pdf` | `books/4_计算机视觉/19_计算机视觉_算法与应用_Szeliski.pdf` |
| `Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf` | `books/4_计算机视觉/20_计算机视觉中的多视图几何_Hartley_Zisserman.pdf` |
| `Foundations of Computer Vision - Antonio Torralba.pdf` | `books/4_计算机视觉/21_计算机视觉基础_Torralba_Isola_Freeman.pdf` |
| `ProgrammingComputerVision_CCdraft.pdf` | `books/4_计算机视觉/22_Python计算机视觉编程_Solem.pdf` |
| `Deep Learning for Vision Systems -- Mohamed Elgendy.pdf` | `books/4_计算机视觉/23_视觉系统深度学习_Elgendy.pdf` |
| `Modern Computer Vision with PyTorch.pdf` | `books/4_计算机视觉/24_PyTorch现代计算机视觉_第二版_Ayyadevara_Reddy.pdf` |
| `Deep Learning for Medical Image Analysis.pdf` | `books/4_计算机视觉/25_医学图像分析中的深度学习_Zhou_Greenspan_Shen.pdf` |
| `Generative Deep Lear... by David Foster (z-lib.org).pdf` | `books/5_生成与对齐/26_生成式深度学习_Foster.pdf` |
| `Reinforcement Learning from Human Feedback.pdf` | `books/5_生成与对齐/27_基于人类反馈的强化学习_RLHF_Lambert.pdf` |
| `ProbabilisticRobotics.pdf` | `books/6_机器人与工程/28_概率机器人_Thrun_Burgard_Fox.pdf` |
| `Modern Operating Systems.pdf` | `books/6_机器人与工程/29_现代操作系统_Tanenbaum.pdf` |
| `Learning_Python.pdf` | `books/6_机器人与工程/30_Python学习手册_Lutz.pdf` |
| `Chapter 01 - A Review of Machine Learning - Full.mp3` | `audiobooks/15_深度学习实践者之道_Patterson_Gibson_第01章_机器学习回顾.mp3` |
| `tts_scripts` | `audiobooks/tts_scripts` |
| `d2l-en` | `book_code/d2l_动手学深度学习` |
| `machine-learning-book-main` | `book_code/raschka_machine-learning-book` |
| `deep_learning_for_vision_systems` | `book_code/elgendy_deep-learning-for-vision-systems` |
| `深度学习计算机视觉源代码和附录` | 已删除（移至废纸篓，2026-10-09） |
| `d2l-en.zip` | `book_code/_archives/d2l-en.zip` |
| `deepSeekLearn` | `notes/deepseek_pytorch学习笔记` |
| `netty` | 已删除（移至废纸篓，2026-10-09） |
