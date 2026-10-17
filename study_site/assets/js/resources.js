/* 手工维护的学习辅助数据（不是构建产物，可直接编辑）
 * - chapters[id].level   难度 1–5
 * - chapters[id].hours   预计学完（讲解+例题+测验+词汇）所需小时
 * - chapters[id].prereq  先修章节 id（不是硬性要求，卡住时回看）
 * - chapters[id].goal    学完你能做什么
 * - chapters[id].core    “入门必读”小节序号（从 0 开始），其余为“进阶选读”
 * - chapters[id].links   精选外部资源：type = video | interactive | read | practice；lang = zh | en
 * 链接以官方站点/权威课程为主；个别站点改版时请直接修改本文件。 */
window.RESOURCES = {
  tracks: [
    {
      id: "zero", name: "零基础入门", icon: "🌱", weeks: "6–8 周",
      who: "几乎没学过机器学习，数学和编程都只有高中/大一基础。",
      goal: "理解“模型是怎么从数据里学出来的”，能读懂并改写一个简单的 CNN 训练脚本。",
      chapters: ["ch03", "ch01", "ch04", "ch02", "ch06", "ch07", "ch08"],
      tip: "先读第 3 章建立整体印象，再回头补数学：数学是为理解模型服务的，不要在第 1 章停留太久。每章只学“入门必读”小节。",
    },
    {
      id: "coder", name: "会编程 · 直奔 CNN", icon: "⚡", weeks: "3–4 周",
      who: "会写 Python，想最快搞懂卷积神经网络和图像任务。",
      goal: "能用 PyTorch 训练图像分类模型，并理解迁移学习、检测与分割的基本思路。",
      chapters: ["ch03", "ch04", "ch06", "ch07", "ch08", "ch10"],
      tip: "遇到不懂的数学符号，再回查第 1、2 章对应小节；代码区请务必改参数跑几遍。",
    },
    {
      id: "vision", name: "计算机视觉进阶", icon: "👁️", weeks: "6–8 周",
      who: "已经学过 CNN，想系统学习传统图像处理、几何视觉和生成模型。",
      goal: "掌握图像处理、三维几何、Transformer 与生成模型的核心原理。",
      chapters: ["ch09", "ch10", "ch11", "ch12", "ch13"],
      tip: "第 11 章数学最重，建议配合 OpenCV 实操；第 12、13 章可以与论文/开源代码对照着读。",
    },
    {
      id: "full", name: "完整路线", icon: "🧭", weeks: "12–16 周",
      who: "想把 16 章全部学一遍的系统学习者。",
      goal: "从线性代数到 RLHF 与概率机器人，完整建立知识体系。",
      chapters: ["ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09", "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "ch16"],
      tip: "按章节顺序走即可；进阶选读小节可以在第二遍时再学。",
    },
  ],

  methods: [
    { t: "主动回忆", d: "合上页面，先试着自己说出这一节的 3 个要点，再对照。想起来比重读有效得多。" },
    { t: "费曼讲解", d: "每节末尾的“我的笔记”里，用大白话写给“完全不懂的朋友”。写不出来，就是还没懂。" },
    { t: "动手改代码", d: "代码区可直接在浏览器运行。改学习率、卷积核、网络层数，观察输出怎么变。" },
    { t: "间隔重复", d: "每天花 10 分钟复习到期闪卡；也可以一键导出到 Anki，在手机上随时刷。" },
    { t: "先粗后细", d: "第一遍只读“入门必读”小节和例题，建立全局印象；第二遍再读进阶小节。" },
    { t: "错题复盘", d: "测验答错的题会进入错题本。每周清一次，并回到对应小节找原因。" },
  ],

  tools: [
    { name: "Google Colab", url: "https://colab.research.google.com/", note: "免费 GPU 的在线 Notebook，跑 PyTorch 训练最方便。" },
    { name: "Kaggle Learn", url: "https://www.kaggle.com/learn", note: "短小的互动式微课程 + 免费 GPU 额度 + 数据集。" },
    { name: "Hugging Face", url: "https://huggingface.co/", note: "预训练模型、数据集和课程的集散地。" },
    { name: "Anki", url: "https://apps.ankiweb.net/", note: "间隔重复软件。本站可一键导出闪卡到 Anki。" },
    { name: "PyTorch 官方教程", url: "https://pytorch.org/tutorials/beginner/basics/intro.html", note: "Learn the Basics：张量、自动求导、训练循环。" },
    { name: "arXiv", url: "https://arxiv.org/", note: "论文预印本；初学者可从 arXiv 上的综述和经典论文读起。" },
  ],

  chapters: {
    ch01: {
      level: 2, hours: 5, prereq: [], core: [0, 2],
      goal: "能把 Ax=b 看成“列的线性组合”，会用最小二乘拟合直线，知道 SVD 是做什么的。",
      links: [
        { type: "video", lang: "en", title: "3Blue1Brown · 线性代数的本质", url: "https://www.3blue1brown.com/topics/linear-algebra", note: "最好的几何直觉入门，有中文字幕。" },
        { type: "video", lang: "en", title: "MIT 18.06 · Gilbert Strang 公开课", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", note: "本站第 1 章的主要来源，教材作者亲自授课。" },
        { type: "interactive", lang: "en", title: "Immersive Linear Algebra", url: "https://immersivemath.com/ila/index.html", note: "可交互拖动的线性代数在线教材。" },
        { type: "interactive", lang: "en", title: "特征值与特征向量的可视化", url: "https://setosa.io/ev/eigenvectors-and-eigenvalues/", note: "拖动向量，直接看到矩阵如何作用。" },
        { type: "read", lang: "en", title: "Khan Academy · 线性代数", url: "https://www.khanacademy.org/math/linear-algebra", note: "从零开始、带练习题。" },
        { type: "practice", lang: "en", title: "NumPy 绝对入门", url: "https://numpy.org/doc/stable/user/absolute_beginners.html", note: "把公式变成代码前必读。" },
      ],
    },
    ch02: {
      level: 2, hours: 5, prereq: [], core: [0, 1, 3],
      goal: "会用贝叶斯公式做推断，理解分布与期望，知道交叉熵损失从哪来。",
      links: [
        { type: "interactive", lang: "en", title: "Seeing Theory · 概率统计可视化", url: "https://seeing-theory.brown.edu/", note: "布朗大学出品，概率与统计的交互式入门。" },
        { type: "interactive", lang: "en", title: "条件概率的可视化", url: "https://setosa.io/conditional/", note: "直观理解 P(A|B)。" },
        { type: "video", lang: "en", title: "StatQuest · 统计与机器学习", url: "https://www.youtube.com/@statquest", note: "用最直白的语言和动画讲统计概念。" },
        { type: "read", lang: "en", title: "Visual Information Theory（colah）", url: "https://colah.github.io/posts/2015-09-Visual-Information/", note: "用图理解熵、交叉熵与 KL 散度。" },
        { type: "read", lang: "en", title: "MacKay《信息论、推断与学习算法》免费在线", url: "http://www.inference.org.uk/itila/book.html", note: "本站第 2 章参考书之一。" },
      ],
    },
    ch03: {
      level: 1, hours: 4, prereq: [], core: [0, 1, 2, 3],
      goal: "说清楚什么是过拟合，会划分训练/验证/测试集，会看混淆矩阵与 ROC。",
      links: [
        { type: "read", lang: "zh", title: "Google 机器学习速成课程", url: "https://developers.google.com/machine-learning/crash-course", note: "官方入门课程，有中文版，配练习。" },
        { type: "interactive", lang: "en", title: "MLU-Explain · 机器学习可视化讲解", url: "https://mlu-explain.github.io/", note: "亚马逊出品的滚动式动画，覆盖偏差-方差、交叉验证、ROC 等。" },
        { type: "interactive", lang: "en", title: "MLU-Explain · 偏差与方差", url: "https://mlu-explain.github.io/bias-variance/", note: "用动画看懂过拟合与欠拟合，适合第一次接触。" },
        { type: "interactive", lang: "en", title: "MLU-Explain · 交叉验证 / ROC", url: "https://mlu-explain.github.io/cross-validation/", note: "配套的交叉验证讲解；ROC 见 mlu-explain.github.io/roc-auc/。" },
        { type: "video", lang: "en", title: "吴恩达 · Machine Learning 专项课程", url: "https://www.coursera.org/specializations/machine-learning-introduction", note: "公认最友好的入门课，可旁听。" },
        { type: "practice", lang: "en", title: "Kaggle · Intro to Machine Learning", url: "https://www.kaggle.com/learn/intro-to-machine-learning", note: "3 小时动手做出第一个模型。" },
        { type: "practice", lang: "en", title: "scikit-learn 用户指南", url: "https://scikit-learn.org/stable/user_guide.html", note: "经典算法的官方文档与示例。" },
      ],
    },
    ch04: {
      level: 3, hours: 5, prereq: ["ch01", "ch03"], core: [0, 1, 3],
      goal: "能手写梯度下降，理解学习率的影响，知道 L1/L2 正则化各自的作用。",
      links: [
        { type: "video", lang: "en", title: "3Blue1Brown · 梯度下降", url: "https://www.3blue1brown.com/lessons/gradient-descent", note: "神经网络系列第 2 集，把“下山”讲透。" },
        { type: "interactive", lang: "en", title: "普通最小二乘回归的可视化", url: "https://setosa.io/ev/ordinary-least-squares-regression/", note: "拖动数据点，看回归线怎么变。" },
        { type: "interactive", lang: "en", title: "MLU-Explain · 线性回归", url: "https://mlu-explain.github.io/linear-regression/", note: "滚动式动画讲解线性回归。" },
        { type: "interactive", lang: "en", title: "MLU-Explain · 逻辑回归", url: "https://mlu-explain.github.io/logistic-regression/", note: "分类的最基础模型。" },
        { type: "read", lang: "en", title: "Why Momentum Really Works（Distill）", url: "https://distill.pub/2017/momentum/", note: "带交互的动量法讲解，进阶阅读。" },
      ],
    },
    ch05: {
      level: 3, hours: 5, prereq: ["ch03", "ch04"], core: [0, 1, 3],
      goal: "知道树、随机森林、K 均值、PCA 各自解决什么问题，会用 scikit-learn 调用。",
      links: [
        { type: "interactive", lang: "en", title: "主成分分析（PCA）的可视化", url: "https://setosa.io/ev/principal-component-analysis/", note: "直观看懂降维。" },
        { type: "interactive", lang: "en", title: "MLU-Explain · 决策树", url: "https://mlu-explain.github.io/decision-tree/", note: "一步步看树如何切分。" },
        { type: "interactive", lang: "en", title: "MLU-Explain · 随机森林", url: "https://mlu-explain.github.io/random-forest/", note: "为什么多棵树比一棵好。" },
        { type: "interactive", lang: "en", title: "K-means 聚类可视化", url: "https://www.naftaliharris.com/blog/visualizing-k-means-clustering/", note: "自己点出数据，观察聚类收敛。" },
        { type: "video", lang: "en", title: "StatQuest · 决策树、随机森林、SVM", url: "https://www.youtube.com/@statquest", note: "在频道内搜索对应主题。" },
      ],
    },
    ch06: {
      level: 3, hours: 6, prereq: ["ch01", "ch04"], core: [0, 1, 2, 3],
      goal: "理解神经网络“前向计算 + 反向传播”的全过程，能不用框架手写一个小网络。",
      links: [
        { type: "video", lang: "en", title: "3Blue1Brown · 神经网络系列", url: "https://www.3blue1brown.com/topics/neural-networks", note: "神经网络与反向传播的最佳动画讲解，有中文字幕。" },
        { type: "interactive", lang: "en", title: "TensorFlow Playground", url: "https://playground.tensorflow.org", note: "在浏览器里调层数、激活函数，实时看决策边界。强烈推荐。" },
        { type: "read", lang: "en", title: "Nielsen《Neural Networks and Deep Learning》", url: "http://neuralnetworksanddeeplearning.com/", note: "免费在线书，反向传播讲得很细（本站第 6 章来源）。" },
        { type: "read", lang: "en", title: "Calculus on Computational Graphs: Backpropagation", url: "https://colah.github.io/posts/2015-08-Backprop/", note: "用计算图理解链式法则。" },
        { type: "video", lang: "en", title: "Karpathy · Neural Networks: Zero to Hero", url: "https://karpathy.ai/zero-to-hero.html", note: "从零手写自动求导，看完你就真懂反向传播。" },
        { type: "read", lang: "zh", title: "动手学深度学习 · 多层感知机", url: "https://zh.d2l.ai/chapter_multilayer-perceptrons/index.html", note: "中文版，代码和讲解并重。" },
      ],
    },
    ch07: {
      level: 3, hours: 5, prereq: ["ch06"], core: [0, 1, 2],
      goal: "知道训练不收敛时该查什么：初始化、学习率、优化器、正则化。",
      links: [
        { type: "read", lang: "en", title: "Karpathy · A Recipe for Training Neural Networks", url: "http://karpathy.github.io/2019/04/25/recipe/", note: "训练神经网络的实战清单，强烈建议收藏。" },
        { type: "read", lang: "en", title: "Why Momentum Really Works（Distill）", url: "https://distill.pub/2017/momentum/", note: "优化器的可视化解释。" },
        { type: "read", lang: "en", title: "CS231n · 训练神经网络（二）", url: "https://cs231n.github.io/neural-networks-3/", note: "学习率、Adam、梯度检查、过拟合诊断。" },
        { type: "read", lang: "zh", title: "动手学深度学习 · 优化算法", url: "https://zh.d2l.ai/chapter_optimization/index.html", note: "中文版优化章节。" },
      ],
    },
    ch08: {
      level: 3, hours: 6, prereq: ["ch06", "ch07"], core: [0, 1, 2, 3],
      goal: "能说清卷积、步长、填充、池化、感受野，能读懂 LeNet 到 ResNet 的演进。",
      links: [
        { type: "interactive", lang: "en", title: "CNN Explainer", url: "https://poloclub.github.io/cnn-explainer/", note: "浏览器里的卷积网络全流程可视化，边看边点。强烈推荐。" },
        { type: "interactive", lang: "en", title: "Image Kernels Explained Visually", url: "https://setosa.io/ev/image-kernels/", note: "亲手改卷积核，看图像怎么变。" },
        { type: "read", lang: "en", title: "CS231n · 卷积神经网络", url: "https://cs231n.github.io/convolutional-networks/", note: "斯坦福经典讲义。" },
        { type: "read", lang: "zh", title: "动手学深度学习 · 卷积神经网络", url: "https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html", note: "中文版，含 LeNet 到 ResNet 的代码。" },
        { type: "read", lang: "en", title: "卷积算术动画（conv_arithmetic）", url: "https://github.com/vdumoulin/conv_arithmetic", note: "步长、填充、转置卷积的动图。" },
        { type: "practice", lang: "en", title: "PyTorch · 构建神经网络", url: "https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html", note: "自己写一个 nn.Module。" },
      ],
    },
    ch09: {
      level: 3, hours: 5, prereq: ["ch01"], core: [0, 2],
      goal: "理解图像是怎样的数组，会做灰度变换、滤波和边缘检测。",
      links: [
        { type: "practice", lang: "en", title: "OpenCV-Python 教程", url: "https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html", note: "官方教程，配合本章代码最合适。" },
        { type: "interactive", lang: "en", title: "Image Kernels Explained Visually", url: "https://setosa.io/ev/image-kernels/", note: "空间滤波的直观演示。" },
        { type: "video", lang: "en", title: "3Blue1Brown · 傅里叶变换", url: "https://www.youtube.com/watch?v=spUNpyF58BY", note: "频率域滤波之前先看这个。" },
        { type: "read", lang: "en", title: "scikit-image 示例库", url: "https://scikit-image.org/docs/stable/auto_examples/", note: "分割、形态学、滤波的现成示例。" },
        { type: "video", lang: "en", title: "First Principles of Computer Vision（搜索）", url: "https://www.youtube.com/results?search_query=First+Principles+of+Computer+Vision+Shree+Nayar", note: "哥伦比亚大学 Shree Nayar 的计算机视觉系列。" },
      ],
    },
    ch10: {
      level: 4, hours: 6, prereq: ["ch08"], core: [0, 1],
      goal: "会用预训练模型做迁移学习，知道 IoU/NMS/mAP，理解 YOLO、U-Net 的用途。",
      links: [
        { type: "practice", lang: "en", title: "PyTorch · 迁移学习教程", url: "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html", note: "微调 ResNet 的标准写法。" },
        { type: "read", lang: "en", title: "CS231n · 迁移学习", url: "https://cs231n.github.io/transfer-learning/", note: "什么时候冻结、什么时候微调。" },
        { type: "read", lang: "en", title: "Ultralytics YOLO 文档", url: "https://docs.ultralytics.com/", note: "几行代码跑通检测、分割，适合快速上手。" },
        { type: "practice", lang: "en", title: "TorchVision · 目标检测微调", url: "https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html", note: "Mask R-CNN 微调完整流程。" },
        { type: "read", lang: "zh", title: "动手学深度学习 · 计算机视觉", url: "https://zh.d2l.ai/chapter_computer-vision/index.html", note: "中文版：数据增强、微调、检测、分割。" },
      ],
    },
    ch11: {
      level: 5, hours: 8, prereq: ["ch01", "ch09"], core: [0],
      goal: "理解针孔相机、单应变换、对极几何，知道三维重建的基本流程。",
      links: [
        { type: "read", lang: "en", title: "Szeliski《Computer Vision》免费在线版", url: "https://szeliski.org/Book/", note: "本站第 11 章主要来源之一。" },
        { type: "read", lang: "en", title: "Hartley & Zisserman 配套网站", url: "https://www.robots.ox.ac.uk/~vgg/hzbook/", note: "多视图几何的圣经，含章节资源。" },
        { type: "practice", lang: "en", title: "OpenCV · 相机标定与三维重建", url: "https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html", note: "标定、对极几何的代码教程。" },
        { type: "practice", lang: "en", title: "COLMAP 文档", url: "https://colmap.github.io/", note: "经典 SfM/MVS 工具，亲手重建一个场景。" },
        { type: "video", lang: "en", title: "Cyrill Stachniss · Photogrammetry（搜索）", url: "https://www.youtube.com/results?search_query=Cyrill+Stachniss+photogrammetry", note: "波恩大学的几何视觉公开课，讲得非常清楚。" },
      ],
    },
    ch12: {
      level: 4, hours: 6, prereq: ["ch06", "ch07"], core: [0, 2, 3],
      goal: "理解词嵌入、注意力与 Transformer 的结构，知道它为什么取代了 RNN。",
      links: [
        { type: "read", lang: "en", title: "The Illustrated Transformer（Jay Alammar）", url: "https://jalammar.github.io/illustrated-transformer/", note: "Transformer 最经典的图解。" },
        { type: "interactive", lang: "en", title: "Transformer Explainer", url: "https://poloclub.github.io/transformer-explainer/", note: "在浏览器里实时运行 GPT-2，看注意力如何计算。" },
        { type: "video", lang: "en", title: "3Blue1Brown · Transformer 与 GPT", url: "https://www.3blue1brown.com/lessons/gpt", note: "动画讲解词嵌入与注意力。" },
        { type: "video", lang: "en", title: "Karpathy · Let's build GPT", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", note: "两小时从零手写一个小 GPT。" },
        { type: "read", lang: "en", title: "Understanding LSTM Networks（colah）", url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", note: "LSTM 门控机制的最佳图解。" },
        { type: "read", lang: "en", title: "Hugging Face · LLM 课程", url: "https://huggingface.co/learn/llm-course", note: "用 Transformers 库实战。" },
      ],
    },
    ch13: {
      level: 4, hours: 7, prereq: ["ch08", "ch12"], core: [0, 1, 3],
      goal: "理解 VAE、GAN、扩散模型各自的训练目标，知道 Stable Diffusion 的组成。",
      links: [
        { type: "interactive", lang: "en", title: "GAN Lab", url: "https://poloclub.github.io/ganlab/", note: "在浏览器里训练 GAN，看生成器和判别器博弈。" },
        { type: "interactive", lang: "en", title: "Diffusion Explainer", url: "https://poloclub.github.io/diffusion-explainer/", note: "Stable Diffusion 的交互式图解。" },
        { type: "read", lang: "en", title: "What are Diffusion Models?（Lilian Weng）", url: "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/", note: "扩散模型的系统讲解（偏数学）。" },
        { type: "read", lang: "en", title: "The Illustrated Stable Diffusion", url: "https://jalammar.github.io/illustrated-stable-diffusion/", note: "图解 Stable Diffusion。" },
        { type: "read", lang: "en", title: "From Autoencoder to VAE（Lilian Weng）", url: "https://lilianweng.github.io/posts/2018-08-12-vae/", note: "从自编码器讲到 VAE。" },
        { type: "read", lang: "en", title: "Hugging Face · Diffusion 课程", url: "https://huggingface.co/learn/diffusion-course", note: "用 diffusers 库实战。" },
      ],
    },
    ch14: {
      level: 4, hours: 6, prereq: ["ch06", "ch02"], core: [0, 1, 3],
      goal: "理解 MDP、Q 学习与策略梯度，知道 RLHF 如何让大模型更符合人类偏好。",
      links: [
        { type: "read", lang: "en", title: "Sutton & Barto《强化学习》免费在线", url: "http://incompleteideas.net/book/the-book-2nd.html", note: "强化学习的标准教材。" },
        { type: "read", lang: "en", title: "OpenAI Spinning Up", url: "https://spinningup.openai.com/", note: "深度强化学习入门：概念 + 算法 + 代码。" },
        { type: "read", lang: "en", title: "Hugging Face · Deep RL 课程", url: "https://huggingface.co/learn/deep-rl-course", note: "边学边在游戏环境里训练智能体。" },
        { type: "read", lang: "en", title: "RLHF Book（Nathan Lambert）", url: "https://rlhfbook.com/", note: "本站第 14 章来源，在线可读。" },
        { type: "read", lang: "en", title: "Illustrating RLHF（Hugging Face 博客）", url: "https://huggingface.co/blog/rlhf", note: "RLHF 全流程图解。" },
        { type: "video", lang: "en", title: "David Silver · 强化学习课程", url: "https://www.davidsilver.uk/teaching/", note: "DeepMind 经典公开课，含课件。" },
      ],
    },
    ch15: {
      level: 5, hours: 7, prereq: ["ch02"], core: [0],
      goal: "理解贝叶斯滤波，知道卡尔曼滤波、粒子滤波和 SLAM 解决什么问题。",
      links: [
        { type: "interactive", lang: "en", title: "kalmanfilter.net", url: "https://www.kalmanfilter.net/", note: "从一维例子讲起的卡尔曼滤波教程。" },
        { type: "read", lang: "en", title: "Kalman and Bayesian Filters in Python", url: "https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python", note: "Jupyter 形式的免费书，边读边跑代码。" },
        { type: "practice", lang: "en", title: "PythonRobotics", url: "https://github.com/AtsushiSakai/PythonRobotics", note: "定位、SLAM、路径规划的 Python 实现合集，带动图。" },
        { type: "practice", lang: "en", title: "ORB-SLAM3", url: "https://github.com/UZ-SLAMLab/ORB_SLAM3", note: "主流视觉 SLAM 开源系统。" },
        { type: "video", lang: "en", title: "Cyrill Stachniss · SLAM（搜索）", url: "https://www.youtube.com/results?search_query=Cyrill+Stachniss+SLAM+course", note: "波恩大学的 SLAM 公开课。" },
      ],
    },
    ch16: {
      level: 2, hours: 5, prereq: [], core: [0, 1],
      goal: "读懂 PyTorch 代码里的 Python 机制，知道模型从实验到上线要注意什么。",
      links: [
        { type: "video", lang: "en", title: "CS50P · Harvard Python 入门", url: "https://cs50.harvard.edu/python/", note: "零基础学 Python 的最佳课程之一，免费。" },
        { type: "read", lang: "zh", title: "Python 官方教程（中文）", url: "https://docs.python.org/zh-cn/3/tutorial/", note: "权威、简洁。" },
        { type: "practice", lang: "en", title: "PyTorch · 自动求导", url: "https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html", note: "理解 loss.backward() 在做什么。" },
        { type: "read", lang: "en", title: "OSTEP · 操作系统导论（免费）", url: "https://pages.cs.wisc.edu/~remzi/OSTEP/", note: "进程、线程、内存的通俗教材。" },
        { type: "read", lang: "en", title: "Made With ML", url: "https://madewithml.com/", note: "把模型做成可上线的产品（MLOps）。" },
        { type: "read", lang: "en", title: "Full Stack Deep Learning", url: "https://fullstackdeeplearning.com/", note: "工业界视角的深度学习工程课程。" },
      ],
    },
  },
};
