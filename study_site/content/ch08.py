CHAPTER = dict(
    no=8,
    title="卷积神经网络与经典架构",
    title_en="Convolutional Neural Networks and Classic Architectures",
    group="深度学习",
    summary="从“为什么全连接网络不适合图像”出发，讲清卷积、填充、步长、池化和感受野，再沿着 LeNet-5 → AlexNet → VGG → GoogLeNet → ResNet 的演进，理解每一代架构解决了什么问题。",
    sources=["Elgendy《视觉系统深度学习》第 3、5 章", "Chollet《Python 深度学习》第 8–9 章", "邱锡鹏《神经网络与深度学习》第 5 章",
             "Géron《机器学习实战》第 14 章", "Nielsen《神经网络与深度学习》第 6 章"],
    learn=[
        dict(
            title="为什么图像需要卷积",
            title_en="Why convolution for images",
            body=r"""
用多层感知机（MLP）处理图像，要先把图像**展平**成一个长向量。一张 $224\times224\times3$ 的彩色图像展平后有 150,528 个数，如果第一层有 1,000 个神经元，仅这一层就需要约 1.5 亿个权重。参数太多会带来两个问题：计算和存储开销巨大，而且极易过拟合。

更根本的问题是：展平**破坏了像素之间的空间关系**。图像里有意义的结构（边缘、角点、纹理）都是**局部**的，而且同一种结构可能出现在图像的任何位置。卷积神经网络（CNN）正是围绕这两点设计的：

- **局部连接**：每个神经元只看输入中的一个小窗口（例如 $3\times3$），而不是整张图。
- **权值共享**：同一个卷积核在整张图上滑动使用，检测“猫耳朵”的特征检测器不必在每个位置各学一遍。
- **平移等变**：输入平移，特征图也跟着平移；再配合池化，就得到对小平移的不变性。

> 直观理解：卷积核就是一个可学习的“模板”，卷积运算就是拿这个模板在图像上逐处比对，相似的地方响应大。
""",
            points=[
                "MLP 的参数量随输入像素数线性增长，CNN 的参数量只取决于卷积核大小和通道数，与图像尺寸无关。",
                "**局部连接**利用了图像的局部相关性；**权值共享**利用了特征在空间上的可重复性。",
                "卷积对平移是**等变**的（equivariant），池化让网络对小平移近似**不变**（invariant）。",
                "CNN 的浅层学到边缘、颜色块，深层组合出纹理、部件，最后到整个物体，这是一种层次化表示。",
            ],
            src="Elgendy 第 3 章 3.1；邱锡鹏 5.1–5.2；Chollet 8.1",
        ),
        dict(
            title="卷积运算：卷积核、步长、填充与通道",
            title_en="The convolution operation",
            body=r"""
深度学习中的“卷积”实际上是**互相关**（cross-correlation）：把卷积核放在输入的一个窗口上，对应元素相乘再求和，得到输出特征图上的一个值；然后按**步长**（stride）移动窗口，重复计算。

输出尺寸公式（输入边长 $n$，核大小 $k$，填充 $p$，步长 $s$）：

$$o = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

- **填充**（padding）：在边缘补 0。`valid` 表示不填充，输出变小；`same` 表示填充 $p=(k-1)/2$，在步长为 1 时输出尺寸不变。
- **步长**：大于 1 时相当于降采样，输出尺寸约缩小为原来的 $1/s$。
- **多通道**：输入有 $C_{in}$ 个通道时，每个卷积核的形状是 $k\times k\times C_{in}$；有 $C_{out}$ 个卷积核，就输出 $C_{out}$ 个通道。

一个卷积层的参数量为 $(k\cdot k\cdot C_{in}+1)\cdot C_{out}$，其中 $+1$ 是每个输出通道的偏置。
""",
            points=[
                "记住输出尺寸公式，几乎所有 CNN 形状题都靠它。",
                "`same` 填充 + 步长 1 时尺寸不变；步长 2 时尺寸大约减半。",
                "卷积核的深度总是等于输入通道数，卷积核的**个数**决定输出通道数。",
                "参数量与输入图像的宽高无关，计算量（乘法次数）则与输出特征图大小成正比。",
            ],
            code="ch08_conv2d.py",
            code_note="用 numpy 手写二维卷积，并用 Sobel 核检测一条竖直边缘；注意输出尺寸如何随填充和步长变化。",
            src="Elgendy 3.2；邱锡鹏 5.1；Géron 14 章“卷积层”",
        ),
        dict(
            title="池化、感受野与层次特征",
            title_en="Pooling, receptive field and feature hierarchy",
            body=r"""
**池化层**（pooling）对特征图做降采样，没有可学习参数。最常见的是 $2\times2$、步长 2 的**最大池化**：每个窗口只保留最大值。它的作用有三：减小特征图尺寸、降低后续计算量；让网络对小的平移和形变更鲁棒；扩大后续层的感受野。另一种是**全局平均池化**，把每个通道整张特征图平均成一个数，常用来替代网络末端的全连接层。

**感受野**（receptive field）指输出特征图上一个单元对应到原始输入上的区域大小。堆叠卷积层时，感受野按下式逐层增长（$j$ 为到上一层为止的累计步长）：

$$r_l = r_{l-1} + (k_l - 1)\, j_{l-1}$$

三个 $3\times3$ 卷积堆叠的感受野是 $7\times7$；中间如果夹着步长为 2 的池化，感受野会增长得更快。这就是 CNN 能够“由小见大”的原因：浅层看局部的边缘，深层看到更大的区域，组合出物体的部件乃至整体。
""",
            points=[
                "池化没有参数，最大池化保留最强响应，平均池化保留整体强度。",
                "池化会丢失精确的位置信息，所以分割、检测等需要定位的任务往往要配合上采样或跳跃连接。",
                "感受野越大，单元能“看到”的上下文越多；增加深度和使用步长都能扩大感受野。",
                "现代网络常用步长为 2 的卷积代替池化，用全局平均池化代替大的全连接层。",
            ],
            code="ch08_pooling.py",
            code_note="验证最大池化的降采样效果，以及特征平移一个像素后池化结果保持不变。",
            src="Elgendy 3.2.3；Chollet 8.1.2；邱锡鹏 5.3",
        ),
        dict(
            title="经典架构演进：从 LeNet 到 ResNet",
            title_en="From LeNet to ResNet",
            body=r"""
- **LeNet-5（1998）**：LeCun 用于手写数字识别，奠定了“卷积 → 池化 → 卷积 → 池化 → 全连接”的基本模式。
- **AlexNet（2012）**：在 ImageNet 竞赛中把 top-5 错误率从约 26% 降到约 15%，引爆深度学习。关键做法：ReLU 激活、Dropout、数据增强、用 GPU 训练。
- **VGG（2014）**：全部使用 $3\times3$ 小卷积核堆叠。两个 $3\times3$ 的感受野等于一个 $5\times5$，但参数更少、非线性更多。结构规整，16/19 层。
- **GoogLeNet / Inception（2014）**：在同一模块里并行使用 $1\times1$、$3\times3$、$5\times5$ 卷积和池化，并先用 $1\times1$ 卷积降维（“瓶颈”），大幅降低计算量。
- **ResNet（2015）**：网络加深后出现**退化**现象：更深的网络训练误差反而更高，这并不是过拟合。ResNet 让每个模块学习残差 $F(x)$，输出 $y = F(x) + x$。恒等的**跳跃连接**让梯度可以直接回传，使 152 层乃至上千层的网络都能训练。

此后还有 DenseNet（每层连接之前所有层）、MobileNet（深度可分离卷积，面向移动端）、EfficientNet（按比例同时缩放深度、宽度和分辨率）等。演进主线始终是：**用更少的参数和计算，训练更深、更好泛化的网络**。
""",
            points=[
                "AlexNet 的成功来自算法（ReLU、Dropout）、数据（ImageNet）和算力（GPU）三者的结合。",
                "VGG 的经验：小卷积核堆叠优于单个大卷积核。",
                "$1\\times1$ 卷积只在通道维上做线性组合，用于降维/升维，是 Inception、ResNet 瓶颈块和 MobileNet 的关键部件。",
                "残差连接解决的是**优化困难**（梯度消失与退化），而不是过拟合。",
                "批量归一化（BatchNorm）让深网络训练更稳定，几乎成为现代 CNN 的标配。",
            ],
            code="ch08_residual.py",
            code_note="把 50 层网络近似为线性映射，比较普通网络与残差网络反向传播时梯度范数的变化。",
            src="Elgendy 第 5 章；Géron 14 章“CNN 架构”；邱锡鹏 5.4；本项目原有的 LeNet、AlexNet、VGG、GoogLeNet、ResNet、DenseNet、MobileNet、ShuffleNet、EfficientNet 论文",
        ),
    ],
    examples=[
        dict(
            title="计算输出尺寸与参数量",
            problem=r"""输入为 $32\times32\times3$ 的彩色图像，经过一个卷积层：6 个 $5\times5$ 卷积核，步长 1，不填充。求输出特征图的形状和该层的参数量。如果改为 `same` 填充，输出形状是什么？""",
            steps=[
                r"输出边长：$o=\lfloor(32+0-5)/1\rfloor+1=28$。",
                r"输出通道数等于卷积核个数 6，所以输出形状为 $28\times28\times6$。",
                r"每个卷积核的形状是 $5\times5\times3$，有 75 个权重，加 1 个偏置共 76 个参数；6 个核共 $76\times6=456$ 个参数。",
                r"`same` 填充时 $p=(5-1)/2=2$，$o=(32+4-5)/1+1=32$，输出为 $32\times32\times6$，参数量不变。",
            ],
            answer=r"$28\times28\times6$，456 个参数；`same` 填充时为 $32\times32\times6$。",
            code="ch08_shapes_params.py",
        ),
        dict(
            title="堆叠小卷积核的感受野",
            problem=r"""(1) 三个步长为 1 的 $3\times3$ 卷积层堆叠后，最后一层一个单元的感受野多大？(2) 网络 conv3×3 → maxpool2×2(步长 2) → conv3×3 → maxpool2×2 → conv3×3 的最终感受野是多少？""",
            steps=[
                r"使用递推式 $r_l=r_{l-1}+(k_l-1)j_{l-1}$，初始 $r_0=1,\ j_0=1$。",
                r"(1) 三层 $3\times3$、步长均为 1：$r=1+2+2+2=7$，即 $7\times7$。",
                r"(2) conv3×3：$r=3,\ j=1$；pool2×2：$r=3+1\times1=4,\ j=2$；conv3×3：$r=4+2\times2=8$；pool2×2：$r=8+1\times2=10,\ j=4$；conv3×3：$r=10+2\times4=18$。",
                r"可见池化（步长）放大了后续层每一步的“跨度”，感受野增长得更快。",
            ],
            answer=r"(1) $7\times7$；(2) $18\times18$。",
            code="ch08_receptive.py",
        ),
        dict(
            title="1×1 卷积瓶颈能省多少计算",
            problem=r"""输入特征图 $28\times28\times192$，要得到 32 个通道的 $5\times5$ 卷积输出（`same` 填充）。比较两种方案的乘法次数：A. 直接 $5\times5$ 卷积；B. 先用 16 个 $1\times1$ 卷积降维，再做 $5\times5$ 卷积。""",
            steps=[
                r"卷积的乘法次数 = 输出元素个数 × 每个输出元素的乘法次数 = $H\cdot W\cdot C_{out}\cdot k\cdot k\cdot C_{in}$。",
                r"方案 A：$28\cdot28\cdot32\cdot5\cdot5\cdot192 = 120{,}422{,}400$，约 1.2 亿次。",
                r"方案 B 第一步（$1\times1$ 降到 16 通道）：$28\cdot28\cdot16\cdot192=2{,}408{,}448$；第二步（$5\times5$，输入 16 通道）：$28\cdot28\cdot32\cdot25\cdot16=10{,}035{,}200$。",
                r"方案 B 合计约 1244 万次，只有方案 A 的约 10%。这就是 Inception 模块使用瓶颈层的原因。",
            ],
            answer="方案 A 约 1.2 亿次乘法，方案 B 约 1244 万次，节省约 90%。",
            code="ch08_bottleneck.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="输入 $64\\times64$，卷积核 $3\\times3$，步长 2，填充 1，输出边长是多少？",
                 options=["31", "32", "33", "64"], answer=1,
                 explain="$\\lfloor(64+2-3)/2\\rfloor+1=31+1=32$。"),
            dict(q="一个卷积层输入 16 通道、输出 32 通道、核大小 $3\\times3$，含偏置的参数量是？",
                 options=["4,608", "4,640", "9,248", "288"], answer=1,
                 explain="$(3\\times3\\times16+1)\\times32=145\\times32=4640$。"),
            dict(q="CNN 相比全连接网络处理图像时，参数大幅减少主要是因为：",
                 options=["使用了 ReLU", "局部连接与权值共享", "使用了 Dropout", "使用了 Softmax"], answer=1,
                 explain="每个卷积核只连接局部窗口，并在所有位置共享同一组权重。"),
            dict(q="关于最大池化，下列说法正确的是：",
                 options=["有可学习的权重", "会增大特征图尺寸", "能提高对小平移的鲁棒性", "只能用于输入层"], answer=2,
                 explain="最大池化没有参数，作用是降采样并带来对小平移的近似不变性。"),
            dict(q="ResNet 中残差连接主要解决的问题是：",
                 options=["过拟合", "深层网络的退化与梯度难以回传", "输入图像尺寸不一致", "类别不平衡"], answer=1,
                 explain="更深的普通网络训练误差反而更高（退化），残差连接让恒等映射容易学到，梯度可直接回传。"),
            dict(q="在 Inception 模块中，$1\\times1$ 卷积的主要作用是：",
                 options=["扩大感受野", "降低通道数以减少计算量", "替代池化", "实现数据增强"], answer=1,
                 explain="$1\\times1$ 卷积在通道维做线性组合，常用作瓶颈层降维。"),
        ],
        tf=[
            dict(q="深度学习框架中的“卷积”在数学上严格来说是互相关运算。", answer=True,
                 explain="框架实现时卷积核不翻转，即互相关；因为核是学出来的，二者效果等价。"),
            dict(q="卷积层的参数量会随着输入图像宽高的增大而增大。", answer=False,
                 explain="参数量只由 $k$、$C_{in}$、$C_{out}$ 决定；增大的是计算量。"),
            dict(q="两个堆叠的 $3\\times3$ 卷积与一个 $5\\times5$ 卷积具有相同的感受野。", answer=True,
                 explain="$1+2+2=5$，并且前者参数更少、非线性更多。"),
            dict(q="AlexNet 的成功只依靠更深的网络结构，与 GPU 和数据无关。", answer=False,
                 explain="ImageNet 大规模数据、GPU 训练、ReLU 和 Dropout 共同促成了它的突破。"),
            dict(q="`same` 填充在步长为 1 时可以保持输出特征图的空间尺寸不变。", answer=True,
                 explain="填充 $p=(k-1)/2$ 时，$o=n$。"),
        ],
        fill=[
            dict(q="卷积输出尺寸公式为 $o=\\lfloor (n+2p-k)/s \\rfloor + $ ____。", answer=["1"],
                 explain="不要漏掉最后的 +1。"),
            dict(q="卷积核在整张图像上重复使用同一组权重，这一性质叫做 ____（中文或英文）。",
                 answer=["权值共享", "权重共享", "参数共享", "weight sharing", "parameter sharing"],
                 explain="weight sharing / 权值共享。"),
            dict(q="输出特征图上一个单元对应到原始输入的区域大小叫做 ____。", answer=["感受野", "receptive field"],
                 explain="receptive field / 感受野。"),
            dict(q="ResNet 模块的输出写作 $y=F(x)+$ ____。", answer=["x"],
                 explain="恒等跳跃连接把输入 $x$ 直接加到残差分支的输出上。"),
        ],
        reading=[
            dict(
                title="The Degradation Problem",
                passage="""By 2015, researchers knew that deeper networks could represent richer functions, so they kept stacking layers. Yet something surprising happened: beyond a certain depth, adding more layers made the **training** error go up, not just the test error. This was not overfitting, because an overfitted model fits its training data well. It was an optimization problem.

The authors of ResNet reasoned that a deeper network should never be worse than a shallower one: the extra layers could simply learn to copy their input. In practice, however, plain layers find it hard to learn an exact identity mapping. Residual learning changes the target. Instead of asking a block to produce the desired output H(x) directly, it asks the block to learn the difference F(x) = H(x) − x and then adds x back through a shortcut connection.

If the identity is already close to optimal, the block only needs to push F(x) toward zero, which is easy. The shortcut also gives gradients a direct path back to early layers. With this simple change, networks with more than a hundred layers trained smoothly and won the 2015 ImageNet competition.""",
                zh="""到 2015 年，研究者知道更深的网络能表示更丰富的函数，于是不断堆叠层数。但出现了一个令人意外的现象：超过一定深度后，继续加层会让**训练**误差上升，而不仅是测试误差上升。这不是过拟合，因为过拟合的模型在训练集上拟合得很好；这是一个优化问题。

ResNet 的作者认为，更深的网络不应该比浅的更差：多出来的层只要学会复制输入即可。然而在实践中，普通的层很难精确地学到恒等映射。残差学习改变了学习目标：不再要求一个模块直接输出期望的 H(x)，而是让它学习差值 F(x) = H(x) − x，再通过捷径连接把 x 加回来。

如果恒等映射已经接近最优，模块只需把 F(x) 推向零，这很容易。捷径连接还给梯度提供了一条直通浅层的路径。凭借这个简单的改动，一百多层的网络也能平稳训练，并赢得了 2015 年的 ImageNet 竞赛。""",
                questions=[
                    dict(q="According to the passage, why is the degradation problem **not** overfitting?",
                         options=["Because test error went down", "Because training error itself increased", "Because the dataset was too small", "Because dropout was used"],
                         answer=1, explain="文中说加深后训练误差上升，而过拟合的模型训练误差应该很低。"),
                    dict(q="What does a residual block learn?",
                         options=["The output H(x) directly", "The input x", "The difference F(x) = H(x) − x", "The gradient of the loss"],
                         answer=2, explain="残差块学习的是 H(x) 与 x 的差。"),
                    dict(q="Which is NOT mentioned as a benefit of the shortcut connection?",
                         options=["It makes identity mappings easy", "It provides a direct path for gradients", "It reduces the number of input pixels", "It helps very deep networks train"],
                         answer=2, explain="文中没有提到减少输入像素。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("convolution", "卷积", "An operation that slides a small kernel over the input and computes weighted sums at each position.", "用小卷积核在输入上滑动、在每个位置做加权求和的运算。", "Each convolution layer learns a set of filters from data."),
        ("kernel", "卷积核", "The small learnable weight matrix used in a convolution; also called a filter.", "卷积中使用的小型可学习权重矩阵，也叫滤波器。", "A 3 by 3 kernel looks at nine neighboring pixels at a time."),
        ("feature map", "特征图", "The output of applying one filter across the whole input.", "一个卷积核作用于整个输入后得到的输出。", "Deeper layers produce feature maps that respond to object parts."),
        ("stride", "步长", "The number of pixels the kernel moves at each step.", "卷积核每次移动的像素数。", "A stride of two roughly halves the width and height of the output."),
        ("padding", "填充", "Adding extra values, usually zeros, around the border of the input.", "在输入边界周围补值（通常为零）。", "Same padding keeps the spatial size unchanged when the stride is one."),
        ("receptive field", "感受野", "The region of the original input that affects one output unit.", "影响某个输出单元的原始输入区域。", "Stacking layers enlarges the receptive field of each unit."),
        ("pooling", "池化", "A downsampling operation that summarizes each local window, for example by its maximum.", "对每个局部窗口做汇总（如取最大值）的降采样运算。", "Max pooling keeps the strongest activation in each window."),
        ("weight sharing", "权值共享", "Using the same weights at every spatial position of the input.", "在输入的所有空间位置使用同一组权重。", "Weight sharing is why a CNN has far fewer parameters than an MLP."),
        ("translation invariance", "平移不变性", "The property that a small shift of the input does not change the output much.", "输入发生小的平移时输出基本不变的性质。", "Pooling gives the network some translation invariance."),
        ("fully connected layer", "全连接层", "A layer in which every neuron is connected to every input.", "每个神经元都与所有输入相连的层。", "The final fully connected layer maps features to class scores."),
        ("residual connection", "残差连接", "A shortcut that adds a block's input to its output.", "把模块的输入直接加到其输出上的捷径连接。", "Residual connections let gradients flow through very deep networks."),
        ("batch normalization", "批量归一化", "Normalizing layer activations using the mean and variance of the current mini-batch.", "用当前小批量的均值和方差对层的激活值进行归一化。", "Batch normalization usually allows a higher learning rate."),
        ("depthwise separable convolution", "深度可分离卷积", "A convolution split into a per-channel spatial step and a 1 by 1 channel-mixing step.", "把卷积拆成逐通道空间卷积和 1×1 通道混合两步。", "MobileNet relies on depthwise separable convolutions to stay small."),
        ("data augmentation", "数据增强", "Creating new training samples by randomly transforming existing ones.", "通过随机变换已有样本来生成新的训练样本。", "Random flips and crops are common forms of data augmentation."),
    ],
    cards=[
        ("卷积输出尺寸公式？", r"$o=\lfloor (n+2p-k)/s \rfloor+1$"),
        ("卷积层参数量公式？", r"$(k\cdot k\cdot C_{in}+1)\cdot C_{out}$，与图像宽高无关"),
        ("为什么 VGG 用 3×3 小卷积核？", "两个 3×3 的感受野等于一个 5×5，但参数更少（约少 28%），且多一次非线性。"),
        ("ResNet 解决了什么问题？", "深层网络的**退化**：加深后训练误差反而上升。残差连接让恒等映射易学、梯度可直达浅层。"),
    ],
)
