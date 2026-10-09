CHAPTER = dict(
    no=7,
    title="训练深度网络：初始化、优化与正则化",
    title_en="Training Deep Networks: Initialization, Optimization and Regularization",
    group="深度学习",
    summary="网络一深就难训练。本章讲梯度消失与爆炸、Xavier/He 初始化、动量/RMSProp/Adam 等优化器、学习率调度，以及 L2 权重衰减、Dropout、早停、数据增强和批量归一化等让深网络训得动、泛化好的技巧。",
    sources=["Nielsen 第 3、5 章", "Buduma《深度学习基础》第 6 章", "邱锡鹏 第 7 章", "Géron 第 11 章", "Patterson & Gibson 第 6 章", "Elgendy 第 4 章"],
    learn=[
        dict(
            title="梯度消失、梯度爆炸与权重初始化",
            title_en="Vanishing gradients and weight initialization",
            body=r"""
Nielsen 在第 5 章发现：给网络加更多隐藏层，效果并没有变好。原因是**梯度不稳定**。反向传播时，梯度要连续乘以每一层的权重和激活函数导数。sigmoid 的导数最大只有 $1/4$，乘上 20 次，浅层的梯度几乎为零——**梯度消失**，浅层学得极慢；反过来，如果权重很大，梯度会指数级增长——**梯度爆炸**，训练直接崩溃。

解决思路之一是让每层输出的方差保持不变，这取决于**初始化**：

- **Xavier/Glorot 初始化**：权重方差 $\mathrm{Var}(W)=\dfrac{1}{n_{in}}$（或 $\dfrac{2}{n_{in}+n_{out}}$），适合 tanh、sigmoid。
- **He 初始化**：$\mathrm{Var}(W)=\dfrac{2}{n_{in}}$，适合 ReLU。ReLU 会把一半的输入置零，方差减半，所以要乘 2 补偿。

再配合**不饱和的激活函数**（ReLU 及其变体）、**批量归一化**、**残差连接**（第 8 章）和**梯度裁剪**（把梯度范数限制在阈值以内，常用于循环网络），深层网络就能稳定训练了。
""",
            points=[
                "梯度消失：浅层几乎不更新；梯度爆炸：损失突然变成 NaN。",
                "sigmoid/tanh 在输入很大或很小时饱和，导数接近 0，是梯度消失的主要来源。",
                "ReLU 配 He 初始化，tanh/sigmoid 配 Xavier 初始化。",
                "Patterson & Gibson 建议监控“参数更新量 / 参数大小”的比值（约 $10^{-3}$ 为宜）来诊断学习率和初始化。",
            ],
            code="ch07_init_vanishing.py",
            code_note="在 20 层网络中比较 sigmoid+Xavier、ReLU+Xavier、ReLU+He 三种组合下激活值和梯度的尺度。",
            src="Nielsen 第 5 章；Géron 11.1；邱锡鹏 7.3；Patterson & Gibson 第 6 章“权重初始化”",
        ),
        dict(
            title="超越梯度下降：动量、RMSProp 与 Adam",
            title_en="Beyond plain gradient descent",
            body=r"""
Buduma 第 6 章指出，深网络误差曲面的真正麻烦不是坏的局部极小值，而是**平坦区域、鞍点**和**病态曲率**：在狭长的“峡谷”里，陡峭方向要求小学习率，平缓方向却需要大步子，普通 SGD 只能折中，走成“之”字形。

- **动量（Momentum）**：$\mathbf v\leftarrow\beta\mathbf v+\nabla L$，$\mathbf w\leftarrow\mathbf w-\eta\mathbf v$。像一个有惯性的小球，在一致的方向上不断加速，来回震荡的方向相互抵消。$\beta$ 常取 0.9。
- **AdaGrad**：按历史梯度平方和缩放每个参数的学习率，频繁更新的参数步子变小；缺点是学习率单调衰减到 0。
- **RMSProp**：改用梯度平方的**指数滑动平均**，避免学习率过早衰减。
- **Adam**：结合动量（一阶矩 $m$）与 RMSProp（二阶矩 $v$），并做偏差校正：

$$m\leftarrow\beta_1 m+(1-\beta_1)g,\quad v\leftarrow\beta_2 v+(1-\beta_2)g^2,\quad \mathbf w\leftarrow\mathbf w-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

默认 $\beta_1=0.9,\ \beta_2=0.999$。Adam 对超参数不敏感，是最常用的默认选择；但在一些任务（如图像分类）上，精调过的 SGD+动量泛化往往更好。

**学习率调度**同样重要：预热（warmup）、按步衰减、余弦退火等。
""",
            points=[
                "Adam 第一步的更新量约等于学习率本身，与梯度的绝对大小无关。",
                "自适应方法在最小值附近可能来回波动，配合学习率衰减效果更好。",
                "没有在所有问题上都最好的优化器（Buduma：优化器选择的哲学）。",
                "AdamW 把权重衰减与自适应学习率解耦，是训练 Transformer 的常用选择。",
            ],
            code="ch07_optimizers.py",
            code_note="在条件数为 50 的二次函数“峡谷”上比较四种优化器在第 10、50、200 步的损失。",
            src="Buduma 第 6 章；Géron 11.3；邱锡鹏 7.2",
        ),
        dict(
            title="正则化：权重衰减、Dropout、早停与数据增强",
            title_en="Regularization for deep networks",
            body=r"""
深度网络参数极多，很容易把训练数据（连同噪声）背下来。常用的正则化手段：

- **L2 权重衰减**：损失加 $\frac\lambda2\|W\|^2$，梯度多出 $\lambda W$，每步把权重按比例缩小一点。Nielsen 从奥卡姆剃刀的角度解释：小权重意味着网络对输入的微小变化不那么敏感，更“简单”。
- **Dropout**：训练时以概率 $p$ 随机让神经元暂时“失活”，每一步都相当于训练一个不同的子网络；测试时使用全部神经元。实现上常用 **inverted dropout**：训练时把保留的输出除以 $1-p$，保证期望不变。它可以看作是对大量子网络的廉价集成。
- **早停**：监控验证集损失，不再下降时停止训练，并恢复最好的参数。
- **数据增强**：对图像做随机翻转、裁剪、旋转、颜色抖动，对文本做同义词替换、回译（Pointer 第 5 章）等，人为扩大训练集。Nielsen 把 MNIST 图片轻微旋转来扩充数据，就带来了明显提升。
- **标签平滑**：把 one-hot 标签 1 改为 $1-\epsilon$，防止模型过度自信。

Elgendy 第 4 章给出了一套调参流程：先建立基线，判断是欠拟合还是过拟合，再对症下药——欠拟合就加大模型、多训练；过拟合就加数据、加正则化。
""",
            points=[
                "正则化总是用一点训练表现换取更好的泛化。",
                "Dropout 只在训练时开启，推理时要关闭（PyTorch 中的 `model.eval()`）。",
                "获取更多真实数据通常是最有效的“正则化”。",
                "多种正则化可以组合使用，但强度要用验证集调。",
            ],
            code="ch07_regularization.py",
            code_note="60 个样本、20 维输入中只有 2 维有用：比较无正则化、L2 和 Dropout 在干净测试集上的准确率。",
            src="Nielsen 3.2；Géron 11.4；邱锡鹏 7.6–7.7；Elgendy 4.4；Patterson & Gibson 第 6 章“正则化”",
        ),
        dict(
            title="批量归一化与训练流程中的诊断",
            title_en="Batch normalization and diagnosing training",
            body=r"""
**批量归一化（BatchNorm）**在每一层对小批量数据逐特征做标准化，再用可学习的 $\gamma,\beta$ 恢复表达能力：

$$\hat z=\frac{z-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat z+\beta$$

它让每层输入的分布更稳定，允许更大的学习率，减轻对初始化的敏感，还带有轻微的正则化效果。注意**训练和推理的行为不同**：训练时用当前批的统计量，推理时用训练期间累积的滑动平均。批很小时 BatchNorm 不稳定，可改用**层归一化**（LayerNorm，Transformer 的标配）或**组归一化**（GroupNorm，Buduma 第 7 章）。

训练中的常见诊断（Patterson & Gibson 第 6 章、Elgendy 第 4 章）：

- 损失不下降：学习率太小、初始化不当、数据没打乱、标签错位；
- 损失变成 NaN：学习率太大、梯度爆炸、数值溢出（如 $\log 0$）；
- 训练好、验证差：过拟合；两者都差：欠拟合或模型容量不足；
- 类别不平衡：使用加权损失或重采样；
- 先在一个很小的子集上让模型**过拟合**，确认整条流水线没有 bug，再放大规模。
""",
            points=[
                "BatchNorm 后的层通常不再需要偏置（$\\beta$ 起到了偏置的作用）。",
                "推理时忘记切换到 eval 模式，是 BatchNorm/Dropout 最常见的 bug。",
                "小批量越大，梯度估计越准，但泛化不一定更好，且需要相应调大学习率。",
                "用 TensorBoard 等工具同时画出训练和验证曲线。",
            ],
            code="ch07_batchnorm.py",
            code_note="对三个尺度相差悬殊的特征做批量归一化，验证输出的均值和标准差等于 $\\beta$ 和 $\\gamma$，并模拟推理时使用的滑动平均。",
            src="Géron 11.1.3；Buduma 第 7 章；邱锡鹏 7.5；Patterson & Gibson 第 6 章",
        ),
    ],
    examples=[
        dict(
            title="手算 Adam 的第一步",
            problem=r"参数当前梯度 $g=0.5$，Adam 超参数 $\eta=0.001,\ \beta_1=0.9,\ \beta_2=0.999$，$m_0=v_0=0$。求第一步的更新量。若梯度是 50 呢？",
            steps=[
                r"$m_1=0.1\times0.5=0.05$，$v_1=0.001\times0.25=0.00025$。",
                r"偏差校正：$\hat m=0.05/(1-0.9)=0.5$，$\hat v=0.00025/(1-0.999)=0.25$。",
                r"更新量 $=\eta\,\hat m/\sqrt{\hat v}=0.001\times0.5/0.5=0.001$。",
                r"若梯度为 50：$\hat m=50$，$\hat v=2500$，更新量仍为 $0.001\times50/50=0.001$。没有偏差校正时 $m_1$、$v_1$ 都偏向 0，校正后第一步的步长就是 $\eta$。",
            ],
            answer="更新量都是 0.001，等于学习率。",
            code="ch07_ex_adam.py",
        ),
        dict(
            title="选择初始化的标准差",
            problem=r"一个全连接层有 512 个输入、256 个输出，激活函数为 ReLU。He 初始化的权重标准差是多少？若改用 tanh 和 Xavier 初始化 $\mathrm{Var}=2/(n_{in}+n_{out})$ 呢？",
            steps=[
                r"He 初始化：$\mathrm{Var}(W)=2/n_{in}=2/512$，标准差 $=\sqrt{2/512}=0.0625$。",
                r"Xavier（Glorot）：$\mathrm{Var}(W)=2/(512+256)=2/768$，标准差 $=\sqrt{2/768}\approx0.051$。",
                r"直觉：ReLU 让一半输出为 0，信号方差减半，所以 He 初始化的方差是 $1/n_{in}$ 的 2 倍。",
                r"如果误用 $\mathcal N(0,1)$ 初始化，每层输出方差约放大 $n_{in}/2=256$ 倍，几层之后就会数值爆炸。",
            ],
            answer="He：0.0625；Xavier：约 0.051。",
            code="ch07_ex_adam.py",
        ),
        dict(
            title="根据训练曲线诊断问题",
            problem=r"三次实验的结果：(A) 训练准确率 99%、验证 70%；(B) 训练 62%、验证 60%；(C) 训练几步后损失变成 NaN。分别是什么问题？怎么改？",
            steps=[
                r"(A) 训练与验证差距大：**过拟合**。加 L2/Dropout、数据增强、早停，或获取更多数据。",
                r"(B) 两者都低且接近：**欠拟合**。加大模型（更深、更宽）、训练更久、减小正则化、改进特征。",
                r"(C) NaN：多半是**学习率过大或梯度爆炸**，也可能是 $\log 0$ 之类的数值问题。降低学习率、加梯度裁剪、检查损失函数的实现。",
                r"通用做法：先在 100 个样本上确认模型能把训练损失降到接近 0，排除流水线 bug。",
            ],
            answer="A 过拟合，B 欠拟合，C 学习率过大/梯度爆炸。",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="sigmoid 函数导数的最大值是：", options=["1", "0.5", "0.25", "0.1"], answer=2, explain="在 $z=0$ 处 $\\sigma'=0.25$。"),
            dict(q="ReLU 网络推荐使用的初始化方法是：", options=["全零初始化", "He 初始化", "均匀分布 [0,1]", "Xavier 初始化（不加修正）"], answer=1, explain="$\\mathrm{Var}=2/n_{in}$。"),
            dict(q="Adam 优化器结合了哪两种思想？", options=["动量与 RMSProp", "牛顿法与 SGD", "Dropout 与 BatchNorm", "L1 与 L2"], answer=0, explain="一阶矩 + 二阶矩自适应缩放。"),
            dict(q="Dropout 在推理（测试）阶段应该：", options=["保持相同的丢弃概率", "关闭", "提高丢弃概率", "只作用于输出层"], answer=1, explain="推理时使用完整网络。"),
            dict(q="批量归一化在推理时使用的均值和方差来自：", options=["当前测试批", "训练期间累积的滑动平均", "整个测试集", "随机生成"], answer=1, explain="保证单样本推理结果确定。"),
            dict(q="训练准确率 99%、验证准确率 70%，首先应该尝试：", options=["增大模型", "加强正则化或增加数据", "提高学习率", "减少训练数据"], answer=1, explain="典型过拟合。"),
        ],
        tf=[
            dict(q="动量法可以减轻在狭长峡谷中的“之”字形震荡。", answer=True, explain="震荡方向的速度相互抵消。"),
            dict(q="梯度裁剪主要用于解决梯度消失问题。", answer=False, explain="梯度裁剪用于防止梯度爆炸。"),
            dict(q="Inverted dropout 在训练时把保留的激活除以保留概率，以保持期望不变。", answer=True, explain="这样推理时无需再缩放。"),
            dict(q="批量归一化在训练和推理时的计算方式完全一样。", answer=False, explain="训练用批统计量，推理用滑动平均。"),
            dict(q="Buduma 认为，深度网络训练中更麻烦的往往是平坦区域、鞍点和病态曲率，而不是坏的局部极小值。", answer=True, explain="见《深度学习基础》第 6 章。"),
        ],
        fill=[
            dict(q="He 初始化的权重方差为 $2/$____。", answer=["n_in", "nin", "fan_in", "输入数", "输入神经元数"], explain="$2/n_{in}$。"),
            dict(q="验证损失不再下降时停止训练的方法叫 ____。", answer=["早停", "提前停止", "early stopping"], explain="early stopping。"),
            dict(q="Adam 默认的 $\\beta_1$ 是 0.9，$\\beta_2$ 是 ____。", answer=["0.999"], explain="二阶矩的衰减率。"),
            dict(q="Transformer 中常用的归一化方式是 ____ 归一化（中文或英文）。", answer=["层", "LayerNorm", "layer", "layer normalization"], explain="LayerNorm 不依赖批大小。"),
        ],
        reading=[
            dict(
                title="Why Deeper Was Not Better",
                passage="""In the early 2000s, a frustrating pattern kept appearing. Researchers would add more hidden layers to a network, expecting it to learn richer features, and instead the deeper network trained more slowly and often ended up worse. The culprit was the way gradients travel backward.

During backpropagation, the gradient that reaches an early layer is a product of many factors, one for each layer above it. If those factors are typically smaller than one, as they are for saturated sigmoid units, the product shrinks exponentially and the early layers barely learn. If the factors are larger than one, the gradient explodes instead. Either way, deep stacks of layers are unstable by default.

The fixes came in pieces. Activation functions such as ReLU do not saturate for positive inputs. Careful initialization keeps the variance of signals roughly constant from layer to layer. Batch normalization re-centers activations during training, and residual connections give gradients a shortcut. None of these ideas is complicated on its own, but together they turned very deep networks from a curiosity into the standard tool of modern machine learning.""",
                zh="""在 21 世纪初，一个令人沮丧的现象反复出现。研究者给网络增加更多隐藏层，期望它学到更丰富的特征，结果更深的网络训练得更慢，最终往往还更差。罪魁祸首是梯度向后传递的方式。

反向传播时，到达浅层的梯度是许多因子的乘积，上面的每一层贡献一个因子。如果这些因子通常小于 1（饱和的 sigmoid 单元正是如此），乘积会指数级缩小，浅层几乎学不到东西；如果因子大于 1，梯度反而会爆炸。无论哪种情况，深层堆叠在默认情况下都是不稳定的。

解决办法是一点点出现的。ReLU 这样的激活函数对正输入不会饱和；精心的初始化让信号的方差在层与层之间大致保持不变；批量归一化在训练中重新校准激活值；残差连接则给梯度提供了一条捷径。这些想法单独看都不复杂，但它们合在一起，把非常深的网络从一种稀奇之物变成了现代机器学习的标准工具。""",
                questions=[
                    dict(q="What happens to the gradient when the per-layer factors are usually smaller than one?", options=["It explodes", "It shrinks exponentially", "It stays constant", "It becomes negative"], answer=1, explain="小于 1 的因子连乘 → 梯度消失。"),
                    dict(q="Which fix is described as keeping signal variance roughly constant across layers?", options=["Residual connections", "Careful initialization", "Dropout", "Data augmentation"], answer=1, explain="初始化的作用。"),
                    dict(q="What is the main message of the last paragraph?", options=["One single trick solved deep learning", "Several simple ideas combined made deep networks trainable", "Deep networks are still unusable", "Sigmoid is the best activation"], answer=1, explain="多个简单想法的组合。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("vanishing gradient", "梯度消失", "Gradients that shrink toward zero as they flow back through many layers.", "梯度在反向穿过许多层时缩小到接近零。", "Vanishing gradients stopped the early layers from learning."),
        ("exploding gradient", "梯度爆炸", "Gradients that grow uncontrollably large during backpropagation.", "反向传播中梯度失控地变得极大。", "Gradient clipping is a standard fix for exploding gradients."),
        ("weight initialization", "权重初始化", "The scheme for setting network weights before training starts.", "训练开始前设定网络权重的方案。", "Good weight initialization keeps activations in a healthy range."),
        ("He initialization", "He 初始化", "Drawing weights with variance two over fan-in, suited to ReLU.", "以方差 2/输入数 采样权重，适合 ReLU。", "We used He initialization for every convolution layer."),
        ("momentum", "动量", "Accumulating past gradients so that updates gain speed in consistent directions.", "累积历史梯度，使更新在一致方向上加速。", "Momentum smooths out the zigzag path of plain SGD."),
        ("Adam", "Adam 优化器", "An optimizer combining momentum with per-parameter adaptive learning rates.", "结合动量与逐参数自适应学习率的优化器。", "Adam is a reliable default optimizer."),
        ("learning rate schedule", "学习率调度", "A plan for changing the learning rate during training.", "训练过程中改变学习率的计划。", "A cosine learning rate schedule slowly lowers the step size."),
        ("saddle point", "鞍点", "A point where the gradient is zero but which is a minimum in some directions and a maximum in others.", "梯度为零、但在某些方向是极小在另一些方向是极大的点。", "High-dimensional loss surfaces contain many saddle points."),
        ("weight decay", "权重衰减", "Shrinking weights a little at every step, equivalent to L2 regularization.", "每一步都把权重缩小一点，等价于 L2 正则化。", "Weight decay keeps the weights small."),
        ("dropout", "随机失活", "Randomly turning off neurons during training to prevent co-adaptation.", "训练时随机关闭神经元以防止相互依赖。", "Dropout acts like training an ensemble of thinned networks."),
        ("early stopping", "早停", "Stopping training when validation performance stops improving.", "验证集表现不再提升时停止训练。", "Early stopping saved the best checkpoint at epoch twelve."),
        ("label smoothing", "标签平滑", "Softening one-hot targets so the model is less overconfident.", "软化 one-hot 目标，使模型不过度自信。", "Label smoothing slightly improved calibration."),
        ("layer normalization", "层归一化", "Normalizing across the features of each individual sample.", "在单个样本的各特征之间做归一化。", "Transformers rely on layer normalization."),
        ("gradient clipping", "梯度裁剪", "Rescaling gradients whose norm exceeds a threshold.", "把范数超过阈值的梯度按比例缩小。", "Gradient clipping kept the recurrent network stable."),
    ],
    cards=[
        ("Xavier vs He 初始化？", r"Xavier：$\mathrm{Var}=1/n_{in}$（tanh/sigmoid）；He：$\mathrm{Var}=2/n_{in}$（ReLU）。"),
        ("Adam 更新公式？", r"$\mathbf w\leftarrow\mathbf w-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$，$\hat m,\hat v$ 为偏差校正后的一阶、二阶矩。"),
        ("Dropout 训练和推理有什么不同？", "训练时随机失活并按 $1/(1-p)$ 放大；推理时关闭，使用完整网络。"),
        ("BatchNorm 公式？", r"$y=\gamma\frac{z-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta$；推理用滑动平均。"),
    ],
)
