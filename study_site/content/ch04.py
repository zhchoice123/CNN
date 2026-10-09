CHAPTER = dict(
    no=4,
    title="线性模型、正则化与梯度下降",
    title_en="Linear Models, Regularization and Gradient Descent",
    group="经典机器学习",
    summary="线性回归和逻辑回归是理解一切模型的起点。本章讲损失函数与最大似然、梯度下降的三种变体与学习率、岭回归与 Lasso 正则化，以及逻辑回归和 softmax 回归。",
    sources=["ISLR 第 3、4、6 章", "Bishop《PRML》第 3–4 章", "MML 第 7、9 章", "Géron《机器学习实战》第 4 章", "Flach 第 7 章", "邱锡鹏 第 3 章"],
    learn=[
        dict(
            title="线性回归、损失函数与最大似然",
            title_en="Linear regression and maximum likelihood",
            body=r"""
线性回归假设 $\hat y=\mathbf w^T\mathbf x+b$。把截距并入权重（给 $\mathbf x$ 添一个常数 1），对 $n$ 个样本就是 $\hat{\mathbf y}=X\mathbf w$。用**均方误差**衡量好坏：

$$L(\mathbf w)=\frac1n\|X\mathbf w-\mathbf y\|^2$$

令梯度为零得到闭式解（正规方程）$\mathbf w^*=(X^TX)^{-1}X^T\mathbf y$，这正是第 1 章的投影。

为什么用平方误差？从概率角度看：假设 $y=\mathbf w^T\mathbf x+\varepsilon$，噪声 $\varepsilon\sim\mathcal N(0,\sigma^2)$，那么最大化数据的**似然**，取对数后恰好等价于最小化平方误差。所以**最小二乘 = 高斯噪声下的最大似然估计**（Bishop、MML 第 9 章）。

“线性”指对**参数**线性，而不是对输入线性。用**基函数** $\phi(\mathbf x)$（多项式、高斯函数等）先变换输入，线性模型同样能拟合非线性关系。ISLR 还强调了几个实践问题：交互项、类别变量的哑变量编码、异方差、离群点和**多重共线性**。
""",
            points=[
                "闭式解需要求 $X^TX$ 的逆，特征数很大时代价为 $O(d^3)$，这时改用梯度下降。",
                "最小二乘 ⇔ 高斯噪声下的最大似然；L1 损失 ⇔ 拉普拉斯噪声，对离群点更鲁棒。",
                "$R^2$ 衡量模型解释了多少方差；系数的显著性检验用于推断而非预测。",
                "特征尺度相差很大时要先标准化，否则梯度下降会很慢。",
            ],
            src="ISLR 第 3 章；Bishop 3.1；MML 9.2；Géron 第 4 章",
        ),
        dict(
            title="梯度下降：批量、随机与小批量",
            title_en="Gradient descent and its variants",
            body=r"""
梯度指向函数增长最快的方向，所以沿负梯度走一小步就能让损失下降：

$$\mathbf w\leftarrow\mathbf w-\eta\,\nabla L(\mathbf w)$$

$\eta$ 是**学习率**。对均方误差，$\nabla L=\frac2n X^T(X\mathbf w-\mathbf y)$。

- **批量梯度下降**：每步用全部数据计算梯度，稳定但慢。
- **随机梯度下降（SGD）**：每步只用一个样本，快且能跳出浅的局部极小，但路径抖动。
- **小批量梯度下降**：每步用一小批（如 32、64 个）样本，兼顾效率与稳定，也最适合 GPU 并行，是深度学习的默认选择。

学习率太小收敛慢；太大会来回震荡甚至**发散**。对二次型损失，当 $\eta>2/\lambda_{\max}$（$\lambda_{\max}$ 为海森矩阵的最大特征值）时必然发散。各个特征尺度差异大时，损失曲面是狭长的“峡谷”，梯度下降会走“之”字形，这就是要做**特征标准化**的原因。

线性回归的损失是**凸**函数，只有一个全局最小值，梯度下降一定能找到它；神经网络的损失非凸，问题会复杂得多（第 7 章）。
""",
            points=[
                "一轮（epoch）= 把全部训练数据用一遍。",
                "学习率是最重要的超参数，通常从 0.1、0.01、0.001 这样按 10 倍尝试。",
                "学习率调度：训练后期减小学习率，让参数在最小值附近稳定下来。",
                "凸函数的任何局部最小值都是全局最小值。",
            ],
            code="ch04_gradient_descent.py",
            code_note="比较正规方程、不同学习率的批量梯度下降和小批量 SGD，观察学习率过大时的发散。",
            src="Géron 第 4 章；MML 第 7 章；Buduma 第 4 章；Patterson & Gibson 第 1 章“优化”",
        ),
        dict(
            title="正则化：岭回归与 Lasso",
            title_en="Regularization: ridge and lasso",
            body=r"""
特征多、样本少时，最小二乘会过拟合，系数变得很大且不稳定。**正则化**在损失上加一项惩罚，把系数往 0 的方向“收缩”：

$$\text{岭回归：}\ L+\lambda\|\mathbf w\|_2^2\qquad\qquad \text{Lasso：}\ L+\lambda\|\mathbf w\|_1$$

- **岭回归**（L2）有闭式解 $(X^TX+\lambda I)^{-1}X^T\mathbf y$。加上 $\lambda I$ 后矩阵总是可逆的，所以也能处理多重共线性。系数整体变小，但不会恰好为 0。
- **Lasso**（L1）能把不重要特征的系数压成**恰好为 0**，同时完成特征选择。几何解释：L1 约束区域是菱形，最优点常常落在坐标轴的尖角上。
- **弹性网络**同时使用 L1 和 L2。

贝叶斯解释：给权重加高斯先验，最大后验估计就是岭回归；加拉普拉斯先验就是 Lasso。$\lambda$ 是超参数，用交叉验证选择。正则化前一定要先**标准化**特征，否则惩罚对不同尺度的特征不公平。

在神经网络里，L2 正则就叫**权重衰减**，还有早停、Dropout 等其他正则化手段（第 7 章）。
""",
            points=[
                "$\\lambda\\to0$ 退化为最小二乘，$\\lambda\\to\\infty$ 所有系数趋于 0。",
                "岭回归降低方差、略微增加偏差，是偏差—方差权衡的直接应用。",
                "Lasso 的稀疏解让模型更容易解释。",
                "早停也是一种正则化：在验证误差开始上升时停止训练。",
            ],
            code="ch04_ridge_lasso.py",
            code_note="8 个特征里只有 3 个有用，比较最小二乘、岭回归和 Lasso（坐标下降实现）的系数。",
            src="ISLR 第 6 章；Bishop 3.1.4；MML 9.2.3；Géron 第 4 章“正则化线性模型”",
        ),
        dict(
            title="逻辑回归与 softmax 回归",
            title_en="Logistic and softmax regression",
            body=r"""
分类为什么不能直接用线性回归？因为输出不是概率，也会被离群点拉偏。**逻辑回归**把线性得分 $z=\mathbf w^T\mathbf x+b$ 送进 **sigmoid** 函数：

$$p(y=1\mid\mathbf x)=\sigma(z)=\frac{1}{1+e^{-z}}$$

等价地，$z=\log\dfrac{p}{1-p}$，即线性模型拟合的是**对数几率**（log-odds）。系数 $w_j$ 的含义：$x_j$ 增加 1，几率乘以 $e^{w_j}$。

损失用**交叉熵**（负对数似然）：$L=-\frac1n\sum[y\log p+(1-y)\log(1-p)]$。它是凸的，梯度形式非常简洁：$\nabla L=\frac1n X^T(\mathbf p-\mathbf y)$，与线性回归的形式完全一样。

多分类用 **softmax 回归**：每个类别一个得分 $z_k$，概率 $p_k=e^{z_k}/\sum_j e^{z_j}$，损失仍是交叉熵，梯度为 $X^T(P-Y)/n$。神经网络分类器的最后一层，就是一个 softmax 回归。

Bishop 把分类方法分成三类：直接给出决策边界的**判别函数**（如感知机）、先建模 $p(\mathbf x\mid y)$ 再用贝叶斯定理的**生成模型**（如 LDA、朴素贝叶斯），以及直接建模 $p(y\mid\mathbf x)$ 的**判别模型**（如逻辑回归）。
""",
            points=[
                "逻辑回归是线性分类器：决策边界 $\\mathbf w^T\\mathbf x+b=0$ 是一个超平面。",
                "数据线性可分时，逻辑回归的权重会无限增大，需要正则化。",
                "softmax 计算时先减去最大得分，防止指数溢出（数值稳定技巧）。",
                "实现完梯度后，用数值梯度做**梯度检验**，是排查错误的好习惯。",
            ],
            code="ch04_softmax_gradcheck.py",
            code_note="实现 softmax 回归的交叉熵损失与解析梯度，用中心差分做梯度检验。",
            src="ISLR 第 4 章；Bishop 第 4 章；Géron 第 4 章；邱锡鹏 3.2–3.3；Patterson & Gibson 第 1 章“逻辑回归”",
        ),
    ],
    examples=[
        dict(
            title="手算几步梯度下降",
            problem=r"对 $f(w)=(w-3)^2$，从 $w_0=0$ 出发，学习率 $\eta=0.1$，计算前三步，并分析学习率为 0.5、1.0、1.1 时会发生什么。",
            steps=[
                r"$f'(w)=2(w-3)$。第 1 步：梯度 $-6$，$w_1=0-0.1\times(-6)=0.6$。",
                r"第 2 步：梯度 $2(0.6-3)=-4.8$，$w_2=1.08$；第 3 步：梯度 $-3.84$，$w_3=1.464$。",
                r"一般地 $w_{t+1}-3=(1-2\eta)(w_t-3)$，误差每步乘以 $1-2\eta=0.8$。",
                r"$\eta=0.5$ 时因子为 0，一步到位；$\eta=1.0$ 时因子为 $-1$，在 0 和 6 之间来回跳；$\eta=1.1$ 时 $|{-1.2}|>1$，发散。",
            ],
            answer=r"$w_1=0.6,\ w_2=1.08,\ w_3=1.464$；$\eta<1$ 收敛，$\eta=1$ 震荡，$\eta>1$ 发散。",
            code="ch04_ex_gd_steps.py",
        ),
        dict(
            title="解读逻辑回归的系数",
            problem=r"某逻辑回归模型 $z=-0.37+2.25x_1+1.97x_2$。(1) 样本 $x=(0.5,0)$ 属于正类的概率是多少？(2) $x_1$ 增加 1 对几率有什么影响？",
            steps=[
                r"(1) $z=-0.37+2.25\times0.5=0.755$，$p=\sigma(0.755)=1/(1+e^{-0.755})\approx0.68$。",
                r"(2) 对数几率 $\log\frac{p}{1-p}=z$，$x_1$ 增加 1 时 $z$ 增加 2.25。",
                r"所以几率乘以 $e^{2.25}\approx9.5$。注意是几率翻了约 9.5 倍，不是概率。",
                r"决策边界是 $-0.37+2.25x_1+1.97x_2=0$，一条直线。",
            ],
            answer="概率约 0.68；$x_1$ 每增加 1，几率约乘以 9.5。",
            code="ch04_logistic.py",
        ),
        dict(
            title="岭回归如何收缩系数",
            problem=r"只有一个特征（已中心化）时，岭回归的解是什么？若 $\sum x_iy_i=50$，$\sum x_i^2=25$，分别求 $\lambda=0$ 和 $\lambda=25$ 时的系数。",
            steps=[
                r"损失 $\sum(y_i-wx_i)^2+\lambda w^2$，对 $w$ 求导令其为 0：$-2\sum x_i(y_i-wx_i)+2\lambda w=0$。",
                r"解得 $w=\dfrac{\sum x_iy_i}{\sum x_i^2+\lambda}$。",
                r"$\lambda=0$：$w=50/25=2$（普通最小二乘）。$\lambda=25$：$w=50/50=1$，系数被收缩了一半。",
                r"$\lambda$ 越大，分母越大，$w$ 越接近 0，但永远不会恰好等于 0——这正是 L2 与 L1 的区别。",
            ],
            answer=r"$w=\sum x_iy_i/(\sum x_i^2+\lambda)$；$\lambda=0$ 时为 2，$\lambda=25$ 时为 1。",
            code="ch04_ridge_lasso.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="在高斯噪声假设下，最大似然估计等价于最小化：", options=["绝对误差", "平方误差", "交叉熵", "合页损失"], answer=1, explain="高斯对数似然中只剩平方误差项。"),
            dict(q="能产生稀疏解（部分系数恰好为 0）的是：", options=["岭回归", "Lasso", "普通最小二乘", "加大学习率"], answer=1, explain="L1 惩罚的菱形约束。"),
            dict(q="深度学习中最常用的梯度下降变体是：", options=["批量梯度下降", "单样本 SGD", "小批量梯度下降", "牛顿法"], answer=2, explain="兼顾效率、稳定与 GPU 并行。"),
            dict(q="逻辑回归中 $\\sigma(z)$ 的取值范围是：", options=["$(-\\infty,\\infty)$", "$[0,\\infty)$", "$(0,1)$", "$\\{0,1\\}$"], answer=2, explain="sigmoid 把实数映射到 (0,1)。"),
            dict(q="学习率过大最可能导致：", options=["收敛更稳定", "损失震荡或发散", "模型欠拟合", "正则化增强"], answer=1, explain="步子太大会越过最小值。"),
            dict(q="逻辑回归的交叉熵损失关于权重的梯度是：", options=["$X^T(\\mathbf p-\\mathbf y)/n$", "$X(\\mathbf p-\\mathbf y)$", "$\\mathbf p\\odot(1-\\mathbf p)$", "$\\mathbf y-\\mathbf p$"], answer=0, explain="与线性回归形式相同。"),
        ],
        tf=[
            dict(q="线性回归的“线性”是指模型对参数是线性的。", answer=True, explain="用基函数变换输入后仍是线性模型。"),
            dict(q="岭回归可以把系数压缩到恰好为 0。", answer=False, explain="L2 只会让系数变小，不会恰好为 0。"),
            dict(q="正则化之前应当对特征进行标准化。", answer=True, explain="否则惩罚对不同尺度的特征不公平。"),
            dict(q="逻辑回归的交叉熵损失是凸函数。", answer=True, explain="所以梯度下降能找到全局最优。"),
            dict(q="softmax 输出的各类概率之和等于 1。", answer=True, explain="分母是所有类别指数之和。"),
        ],
        fill=[
            dict(q="参数更新公式：$\\mathbf w\\leftarrow\\mathbf w-\\eta\\,$____。", answer=["梯度", "∇L", "gradient", "grad"], explain="沿负梯度方向更新。"),
            dict(q="L2 正则化在神经网络中又被称为权重 ____。", answer=["衰减", "decay"], explain="weight decay。"),
            dict(q="逻辑回归中，线性得分 $z$ 等于正类的对数 ____。", answer=["几率", "odds", "log-odds", "赔率"], explain="$z=\\log\\frac{p}{1-p}$。"),
            dict(q="把全部训练数据完整使用一遍称为一个 ____（英文）。", answer=["epoch", "轮", "周期"], explain="epoch。"),
        ],
        reading=[
            dict(
                title="Walking Downhill in the Fog",
                passage="""Picture yourself on a mountainside covered in thick fog. You want to reach the valley floor, but you can only feel the slope directly under your feet. A sensible strategy is to take a small step in the steepest downhill direction, feel the ground again, and repeat. This is gradient descent, the workhorse behind almost every modern learning algorithm.

The size of each step matters a great deal. Tiny steps are safe but painfully slow. Huge steps can carry you across the valley and higher up the opposite slope, so that every move makes things worse. In practice, people try a few learning rates that differ by a factor of ten and watch how the loss behaves.

Computing the exact slope requires looking at every training example, which is expensive when there are millions of them. Stochastic gradient descent estimates the slope from a small random batch instead. Each estimate is noisy, but the noise averages out over many steps, and the computer can take far more steps in the same amount of time. The jitter can even help the search escape shallow dips that are not the true bottom.""",
                zh="""想象你站在一座被浓雾笼罩的山坡上。你想走到谷底，却只能感觉到脚下的坡度。一个合理的策略是：朝最陡的下坡方向迈一小步，再感受一下地面，然后重复。这就是梯度下降，几乎所有现代学习算法背后的主力。

每一步的大小非常重要。步子很小很安全，但慢得令人痛苦；步子太大可能越过谷底，走到对面更高的坡上，于是每一步都让情况更糟。实践中，人们会尝试几个相差 10 倍的学习率，观察损失的变化。

计算精确的坡度需要查看每一个训练样本，当样本有数百万个时代价很高。随机梯度下降改为从一小批随机样本中估计坡度。每次估计都有噪声，但噪声会在多步中相互抵消，而计算机在同样时间内能走更多步。这种抖动甚至能帮助搜索跳出并非真正谷底的浅坑。""",
                questions=[
                    dict(q="In the analogy, what does the fog represent?", options=["The learning rate", "Only knowing the local slope, not the whole landscape", "Overfitting", "The training data"], answer=1, explain="雾代表只能感知局部梯度。"),
                    dict(q="What can happen when the steps are too large?", options=["Training becomes very slow", "You may overshoot and the loss gets worse", "The gradient becomes zero", "The model underfits"], answer=1, explain="步子太大会越过谷底。"),
                    dict(q="Why is stochastic gradient descent faster in practice?", options=["It computes the exact gradient", "It uses small batches, so it can take many more steps", "It needs no learning rate", "It always finds the global minimum"], answer=1, explain="小批量估计梯度，单位时间步数更多。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("linear regression", "线性回归", "Modeling a continuous target as a weighted sum of input features.", "把连续目标建模为输入特征的加权和。", "Linear regression is the first model most students learn."),
        ("loss function", "损失函数", "A function measuring how wrong the model's predictions are.", "衡量模型预测有多错误的函数。", "Mean squared error is the usual loss function for regression."),
        ("mean squared error", "均方误差", "The average of the squared differences between predictions and targets.", "预测值与目标值之差的平方的平均。", "The mean squared error dropped after normalization."),
        ("maximum likelihood", "最大似然", "Choosing parameters that make the observed data most probable.", "选择使观测数据概率最大的参数。", "Least squares is maximum likelihood under Gaussian noise."),
        ("gradient descent", "梯度下降", "Iteratively moving parameters against the gradient to reduce the loss.", "沿梯度反方向迭代更新参数以减小损失。", "Gradient descent converges slowly when features are not scaled."),
        ("learning rate", "学习率", "The step size used in each gradient update.", "每次梯度更新所用的步长。", "A learning rate that is too large makes training diverge."),
        ("stochastic gradient descent", "随机梯度下降", "Gradient descent that estimates the gradient from random samples or mini-batches.", "用随机样本或小批量估计梯度的梯度下降。", "Stochastic gradient descent scales to millions of examples."),
        ("mini-batch", "小批量", "A small random subset of training examples used for one update.", "一次更新所用的一小部分随机训练样本。", "We used a mini-batch size of sixty-four."),
        ("convex function", "凸函数", "A function whose every local minimum is a global minimum.", "任何局部最小值都是全局最小值的函数。", "The logistic regression loss is a convex function."),
        ("regularization", "正则化", "Adding a penalty or constraint to reduce overfitting.", "通过加入惩罚或约束来减少过拟合。", "Regularization trades a little bias for much lower variance."),
        ("ridge regression", "岭回归", "Linear regression with an L2 penalty on the weights.", "对权重施加 L2 惩罚的线性回归。", "Ridge regression handles correlated features gracefully."),
        ("lasso", "Lasso 回归", "Linear regression with an L1 penalty that can set weights exactly to zero.", "带 L1 惩罚、能把权重压为零的线性回归。", "The lasso kept only three of the eight features."),
        ("logistic regression", "逻辑回归", "A linear classifier that outputs probabilities through the sigmoid function.", "通过 sigmoid 函数输出概率的线性分类器。", "Logistic regression is a strong baseline for tabular data."),
        ("softmax", "softmax 函数", "A function turning a vector of scores into probabilities that sum to one.", "把得分向量转换为和为一的概率的函数。", "The softmax layer outputs a probability for each class."),
    ],
    cards=[
        ("线性回归闭式解？", r"$\mathbf w^*=(X^TX)^{-1}X^T\mathbf y$"),
        ("岭回归 vs Lasso？", "L2 让系数整体收缩、不为 0；L1 产生稀疏解，可做特征选择。"),
        ("sigmoid 与对数几率？", r"$\sigma(z)=\frac1{1+e^{-z}}$，$z=\log\frac{p}{1-p}$"),
        ("学习率太大会怎样？", r"震荡或发散；二次损失下 $\eta>2/\lambda_{\max}$ 必然发散。"),
    ],
)
