CHAPTER = dict(
    no=6,
    title="神经网络与反向传播",
    title_en="Neural Networks and Backpropagation",
    group="深度学习",
    summary="从感知机到多层网络：神经元如何加权求和再通过非线性激活，为什么一层不够、为什么非线性必不可少，反向传播如何用链式法则高效算出所有梯度，以及万能近似定理告诉了我们什么。",
    sources=["Nielsen《神经网络与深度学习》第 1–2、4 章", "Kelleher《深度学习》第 3–4、6 章", "邱锡鹏《神经网络与深度学习》第 4 章", "Buduma《深度学习基础》第 3–4 章", "Patterson & Gibson 第 2 章", "Elgendy 第 2 章"],
    learn=[
        dict(
            title="从生物神经元到感知机",
            title_en="From neurons to the perceptron",
            body=r"""
人工神经元是对生物神经元的极简抽象：接收若干输入，各乘一个**权重**，加上**偏置**，再通过一个**激活函数**：

$$a=f\Big(\sum_i w_ix_i+b\Big)=f(\mathbf w^T\mathbf x+b)$$

1958 年罗森布拉特的**感知机**用阶跃函数作激活，输出 0 或 1。学习规则非常简单：预测错了，就把权重往正确的方向挪一点 $\mathbf w\leftarrow\mathbf w+\eta(y-\hat y)\mathbf x$。只要数据线性可分，感知机一定在有限步内收敛。

Nielsen 举了一个生活化的例子：周末要不要去奶酪节？取决于天气好不好、伴侣去不去、交通方不方便——每个因素一个权重，你最在意的那个权重最大，偏置则表示你有多想去。

但 1969 年明斯基和派珀特指出：单个感知机连**异或（XOR）**都学不会，因为 XOR 不是线性可分的——没有一条直线能把 (0,1)、(1,0) 与 (0,0)、(1,1) 分开。这一结论让神经网络研究陷入了第一次寒冬（Kelleher 第 4 章）。解决办法是**多层**：一个隐藏层就能把 XOR 变成线性可分的问题，难点在于怎么训练多层网络——答案就是反向传播。
""",
            points=[
                "单个神经元 = 线性模型 + 激活函数；sigmoid 神经元就是逻辑回归。",
                "感知机只能学习线性可分的函数（AND、OR 可以，XOR 不行）。",
                "阶跃函数的导数几乎处处为 0，无法用梯度下降，所以现代网络使用平滑或分段线性的激活函数。",
                "神经网络发展史：阈值逻辑单元 → 感知机 → 第一次寒冬 → 反向传播 → 第二次低谷 → 2006 年后的深度学习。",
            ],
            code="ch06_perceptron.py",
            code_note="用感知机学习规则训练 AND、OR 和 XOR，验证 XOR 无法收敛。",
            src="Nielsen 1.1；Kelleher 第 4 章；Elgendy 2.1；Patterson & Gibson 2.1",
        ),
        dict(
            title="多层网络、激活函数与前向传播",
            title_en="Multilayer networks and activation functions",
            body=r"""
把神经元排成层：输入层 → 一个或多个**隐藏层** → 输出层。每一层的计算是

$$\mathbf z^{(l)}=W^{(l)}\mathbf a^{(l-1)}+\mathbf b^{(l)},\qquad \mathbf a^{(l)}=f\big(\mathbf z^{(l)}\big)$$

逐层计算直到输出，就是**前向传播**。用矩阵写，一整个小批量可以一次算完，这正是 GPU 擅长的。

**为什么需要非线性激活？** 如果每层都是线性的，多层叠加仍然等价于一个线性变换，深度就毫无意义（Kelleher）。常用激活函数：

| 函数 | 公式 | 特点 |
|---|---|---|
| sigmoid | $1/(1+e^{-z})$ | 输出 (0,1)；两端饱和，梯度易消失 |
| tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | 输出 (−1,1)，零中心 |
| ReLU | $\max(0,z)$ | 计算快、不饱和；负区间梯度为 0 |
| Leaky ReLU | $\max(\alpha z,z)$ | 缓解“死亡 ReLU” |
| softmax | $e^{z_k}/\sum_j e^{z_j}$ | 多分类输出层 |

输出层与损失函数要搭配：回归用线性输出 + 均方误差；二分类用 sigmoid + 二元交叉熵；多分类用 softmax + 交叉熵。Kelleher 还强调了三个“空间”：输入空间、权重空间与激活空间——学习就是在权重空间中寻找最好的点，而每个隐藏层都把数据变换到一个新的激活空间，使问题越来越容易分开。
""",
            points=[
                "参数量：每层 $(\\text{输入数}+1)\\times\\text{输出数}$；784-100-10 网络共 79,510 个参数。",
                "隐藏层学到的是数据的新表示；深度学习的本质是**表示学习**。",
                "现代网络的隐藏层默认使用 ReLU 或其变体（GELU、SiLU 等）。",
                "隐藏层的层数和宽度是超参数，通常从较小的网络开始，再逐步加大。",
            ],
            code="ch06_ex_forward.py",
            code_note="手算一个 2-2-1 网络的前向传播，并计算 784-100-10 网络的参数量。",
            src="Nielsen 1.2–1.4；邱锡鹏 4.1–4.3；Elgendy 2.2–2.4；Kelleher 第 2–3 章",
        ),
        dict(
            title="反向传播：链式法则的高效实现",
            title_en="Backpropagation",
            body=r"""
训练网络需要损失对**每一个**权重的梯度。最朴素的办法是对每个权重做微小扰动看损失变化，但一个有一百万个权重的网络就需要一百万次前向计算。**反向传播**只需一次前向、一次反向，就能算出全部梯度（Nielsen 2.8）。

定义第 $l$ 层的**误差** $\boldsymbol\delta^{(l)}=\partial L/\partial\mathbf z^{(l)}$。Nielsen 把反向传播总结为四个基本方程：

$$\boldsymbol\delta^{(L)}=\nabla_{\mathbf a}L\odot f'(\mathbf z^{(L)})$$
$$\boldsymbol\delta^{(l)}=\big(W^{(l+1)}\big)^T\boldsymbol\delta^{(l+1)}\odot f'(\mathbf z^{(l)})$$
$$\frac{\partial L}{\partial \mathbf b^{(l)}}=\boldsymbol\delta^{(l)},\qquad \frac{\partial L}{\partial W^{(l)}}=\boldsymbol\delta^{(l)}\big(\mathbf a^{(l-1)}\big)^T$$

直观地说：先算出输出层“错了多少”，再沿着网络**反向**把误差按权重分摊给前一层，每一层的权重梯度 = 本层误差 × 上一层的激活。本质上就是**链式法则**，再加上对中间结果的复用。

现代框架用**自动微分**实现反向传播：前向时记录计算图，反向时自动应用链式法则。尽管如此，亲手推导并用**梯度检验**（与数值梯度比较）验证一次，是真正理解它的最好方式。

> 一个容易被忽略的细节：sigmoid + 交叉熵组合时，输出层误差化简为 $\mathbf p-\mathbf y$，导数项被消掉了——这正是交叉熵比平方误差学得快的原因（Nielsen 第 3 章）。
""",
            points=[
                "前向传播保存每层的 $\\mathbf z$ 和 $\\mathbf a$，反向传播要用到它们。",
                "反向传播的计算量与前向传播同一量级，这是它“快”的原因。",
                "梯度检验用中心差分 $\\frac{L(w+\\epsilon)-L(w-\\epsilon)}{2\\epsilon}$，相对误差应小于 $10^{-6}$ 左右。",
                "初始化不能全为零，否则同层神经元的梯度完全相同，永远学到一样的东西（对称性问题）。",
            ],
            code="ch06_gradcheck.py",
            code_note="按四个基本方程手写一个 ReLU 两层网络的反向传播，并逐个参数与数值梯度对比。",
            src="Nielsen 第 2 章；Kelleher 第 6 章；邱锡鹏 4.4–4.5；Buduma 第 4 章",
        ),
        dict(
            title="训练一个网络，以及万能近似定理",
            title_en="Training a network and universal approximation",
            body=r"""
把前面的部件组装起来，训练循环是：

1. 取一个小批量数据，做**前向传播**得到预测；
2. 计算**损失**；
3. **反向传播**得到梯度；
4. 用梯度下降**更新参数**；
5. 重复直到验证集表现不再提升。

用一个只有 8 个隐藏神经元的两层网络，就能在几千步内学会感知机学不会的 XOR。不过训练并不总是顺利：同样的代码如果只用 4 个隐藏神经元，某些随机初始化会卡在“两个样本永远输出 0.5”的局部解上——这提醒我们，神经网络的损失是**非凸**的，初始化和网络宽度都会影响结果。Nielsen 用不到 100 行 Python 实现的网络，在 MNIST 手写数字上就能达到约 95% 的准确率。

**万能近似定理**：只要隐藏神经元足够多，单隐藏层网络能以任意精度逼近任何连续函数。Nielsen 给了一个漂亮的图示证明：两个很陡的 sigmoid 相减得到一个“台阶”，许多台阶拼起来就能近似任意曲线。但他也提醒两点：这个定理只保证**能表示**，不保证**能学到**；而且可能需要极多的神经元。在实践中，**深**的网络往往比**宽**的网络更高效，因为它能层层组合出层次化的特征——就像深层电路比浅层电路更省门电路。
""",
            points=[
                "神经网络损失非凸，存在多个局部解和鞍点；初始化、宽度、学习率都会影响能否训练成功。",
                "万能近似 ≠ 一定学得会，也不代表浅网络就够用。",
                "深度带来层次化表示：边缘 → 纹理 → 部件 → 物体。",
                "训练时同时监控训练损失和验证损失，判断是否过拟合。",
            ],
            code="ch06_mlp_xor.py",
            code_note="不用任何框架，用 numpy 实现两层网络的前向、反向和参数更新，学会 XOR。",
            src="Nielsen 1.6、第 4 章、第 5 章开头；邱锡鹏 4.3.1；Buduma 第 4 章",
        ),
    ],
    examples=[
        dict(
            title="手算前向传播",
            problem=r"输入 $\mathbf x=(1,2)$。隐藏层两个 ReLU 神经元：$h_1=\mathrm{ReLU}(0.5x_1+0.25x_2)$，$h_2=\mathrm{ReLU}(-x_1+0.5x_2+0.5)$。输出 $y=\sigma(h_1-2h_2+0.5)$。求输出。",
            steps=[
                r"$z_1=0.5\times1+0.25\times2=1$，$h_1=\mathrm{ReLU}(1)=1$。",
                r"$z_2=-1+0.5\times2+0.5=0.5$，$h_2=0.5$。",
                r"输出层 $z=1-2\times0.5+0.5=0.5$。",
                r"$y=\sigma(0.5)=1/(1+e^{-0.5})\approx0.6225$。",
            ],
            answer=r"$y\approx0.6225$。",
            code="ch06_ex_forward.py",
        ),
        dict(
            title="单个神经元的反向传播",
            problem=r"单个 sigmoid 神经元 $a=\sigma(wx+b)$，损失 $L=\frac12(a-y)^2$。已知 $x=1.5,\ w=0.8,\ b=-0.2,\ y=1$。用链式法则求 $\partial L/\partial w$ 和 $\partial L/\partial b$。",
            steps=[
                r"前向：$z=0.8\times1.5-0.2=1.0$，$a=\sigma(1)\approx0.7311$。",
                r"$\dfrac{\partial L}{\partial a}=a-y=-0.2689$；$\dfrac{\partial a}{\partial z}=a(1-a)\approx0.1966$。",
                r"所以 $\delta=\dfrac{\partial L}{\partial z}=-0.2689\times0.1966\approx-0.0529$。",
                r"$\dfrac{\partial L}{\partial w}=\delta\cdot x\approx-0.0793$，$\dfrac{\partial L}{\partial b}=\delta\approx-0.0529$。梯度为负，说明应增大 $w$ 和 $b$，让输出更接近 1。",
            ],
            answer=r"$\partial L/\partial w\approx-0.079$，$\partial L/\partial b\approx-0.053$。",
        ),
        dict(
            title="为什么感知机学不会 XOR，而两层网络可以",
            problem=r"说明单层感知机无法实现 XOR，并构造一个两个隐藏单元的网络实现它（使用阶跃激活）。",
            steps=[
                r"若存在 $w_1x_1+w_2x_2+b>0$ 实现 XOR，则 (0,0)：$b\le0$；(1,1)：$w_1+w_2+b\le0$；(1,0)：$w_1+b>0$；(0,1)：$w_2+b>0$。",
                r"后两式相加得 $w_1+w_2+2b>0$，即 $w_1+w_2+b>-b\ge0$，与第二式矛盾。所以不存在这样的直线。",
                r"构造：$h_1=\mathrm{step}(x_1+x_2-0.5)$（OR），$h_2=\mathrm{step}(x_1+x_2-1.5)$（AND）。",
                r"输出 $y=\mathrm{step}(h_1-h_2-0.5)$，即“OR 且非 AND”，恰好是 XOR。隐藏层把输入映射到了一个线性可分的新空间。",
            ],
            answer="单层不可行（线性不可分）；XOR = OR − AND，两层即可实现。",
            code="ch06_perceptron.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="如果神经网络所有层都使用线性激活函数，那么：", options=["网络表达能力随层数指数增长", "整个网络等价于一个线性模型", "训练会更快收敛到非线性解", "必须使用 softmax"], answer=1, explain="线性变换的复合仍是线性变换。"),
            dict(q="下列哪个函数单个感知机无法实现？", options=["AND", "OR", "NOT", "XOR"], answer=3, explain="XOR 线性不可分。"),
            dict(q="反向传播的数学本质是：", options=["蒙特卡洛采样", "链式法则", "拉格朗日乘子", "傅里叶变换"], answer=1, explain="逐层应用链式法则并复用中间结果。"),
            dict(q="一个 784-100-10 的全连接网络（含偏置）共有多少参数？", options=["78,400", "79,400", "79,510", "88,410"], answer=2, explain="$784\\times100+100+100\\times10+10=79510$。"),
            dict(q="多分类输出层通常搭配的激活函数与损失是：", options=["ReLU + 均方误差", "softmax + 交叉熵", "sigmoid + 合页损失", "tanh + L1 损失"], answer=1, explain="标准组合。"),
            dict(q="万能近似定理说明：", options=["任何网络都能学会任何函数", "足够宽的单隐层网络能以任意精度逼近连续函数", "深网络一定优于浅网络", "梯度下降一定收敛到全局最优"], answer=1, explain="只保证表示能力，不保证可学性。"),
        ],
        tf=[
            dict(q="反向传播求全部梯度的计算量与一次前向传播处于同一量级。", answer=True, explain="这是它比逐个扰动权重高效得多的原因。"),
            dict(q="把所有权重初始化为 0 是训练神经网络的好办法。", answer=False, explain="会导致对称性问题，同层神经元学到相同的东西。"),
            dict(q="ReLU 在 $z>0$ 时梯度为 1，不会饱和。", answer=True, explain="这缓解了梯度消失。"),
            dict(q="sigmoid 与交叉熵搭配时，输出层误差可化简为 $p-y$。", answer=True, explain="sigmoid 导数项被约掉。"),
            dict(q="只要数据线性可分，感知机学习算法一定能在有限步内收敛。", answer=True, explain="感知机收敛定理。"),
        ],
        fill=[
            dict(q="ReLU 的表达式是 $\\max(0,$ ____ $)$。", answer=["z", "x"], explain="$\\mathrm{ReLU}(z)=\\max(0,z)$。"),
            dict(q="用数值差分检查反向传播是否正确，叫做梯度 ____。", answer=["检验", "检查", "check", "checking"], explain="gradient checking。"),
            dict(q="sigmoid 函数的导数可以写成 $\\sigma(z)(1-$ ____ $)$。", answer=["σ(z)", "sigma(z)", "σ", "sigmoid(z)"], explain="$\\sigma'=\\sigma(1-\\sigma)$。"),
            dict(q="1969 年指出单层感知机局限性的书名是《____》（英文）。", answer=["Perceptrons", "感知机"], explain="Minsky & Papert, Perceptrons。"),
        ],
        reading=[
            dict(
                title="Is That All It Does?",
                passage="""When people first look inside a neural network, they are often surprised by how simple each piece is. A neuron multiplies its inputs by some weights, adds them up together with a bias, and passes the result through a simple nonlinear function. A layer does this for many neurons at once, which is nothing more than a matrix multiplication. Stack a few layers and you have a deep network.

The magic is not in any single operation but in the sheer number of them and in the way their weights are tuned. Training starts with random weights. For each batch of examples, the network makes predictions, a loss function measures how wrong they are, and backpropagation works out how much each weight contributed to the error. Every weight is then nudged a little in the direction that reduces the loss.

Repeated millions of times, these small adjustments turn a random tangle of numbers into a system that can recognize faces, translate languages or play Go. As one author puts it, a deep learning model really is just a lot of multiplications and additions with a few nonlinear mappings added in.""",
                zh="""人们第一次看到神经网络内部时，常常惊讶于每个部件是多么简单。一个神经元把输入乘以一些权重，与偏置一起相加，再把结果送进一个简单的非线性函数。一层就是同时对许多神经元这样做，本质上只是一次矩阵乘法。把几层叠起来，就得到一个深层网络。

神奇之处不在于任何单个运算，而在于运算的数量之多，以及权重被调节的方式。训练从随机权重开始。对每一批样本，网络做出预测，损失函数衡量预测错得有多离谱，反向传播计算出每个权重对误差贡献了多少。然后每个权重都朝着降低损失的方向被轻轻推一下。

重复数百万次之后，这些微小的调整把一团杂乱的随机数，变成了能识别人脸、翻译语言或下围棋的系统。正如一位作者所说，深度学习模型真的只是大量的乘法和加法，再加上一点非线性映射。""",
                questions=[
                    dict(q="According to the passage, what does a single layer amount to?", options=["A decision tree", "Essentially a matrix multiplication (plus a nonlinearity)", "A sorting algorithm", "A random number generator"], answer=1, explain="一层本质上是矩阵乘法。"),
                    dict(q="What is the role of backpropagation in the passage?", options=["To make predictions", "To work out how much each weight contributed to the error", "To collect more data", "To choose the network depth"], answer=1, explain="计算每个权重对误差的贡献。"),
                    dict(q="Where does the passage locate the “magic” of deep learning?", options=["In a single complex operation", "In the number of simple operations and how their weights are tuned", "In the choice of programming language", "In hand-written rules"], answer=1, explain="大量简单运算 + 权重调节。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("neuron", "神经元", "A unit that computes a weighted sum of inputs plus a bias and applies an activation.", "计算输入加权和加偏置再施加激活函数的单元。", "Each neuron in the hidden layer detects a different pattern."),
        ("perceptron", "感知机", "An early neural model that outputs one or zero using a step function.", "用阶跃函数输出 1 或 0 的早期神经元模型。", "The perceptron cannot learn the XOR function."),
        ("weight", "权重", "A learnable number that scales an input's influence.", "调节某个输入影响大小的可学习数值。", "Training adjusts every weight a little at each step."),
        ("bias", "偏置", "A learnable constant added before the activation function.", "在激活函数前加上的可学习常数。", "The bias lets a neuron shift its decision threshold."),
        ("activation function", "激活函数", "The nonlinear function applied to a neuron's weighted sum.", "作用于神经元加权和的非线性函数。", "Without an activation function, layers collapse into one linear map."),
        ("sigmoid", "S 型函数", "A smooth function squashing any real number into the range zero to one.", "把任意实数压缩到 0 到 1 之间的平滑函数。", "The sigmoid saturates for large positive or negative inputs."),
        ("ReLU", "修正线性单元", "The activation max(0, z), simple and non-saturating for positive inputs.", "激活函数 max(0, z)，简单且对正输入不饱和。", "ReLU made deep networks much easier to train."),
        ("hidden layer", "隐藏层", "A layer between the input and the output whose values are not directly observed.", "位于输入和输出之间、其值不被直接观察的层。", "One hidden layer is enough to solve XOR."),
        ("forward propagation", "前向传播", "Computing outputs layer by layer from the input.", "从输入开始逐层计算输出。", "Forward propagation stores activations for the backward pass."),
        ("backpropagation", "反向传播", "An algorithm that applies the chain rule backward to get gradients of all weights.", "反向应用链式法则以求出所有权重梯度的算法。", "Backpropagation computes all gradients in one backward pass."),
        ("chain rule", "链式法则", "The calculus rule for differentiating a composition of functions.", "对复合函数求导的微积分法则。", "Backpropagation is the chain rule applied systematically."),
        ("automatic differentiation", "自动微分", "Software that records operations and computes exact derivatives automatically.", "记录运算并自动计算精确导数的软件技术。", "PyTorch uses automatic differentiation to build gradients."),
        ("XOR problem", "异或问题", "The task of outputting one when exactly one of two inputs is one; not linearly separable.", "两个输入中恰有一个为 1 时输出 1 的任务，线性不可分。", "The XOR problem showed the limits of single-layer networks."),
        ("universal approximation theorem", "万能近似定理", "A wide enough one-hidden-layer network can approximate any continuous function.", "足够宽的单隐层网络能逼近任意连续函数。", "The universal approximation theorem says nothing about how to learn the weights."),
    ],
    cards=[
        ("为什么必须使用非线性激活？", "线性层叠加仍是线性变换，深度失去意义。"),
        ("反向传播的递推式？", r"$\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot f'(z^{(l)})$，$\partial L/\partial W^{(l)}=\delta^{(l)}(a^{(l-1)})^T$"),
        ("为什么权重不能全初始化为 0？", "同层神经元会得到完全相同的梯度，永远学不到不同的特征（对称性）。"),
        ("万能近似定理的两个局限？", "只保证能表示、不保证能学到；可能需要极多神经元，深网络通常更高效。"),
    ],
)
