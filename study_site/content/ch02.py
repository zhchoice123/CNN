CHAPTER = dict(
    no=2,
    title="概率、统计与信息论",
    title_en="Probability, Statistics and Information Theory",
    group="数学基础",
    summary="用贝叶斯定理更新信念，用抽样分布和置信区间描述不确定性，用假设检验做判断，再用熵、交叉熵和 KL 散度理解分类损失函数从哪里来。",
    sources=["Boslaugh & Watters《统计学概要》第 2、5–8 章", "MacKay《信息论、推断与学习算法》第 1–4、28 章", "Deisenroth 等《机器学习的数学》第 6 章", "Bishop《PRML》第 1–2 章"],
    learn=[
        dict(
            title="概率规则与贝叶斯定理",
            title_en="Probability rules and Bayes' theorem",
            body=r"""
概率论只有两条基本规则：

- **加法规则**（边缘化）：$p(x)=\sum_y p(x,y)$
- **乘法规则**：$p(x,y)=p(y\mid x)\,p(x)$

由二者直接推出**贝叶斯定理**：

$$p(\theta\mid D)=\frac{p(D\mid\theta)\,p(\theta)}{p(D)}\qquad\text{后验}=\frac{\text{似然}\times\text{先验}}{\text{证据}}$$

贝叶斯定理告诉我们如何用数据更新信念。一个经典例子：某病患病率 1%，检测的灵敏度 99%、特异度 95%。检测阳性时，真正患病的概率只有约 16.7%，因为健康人数量远多于病人，5% 的假阳性在绝对数量上压倒了真阳性。这就是**基率谬误**：忽略先验会严重高估结论。

MacKay 把这种观点推广到整个学习过程：学习就是推断，模型参数的后验分布就是学到的知识；用于比较模型的“证据” $p(D\mid M)$ 会自动惩罚过于复杂的模型，这是**贝叶斯奥卡姆剃刀**。
""",
            points=[
                "灵敏度 = $P(\\text{阳性}\\mid\\text{患病})$，特异度 = $P(\\text{阴性}\\mid\\text{健康})$。",
                "后验可以作为下一次观测的先验：连续两次阳性后，患病概率升到约 80%。",
                "最大似然只看 $p(D\\mid\\theta)$；最大后验（MAP）再乘上先验 $p(\\theta)$，相当于加了正则化。",
                "独立：$p(x,y)=p(x)p(y)$；条件独立是朴素贝叶斯和图模型的基础。",
            ],
            code="ch02_bayes.py",
            code_note="用公式和 100 万人的模拟两种方式计算阳性者的患病概率。",
            src="《统计学概要》第 2 章；MacKay 第 2–3、28 章；MML 6.3",
        ),
        dict(
            title="分布、期望与中心极限定理",
            title_en="Distributions, expectation and the central limit theorem",
            body=r"""
**期望** $\mathbb E[X]$ 是加权平均，**方差** $\mathrm{Var}[X]=\mathbb E[(X-\mathbb E X)^2]$ 衡量离散程度。常用分布：

- 伯努利/二项分布：抛硬币、点击与否；
- 类别/多项分布：分类问题的标签；
- **高斯分布** $\mathcal N(\mu,\sigma^2)$：机器学习中最重要的连续分布，边缘、条件和线性变换后仍是高斯。

我们手里只有**样本**，却想了解**总体**。样本均值 $\bar X$ 本身是随机变量，它的分布叫**抽样分布**。**中心极限定理**说：不管总体是什么分布（只要方差有限），当样本量 $n$ 足够大时，$\bar X$ 近似服从 $\mathcal N(\mu,\sigma^2/n)$。于是样本均值的标准误是 $\sigma/\sqrt n$——要把误差减半，样本量要变成 4 倍。

95% **置信区间** $\bar x\pm1.96\,s/\sqrt n$ 的含义是：如果重复抽样很多次，约 95% 的区间会包含真值；它不是“真值有 95% 的概率落在这个区间里”。

> 代码里总体是偏态的指数分布，$n=50$ 时区间的实际覆盖率约 93%，略低于名义上的 95%：中心极限定理是近似，偏态越强，需要的样本越多。
""",
            points=[
                "标准差描述数据本身的离散，标准误描述估计量（如均值）的不确定性。",
                "高斯分布由均值和协方差完全决定，是线性回归噪声、VAE 隐变量等模型的默认选择。",
                "样本方差除以 $n-1$（贝塞尔校正）才是总体方差的无偏估计。",
                "数据一定要先画图（直方图、箱线图），很多问题一眼就能看出来。",
            ],
            code="ch02_clt.py",
            code_note="从偏态的指数分布中反复抽样，观察样本均值的分布怎样随 $n$ 变得对称，并检验置信区间的覆盖率。",
            src="《统计学概要》第 4、7 章；MML 6.4–6.5；Bishop 2.3",
        ),
        dict(
            title="假设检验、p 值与研究设计",
            title_en="Hypothesis testing, p-values and study design",
            body=r"""
假设检验的逻辑是反证：先假设“没有效应”（**零假设** $H_0$），再看在 $H_0$ 成立时观测到这么极端（或更极端）数据的概率，这个概率就是 **p 值**。p 值很小，说明数据与 $H_0$ 不太相容，于是拒绝 $H_0$。

两类错误：

- **第一类错误**（假阳性）：$H_0$ 为真却拒绝了它，概率为显著性水平 $\alpha$；
- **第二类错误**（假阴性）：$H_0$ 为假却没拒绝，概率为 $\beta$；**统计功效** $=1-\beta$。

比较两组均值常用 t 检验；不想依赖正态假设时，可以用**置换检验**：如果两组没有差别，组标签就可以随意打乱，打乱很多次得到差值的零分布，再看观测差值落在哪里。

统计也要讲研究设计：**相关不等于因果**；观察研究中存在**混杂因素**和**选择偏差**；随机对照实验通过随机分组消除混杂。在机器学习里，比较两个模型时也要问：准确率高了 1 个百分点，是真的更好，还是随机划分造成的波动？
""",
            points=[
                "p 值不是“$H_0$ 为真的概率”，也不衡量效应大小。",
                "样本量越大，越小的效应也能“显著”，所以要同时报告效应量和置信区间。",
                "多次检验会累积假阳性（多重比较问题），需要校正。",
                "识破误导性统计：纵轴不从零开始、只报相对风险、样本不具代表性。",
            ],
            code="ch02_permutation_test.py",
            code_note="用 2 万次置换构造零分布，计算两组均值差的 p 值，并给出 Welch t 统计量。",
            src="《统计学概要》第 5–8 章",
        ),
        dict(
            title="熵、交叉熵与 KL 散度",
            title_en="Entropy, cross-entropy and KL divergence",
            body=r"""
香农用**熵**衡量一个随机变量的平均不确定性（以 2 为底时单位是比特）：

$$H(p)=-\sum_x p(x)\log_2 p(x)$$

公平硬币的熵是 1 比特，偏得厉害的硬币熵更小。**信源编码定理**说：数据无损压缩后的平均长度不可能低于熵。

如果真实分布是 $p$，我们却用模型分布 $q$ 来编码，平均码长就是**交叉熵** $H(p,q)=-\sum_x p(x)\log q(x)$，多出来的部分是 **KL 散度**：

$$H(p,q)=H(p)+D_{KL}(p\,\|\,q),\qquad D_{KL}(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)}\ge0$$

这就解释了分类网络为什么用**交叉熵损失**：真实标签 $p$ 是 one-hot 向量，$H(p)=0$，交叉熵只剩 $-\log q(\text{正确类})$；最小化它就是在最小化模型分布与真实分布之间的 KL 散度，也等价于最大化似然。KL 散度不对称，这一点在变分推断和 RLHF 的 KL 正则中都很关键。
""",
            points=[
                "熵越大越不确定；均匀分布的熵最大。",
                "交叉熵 ≥ 熵，二者之差就是 KL 散度。",
                "$D_{KL}(p\\|q)\\ne D_{KL}(q\\|p)$，它不是严格意义上的距离。",
                "最小化交叉熵 = 最大化对数似然 = 最小化编码长度，三种说法等价。",
            ],
            code="ch02_entropy.py",
            code_note="计算几个分布的熵，并验证 $H(p,q)=H(p)+D_{KL}(p\\|q)$ 与 KL 的不对称性。",
            src="MacKay 第 1–2、4 章；Buduma 第 2 章；Bishop 1.6",
        ),
    ],
    examples=[
        dict(
            title="罕见病检测的后验概率",
            problem=r"某病患病率 1%。检测的灵敏度为 99%，特异度为 95%。一人检测阳性，他真正患病的概率是多少？若再测一次仍为阳性呢？",
            steps=[
                r"记 $D$ 为患病，$+$ 为阳性。已知 $P(D)=0.01$，$P(+\mid D)=0.99$，$P(+\mid\bar D)=1-0.95=0.05$。",
                r"全概率公式：$P(+)=0.99\times0.01+0.05\times0.99=0.0099+0.0495=0.0594$。",
                r"贝叶斯定理：$P(D\mid+)=0.0099/0.0594\approx0.1667$。",
                r"第二次检测时把 0.1667 当作新的先验：$P=\dfrac{0.99\times0.1667}{0.99\times0.1667+0.05\times0.8333}\approx0.798$。",
            ],
            answer="一次阳性约 16.7%，两次阳性约 79.8%。",
            code="ch02_bayes.py",
        ),
        dict(
            title="12 个球称重问题的信息论下界",
            problem=r"12 个外观相同的球中有一个重量不同（可能偏重也可能偏轻）。用天平最少称几次才能保证找出它并判断轻重？用信息论给出下界。",
            steps=[
                r"可能的情况共有 $12\times2=24$ 种，确定其中一种需要 $\log_2 24\approx4.585$ 比特信息。",
                r"天平每次有左重、右重、平衡三种结果，最多提供 $\log_2 3\approx1.585$ 比特。",
                r"所以称重次数至少为 $4.585/1.585\approx2.89$，向上取整为 3 次；而 $3^3=27\ge24$，3 次在信息量上足够。",
                r"设计方案的关键是让每次称重的三种结果尽量等可能，这样每次获得的信息最多（例如第一次左右各放 4 个）。",
            ],
            answer="至少 3 次，且 3 次可以做到。",
            code="ch02_ex_weighing.py",
        ),
        dict(
            title="分类损失的交叉熵计算",
            problem=r"三分类问题的真实标签是第 1 类（one-hot 为 $(1,0,0)$），模型输出概率 $q=(0.5,0.3,0.2)$。求以 2 为底的交叉熵损失；若模型输出改为 $(0.9,0.05,0.05)$，损失变为多少？",
            steps=[
                r"one-hot 标签下交叉熵只剩正确类一项：$H(p,q)=-\log_2 q_1$。",
                r"$q_1=0.5$ 时：$-\log_2 0.5=1$ 比特。",
                r"$q_1=0.9$ 时：$-\log_2 0.9\approx0.152$ 比特，损失显著下降。",
                r"实际框架常用自然对数（单位为奈特），数值相差一个常数因子 $\ln 2$，不影响优化。",
            ],
            answer="1 比特；约 0.152 比特。",
            code="ch02_entropy.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="贝叶斯定理中，$p(D\\mid\\theta)$ 称为：", options=["先验", "似然", "后验", "证据"], answer=1, explain="给定参数时数据出现的概率是似然。"),
            dict(q="样本量从 100 增加到 400，样本均值的标准误变为原来的：", options=["1/4", "1/2", "2 倍", "不变"], answer=1, explain="标准误 $\\propto1/\\sqrt n$，$\\sqrt{4}=2$。"),
            dict(q="关于 p 值，正确的是：", options=["p 值是零假设为真的概率", "p 值越小效应越大", "p 值是在零假设成立时观测到至少如此极端数据的概率", "p<0.05 证明了因果关系"], answer=2, explain="p 值是条件在 $H_0$ 下的尾部概率。"),
            dict(q="第一类错误指的是：", options=["零假设为真却被拒绝", "零假设为假却没被拒绝", "样本量太小", "模型过拟合"], answer=0, explain="即假阳性，概率为 $\\alpha$。"),
            dict(q="公平的四面骰子（四种结果等可能）的熵是：", options=["1 比特", "2 比特", "4 比特", "0.25 比特"], answer=1, explain="$\\log_2 4=2$。"),
            dict(q="分类网络使用交叉熵损失，本质上是在最小化：", options=["模型分布与真实分布的 KL 散度", "参数的 L2 范数", "预测值的方差", "训练时间"], answer=0, explain="$H(p,q)=H(p)+KL$，$H(p)$ 与模型无关。"),
        ],
        tf=[
            dict(q="95% 置信区间意味着真值有 95% 的概率落在这一次算出的区间里。", answer=False, explain="频率学派解释：重复抽样时约 95% 的区间会包含真值。"),
            dict(q="KL 散度总是非负的。", answer=True, explain="由 Gibbs 不等式，等号当且仅当两分布相同。"),
            dict(q="KL 散度是对称的，即 $D_{KL}(p\\|q)=D_{KL}(q\\|p)$。", answer=False, explain="一般不相等。"),
            dict(q="两个变量高度相关，就可以断定其中一个导致了另一个。", answer=False, explain="相关不等于因果，可能存在混杂因素。"),
            dict(q="最大后验估计相当于在最大似然的基础上加入了先验，可起到正则化作用。", answer=True, explain="高斯先验对应 L2 正则。"),
        ],
        fill=[
            dict(q="把患病率这样的已有知识忽略而导致误判，叫做 ____ 谬误。", answer=["基率", "base rate", "基础比率"], explain="base rate fallacy。"),
            dict(q="不管总体分布如何，样本均值的分布随样本量增大趋于正态，这叫 ____ 定理。", answer=["中心极限", "central limit", "中心极限定理"], explain="Central Limit Theorem。"),
            dict(q="公平硬币一次抛掷的熵是 ____ 比特。", answer=["1", "一"], explain="$-2\\times0.5\\log_2 0.5=1$。"),
            dict(q="交叉熵等于熵加上 ____ 散度。", answer=["KL", "Kullback-Leibler", "相对熵"], explain="$H(p,q)=H(p)+D_{KL}(p\\|q)$。"),
        ],
        reading=[
            dict(
                title="The Base Rate Trap",
                passage="""Imagine a screening test that catches 99 percent of people who have a rare disease and correctly clears 95 percent of people who do not. If your result comes back positive, how worried should you be? Most people guess that the chance of being ill is close to 99 percent. The true answer, when only one person in a hundred has the disease, is about one in six.

The reason is simple arithmetic. Out of ten thousand people, one hundred are sick and ninety-nine of them test positive. Among the 9,900 healthy people, five percent, almost five hundred, also test positive. A positive result therefore puts you in a group of about six hundred people, of whom only ninety-nine are actually ill.

Ignoring how common a condition is in the first place is called the base rate fallacy. The same trap appears in machine learning. A fraud detector with 99 percent accuracy sounds impressive, but if only one transaction in a thousand is fraudulent, a model that never flags anything scores 99.9 percent. Accuracy alone says very little; precision and recall tell a more honest story.""",
                zh="""设想一种筛查检测，能发现 99% 的患者，并能正确排除 95% 的健康人。如果你的结果是阳性，你该有多担心？大多数人会猜患病的可能接近 99%。但当一百个人中只有一个人患病时，真实答案大约是六分之一。

原因只是简单的算术。一万人中有一百人患病，其中 99 人检测阳性；在 9,900 名健康人中，有 5%，也就是近 500 人，同样检测阳性。因此，阳性结果把你放进了一个约 600 人的群体，其中真正患病的只有 99 人。

忽略某种情况本身有多常见，叫做基率谬误。机器学习中也有同样的陷阱。一个准确率 99% 的欺诈检测器听起来很厉害，但如果每一千笔交易中只有一笔是欺诈，一个从不报警的模型准确率就能达到 99.9%。仅看准确率说明不了什么，精确率和召回率才能讲出更真实的情况。""",
                questions=[
                    dict(q="In a group of 10,000 people, about how many test positive in total?",
                         options=["About 100", "About 600", "About 9,900", "About 5,000"], answer=1,
                         explain="99 个真阳性 + 约 495 个假阳性 ≈ 600。"),
                    dict(q="What is the main point of the fraud detection example?",
                         options=["Fraud is very common", "Accuracy can be misleading when classes are imbalanced", "Precision is always 99 percent", "Models should flag every transaction"],
                         answer=1, explain="类别极不平衡时准确率会误导。"),
                    dict(q="The phrase “base rate” in the passage refers to:",
                         options=["how common a condition is to begin with", "the accuracy of the test", "the cost of the test", "the number of positive results"],
                         answer=0, explain="基率就是先验发生率。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("probability", "概率", "A number between 0 and 1 measuring how likely an event is.", "介于 0 和 1 之间、衡量事件可能性的数。", "The probability of rain tomorrow is thirty percent."),
        ("prior", "先验", "The belief about a quantity before seeing the data.", "在看到数据之前对某个量的信念。", "A strong prior pulls the estimate toward zero."),
        ("posterior", "后验", "The updated belief after combining the prior with the data.", "结合先验和数据后更新的信念。", "The posterior becomes sharper as more data arrive."),
        ("likelihood", "似然", "How probable the observed data are under given parameters.", "在给定参数下观测数据出现的概率。", "Maximum likelihood picks the parameters that best explain the data."),
        ("expectation", "期望", "The probability-weighted average value of a random variable.", "随机变量按概率加权的平均值。", "The expectation of a fair die roll is three and a half."),
        ("variance", "方差", "The expected squared deviation from the mean.", "偏离均值的平方的期望。", "High variance means the estimates change a lot between samples."),
        ("Gaussian distribution", "高斯分布", "The bell-shaped normal distribution defined by a mean and a variance.", "由均值和方差决定的钟形正态分布。", "Measurement noise is often modeled with a Gaussian distribution."),
        ("sampling distribution", "抽样分布", "The distribution of a statistic over repeated samples.", "统计量在反复抽样下的分布。", "The sampling distribution of the mean narrows as n grows."),
        ("confidence interval", "置信区间", "A range constructed so that it captures the true value in a stated fraction of repeated samples.", "按一定程序构造、在重复抽样中以指定比例覆盖真值的区间。", "Report a confidence interval, not just a single number."),
        ("null hypothesis", "零假设", "The default claim of no effect or no difference that a test tries to reject.", "检验试图拒绝的“没有效应或差异”的默认主张。", "The null hypothesis says the two models perform equally well."),
        ("p-value", "p 值", "The probability, under the null hypothesis, of data at least as extreme as observed.", "在零假设下得到至少与观测一样极端数据的概率。", "A small p-value is evidence against the null hypothesis."),
        ("entropy", "熵", "The average uncertainty of a random variable, measured in bits.", "随机变量的平均不确定性，以比特为单位。", "A fair coin has an entropy of one bit."),
        ("cross-entropy", "交叉熵", "The average code length when data from p are encoded with model q.", "用模型 q 编码来自 p 的数据时的平均码长。", "Cross-entropy is the standard loss for classification."),
        ("KL divergence", "KL 散度", "A non-negative, asymmetric measure of how one distribution differs from another.", "衡量一个分布与另一个分布差异的非负、不对称度量。", "The KL divergence term keeps the policy close to the reference model."),
    ],
    cards=[
        ("贝叶斯定理？", r"$p(\theta\mid D)=\dfrac{p(D\mid\theta)p(\theta)}{p(D)}$：后验 ∝ 似然 × 先验"),
        ("标准误与样本量的关系？", r"$SE=\sigma/\sqrt n$，误差减半需要 4 倍样本。"),
        ("两类错误？", "第一类：拒绝了真的 $H_0$（假阳性，$\\alpha$）；第二类：没拒绝假的 $H_0$（假阴性，$\\beta$）。"),
        ("交叉熵、熵与 KL 的关系？", r"$H(p,q)=H(p)+D_{KL}(p\|q)$"),
    ],
)
