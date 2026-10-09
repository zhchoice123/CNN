CHAPTER = dict(
    no=15,
    title="概率机器人：贝叶斯滤波、定位与 SLAM",
    title_en="Probabilistic Robotics: Bayes Filters, Localization and SLAM",
    group="机器人与工程",
    summary="机器人永远面对不确定性。本章用概率显式地表示“我在哪里、世界是什么样”：贝叶斯滤波的预测—更新循环，卡尔曼滤波与扩展卡尔曼滤波，直方图滤波与粒子滤波，运动与测量模型，蒙特卡洛定位、占据栅格建图与同时定位与建图（SLAM）。",
    sources=["Thrun, Burgard & Fox《概率机器人》（早期草稿）第 1–10 章", "Forsyth & Ponce 第 11 章“跟踪”", "Bishop 第 13 章“线性动态系统”"],
    learn=[
        dict(
            title="不确定性与贝叶斯滤波",
            title_en="Uncertainty and the Bayes filter",
            body=r"""
《概率机器人》开篇总结了不确定性的五个来源：**环境**本身不可预测；**传感器**有量程、分辨率限制和噪声；**机器人**的执行机构有控制误差和磨损；**模型**只是对现实的近似；**计算**受实时性约束，算法只能是近似的。

概率机器人学的核心思想是：不只保留一个“最好的猜测”，而是用**概率分布**表示对状态的全部信念 $bel(x_t)=p(x_t\mid z_{1:t},u_{1:t})$。这样机器人能以数学上严谨的方式表达模糊性，面对不确定性时性能平稳退化，而不是突然崩溃。

**贝叶斯滤波**交替执行两步：

$$\text{预测：}\ \overline{bel}(x_t)=\int p(x_t\mid u_t,x_{t-1})\,bel(x_{t-1})\,dx_{t-1}$$
$$\text{更新：}\ bel(x_t)=\eta\,p(z_t\mid x_t)\,\overline{bel}(x_t)$$

预测步用**运动模型**把信念向前推（不确定性增大），更新步用**测量模型**乘以似然再归一化（不确定性减小）。这依赖**马尔可夫假设**：当前状态包含了预测未来所需的全部信息。

书中的经典例子：机器人在有三扇门的走廊里，一开始完全不知道自己在哪（均匀分布）；看到一扇门后，信念在三扇门前各出现一个峰；向前走几步再看到门，只有一个位置能同时解释两次观测，信念就集中到了真实位置。
""",
            points=[
                "预测让分布变“胖”，观测让分布变“瘦”。",
                "直方图滤波把状态空间离散成格子，适合低维问题。",
                "贝叶斯滤波是一个框架，卡尔曼滤波、粒子滤波都是它的具体实现。",
                "代码中的直方图滤波重现了书中的门走廊定位过程。",
            ],
            code="ch15_histogram_filter.py",
            code_note="20 格走廊、3 扇门：从均匀分布出发，交替观测和移动，观察信念如何从多峰收敛到真实位置。",
            src="Thrun 第 1–2 章、4.1",
        ),
        dict(
            title="卡尔曼滤波与扩展卡尔曼滤波",
            title_en="Kalman and extended Kalman filters",
            body=r"""
当运动和测量都是**线性**的，噪声都是**高斯**的，贝叶斯滤波中的信念始终是高斯分布，只需跟踪均值 $\mu$ 和协方差 $\Sigma$——这就是**卡尔曼滤波**：

$$\text{预测：}\ \bar\mu=A\mu+Bu,\quad \bar\Sigma=A\Sigma A^T+R$$
$$\text{更新：}\ K=\bar\Sigma C^T(C\bar\Sigma C^T+Q)^{-1},\quad \mu=\bar\mu+K(z-C\bar\mu),\quad \Sigma=(I-KC)\bar\Sigma$$

**卡尔曼增益** $K$ 权衡了“我有多相信预测”和“我有多相信测量”：测量噪声大时 $K$ 小，更相信预测；预测不确定时 $K$ 大，更相信测量。它还能估计无法直接测量的量（例如只测位置也能估计速度）。

现实中的系统大多是**非线性**的（机器人转弯、相机投影）。**扩展卡尔曼滤波（EKF）**在当前估计附近做一阶泰勒展开，用雅可比矩阵线性化，再套用卡尔曼公式。**信息滤波**用信息矩阵 $\Sigma^{-1}$ 和信息向量表示高斯分布，在建图问题中常常是稀疏的，计算更高效。

卡尔曼滤波在导航、跟踪（Forsyth & Ponce 第 11 章）、传感器融合中无处不在；Bishop 第 13 章则把它看作线性动态系统（隐状态模型）上的推断。
""",
            points=[
                "卡尔曼滤波是线性高斯系统下的最优估计。",
                "增益会随时间收敛到一个稳态值（代码中从 0.83 降到 0.27）。",
                "EKF 在非线性很强或初始误差很大时可能发散；无迹卡尔曼滤波（UKF）是另一种选择。",
                "高斯只能表示单峰信念，无法同时保留“在门 1 前还是门 3 前”这样的多个假设。",
            ],
            code="ch15_kalman.py",
            code_note="跟踪匀速运动的目标：只测位置，同时估计位置和速度，比较滤波前后的误差。",
            src="Thrun 第 3 章；Forsyth & Ponce 11.3；Bishop 13.3",
        ),
        dict(
            title="粒子滤波与蒙特卡洛定位",
            title_en="Particle filters and Monte Carlo localization",
            body=r"""
**粒子滤波**用一大堆随机样本——**粒子**——来表示任意形状的分布，粒子越密集的地方概率越大。每一步：

1. **采样（预测）**：按运动模型移动每个粒子，并加入运动噪声；
2. **加权（更新）**：按测量似然 $p(z_t\mid x_t^{[m]})$ 给每个粒子算权重；
3. **重采样**：按权重有放回地抽取新的粒子集合，权重高的粒子被复制，权重低的被淘汰。

它能表示**多峰**分布，所以能解决**全局定位**：一开始把粒子均匀撒满整个地图，随着观测逐步收敛。用于机器人定位时叫**蒙特卡洛定位（MCL）**，是《概率机器人》作者们最有代表性的贡献之一。

定位问题分三类（第 7 章）：**位置跟踪**（已知初始位置）、**全局定位**（从头开始）、以及最难的**绑架机器人问题**（定位成功后被突然搬走，机器人必须意识到自己错了并重新定位，常用的办法是持续注入少量随机粒子）。

实践细节：**有效粒子数** $1/\sum w_i^2$ 太小时说明粒子退化，应当重采样；粒子数要与状态维数相匹配，高维问题需要的粒子数呈指数增长。
""",
            points=[
                "粒子滤波是非参数方法，不假设分布形状。",
                "重采样避免了粒子权重退化，但会降低多样性。",
                "动态环境（周围有行人）中，测量模型要能处理“意外障碍”。",
                "粒子滤波也用于视觉跟踪（Forsyth & Ponce 11.5）。",
            ],
            code="ch15_particle_filter.py",
            code_note="2000 个粒子在连续的门走廊中做全局定位，打印每次观测后的有效粒子数和粒子密集区。",
            src="Thrun 第 4、7–8 章；Forsyth & Ponce 11.5",
        ),
        dict(
            title="建图与同时定位与建图（SLAM）",
            title_en="Mapping and SLAM",
            body=r"""
**运动模型与测量模型**（第 5–6 章）：速度运动模型和里程计运动模型给出的不是一个确定位置，而是一个概率分布；激光测距仪的**波束模型**考虑正确测到障碍物、被意外物体挡住、测量失败、随机噪声四种情况，**似然场模型**计算更高效。

**占据栅格建图**（第 9 章）：已知机器人位姿时，把环境切成网格，每个格子维护一个被占据的概率。用**对数几率** $l=\log\frac{p}{1-p}$ 表示，每次测量只需做加法：光束穿过的格子 $l$ 减小（更可能空闲），光束末端的格子 $l$ 增大（更可能被占据），从未观测到的格子保持 $l=0$，即概率 0.5。

**SLAM**（第 10 章起）是机器人学的“鸡生蛋”问题：要定位需要地图，要建图需要知道位置。**EKF-SLAM** 把机器人位姿和所有地标位置放进同一个状态向量一起估计，但地标一多，协方差矩阵就变得巨大而稠密。书中的早期草稿讨论了**扩展信息形式**和**稀疏扩展信息滤波**：在信息形式下矩阵近似稀疏，可以高效求解——这些思想后来发展为**图优化 SLAM**，即今天主流 SLAM 系统的后端，与第 11 章的光束法平差一脉相承。

最后两章讲决策：**马尔可夫决策过程**（价值迭代），以及状态不完全可观测时的 **POMDP**——好的决策有时是先去收集信息（比如先看一眼路标确定位置），再执行任务。
""",
            points=[
                "对数几率让多次观测的融合变成简单的加法。",
                "SLAM 中，回到已知地点（回环检测）能大幅减小累积误差。",
                "未知数据关联（哪个观测对应哪个地标）是 SLAM 的核心难点之一，书中用 EM 算法处理。",
                "今天的自动驾驶定位、扫地机器人建图、无人机导航都建立在这些思想之上。",
            ],
            code="ch15_occupancy.py",
            code_note="三次激光扫描都在第 7 格测到障碍：用对数几率累加，观察每个格子的占据概率如何变化。",
            src="Thrun 第 5–6、9–16 章",
        ),
    ],
    examples=[
        dict(
            title="一步贝叶斯更新",
            problem=r"机器人可能位于 A、B、C 三个位置，先验信念为 (0.5, 0.3, 0.2)。只有 B 处有门。传感器在门前报告“门”的概率是 0.8，在非门处误报“门”的概率是 0.1。现在传感器报告“门”，求后验信念。",
            steps=[
                r"似然：A 为 0.1，B 为 0.8，C 为 0.1。",
                r"未归一化后验 = 似然 × 先验：A $0.05$，B $0.24$，C $0.02$。",
                r"归一化常数 $\eta^{-1}=0.05+0.24+0.02=0.31$。",
                r"后验：A $0.161$，B $0.774$，C $0.065$。一次观测就让 B 从 0.3 升到 0.77。",
            ],
            answer="后验约为 (0.161, 0.774, 0.065)。",
        ),
        dict(
            title="一维卡尔曼增益",
            problem=r"一维状态。预测的位置为 10.0，预测方差 $\bar\sigma^2=4$；传感器测得 12.0，测量噪声方差 $\sigma_z^2=1$。求卡尔曼增益、更新后的估计和方差。",
            steps=[
                r"一维时 $K=\bar\sigma^2/(\bar\sigma^2+\sigma_z^2)=4/5=0.8$。",
                r"$\mu=10+0.8\times(12-10)=11.6$：更靠近测量值，因为测量更可靠。",
                r"$\sigma^2=(1-K)\bar\sigma^2=0.2\times4=0.8$，比预测方差 4 和测量方差 1 都小。",
                r"融合两个信息源后的不确定性小于任何一个单独的来源——这就是传感器融合的价值。",
            ],
            answer="K = 0.8，估计 11.6，方差 0.8。",
            code="ch15_kalman.py",
        ),
        dict(
            title="占据栅格的对数几率更新",
            problem=r"某格子初始占据概率 0.5。逆测量模型：光束末端处的格子占据概率为 0.7。连续三次扫描都在该格子测到障碍，最终占据概率是多少？",
            steps=[
                r"初始对数几率 $l_0=\log(0.5/0.5)=0$。每次命中增加 $\log(0.7/0.3)\approx0.847$。",
                r"三次之后 $l=3\times0.847\approx2.542$。",
                r"转回概率：$p=1-\dfrac{1}{1+e^{l}}=1-\dfrac{1}{1+12.70}\approx0.927$。",
                r"与代码中第 3 次扫描后的 0.93 一致。对数几率把概率的乘法更新变成了加法，计算简单且数值稳定。",
            ],
            answer="约 0.93。",
            code="ch15_occupancy.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="贝叶斯滤波的两个步骤是：", options=["编码与解码", "预测与更新", "前向与反向", "采样与拒绝"], answer=1, explain="运动模型预测，测量模型更新。"),
            dict(q="卡尔曼滤波的适用前提是：", options=["任意非线性系统", "线性系统与高斯噪声", "离散状态", "无噪声"], answer=1, explain="此时信念始终是高斯分布。"),
            dict(q="测量噪声越大，卡尔曼增益：", options=["越大", "越小", "不变", "变为负数"], answer=1, explain="更相信预测。"),
            dict(q="能够表示多峰分布、适合全局定位的方法是：", options=["卡尔曼滤波", "扩展卡尔曼滤波", "粒子滤波", "最小二乘"], answer=2, explain="非参数的样本表示。"),
            dict(q="机器人被突然搬到另一个地方，需要意识到并重新定位，这叫：", options=["位置跟踪", "全局定位", "绑架机器人问题", "回环检测"], answer=2, explain="kidnapped robot problem。"),
            dict(q="占据栅格中使用对数几率表示的好处是：", options=["节省内存", "多次观测的融合变成加法", "不需要传感器", "可以表示颜色"], answer=1, explain="乘法 → 加法。"),
        ],
        tf=[
            dict(q="预测步通常会增大信念的不确定性，更新步通常会减小它。", answer=True, explain="运动噪声累积，观测提供信息。"),
            dict(q="粒子滤波要求信念必须是高斯分布。", answer=False, explain="粒子可以表示任意分布。"),
            dict(q="扩展卡尔曼滤波通过一阶泰勒展开对非线性模型做线性化。", answer=True, explain="使用雅可比矩阵。"),
            dict(q="SLAM 需要同时估计机器人位姿和环境地图。", answer=True, explain="Simultaneous Localization and Mapping。"),
            dict(q="占据栅格中从未被观测的格子，占据概率应设为 1。", answer=False, explain="应保持 0.5（未知）。"),
        ],
        fill=[
            dict(q="卡尔曼滤波中权衡预测与测量的系数叫卡尔曼 ____。", answer=["增益", "gain"], explain="Kalman gain。"),
            dict(q="粒子滤波中按权重有放回地抽取新粒子的步骤叫 ____。", answer=["重采样", "resampling"], explain="resampling。"),
            dict(q="用粒子滤波做机器人定位的方法缩写为 ____。", answer=["MCL", "蒙特卡洛定位"], explain="Monte Carlo Localization。"),
            dict(q="当前状态包含预测未来所需全部信息的假设叫 ____ 假设。", answer=["马尔可夫", "Markov"], explain="Markov assumption。"),
        ],
        reading=[
            dict(
                title="Never Trust a Single Guess",
                passage="""Robots used to live on factory floors, where every part arrived in the same place at the same time. Today they drive on highways, vacuum living rooms and fly through warehouses, environments that are far less predictable. Sensors are noisy, wheels slip, and every model of the world is only an approximation.

Probabilistic robotics responds to this mess with a simple principle: instead of committing to one best guess, keep track of everything you might believe and how strongly you believe it. A robot that has just been switched on in an office does not know where it is, so its belief is spread evenly across the whole map. When it sees a door, the belief gathers in front of every door. After it moves a few meters and sees another door, only one location explains both observations, and the belief collapses there.

This cycle of predicting and correcting, the Bayes filter, appears again and again in the field. The Kalman filter implements it with bell-shaped curves; the particle filter implements it with thousands of samples. Both let a robot act sensibly even when it is unsure.""",
                zh="""机器人过去生活在工厂车间里，每个零件都在同一时间到达同一位置。如今它们在高速公路上行驶、在客厅里吸尘、在仓库中飞行，这些环境的可预测性要低得多。传感器有噪声，车轮会打滑，而任何关于世界的模型都只是近似。

概率机器人学用一个简单的原则来应对这种混乱：不要押注于一个最佳猜测，而是记录下你可能相信的一切，以及你相信的程度。一台刚在办公室里开机的机器人不知道自己在哪里，所以它的信念均匀地铺满整张地图。当它看到一扇门时，信念聚集到每一扇门的前面。当它移动几米后又看到另一扇门，只有一个位置能同时解释两次观测，信念就坍缩到那里。

这种预测与修正的循环——贝叶斯滤波——在这个领域中一再出现。卡尔曼滤波用钟形曲线实现它，粒子滤波用成千上万个样本实现它。两者都让机器人即使在不确定时也能明智地行动。""",
                questions=[
                    dict(q="What is the central principle of probabilistic robotics according to the passage?", options=["Always commit to the single best guess", "Keep track of all possible beliefs and their strengths", "Avoid using sensors", "Only work on factory floors"], answer=1, explain="保留全部信念及其强度。"),
                    dict(q="Right after the robot first sees a door, its belief is:", options=["Spread evenly over the map", "Concentrated in front of every door", "Collapsed to the true position", "Zero everywhere"], answer=1, explain="在每扇门前出现峰值。"),
                    dict(q="How does the particle filter represent belief?", options=["With a single bell curve", "With thousands of samples", "With a lookup table of doors", "With a neural network only"], answer=1, explain="成千上万个样本。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("belief", "信念", "The robot's probability distribution over possible states.", "机器人对可能状态的概率分布。", "The belief spreads out after every motion."),
        ("Bayes filter", "贝叶斯滤波", "A recursive algorithm that alternates prediction with measurement updates.", "交替进行预测和测量更新的递归算法。", "Every localization algorithm in the book is a Bayes filter."),
        ("motion model", "运动模型", "A probabilistic description of how the state changes under a control.", "描述控制作用下状态如何变化的概率模型。", "The odometry motion model accounts for wheel slip."),
        ("measurement model", "测量模型", "The probability of a sensor reading given the state.", "给定状态时传感器读数的概率。", "The beam model is a measurement model for laser range finders."),
        ("Kalman filter", "卡尔曼滤波", "The optimal Bayes filter for linear systems with Gaussian noise.", "线性高斯系统下的最优贝叶斯滤波器。", "GPS receivers use a Kalman filter to smooth positions."),
        ("Kalman gain", "卡尔曼增益", "The weight that balances the prediction against the new measurement.", "在预测与新测量之间权衡的权重。", "A small Kalman gain means the sensor is not trusted much."),
        ("extended Kalman filter", "扩展卡尔曼滤波", "A Kalman filter that linearizes nonlinear models around the current estimate.", "在当前估计附近线性化非线性模型的卡尔曼滤波。", "The extended Kalman filter handles the robot's turning motion."),
        ("particle filter", "粒子滤波", "A Bayes filter that represents belief with weighted random samples.", "用带权随机样本表示信念的贝叶斯滤波。", "The particle filter kept two hypotheses alive."),
        ("resampling", "重采样", "Drawing a new particle set in proportion to the particle weights.", "按粒子权重比例抽取新的粒子集合。", "Resampling discards particles with tiny weights."),
        ("localization", "定位", "Estimating the robot's pose within a known map.", "在已知地图中估计机器人的位姿。", "Global localization starts with no idea of the pose."),
        ("kidnapped robot problem", "绑架机器人问题", "Recovering after the robot is moved to an unknown location without being told.", "机器人在不知情时被搬到未知位置后恢复定位的问题。", "Injecting random particles helps with the kidnapped robot problem."),
        ("occupancy grid", "占据栅格", "A map that stores the probability that each cell is occupied.", "存储每个格子被占据概率的地图。", "The vacuum robot builds an occupancy grid of the room."),
        ("log odds", "对数几率", "The logarithm of p over one minus p, which turns updates into additions.", "p/(1−p) 的对数，可把更新变为加法。", "Each laser hit adds a constant to the log odds."),
        ("SLAM", "同时定位与建图", "Simultaneously estimating a map and the robot's position within it.", "同时估计地图和机器人在其中的位置。", "Loop closure greatly reduces drift in SLAM."),
    ],
    cards=[
        ("贝叶斯滤波两步？", r"预测：$\overline{bel}=\int p(x_t|u_t,x_{t-1})bel(x_{t-1})$；更新：$bel=\eta\,p(z_t|x_t)\overline{bel}$"),
        ("卡尔曼增益（一维）？", r"$K=\bar\sigma^2/(\bar\sigma^2+\sigma_z^2)$，测量越准 $K$ 越大。"),
        ("粒子滤波三步？", "按运动模型移动粒子 → 按测量似然加权 → 按权重重采样。"),
        ("定位的三类问题？", "位置跟踪、全局定位、绑架机器人问题。"),
    ],
)
