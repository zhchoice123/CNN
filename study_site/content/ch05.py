CHAPTER = dict(
    no=5,
    title="经典算法：树、集成、SVM、聚类与降维",
    title_en="Classic Algorithms: Trees, Ensembles, SVMs, Clustering and PCA",
    group="经典机器学习",
    summary="深度学习之外的“主力工具箱”：决策树怎样分裂，装袋、随机森林和提升法为什么有效，支持向量机的最大间隔与核技巧，以及 K 均值聚类和主成分分析。处理表格数据时，它们往往仍是首选。",
    sources=["ISLR 第 8–10 章", "Géron《机器学习实战》第 5–9 章", "Flach《机器学习》第 5、7–8、11 章", "MML 第 10–12 章", "Bishop 第 7、9、12、14 章", "Glassner 附加章 B1"],
    learn=[
        dict(
            title="决策树：用一连串问题切分特征空间",
            title_en="Decision trees",
            body=r"""
决策树通过一系列“是 / 否”问题（例如“温度 ≤ 84?”）把特征空间切成一个个矩形区域，每个叶节点给出一个预测：分类取多数类，回归取均值。它直观、可解释，能处理数值和类别特征，也不需要特征缩放。

怎样选择每一步的分裂？目标是让子节点尽量“纯”。常用的**不纯度**度量：

$$\text{基尼不纯度 } G=1-\sum_k p_k^2,\qquad \text{熵 } H=-\sum_k p_k\log_2 p_k$$

对每个特征的每个候选阈值，计算分裂后子节点不纯度的加权平均，选择下降最多的那个（下降量用熵衡量时就是**信息增益**）。这种贪心策略就是 CART 算法。回归树则选择使子节点平方误差之和最小的切分。

单棵树的缺点是**方差大**：数据稍有变化，树的结构就可能完全不同，深树极易过拟合。控制手段有限制最大深度、叶节点最少样本数、剪枝等。Géron 还指出，决策树的边界总是与坐标轴平行，对数据旋转很敏感。
""",
            points=[
                "基尼不纯度为 0 表示节点纯净；二分类时最大为 0.5。",
                "决策树是“分组模型”：同一叶节点内的样本得到相同预测（Flach）。",
                "树可以直接给出特征重要性：某特征带来的不纯度下降总量。",
                "单棵树不稳定，但正是这种不稳定让集成方法大放异彩。",
            ],
            code="ch05_tree_split.py",
            code_note="在一个 12 条记录的小数据集上，枚举所有阈值，计算加权基尼不纯度与信息增益。",
            src="ISLR 8.1；Géron 第 6 章；Flach 第 5 章",
        ),
        dict(
            title="集成学习：装袋、随机森林与提升",
            title_en="Ensembles: bagging, random forests and boosting",
            body=r"""
“三个臭皮匠，顶个诸葛亮”。把很多个模型的预测组合起来，往往比任何单个模型都好。

- **装袋（Bagging）**：对训练集做有放回的**自助采样**，得到许多略有不同的数据集，各训练一棵深树，预测时平均（回归）或投票（分类）。平均能显著降低**方差**。
- **随机森林**：在装袋基础上，每次分裂只允许从随机选出的一部分特征中挑选，让树与树之间更不相关，平均后方差降得更多。还能用**袋外样本**免费估计泛化误差。
- **提升（Boosting）**：一棵接一棵地训练**浅树**，每棵新树专门修正前面所有树的错误。AdaBoost 提高被分错样本的权重；**梯度提升**让新树拟合当前的残差（更一般地，损失函数的负梯度），每步乘一个小的学习率慢慢学。提升主要降低**偏差**。XGBoost、LightGBM 是它的高效实现，是表格数据竞赛的常胜冠军。
- **堆叠（Stacking）**：用一个元模型学习如何组合各个基模型的输出。

集成有效的关键是基模型**足够好且彼此不同**：犯的错误不相关，平均时才能相互抵消。
""",
            points=[
                "Bagging 用深树（低偏差、高方差），Boosting 用浅树（高偏差、低方差）。",
                "随机森林几乎不需要调参就能得到不错的结果，是很好的基线。",
                "梯度提升的学习率与树的数量需要一起调，学习率越小需要的树越多。",
                "Patterson & Gibson 提醒：对结构化表格数据，树模型常常不输深度学习。",
            ],
            code="ch05_ensembles.py",
            code_note="手写一维回归树，比较单棵深树、30 棵装袋树和 100 棵梯度提升浅树的测试误差。",
            src="ISLR 8.2；Géron 第 7 章；Flach 第 11 章；Bishop 14.2–14.3",
        ),
        dict(
            title="支持向量机：最大间隔与核技巧",
            title_en="Support vector machines",
            body=r"""
线性可分时，能把两类分开的直线有无数条。**支持向量机（SVM）**选择离两类样本都最远的那条，即**最大间隔**超平面 $\mathbf w^T\mathbf x+b=0$，间隔宽度为 $2/\|\mathbf w\|$。最大化间隔等价于：

$$\min_{\mathbf w,b}\ \tfrac12\|\mathbf w\|^2\quad\text{s.t. } y_i(\mathbf w^T\mathbf x_i+b)\ge1$$

现实数据往往不可分，于是引入**软间隔**：允许一些样本越界，用超参数 $C$ 权衡间隔宽度与越界惩罚。等价的无约束形式是**合页损失** $\max(0,1-y_i f(\mathbf x_i))$ 加 L2 正则。最终决策只取决于间隔边界上或越界的少数样本——**支持向量**，所以模型是稀疏的。

**核技巧**：SVM 的对偶形式只依赖样本之间的内积 $\mathbf x_i^T\mathbf x_j$。把内积换成核函数 $k(\mathbf x_i,\mathbf x_j)$（如多项式核、高斯 RBF 核），就相当于在高维甚至无限维的特征空间里做线性分类，却不必真的计算那个空间里的坐标。Bishop 和 MML 都从拉格朗日对偶推导了这一点。
""",
            points=[
                "$C$ 大：更少越界、间隔更窄、易过拟合；$C$ 小：间隔更宽、更多越界、更平滑。",
                "SVM 对特征尺度敏感，使用前必须标准化。",
                "RBF 核的 $\\gamma$ 越大，决策边界越弯曲。",
                "SVM 适合中小规模数据；样本量很大时训练代价高。",
            ],
            code="ch05_svm.py",
            code_note="用次梯度下降最小化“合页损失 + L2 正则”训练线性 SVM，计算间隔宽度和支持向量数量。",
            src="ISLR 第 9 章；Géron 第 5 章；MML 第 12 章；Bishop 第 7 章",
        ),
        dict(
            title="无监督学习：K 均值聚类与主成分分析",
            title_en="K-means and PCA",
            body=r"""
**K 均值聚类**把数据分成 $K$ 个簇，交替执行两步直到收敛：

1. **分配**：把每个样本分给最近的中心；
2. **更新**：把每个中心移到其所有样本的均值处。

它在最小化簇内平方和，每一步都不会让目标变大，所以一定收敛，但可能收敛到局部最优，因此常用多次随机初始化（或 k-means++）。$K$ 的选择可以看“肘部”或轮廓系数。K 均值假设簇是球形、大小相近；不满足时可用 DBSCAN 或**高斯混合模型**（用 EM 算法估计，是 K 均值的“软”版本）。

**主成分分析（PCA）**寻找数据方差最大的方向。数据中心化后，协方差矩阵的特征向量就是主成分方向，特征值就是该方向上的方差；等价地，对中心化数据矩阵做 SVD，右奇异向量就是主成分。MML 从两个角度推出同一个答案：**最大化投影方差**，或**最小化重建误差**。

用**解释方差比例**决定保留几个主成分（如累计 95%）。PCA 可用于降维、可视化、去噪和压缩，也能缓解**维数灾难**。注意 PCA 是线性方法，而且对特征尺度敏感。
""",
            points=[
                "K 均值需要事先指定 $K$，并且对初始化和离群点敏感。",
                "PCA 的主成分两两正交，按解释方差从大到小排列。",
                "高维空间中点与点的距离趋于相近，基于距离的方法会变差，这就是维数灾难。",
                "无监督学习没有标准答案，结果需要结合领域知识解读（ISLR）。",
            ],
            code="ch05_kmeans_pca.py",
            code_note="在三团数据上运行 K 均值；再对一个近似位于平面上的三维数据做 PCA，看解释方差比例。",
            src="ISLR 第 10 章；Géron 第 8–9 章；MML 第 10–11 章；Bishop 9.1、12.1",
        ),
    ],
    examples=[
        dict(
            title="计算基尼不纯度与最佳分裂",
            problem=r"一个节点有 12 个样本，其中 8 个正类、4 个负类。(1) 求基尼不纯度和熵。(2) 若按“温度 ≤ 84”分裂，左边 11 个样本中正类 8 个、负类 3 个，右边 1 个是负类，求分裂后的加权基尼不纯度。",
            steps=[
                r"(1) $p=(8/12,4/12)=(2/3,1/3)$，$G=1-(4/9+1/9)=4/9\approx0.444$。",
                r"熵 $H=-\tfrac23\log_2\tfrac23-\tfrac13\log_2\tfrac13\approx0.918$ 比特。",
                r"(2) 左子节点 $p=(8/11,3/11)$，$G_L=1-\tfrac{64+9}{121}=\tfrac{48}{121}\approx0.397$；右子节点只有负类，$G_R=0$。",
                r"加权平均 $=\tfrac{11}{12}\times0.397+\tfrac{1}{12}\times0\approx0.364$，比根节点的 0.444 低，且是所有候选阈值中最低的。",
            ],
            answer=r"根节点 $G\approx0.444$，$H\approx0.918$；该分裂后加权基尼约 0.364。",
            code="ch05_tree_split.py",
        ),
        dict(
            title="由协方差矩阵求主成分",
            problem=r"二维数据的协方差矩阵 $S=\begin{bmatrix}4&2\\2&3\end{bmatrix}$。求两个主成分的方差，以及第一主成分解释了多少比例的方差。",
            steps=[
                r"解 $\det(S-\lambda I)=(4-\lambda)(3-\lambda)-4=\lambda^2-7\lambda+8=0$。",
                r"$\lambda=\dfrac{7\pm\sqrt{17}}{2}$，即 $\lambda_1\approx5.562$，$\lambda_2\approx1.438$。",
                r"特征值就是主成分方向上的方差，总方差 $=\mathrm{tr}(S)=7$。",
                r"第一主成分解释比例 $=5.562/7\approx79.5\%$；方向为对应特征向量，约 $(0.788,0.615)$（符号无关紧要）。",
            ],
            answer="方差约 5.562 和 1.438，第一主成分解释约 79.5%。",
            code="ch05_ex_pca.py",
        ),
        dict(
            title="装袋与提升分别降低了什么",
            problem=r"同一回归任务上：单棵深度 8 的树测试 MSE 为 0.078，30 棵深树装袋后为 0.042，100 棵深度 2 的树做梯度提升后为 0.027。请解释这两种集成分别解决了什么问题。",
            steps=[
                r"深度 8 的树几乎能记住训练数据：偏差低、**方差高**。",
                r"装袋在不同的自助样本上训练多棵这样的树再平均，独立误差相互抵消，方差下降，所以 MSE 从 0.078 降到 0.042。",
                r"深度 2 的树非常简单：方差低、**偏差高**。梯度提升让每棵新树拟合前面的残差，逐步修正系统性误差，偏差不断下降。",
                r"因此：装袋 = 降方差，适合强而不稳定的基模型；提升 = 降偏差，适合弱而稳定的基模型。",
            ],
            answer="装袋主要降低方差，提升主要降低偏差。",
            code="ch05_ensembles.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="一个二分类节点中两类各占一半，其基尼不纯度为：", options=["0", "0.25", "0.5", "1"], answer=2, explain="$1-0.5^2-0.5^2=0.5$。"),
            dict(q="随机森林相比普通装袋的改进是：", options=["使用浅树", "每次分裂只从随机子集的特征中选择", "按顺序训练树", "对错分样本加权"], answer=1, explain="特征随机性让树之间更不相关。"),
            dict(q="梯度提升中，每一棵新树拟合的是：", options=["原始标签", "当前模型的残差（负梯度）", "随机噪声", "测试集"], answer=1, explain="逐步修正前面模型的误差。"),
            dict(q="SVM 的支持向量是指：", options=["所有训练样本", "位于间隔边界上或越界的样本", "离超平面最远的样本", "测试样本"], answer=1, explain="只有它们决定了决策边界。"),
            dict(q="K 均值算法的两个交替步骤是：", options=["前向与反向", "分配与更新", "编码与解码", "采样与拒绝"], answer=1, explain="分配样本到最近中心，再更新中心为均值。"),
            dict(q="PCA 中，主成分的方差等于协方差矩阵的：", options=["行列式", "迹", "特征值", "逆"], answer=2, explain="特征值就是该方向上的方差。"),
        ],
        tf=[
            dict(q="决策树需要先对特征做标准化。", answer=False, explain="树只比较阈值，与特征尺度无关。"),
            dict(q="装袋法主要用于降低模型的方差。", answer=True, explain="平均多个不稳定模型。"),
            dict(q="SVM 中超参数 C 越大，允许越界的样本越多。", answer=False, explain="C 越大越界惩罚越重，越界越少、间隔越窄。"),
            dict(q="K 均值一定能找到全局最优的聚类结果。", answer=False, explain="只保证收敛到局部最优，常需多次初始化。"),
            dict(q="核技巧让 SVM 不必显式计算高维特征就能在高维空间中做线性分类。", answer=True, explain="只需核函数计算内积。"),
        ],
        fill=[
            dict(q="有放回地从训练集中抽样得到新数据集的方法叫 ____ 采样（英文或中文）。", answer=["自助", "bootstrap"], explain="bootstrap。"),
            dict(q="SVM 的间隔宽度为 $2/$____。", answer=["||w||", "‖w‖", "|w|", "w的范数"], explain="$2/\\|\\mathbf w\\|$。"),
            dict(q="用信息熵衡量分裂效果时，熵的下降量叫做信息 ____。", answer=["增益", "gain"], explain="information gain。"),
            dict(q="高维空间中样本变得稀疏、距离失去区分度的现象叫维数 ____。", answer=["灾难", "curse"], explain="curse of dimensionality。"),
        ],
        reading=[
            dict(
                title="The Wisdom of Many Trees",
                passage="""A single decision tree is easy to read but notoriously unstable. Change a handful of training examples and the first split may move, sending the whole tree in a different direction. On its own, a deep tree tends to memorize its data.

Random forests turn this weakness into a strength. Each tree is grown on a bootstrap sample, a random draw of the training set with replacement, and at every split it may only consider a random subset of the features. The trees therefore disagree in useful ways. When their predictions are averaged, the idiosyncratic mistakes of individual trees cancel out, while the signal they share survives.

Boosting follows a different philosophy. Instead of growing many independent trees, it grows small trees one after another. Each new tree studies the errors left by the current ensemble and tries to fix them, so the model improves step by step. Bagging calms down models that are too nervous; boosting sharpens models that are too blunt. Both remain among the most reliable tools for tabular data.""",
                zh="""单棵决策树容易解读，但出了名地不稳定。只要改动几个训练样本，第一次分裂就可能改变，整棵树随之走向完全不同的方向。单独使用时，深树往往会把数据背下来。

随机森林把这个弱点变成了优势。每棵树在一个自助样本（对训练集的有放回随机抽取）上生长，并且每次分裂只能考虑一个随机的特征子集。于是这些树会以有益的方式产生分歧。把它们的预测平均后，各棵树特有的错误相互抵消，而它们共同捕捉到的信号得以保留。

提升法遵循另一种思路。它不是并行地种出许多独立的树，而是一棵接一棵地种小树。每棵新树研究当前集成留下的错误并试图修正，模型就这样一步步改进。装袋让过于“紧张”的模型冷静下来，提升让过于“迟钝”的模型变得敏锐。两者至今仍是处理表格数据最可靠的工具之一。""",
                questions=[
                    dict(q="Why do the trees in a random forest disagree?", options=["They use different loss functions", "Each uses a bootstrap sample and random feature subsets", "They are trained on the test set", "They have different depths only"], answer=1, explain="自助样本 + 随机特征子集。"),
                    dict(q="What happens when the forest averages its trees' predictions?", options=["Shared signal survives while individual mistakes cancel", "All errors are amplified", "Only the first tree matters", "The bias increases sharply"], answer=0, explain="个体错误抵消，共同信号保留。"),
                    dict(q="According to the passage, boosting is best described as:", options=["growing many independent deep trees", "growing small trees sequentially to fix earlier errors", "removing features one by one", "a clustering method"], answer=1, explain="依次种小树修正错误。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("decision tree", "决策树", "A model that predicts by asking a sequence of yes-or-no questions about features.", "通过对特征提出一连串是否问题来预测的模型。", "The decision tree first splits on temperature."),
        ("Gini impurity", "基尼不纯度", "The probability that two randomly drawn samples from a node have different labels.", "从节点中随机取两个样本、其标签不同的概率。", "A pure node has a Gini impurity of zero."),
        ("information gain", "信息增益", "The reduction in entropy achieved by a split.", "一次分裂带来的熵的减少量。", "The algorithm picks the split with the largest information gain."),
        ("ensemble", "集成", "A combination of several models whose predictions are aggregated.", "把多个模型的预测汇总起来的组合。", "An ensemble of weak models can be very strong."),
        ("bagging", "装袋法", "Training models on bootstrap samples and averaging their predictions.", "在自助样本上训练多个模型并平均其预测。", "Bagging mainly reduces variance."),
        ("random forest", "随机森林", "Bagged decision trees that also sample features at each split.", "每次分裂还对特征随机抽样的装袋决策树。", "A random forest is a strong baseline for tabular data."),
        ("boosting", "提升法", "Training weak models sequentially, each correcting its predecessors.", "依次训练弱模型、每个都修正前面模型的错误。", "Gradient boosting won many data science competitions."),
        ("support vector machine", "支持向量机", "A classifier that finds the separating boundary with the largest margin.", "寻找间隔最大的分隔边界的分类器。", "A support vector machine works well on small, clean datasets."),
        ("margin", "间隔", "The distance between the decision boundary and the closest training points.", "决策边界与最近训练样本之间的距离。", "Maximizing the margin improves generalization."),
        ("kernel trick", "核技巧", "Replacing inner products with a kernel to work implicitly in a high-dimensional space.", "用核函数替换内积，从而隐式地在高维空间中计算。", "The kernel trick lets a linear method draw curved boundaries."),
        ("clustering", "聚类", "Grouping similar samples together without labels.", "在没有标签的情况下把相似样本归为一组。", "Clustering revealed three customer segments."),
        ("k-means", "K 均值", "A clustering algorithm that alternates between assigning points and updating centroids.", "交替分配样本和更新中心的聚类算法。", "K-means assumes roughly spherical clusters."),
        ("principal component analysis", "主成分分析", "A linear method that projects data onto directions of maximum variance.", "把数据投影到方差最大方向上的线性方法。", "Principal component analysis reduced 50 features to 5."),
        ("curse of dimensionality", "维数灾难", "The many problems that appear when data live in very high-dimensional spaces.", "数据位于极高维空间时出现的种种问题。", "Nearest-neighbor methods suffer from the curse of dimensionality."),
    ],
    cards=[
        ("基尼不纯度与熵？", r"$G=1-\sum p_k^2$；$H=-\sum p_k\log_2 p_k$"),
        ("装袋 vs 提升？", "装袋：并行、深树、降方差；提升：串行、浅树、降偏差。"),
        ("SVM 的优化目标？", r"$\min\frac12\|\mathbf w\|^2$ s.t. $y_i(\mathbf w^T\mathbf x_i+b)\ge1$；软间隔用合页损失。"),
        ("PCA 的两种等价推导？", "最大化投影方差 = 最小化重建误差；解为协方差矩阵最大特征值对应的特征向量。"),
    ],
)
