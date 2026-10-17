CHAPTER = dict(
    no=3,
    title="机器学习基础：泛化与模型评估",
    title_en="Machine Learning Fundamentals: Generalization and Evaluation",
    group="经典机器学习",
    summary="什么是学习、怎样判断学得好不好。本章讲机器学习的分类、偏差与方差的权衡、训练集/验证集/测试集与交叉验证，以及准确率之外的评估指标：精确率、召回率、F1 和 ROC/AUC。",
    sources=["James 等《统计学习导论》第 2、5 章", "Flach《机器学习》序章、第 1–2 章", "Géron《机器学习实战》第 1–3 章", "Bishop《PRML》1.1–1.3", "Kelleher《深度学习》第 1–2 章"],
    learn=[
        dict(
            title="什么是机器学习：任务、数据与模型",
            title_en="What machine learning is",
            body=r"""
传统编程是“规则 + 数据 → 答案”；机器学习是“数据 + 答案 → 规则”（Chollet）。Kelleher 把一个机器学习项目归纳为三要素：**数据**（历史样例）、**一组候选函数**（模型/假设空间），以及**衡量匹配程度的标准**（损失函数）。邱锡鹏的说法与此对应：**模型、学习准则、优化算法**。

按监督信号分类：

- **有监督学习**：有输入 $\mathbf x$ 和标签 $y$。$y$ 是连续值就是**回归**，是类别就是**分类**。
- **无监督学习**：只有输入，目标是发现结构，例如聚类、降维、密度估计。
- **半监督学习**：少量有标签 + 大量无标签数据。
- **强化学习**：智能体与环境交互，从奖励中学习策略（第 14 章）。

Géron 还从另外两个角度分类：**批量学习**与**在线学习**；**基于实例**（如 K 近邻，直接记住样本）与**基于模型**（如线性回归，学出参数）。

ISLR 区分了建模的两个目的：**预测**只关心预测准不准，模型可以是黑箱；**推断**想知道哪些变量重要、关系是什么，模型必须可解释。灵活性与可解释性往往此消彼长。
""",
            points=[
                "机器学习的主要挑战：数据不足、数据没有代表性、数据质量差、特征不相关、过拟合、欠拟合。",
                "特征是机器学习的“主力”（Flach）：好特征往往比换模型更有效。",
                "“没有免费午餐定理”：不存在在所有问题上都最好的算法。",
                "把数据可视化、建立一个常识性的基线，是每个项目的第一步。",
            ],
            src="ISLR 2.1；Géron 第 1 章；Kelleher 第 1 章；Flach 第 1 章；邱锡鹏 2.2",
        ),
        dict(
            title="偏差—方差权衡、过拟合与欠拟合",
            title_en="The bias–variance trade-off",
            body=r"""
我们真正关心的是模型在**没见过的数据**上的表现，即**泛化误差**。对平方损失，期望测试误差可以分解为：

$$\mathbb E\big[(y-\hat f(x))^2\big]=\underbrace{\big(\mathbb E[\hat f(x)]-f(x)\big)^2}_{\text{偏差}^2}+\underbrace{\mathrm{Var}[\hat f(x)]}_{\text{方差}}+\underbrace{\sigma^2}_{\text{不可约误差}}$$

- **偏差**：模型太简单，连训练数据的基本规律都抓不住，训练误差和测试误差都高，叫**欠拟合**。
- **方差**：模型太复杂，把训练数据的噪声也当成规律，换一批数据结果就面目全非，训练误差很低而测试误差很高，叫**过拟合**。

Bishop 用多项式拟合正弦曲线演示了这一点：10 个数据点，用 9 阶多项式能穿过每一个点，训练误差为 0，但在点与点之间剧烈震荡。对策有三：降低模型复杂度、加**正则化**、增加数据量。

随着模型灵活性增加，训练误差单调下降，测试误差呈 **U 形**——这是 ISLR 中最重要的一张图。
""",
            points=[
                "训练误差低不代表模型好，泛化误差才是目标。",
                "高偏差 → 增加模型容量/特征；高方差 → 正则化、更多数据、简化模型。",
                "学习曲线（误差随训练样本数的变化）可以帮助诊断是偏差问题还是方差问题。",
                "不可约误差来自数据本身的噪声，任何模型都消除不了。",
            ],
            code="ch03_bias_variance.py",
            code_note="复现 Bishop 的多项式曲线拟合：比较不同阶数的训练与测试误差，再看增加数据量的效果。",
            src="ISLR 2.2；Bishop 1.1、3.2；Géron 第 4 章“学习曲线”",
        ),
        dict(
            title="训练集、验证集、测试集与交叉验证",
            title_en="Train, validation, test and cross-validation",
            body=r"""
数据要分三份：**训练集**用于拟合参数；**验证集**用于选择模型和超参数；**测试集**只在最后用一次，估计最终泛化能力。如果反复用测试集调参，测试集的信息就“泄露”进了模型，结果会过于乐观。

数据不多时，单次划分的验证结果噪声很大，于是用 **K 折交叉验证**：把数据分成 $K$ 份，轮流用其中 1 份做验证、其余做训练，取 $K$ 次结果的平均。$K=5$ 或 $10$ 通常在偏差与方差之间取得较好的平衡；$K=n$ 时就是留一法。

划分数据时要注意：

- 类别不平衡或有重要分组时用**分层抽样**，保证各份比例一致（Géron 用收入分层切分加州房价数据）；
- 时间序列不能随机打乱，要用“过去训练、未来验证”；
- 预处理（如标准化）的参数只能在训练集上学，再应用到验证/测试集，否则会发生**数据泄露**。把预处理和模型放进同一个**流水线**可以避免这一点。

**自助法**（bootstrap）通过有放回抽样估计统计量的不确定性，也是装袋法和随机森林的基础。
""",
            points=[
                "测试集是“最后一次考试”，只能用一次。",
                "交叉验证用于选择超参数，例如 K 近邻的 K、正则化强度、多项式阶数。",
                "K 近邻中 K 太小过拟合（训练准确率 100%），K 太大欠拟合。",
                "一切从数据中“学”的步骤（标准化、特征选择、填补缺失值）都要放在交叉验证的每一折内部。",
            ],
            code="ch03_knn_cv.py",
            code_note="手写 K 近邻，用 5 折交叉验证在带噪声的数据上选择 K，观察训练准确率与验证准确率的差距。",
            src="ISLR 第 5 章；Géron 第 2 章；Glassner 附加章 B1“交叉验证与流水线”",
        ),
        dict(
            title="分类评估：混淆矩阵、精确率、召回率与 ROC",
            title_en="Evaluating classifiers",
            body=r"""
**准确率**在类别不平衡时会骗人：如果只有 10% 的样本是正类，一个永远预测负类的模型准确率也有 90%（Géron 的“不是 5”分类器）。所以要看**混淆矩阵**：

| | 预测正 | 预测负 |
|---|---|---|
| **实际正** | TP | FN |
| **实际负** | FP | TN |

- **精确率** $P=\dfrac{TP}{TP+FP}$：预测为正的里面有多少是真的。
- **召回率**（灵敏度）$R=\dfrac{TP}{TP+FN}$：真的正例里找回了多少。
- **F1** $=\dfrac{2PR}{P+R}$：二者的调和平均。

大多数分类器输出的是分数，用阈值把分数变成类别。提高阈值，精确率上升、召回率下降，二者需要按业务权衡：癌症筛查更看重召回率，垃圾邮件过滤更看重精确率。

**ROC 曲线**画出随阈值变化的真阳性率与假阳性率。曲线下面积 **AUC** 有一个直观含义（Flach）：随机取一个正例和一个负例，模型给正例打分更高的概率。AUC 与阈值无关，适合比较模型的排序能力。
""",
            points=[
                "类别不平衡时优先看精确率、召回率、F1 或 PR 曲线，而不是准确率。",
                "阈值是业务决策，不是模型本身的属性。",
                "AUC=0.5 相当于随机猜，AUC=1 表示完美排序。",
                "多分类可以对每一类计算指标再做宏平均或微平均。",
            ],
            code="ch03_metrics.py",
            code_note="在正类只占 10% 的数据上，比较三个阈值下的各项指标，并直接按定义计算 AUC。",
            src="Géron 第 3 章；Flach 第 2 章；Patterson & Gibson 第 1 章“混淆矩阵”",
        ),
    ],
    examples=[
        dict(
            title="由混淆矩阵计算指标",
            problem=r"某疾病分类器在 1000 人上的结果：TP=40，FP=10，FN=20，TN=930。求准确率、精确率、召回率、F1 和特异度，并评价这个模型。",
            steps=[
                r"准确率 $=(40+930)/1000=0.97$。",
                r"精确率 $=40/(40+10)=0.80$；召回率 $=40/(40+20)\approx0.667$。",
                r"F1 $=2\times0.8\times0.667/(0.8+0.667)\approx0.727$；特异度 $=930/(930+10)\approx0.989$。",
                r"准确率看起来很高，但召回率只有 2/3：每 3 个病人就漏掉 1 个。对筛查任务来说，应该降低阈值以提高召回率。",
            ],
            answer="准确率 0.970，精确率 0.800，召回率 0.667，F1 0.727，特异度 0.989。",
            code="ch03_ex_prf.py",
        ),
        dict(
            title="诊断过拟合还是欠拟合",
            problem=r"用 10 个点拟合多项式：1 阶模型训练 RMSE 0.30、测试 RMSE 0.53；9 阶模型训练 RMSE 0.00、测试 RMSE 极大。分别属于什么问题？应该怎么改进？",
            steps=[
                r"1 阶模型训练误差和测试误差都较高，模型连训练数据的规律都抓不住：**高偏差（欠拟合）**。",
                r"改进 1 阶模型：增加模型复杂度，例如用 3 阶多项式（测试 RMSE 降到约 0.28）。",
                r"9 阶模型训练误差为 0、测试误差巨大：**高方差（过拟合）**。",
                r"改进 9 阶模型：增加数据（200 个点时测试 RMSE 约 0.26）、加正则化，或者降低阶数。",
            ],
            answer="1 阶欠拟合，9 阶过拟合；前者增加容量，后者加数据或正则化。",
            code="ch03_bias_variance.py",
        ),
        dict(
            title="用交叉验证选择 K 近邻的 K",
            problem=r"在带 10% 标签噪声的二维数据上，对 $K\in\{1,3,7,15,31,101\}$ 做 5 折交叉验证。为什么 $K=1$ 的训练准确率是 100%，却不是最好的选择？",
            steps=[
                r"$K=1$ 时，每个训练点的最近邻就是它自己，所以训练准确率必然是 100%，包括那些噪声标签也被“记住”了。",
                r"交叉验证在没参与训练的那一折上评估，$K=1$ 的验证准确率只有约 0.82，说明过拟合。",
                r"$K$ 增大时决策边界变平滑，$K=7$ 时验证准确率最高（约 0.86）。",
                r"$K=101$ 时边界过于平滑，几乎变成多数类投票，验证准确率掉到约 0.53，欠拟合。",
            ],
            answer="选择交叉验证准确率最高的 K=7；训练准确率不能用来选模型。",
            code="ch03_knn_cv.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="模型训练误差很低、测试误差很高，最可能的问题是：", options=["高偏差", "高方差", "数据量过大", "学习率过小"], answer=1, explain="过拟合即高方差。"),
            dict(q="以下哪项最不适合用来缓解过拟合？", options=["增加训练数据", "加入正则化", "增加模型参数量", "提前停止"], answer=2, explain="增加参数会进一步提高方差。"),
            dict(q="5 折交叉验证中，每个样本被用作验证数据的次数是：", options=["0 次", "1 次", "4 次", "5 次"], answer=1, explain="每个样本只在属于自己的那一折时做验证。"),
            dict(q="垃圾邮件过滤中，把正常邮件误判为垃圾邮件代价很高，应优先关注：", options=["召回率", "精确率", "训练速度", "AUC 之外的一切"], answer=1, explain="减少假阳性 FP 即提高精确率。"),
            dict(q="AUC 的一个直观解释是：", options=["模型的准确率", "随机取一正一负，正例得分更高的概率", "最佳阈值", "召回率与精确率之和"], answer=1, explain="Flach 对 AUC 的排序解释。"),
            dict(q="标准化等预处理的参数应该在哪里计算？", options=["全部数据", "只在训练集上", "只在测试集上", "验证集与测试集合并"], answer=1, explain="在全部数据上计算会造成数据泄露。"),
        ],
        tf=[
            dict(q="测试误差可以分解为偏差的平方、方差和不可约误差三部分。", answer=True, explain="平方损失下的偏差—方差分解。"),
            dict(q="类别极不平衡时，准确率是最可靠的评估指标。", answer=False, explain="傻瓜模型也能得到很高准确率。"),
            dict(q="K 近邻中 K 越大，决策边界越平滑。", answer=True, explain="更多邻居投票，模型方差降低、偏差升高。"),
            dict(q="可以根据测试集上的表现反复调整超参数。", answer=False, explain="这会让测试结果过于乐观，应使用验证集或交叉验证。"),
            dict(q="提高分类阈值通常会提高精确率、降低召回率。", answer=True, explain="预测为正的门槛更高，FP 减少、FN 增加。"),
        ],
        fill=[
            dict(q="模型连训练数据的规律都抓不住，叫做 ____ 拟合。", answer=["欠", "under", "underfitting", "欠拟合"], explain="underfitting。"),
            dict(q="F1 是精确率和召回率的 ____ 平均。", answer=["调和", "harmonic"], explain="harmonic mean。"),
            dict(q="召回率的公式是 TP / (TP + ____)。", answer=["FN", "假阴性"], explain="$R=TP/(TP+FN)$。"),
            dict(q="按类别比例划分数据集的方法叫 ____ 抽样。", answer=["分层", "stratified"], explain="stratified sampling。"),
        ],
        reading=[
            dict(
                title="Memorizing Is Not Learning",
                passage="""A student who memorizes last year's exam answers may score perfectly on a practice test made from the same questions, yet fail when the teacher writes new ones. Machine learning models can fall into exactly the same trap. A flexible model can memorize its training set, including the random noise in it, and still predict poorly on fresh data. This is called overfitting.

The opposite failure is a model that is too rigid to capture the real pattern. A straight line drawn through data that curves will be wrong everywhere, on old and new examples alike. This is underfitting.

To tell the two apart, we hold some data back. The model never trains on the held-out set, so its score there is an honest estimate of how it will behave in the real world. When data are scarce, cross-validation repeats this idea several times with different held-out slices and averages the results. The golden rule is simple: whatever you use to make decisions about the model must never include the data you use for the final report.""",
                zh="""一个背下去年考题答案的学生，在用同样题目出的练习卷上也许能拿满分，但老师一出新题就可能不及格。机器学习模型也会掉进同样的陷阱。一个灵活的模型能把训练集连同其中的随机噪声一起记下来，却在新数据上预测得很差。这叫过拟合。

相反的失败是模型太死板，抓不住真实的规律。在弯曲的数据上画一条直线，无论对旧样本还是新样本都会出错。这叫欠拟合。

为了区分二者，我们会预留一部分数据。模型从不在留出的数据上训练，所以它在那里的得分是对真实表现的诚实估计。数据稀缺时，交叉验证会用不同的留出部分把这个过程重复几次，再取平均。黄金法则很简单：任何用来对模型做决策的东西，都绝不能包含用于最终报告的数据。""",
                questions=[
                    dict(q="The student example in the passage illustrates:", options=["underfitting", "overfitting", "data augmentation", "regularization"], answer=1, explain="记住答案却不会新题，即过拟合。"),
                    dict(q="Why is the score on held-out data considered honest?", options=["Because it is always higher", "Because the model never trained on it", "Because it contains no noise", "Because it is larger than the training set"], answer=1, explain="模型没在留出数据上训练过。"),
                    dict(q="According to the golden rule, the data used for the final report should:", options=["also be used to tune the model", "be excluded from all model decisions", "be the training set", "be chosen after seeing results"], answer=1, explain="最终报告用的数据不能参与任何决策。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("supervised learning", "有监督学习", "Learning a mapping from inputs to known target labels.", "从输入到已知目标标签的映射学习。", "Spam filtering is a classic supervised learning task."),
        ("unsupervised learning", "无监督学习", "Finding structure in data without target labels.", "在没有标签的数据中发现结构。", "Clustering customers is a form of unsupervised learning."),
        ("regression", "回归", "Predicting a continuous numerical value.", "预测一个连续的数值。", "House price prediction is a regression problem."),
        ("classification", "分类", "Predicting which discrete category an input belongs to.", "预测输入属于哪个离散类别。", "Digit recognition is a ten-class classification problem."),
        ("generalization", "泛化", "How well a model performs on new, unseen data.", "模型在新的、未见过的数据上的表现。", "The goal of training is good generalization, not a perfect training score."),
        ("overfitting", "过拟合", "Fitting the noise in the training data so that test performance suffers.", "拟合了训练数据中的噪声，导致测试表现变差。", "A ninth-degree polynomial overfits ten noisy points."),
        ("underfitting", "欠拟合", "A model too simple to capture the true pattern in the data.", "模型过于简单，抓不住数据中的真实规律。", "A straight line underfits data that clearly curve."),
        ("bias", "偏差", "Error from overly simple assumptions in the model.", "模型过于简单的假设带来的误差。", "Adding features can reduce the bias of a linear model."),
        ("hyperparameter", "超参数", "A setting chosen before training rather than learned from data.", "训练前设定、而非从数据中学到的参数。", "The number of neighbors K is a hyperparameter."),
        ("validation set", "验证集", "Held-out data used to choose models and hyperparameters.", "用于选择模型和超参数的留出数据。", "We tuned the learning rate on the validation set."),
        ("cross-validation", "交叉验证", "Rotating which part of the data is held out and averaging the results.", "轮流留出不同部分数据并对结果取平均。", "Five-fold cross-validation gives a more stable estimate."),
        ("confusion matrix", "混淆矩阵", "A table counting true and false positives and negatives.", "统计真假阳性与真假阴性数量的表格。", "The confusion matrix shows which classes the model mixes up."),
        ("precision", "精确率", "The fraction of predicted positives that are truly positive.", "预测为正的样本中真正为正的比例。", "High precision means few false alarms."),
        ("recall", "召回率", "The fraction of actual positives that the model finds.", "实际为正的样本中被模型找出的比例。", "A cancer screening test needs very high recall."),
    ],
    cards=[
        ("偏差—方差分解？", r"测试误差 = 偏差² + 方差 + 不可约误差"),
        ("精确率与召回率？", r"$P=\frac{TP}{TP+FP}$，$R=\frac{TP}{TP+FN}$"),
        ("为什么不能用测试集调参？", "测试集的信息会泄露进模型，最终评估变得过于乐观。"),
        ("机器学习三要素（邱锡鹏）？", "模型、学习准则（损失）、优化算法。"),
    ],
)
