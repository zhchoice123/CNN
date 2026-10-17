CHAPTER = dict(
    no=1,
    title="线性代数：向量、矩阵与分解",
    title_en="Linear Algebra: Vectors, Matrices and Factorizations",
    group="数学基础",
    summary="机器学习的数据、参数和每一层网络都是矩阵运算。本章用列视角理解 $Ax=b$，掌握四个基本子空间、最小二乘投影、特征分解和奇异值分解（SVD），它们是线性回归、PCA 和神经网络的共同语言。",
    sources=["Strang《线性代数导论》第 1–7 章", "Deisenroth 等《机器学习的数学》第 2–4 章", "Patterson & Gibson 第 1 章“线性代数”"],
    learn=[
        dict(
            title="向量、线性组合与矩阵乘法的列视角",
            title_en="Vectors, linear combinations and the column picture",
            body=r"""
线性代数的核心只有两个运算：**向量相加**和**数乘**。把它们合起来就得到**线性组合** $c\,\mathbf v + d\,\mathbf w$。所有更高级的概念都从这里出发。

看待 $A\mathbf x$ 有两种视角：

- **行视角**：$A\mathbf x$ 的第 $i$ 个分量是 $A$ 的第 $i$ 行与 $\mathbf x$ 的点积。方程组的每一行是一个超平面，解是它们的交点。
- **列视角**：$A\mathbf x = x_1\mathbf a_1 + x_2\mathbf a_2 + \dots + x_n\mathbf a_n$，即用 $\mathbf x$ 的分量把 $A$ 的各列线性组合起来。

于是“$A\mathbf x=\mathbf b$ 有没有解”就变成了“$\mathbf b$ 能不能由 $A$ 的列组合出来”。在机器学习里，一个全连接层 $\mathbf y=W\mathbf x+\mathbf b$ 正是把输入各分量当作系数，对权重矩阵的列做线性组合。

**点积**与长度：$\mathbf v\cdot\mathbf w=\sum_i v_iw_i$，$\|\mathbf v\|=\sqrt{\mathbf v\cdot\mathbf v}$，夹角满足 $\cos\theta=\dfrac{\mathbf v\cdot\mathbf w}{\|\mathbf v\|\|\mathbf w\|}$。余弦相似度就是从这里来的。
""",
            points=[
                "矩阵乘向量 = 列的线性组合；矩阵乘矩阵 = 对右边矩阵的每一列分别做这件事。",
                "点积为 0 表示两个向量**正交**；余弦相似度只看方向、不看长度。",
                "在 $\\mathbb R^n$ 中，二维、三维空间里建立的几何直觉仍然成立，只是无法画出来。",
            ],
            code="ch01_column_picture.py",
            code_note="验证 $A\\mathbf x$ 等于列的线性组合，并用 SVD 求秩和零空间。",
            src="Strang 第 1–2 章；MML 第 2 章",
        ),
        dict(
            title="秩与四个基本子空间",
            title_en="Rank and the four fundamental subspaces",
            body=r"""
任何 $m\times n$ 矩阵 $A$ 都自带四个子空间，它们的维数都由**秩** $r$ 决定：

- **列空间** $C(A)\subseteq\mathbb R^m$：所有 $A\mathbf x$ 的集合，维数 $r$。
- **零空间** $N(A)\subseteq\mathbb R^n$：所有满足 $A\mathbf x=\mathbf 0$ 的 $\mathbf x$，维数 $n-r$。
- **行空间** $C(A^T)\subseteq\mathbb R^n$：维数 $r$。
- **左零空间** $N(A^T)\subseteq\mathbb R^m$：维数 $m-r$。

并且行空间 $\perp$ 零空间，列空间 $\perp$ 左零空间。这就是 Strang 所说的**线性代数基本定理**。

它直接回答了方程组的问题：$A\mathbf x=\mathbf b$ 有解当且仅当 $\mathbf b\in C(A)$；如果有解，完整解 = 一个特解 + 零空间中的任意向量。秩 $r$ 也说明了矩阵中真正“独立”的信息有多少：秩小于列数，意味着某些特征可以被其他特征线性表示（**多重共线性**）。
""",
            points=[
                "秩 = 主元个数 = 线性无关的列数 = 线性无关的行数。",
                "$r=n$（列满秩）时零空间只有零向量，解若存在则唯一。",
                "$r=m$（行满秩）时对任意 $\\mathbf b$ 都有解。",
                "数据矩阵秩不满时，$X^TX$ 不可逆，线性回归需要正则化或删去冗余特征。",
            ],
            src="Strang 第 3 章；MML 2.6–2.7",
        ),
        dict(
            title="正交、投影与最小二乘",
            title_en="Orthogonality, projection and least squares",
            body=r"""
当方程比未知数多（数据点多于参数）时，$A\mathbf x=\mathbf b$ 通常无解。退而求其次：找到使误差 $\|A\mathbf x-\mathbf b\|^2$ 最小的 $\hat{\mathbf x}$。几何上，这就是把 $\mathbf b$ **投影**到 $A$ 的列空间上，投影点 $\mathbf p=A\hat{\mathbf x}$，残差 $\mathbf e=\mathbf b-\mathbf p$ 与列空间垂直：

$$A^T(\mathbf b-A\hat{\mathbf x})=\mathbf 0 \quad\Longrightarrow\quad A^TA\,\hat{\mathbf x}=A^T\mathbf b$$

这就是**正规方程**。投影矩阵 $P=A(A^TA)^{-1}A^T$ 满足 $P^2=P$、$P^T=P$。

线性回归就是这个投影问题：设计矩阵的每一列是一个特征（第一列全为 1 表示截距），$\hat{\mathbf x}$ 就是回归系数。实际计算中通常不直接求逆，而用 **QR 分解**（$A=QR$，$Q$ 的列标准正交）或 SVD，数值上更稳定。Gram–Schmidt 正交化就是得到 $Q$ 的过程。
""",
            points=[
                "最小二乘的核心条件：残差与每一个特征列都正交（$A^T\\mathbf e=\\mathbf 0$）。",
                "投影矩阵投影两次与投影一次结果相同（$P^2=P$）。",
                "标准正交矩阵 $Q$ 满足 $Q^TQ=I$，用它计算既简单又稳定。",
                "从概率角度看，高斯噪声下的最大似然估计正好等价于最小二乘（见第 4 章）。",
            ],
            code="ch01_least_squares.py",
            code_note="用正规方程和 `lstsq` 拟合一条带噪声的直线，验证残差与列空间正交、投影矩阵幂等。",
            src="Strang 第 4 章；MML 3.8、第 9 章",
        ),
        dict(
            title="特征值、正定矩阵与奇异值分解",
            title_en="Eigenvalues, positive definiteness and SVD",
            body=r"""
**特征向量**是被矩阵作用后方向不变的向量：$A\mathbf x=\lambda\mathbf x$。若 $A$ 有 $n$ 个线性无关的特征向量，就能对角化：$A=S\Lambda S^{-1}$，于是 $A^k=S\Lambda^kS^{-1}$，矩阵的幂和动力系统的长期行为由特征值决定（$|\lambda|<1$ 的分量衰减）。

**对称矩阵**最好：特征值都是实数，特征向量可选成正交的，$A=Q\Lambda Q^T$（谱定理）。若所有特征值都大于 0，矩阵是**正定**的，此时 $\mathbf x^TA\mathbf x>0$ 对任意非零 $\mathbf x$ 成立，对应一个“碗形”的二次函数——优化中的海森矩阵正定，意味着到达了局部极小值。协方差矩阵总是对称半正定的。

**奇异值分解（SVD）**对任意矩阵都成立：

$$A=U\Sigma V^T=\sum_{i=1}^{r}\sigma_i\,\mathbf u_i\mathbf v_i^T$$

$U$、$V$ 是正交矩阵，奇异值 $\sigma_1\ge\sigma_2\ge\dots\ge0$。只保留前 $k$ 项，就得到误差最小的秩 $k$ 近似（Eckart–Young 定理）。这正是 **PCA**、图像压缩、推荐系统矩阵分解和潜在语义分析的数学核心。
""",
            points=[
                "特征值分解只适用于方阵，SVD 适用于任意形状的矩阵。",
                "$A^TA$ 的特征值是 $A$ 的奇异值的平方，$V$ 的列是 $A^TA$ 的特征向量。",
                "奇异值下降得越快，矩阵越接近低秩，可压缩性越强。",
                "数据中心化后做 SVD，右奇异向量就是主成分方向（第 5 章）。",
            ],
            code="ch01_svd.py",
            code_note="构造一个近似秩 3 的矩阵，观察奇异值的断崖和不同秩的近似误差。",
            src="Strang 第 6 章（6.1–6.7）；MML 第 4 章",
        ),
    ],
    examples=[
        dict(
            title="四个子空间的维数",
            problem=r"矩阵 $A=\begin{bmatrix}1&2&3\\2&4&6\\1&0&1\end{bmatrix}$。求秩，写出四个子空间的维数，并找出零空间的一个基向量。",
            steps=[
                r"第二行是第一行的 2 倍，消元后只剩两个主元，所以 $r=2$。",
                r"$m=n=3$：列空间维数 $r=2$，零空间 $n-r=1$，行空间 $2$，左零空间 $m-r=1$。",
                r"解 $A\mathbf x=\mathbf 0$：由第三行 $x_1+x_3=0$，第一行 $x_1+2x_2+3x_3=0$。令 $x_3=1$，得 $x_1=-1$，$x_2=-1$。",
                r"零空间基向量 $(-1,-1,1)$。代码中 SVD 给出的是它的单位化版本 $(-0.577,-0.577,0.577)$。",
            ],
            answer=r"$r=2$；维数 2、1、2、1；零空间由 $(-1,-1,1)$ 张成。",
            code="ch01_column_picture.py",
        ),
        dict(
            title="最小二乘拟合直线",
            problem=r"用直线 $b=C+Dt$ 拟合三个点 $(0,6),(1,0),(2,0)$，求最佳的 $C,D$ 和误差平方和。",
            steps=[
                r"写成 $A\mathbf x=\mathbf b$：$A=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}$，$\mathbf b=(6,0,0)$，三个方程两个未知数，无精确解。",
                r"计算 $A^TA=\begin{bmatrix}3&3\\3&5\end{bmatrix}$，$A^T\mathbf b=(6,0)$。",
                r"解正规方程 $3C+3D=6,\ 3C+5D=0$，得 $D=-3$，$C=5$。",
                r"误差 $\mathbf e=\mathbf b-A\hat{\mathbf x}=(6,0,0)-(5,2,-1)=(1,-2,1)$，误差平方和为 $6$；可验证 $A^T\mathbf e=\mathbf 0$。",
            ],
            answer=r"$b=5-3t$，误差平方和 6。",
            code="ch01_ex_line.py",
        ),
        dict(
            title="用特征值计算矩阵的高次幂",
            problem=r"$A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$，求特征值、特征向量，并计算 $A^{10}$。",
            steps=[
                r"特征方程 $\det(A-\lambda I)=(2-\lambda)^2-1=0$，得 $\lambda_1=1,\ \lambda_2=3$。",
                r"$\lambda=1$：$(A-I)\mathbf x=0$ 得 $\mathbf x=(1,-1)$；$\lambda=3$：得 $\mathbf x=(1,1)$。两者正交，因为 $A$ 对称。",
                r"$A=Q\Lambda Q^T$，$A^{10}=Q\Lambda^{10}Q^T=\tfrac12\begin{bmatrix}3^{10}+1&3^{10}-1\\3^{10}-1&3^{10}+1\end{bmatrix}$。",
                r"$3^{10}=59049$，所以 $A^{10}=\begin{bmatrix}29525&29524\\29524&29525\end{bmatrix}$。",
            ],
            answer=r"$\lambda=1,3$；$A^{10}=\begin{bmatrix}29525&29524\\29524&29525\end{bmatrix}$。",
            code="ch01_ex_eigen.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="一个 $5\\times8$ 矩阵的秩为 3，它的零空间维数是：", options=["2", "3", "5", "8"], answer=2,
                 explain="零空间维数 = $n-r=8-3=5$。"),
            dict(q="$A\\mathbf x=\\mathbf b$ 有解的充要条件是：", options=["$A$ 是方阵", "$\\mathbf b$ 在 $A$ 的列空间中", "$\\mathbf b=\\mathbf 0$", "$A$ 的行列式不为 0"], answer=1,
                 explain="$A\\mathbf x$ 就是列的线性组合，所以 $\\mathbf b$ 必须能由列组合出来。"),
            dict(q="最小二乘解满足的正规方程是：", options=["$A\\hat{\\mathbf x}=\\mathbf b$", "$AA^T\\hat{\\mathbf x}=\\mathbf b$", "$A^TA\\hat{\\mathbf x}=A^T\\mathbf b$", "$A^T\\hat{\\mathbf x}=\\mathbf b$"], answer=2,
                 explain="由残差与列空间正交 $A^T(\\mathbf b-A\\hat{\\mathbf x})=0$ 推出。"),
            dict(q="下列关于实对称矩阵的说法，错误的是：", options=["特征值都是实数", "不同特征值的特征向量正交", "一定可以对角化", "一定是正定的"], answer=3,
                 explain="对称矩阵可以有负特征值，不一定正定。"),
            dict(q="对一个矩阵做 SVD 后只保留前 $k$ 个奇异值，得到的是：", options=["误差最小的秩 $k$ 近似", "矩阵的逆", "矩阵的转置", "矩阵的特征向量"], answer=0,
                 explain="Eckart–Young 定理。"),
            dict(q="两个向量的余弦相似度为 0，说明它们：", options=["长度相等", "方向相同", "正交", "线性相关"], answer=2,
                 explain="$\\cos\\theta=0$ 即夹角 90°。"),
        ],
        tf=[
            dict(q="矩阵乘向量 $A\\mathbf x$ 可以看成 $A$ 的各列以 $\\mathbf x$ 的分量为系数的线性组合。", answer=True, explain="这是列视角。"),
            dict(q="行空间与列空间的维数一般不相等。", answer=False, explain="两者维数都等于秩 $r$。"),
            dict(q="投影矩阵 $P$ 满足 $P^2=P$。", answer=True, explain="投影两次与投影一次相同。"),
            dict(q="特征值分解适用于任意形状的矩阵。", answer=False, explain="特征值分解只对方阵定义；任意矩阵用 SVD。"),
            dict(q="$A^TA$ 的特征值等于 $A$ 的奇异值的平方。", answer=True, explain="$A^TA=V\\Sigma^2V^T$。"),
        ],
        fill=[
            dict(q="零空间与 ____ 空间互为正交补（中文填写）。", answer=["行", "行空间"], explain="零空间 ⊥ 行空间。"),
            dict(q="把任意矩阵分解为 $U\\Sigma V^T$ 的方法叫做 ____（英文缩写）。", answer=["SVD", "奇异值分解"], explain="Singular Value Decomposition。"),
            dict(q="所有特征值都大于 0 的对称矩阵称为 ____ 矩阵。", answer=["正定", "positive definite", "正定矩阵"], explain="positive definite。"),
            dict(q="用 Gram–Schmidt 正交化可以得到矩阵的 ____ 分解（两个字母）。", answer=["QR"], explain="$A=QR$。"),
        ],
        reading=[
            dict(
                title="Why Matrices Sit at the Heart of Machine Learning",
                passage="""A dataset is usually stored as a matrix: each row is an example and each column is a feature. Once data takes this form, many learning problems become questions about matrices. Fitting a linear model asks which combination of the columns comes closest to the target vector. The answer is a projection, and the leftover error is perpendicular to every feature column.

Factorizations reveal structure that raw numbers hide. The singular value decomposition writes any matrix as a sum of simple rank-one pieces, ordered from most to least important. When the first few singular values are large and the rest are tiny, the data really lives in a low-dimensional subspace. Keeping only the leading pieces compresses the data while losing very little information. This single idea underlies principal component analysis, image compression and many recommender systems.

Neural networks are matrices too. Every fully connected layer multiplies its input by a weight matrix and adds a bias. Training a network is, to a large extent, a long sequence of matrix multiplications running on hardware built to do exactly that.""",
                zh="""数据集通常以矩阵形式存储：每一行是一个样本，每一列是一个特征。数据一旦变成这种形式，许多学习问题就变成了关于矩阵的问题。拟合线性模型就是在问：哪种列的组合最接近目标向量？答案是一个投影，剩下的误差与每一个特征列都垂直。

矩阵分解能揭示原始数字背后隐藏的结构。奇异值分解把任意矩阵写成一系列简单的秩一部分之和，并按重要性从大到小排列。如果前几个奇异值很大、其余很小，说明数据实际上位于一个低维子空间中。只保留最重要的几项，就能在几乎不损失信息的情况下压缩数据。主成分分析、图像压缩和许多推荐系统都建立在这一思想之上。

神经网络同样是矩阵。每一个全连接层都把输入乘以权重矩阵再加上偏置。训练网络在很大程度上就是一长串矩阵乘法，运行在专门为此设计的硬件上。""",
                questions=[
                    dict(q="In the passage, the error of a fitted linear model is described as:",
                         options=["parallel to the target", "perpendicular to every feature column", "always zero", "equal to the largest singular value"],
                         answer=1, explain="原文：leftover error is perpendicular to every feature column。"),
                    dict(q="What does it suggest when only the first few singular values are large?",
                         options=["The matrix cannot be factorized", "The data lives in a low-dimensional subspace", "The model is overfitting", "The features are all orthogonal"],
                         answer=1, explain="少数大奇异值说明数据近似低维。"),
                    dict(q="Which application is NOT mentioned as relying on SVD-style low-rank ideas?",
                         options=["Principal component analysis", "Image compression", "Recommender systems", "Password hashing"],
                         answer=3, explain="文中没有提到密码哈希。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("vector", "向量", "An ordered list of numbers, viewed as a point or an arrow in space.", "一组有序的数，可看作空间中的点或箭头。", "Each image can be flattened into a long vector of pixel values."),
        ("matrix", "矩阵", "A rectangular array of numbers arranged in rows and columns.", "按行和列排列的矩形数表。", "The weight matrix of this layer has 512 rows and 784 columns."),
        ("linear combination", "线性组合", "A sum of vectors each multiplied by a scalar coefficient.", "各向量乘以标量系数后求和。", "The output is a linear combination of the input features."),
        ("dot product", "点积", "The sum of products of corresponding entries of two vectors.", "两个向量对应分量乘积之和。", "Cosine similarity divides the dot product by the two lengths."),
        ("rank", "秩", "The number of linearly independent columns (or rows) of a matrix.", "矩阵中线性无关的列（或行）的个数。", "A rank-deficient design matrix signals redundant features."),
        ("column space", "列空间", "The set of all linear combinations of a matrix's columns.", "矩阵各列所有线性组合构成的集合。", "The system has a solution only if b lies in the column space."),
        ("null space", "零空间", "All vectors x such that Ax equals the zero vector.", "所有使 Ax 等于零向量的 x 构成的集合。", "Adding any null space vector does not change Ax."),
        ("orthogonal", "正交的", "Having a dot product of zero; perpendicular.", "点积为零，即相互垂直。", "The residual is orthogonal to every column of A."),
        ("projection", "投影", "The closest point to a vector within a given subspace.", "向量在给定子空间中最近的那个点。", "Least squares finds the projection of b onto the column space."),
        ("least squares", "最小二乘", "Choosing parameters that minimize the sum of squared errors.", "选择使误差平方和最小的参数。", "Linear regression is usually solved by least squares."),
        ("eigenvalue", "特征值", "A scalar lambda such that Ax equals lambda times x for some nonzero x.", "使 Ax 等于 λx（x 非零）成立的标量 λ。", "The largest eigenvalue controls how fast the powers of A grow."),
        ("eigenvector", "特征向量", "A nonzero vector whose direction is unchanged by the matrix.", "被矩阵作用后方向不变的非零向量。", "The steady state of a Markov chain is an eigenvector."),
        ("positive definite", "正定的", "A symmetric matrix whose eigenvalues are all positive.", "所有特征值都为正的对称矩阵。", "A positive definite Hessian means we are at a local minimum."),
        ("singular value decomposition", "奇异值分解", "Factoring any matrix as U times Sigma times V transpose.", "把任意矩阵分解为 UΣVᵀ。", "The singular value decomposition gives the best low-rank approximation."),
    ],
    cards=[
        ("四个基本子空间的维数？", "列空间 $r$、零空间 $n-r$、行空间 $r$、左零空间 $m-r$。"),
        ("正规方程？", r"$A^TA\hat{\mathbf x}=A^T\mathbf b$，来自残差与列空间正交。"),
        ("对称矩阵的谱分解？", r"$A=Q\Lambda Q^T$，特征值为实数，特征向量正交。"),
        ("Eckart–Young 定理说什么？", "截断 SVD（保留前 $k$ 个奇异值）给出 Frobenius 范数意义下误差最小的秩 $k$ 近似。"),
    ],
)
