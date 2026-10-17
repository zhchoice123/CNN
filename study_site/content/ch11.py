CHAPTER = dict(
    no=11,
    title="几何视觉：相机、对极几何与三维重建",
    title_en="Geometric Vision: Cameras, Epipolar Geometry and 3D Reconstruction",
    group="计算机视觉",
    summary="从二维照片恢复三维世界：针孔相机与齐次坐标、单应变换与全景拼接、特征匹配与 RANSAC、对极几何与基础矩阵、三角化与光束法平差。这些几何约束是 SLAM、三维重建、AR 和 NeRF 的共同地基。",
    sources=["Hartley & Zisserman《计算机视觉中的多视图几何》第 2、4、6、9–12、18 章", "Szeliski《计算机视觉：算法与应用》第 2、4、6–7、9、11 章", "Forsyth & Ponce 第 1、5、7–8、10 章", "Solem《Python 计算机视觉编程》第 2–5 章", "Torralba 等第 38–45 章"],
    learn=[
        dict(
            title="针孔相机模型与齐次坐标",
            title_en="The pinhole camera and homogeneous coordinates",
            body=r"""
**齐次坐标**给每个点多加一个分量：二维点 $(x,y)$ 写成 $(x,y,1)$，且 $(x,y,w)$ 与 $(kx,ky,kw)$ 表示同一个点。好处是：平移、旋转和透视投影都变成了矩阵乘法；平行线在“无穷远点”相交，所有无穷远点构成无穷远直线（Hartley & Zisserman 第 2 章）。

**针孔相机**把三维点 $\mathbf X$ 投影到图像点 $\mathbf x$：

$$\mathbf x \simeq K\,[R\mid\mathbf t]\,\mathbf X,\qquad K=\begin{bmatrix}f_x&s&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}$$

- **内参** $K$：焦距 $f$、主点 $(c_x,c_y)$、倾斜 $s$，描述相机自身；
- **外参** $R,\mathbf t$：相机在世界中的朝向和位置；
- 投影后要做**透视除法**（除以第三个分量），这就是“近大远小”的来源。

现实中的镜头还有**径向畸变**，需要用多项式模型校正。**相机标定**用已知的标定板（如棋盘格）求出内参和畸变系数。

透视投影不保持平行：现实中平行的直线在图像中汇聚到同一个**消失点**，其位置是 $K R\mathbf d$（$\mathbf d$ 为直线方向）。利用消失点可以从单张图像中测量几何量，甚至标定相机。
""",
            points=[
                "射影变换只保持共线性和交比；仿射变换还保持平行；相似变换还保持角度；欧氏变换保持一切距离。",
                "相机矩阵 $P=K[R\\mid\\mathbf t]$ 是 $3\\times4$ 的，有 11 个自由度。",
                "深度信息在投影中丢失了：同一条视线上的所有三维点投影到同一个像素。",
                "Forsyth & Ponce 第 1 章、Szeliski 2.1 都从针孔模型开始讲视觉。",
            ],
            code="ch11_pinhole.py",
            code_note="构造内外参，投影几个三维点；让两条平行线逐渐远去，验证它们汇聚到理论消失点 $KR\\mathbf d$。",
            src="Hartley & Zisserman 第 2、6、8 章；Szeliski 2.1；Forsyth & Ponce 第 1 章",
        ),
        dict(
            title="单应变换、特征匹配与 RANSAC",
            title_en="Homographies, feature matching and RANSAC",
            body=r"""
同一个**平面**在两幅图像之间的对应关系，或者相机绕光心**纯旋转**时拍到的两幅图像之间，都由一个 $3\times3$ 的**单应矩阵** $H$ 联系：$\mathbf x'\simeq H\mathbf x$。$H$ 有 8 个自由度，至少需要 4 对对应点。这是全景拼接、文档矫正和 AR 贴图的基础。

**求解流程**（Hartley & Zisserman 第 4 章确立的“标准方法论”）：

1. **检测并匹配特征**：用 Harris 角点或 **SIFT** 关键点及描述子，在两幅图之间按描述子距离匹配（常用比值检验去掉模糊匹配）。
2. **直接线性变换（DLT）**：每对点给出两个线性方程，堆成矩阵 $A$，用 SVD 求 $A\mathbf h=0$ 的最小二乘解。**一定要先做坐标归一化**（平移到原点、缩放到平均距离 $\sqrt2$），否则数值精度很差。
3. **RANSAC 去除错误匹配**：反复随机抽取最少样本（4 对点）拟合模型，统计误差小于阈值的“内点”数量，选择内点最多的模型，最后用全部内点重新估计。
4. **非线性优化**：最小化几何误差（重投影误差），即“金标准”算法。

RANSAC 所需的迭代次数 $N=\dfrac{\log(1-p)}{\log(1-w^s)}$（$w$ 为内点比例，$s$ 为最小样本数，$p$ 为成功概率）。样本数 $s$ 越大，需要的迭代越多。
""",
            points=[
                "代数误差容易计算，几何误差才有真实的物理意义。",
                "SIFT 对尺度、旋转和光照变化都比较稳定（Szeliski 4.1、Forsyth & Ponce 第 5 章）。",
                "最小二乘会被离群点严重拉偏；RANSAC 用“投票”找到大多数点支持的模型。",
                "Solem 第 3 章用 SIFT + RANSAC + 单应矩阵实现了全景拼接。",
            ],
            code="ch11_homography.py",
            code_note="60 对点中 30% 是错误匹配：比较直接 DLT 与 RANSAC+DLT 的结果。",
            src="Hartley & Zisserman 第 4 章；Szeliski 第 4、6、9 章；Solem 第 2–3 章；Forsyth & Ponce 10.4",
        ),
        dict(
            title="对极几何、基础矩阵与本质矩阵",
            title_en="Epipolar geometry",
            body=r"""
两台相机拍同一个场景。第一幅图中的一个点 $\mathbf x$，在第二幅图中的对应点 $\mathbf x'$ 必然落在一条直线上——**对极线**。这个约束由 $3\times3$ 的**基础矩阵** $F$ 描述：

$$\mathbf x'^{T}F\,\mathbf x=0$$

$F$ 的秩为 2，有 7 个自由度，只依赖于两台相机的相对几何，与场景无关。它把二维的匹配搜索缩小成沿对极线的一维搜索，这正是**立体匹配**的基础。

- **八点算法**：每对匹配点给出一个关于 $F$ 的线性方程，8 对点即可求解；求解后要强制秩为 2。Hartley 在 1997 年的著名论文“为八点算法辩护”中证明：只要做好**坐标归一化**，这个简单的算法就能达到很高的精度。
- **本质矩阵** $E=K'^TFK$：内参已知时使用，满足两个奇异值相等、第三个为 0，可以分解出两台相机之间的旋转 $R$ 和平移方向 $\mathbf t$（平移的绝对尺度无法从图像中恢复）。

**双目立体视觉**是特例：两台相机平行放置时，对极线是水平的，深度与视差成反比：$Z=\dfrac{fB}{d}$（$B$ 为基线长度，$d$ 为视差）。
""",
            points=[
                "对极约束把对应点的搜索从二维降到一维。",
                "只有两幅图像而不知道内参时，只能把场景重建到一个射影变换的程度（射影歧义）。",
                "单目视觉无法确定绝对尺度；双目或 IMU 可以提供尺度。",
                "代码中本质矩阵的两个非零奇异值几乎相等，验证了理论性质。",
            ],
            code="ch11_epipolar.py",
            code_note="生成两台相机和 50 个三维点，用归一化八点法估计 $F$，检验对极约束、$E$ 的奇异值，并三角化恢复三维点。",
            src="Hartley & Zisserman 第 9–11 章；Szeliski 7.2、第 11 章；Forsyth & Ponce 第 7 章；Solem 第 5 章",
        ),
        dict(
            title="三角化、运动恢复结构与光束法平差",
            title_en="Triangulation, structure from motion and bundle adjustment",
            body=r"""
已知两台相机和一对匹配点，求三维点的位置就是**三角化**：两条反投影射线的交点。由于噪声，射线通常不会恰好相交，可以用 DLT 求最小二乘解，或者最小化重投影误差。

**运动恢复结构（SfM）**从一组无序照片中同时恢复相机位姿和场景三维结构：

1. 提取并匹配所有图像的特征；
2. 用 RANSAC 估计图像对之间的几何关系（$F$ 或 $E$），选一对初始化；
3. 三角化得到初始点云，再逐张加入新图像（PnP 求位姿），不断三角化新点；
4. **光束法平差（Bundle Adjustment）**：同时优化所有相机参数和所有三维点，使所有点在所有图像上的重投影误差之和最小，通常用列文伯格—马夸尔特算法求解，并利用问题的稀疏结构加速。

Szeliski 参与的“照片旅游”（Photo Tourism）项目，就是用这种方法从网上成千上万张游客照片中重建出著名景点的三维模型。今天的开源工具 COLMAP、视觉 SLAM 系统（如 ORB-SLAM）的后端，以及 NeRF 和三维高斯泼溅训练前的相机位姿估计，用的都是这一套几何方法。深度学习可以替代特征提取与匹配（如学习得到的特征和匹配网络），但几何约束本身是物理规律，不会过时。
""",
            points=[
                "重投影误差 = 三维点投影到图像上的位置与实际观测位置之间的距离。",
                "光束法平差是一个大规模的非线性最小二乘问题。",
                "仿射相机假设下，因子分解法可以用一次 SVD 同时求出结构和运动（Tomasi–Kanade）。",
                "Torralba 等第 43–45 章讨论了从单张图像学习深度，以及辐射场（NeRF）等新方法。",
            ],
            code="ch11_ransac.py",
            code_note="用 RANSAC 在 40% 离群点中拟合直线，并计算不同内点比例、最小样本数下所需的迭代次数。",
            src="Hartley & Zisserman 第 12、18 章；Szeliski 第 7 章；Forsyth & Ponce 第 8 章；Torralba 第 44–45 章",
        ),
    ],
    examples=[
        dict(
            title="手算针孔投影",
            problem=r"相机内参 $f=800$，主点 $(320,240)$，相机位于世界原点、朝向 $z$ 轴（$R=I,\ \mathbf t=0$）。求三维点 $(1,0.5,4)$ 的像素坐标。若该点沿视线移到 $(2,1,8)$ 呢？",
            steps=[
                r"$K\mathbf X=(800\times1+320\times4,\ 800\times0.5+240\times4,\ 4)=(2080,1360,4)$。",
                r"透视除法：$u=2080/4=520$，$v=1360/4=340$。",
                r"点 $(2,1,8)$：$K\mathbf X=(4160,2720,8)$，除以 8 后同样是 $(520,340)$。",
                r"同一条视线上的点投影到同一个像素，深度信息丢失——这正是需要第二个视角的原因。",
            ],
            answer="两点都投影到像素 (520, 340)。",
            code="ch11_pinhole.py",
        ),
        dict(
            title="RANSAC 需要多少次迭代",
            problem=r"匹配点中内点比例为 50%。要以 99% 的概率至少抽到一组全是内点的样本，估计单应矩阵（最少 4 对点）和基础矩阵（八点法，8 对点）分别需要多少次迭代？",
            steps=[
                r"一次抽样全是内点的概率为 $w^s$：单应 $0.5^4=0.0625$，基础矩阵 $0.5^8\approx0.0039$。",
                r"$N=\log(1-p)/\log(1-w^s)$，$p=0.99$。",
                r"单应：$N=\log0.01/\log0.9375\approx71.4$，约 72 次。基础矩阵：$N=\log0.01/\log0.9961\approx1177$ 次。",
                r"最小样本数越少越好，这就是七点法、五点法（求本质矩阵）受欢迎的原因。内点比例升到 70% 时，八点法只需约 78 次。",
            ],
            answer="单应约 72 次，基础矩阵约 1177 次。",
            code="ch11_ransac.py",
        ),
        dict(
            title="双目视差求深度",
            problem=r"一对平行放置的双目相机，焦距 $f=700$ 像素，基线 $B=0.12$ 米。某点在左右图像中的横坐标分别为 410 和 382。求该点的深度。若视差测量误差为 1 像素，深度误差大约多少？",
            steps=[
                r"视差 $d=410-382=28$ 像素。",
                r"深度 $Z=fB/d=700\times0.12/28=3.0$ 米。",
                r"对 $d$ 求导：$|\partial Z/\partial d|=fB/d^2=84/784\approx0.107$ 米/像素。",
                r"1 像素的视差误差带来约 11 厘米的深度误差；距离越远视差越小，误差按 $Z^2$ 增长，所以双目测距在远处不准。",
            ],
            answer="深度 3.0 米，1 像素视差误差约对应 0.11 米的深度误差。",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="单应矩阵有多少个自由度？", options=["6", "7", "8", "9"], answer=2, explain="$3\\times3$ 矩阵，去掉一个尺度因子。"),
            dict(q="基础矩阵 $F$ 的秩是：", options=["1", "2", "3", "满秩"], answer=1, explain="对极约束要求 $\\det F=0$。"),
            dict(q="针孔模型中，描述焦距和主点的是：", options=["外参", "内参矩阵 $K$", "本质矩阵", "单应矩阵"], answer=1, explain="内参描述相机自身。"),
            dict(q="RANSAC 的主要作用是：", options=["加速矩阵求逆", "在存在大量离群点时稳健地估计模型", "检测角点", "压缩图像"], answer=1, explain="通过随机抽样与投票。"),
            dict(q="双目视觉中，视差越大说明物体：", options=["越远", "越近", "越亮", "越大"], answer=1, explain="$Z=fB/d$。"),
            dict(q="同时优化所有相机参数和三维点以最小化重投影误差的方法叫：", options=["八点法", "光束法平差", "直方图均衡化", "非极大值抑制"], answer=1, explain="Bundle Adjustment。"),
        ],
        tf=[
            dict(q="在齐次坐标中，$(2,4,2)$ 与 $(1,2,1)$ 表示同一个二维点。", answer=True, explain="都对应 $(1,2)$。"),
            dict(q="透视投影保持直线的平行关系。", answer=False, explain="平行线会汇聚到消失点。"),
            dict(q="使用 DLT 估计单应矩阵之前，应先对点坐标做归一化。", answer=True, explain="Hartley 归一化显著提高数值稳定性。"),
            dict(q="仅凭两幅图像，单目系统就能恢复场景的绝对尺度。", answer=False, explain="只能恢复到一个尺度因子。"),
            dict(q="对极约束把对应点的搜索范围从整幅图像缩小到一条直线。", answer=True, explain="对极线。"),
        ],
        fill=[
            dict(q="对极约束的形式是 $\\mathbf x'^TF\\mathbf x=$ ____。", answer=["0", "零"], explain="对应点必须满足。"),
            dict(q="内参已知时，与基础矩阵对应的矩阵叫 ____ 矩阵。", answer=["本质", "essential"], explain="$E=K'^TFK$。"),
            dict(q="现实中平行的直线在图像中汇聚的点叫做 ____ 点。", answer=["消失", "vanishing"], explain="vanishing point。"),
            dict(q="双目深度公式 $Z=fB/$____。", answer=["d", "视差"], explain="视差 $d$。"),
        ],
        reading=[
            dict(
                title="Rebuilding Rome from Tourist Photos",
                passage="""Millions of people photograph famous landmarks and share the pictures online. Each photo is taken from a different position, with a different camera, at a different time of day. Hidden in this messy collection is enough information to rebuild the landmark in three dimensions.

The process begins by finding distinctive points, such as corners and textured patches, in every image and matching them across photos. Many matches are wrong, so a robust method called RANSAC repeatedly tests small random subsets of matches and keeps the geometric model that most matches agree with. From a reliable pair of images, the relative position of the two cameras can be estimated and the matched points can be triangulated into 3D.

New photos are then added one at a time. Finally, an optimization called bundle adjustment fine-tunes every camera and every 3D point together so that the reconstructed points project back onto the images as accurately as possible. The result is a cloud of millions of points and the exact place from which each tourist took their picture.""",
                zh="""数以百万计的人拍摄著名地标并把照片分享到网上。每张照片的拍摄位置不同、相机不同、时间也不同。在这堆杂乱的照片中，隐藏着足以把地标重建成三维模型的信息。

这个过程首先在每张图像中找出有辨识度的点，比如角点和有纹理的小块，并在照片之间进行匹配。很多匹配是错误的，于是一种叫 RANSAC 的鲁棒方法会反复测试一小组随机挑选的匹配，保留大多数匹配都认可的几何模型。从一对可靠的图像出发，可以估计出两台相机的相对位置，并把匹配点三角化成三维点。

接着，新照片被一张张地加入。最后，一种叫光束法平差的优化会同时微调每一台相机和每一个三维点，使重建出的点投影回图像时尽可能准确。结果是一个由数百万个点组成的点云，以及每位游客拍照时所在的确切位置。""",
                questions=[
                    dict(q="Why is RANSAC needed in this pipeline?", options=["To make the photos brighter", "Because many feature matches are wrong", "To compress the images", "To choose the best tourist"], answer=1, explain="大量错误匹配需要鲁棒估计。"),
                    dict(q="What does bundle adjustment optimize?", options=["Only the image brightness", "All cameras and all 3D points together", "Only the first two cameras", "The number of photos"], answer=1, explain="同时优化所有相机与点。"),
                    dict(q="Besides a 3D point cloud, what else does the method recover?", options=["The names of the tourists", "Where each photo was taken from", "The weather forecast", "The camera prices"], answer=1, explain="每张照片的拍摄位置（相机位姿）。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("homogeneous coordinates", "齐次坐标", "Coordinates with an extra component so that projective transforms become matrix products.", "多加一个分量、使射影变换成为矩阵乘法的坐标表示。", "In homogeneous coordinates, parallel lines meet at infinity."),
        ("pinhole camera", "针孔相机", "An idealized camera model in which every ray passes through a single point.", "所有光线都经过同一点的理想相机模型。", "The pinhole camera model ignores lens distortion."),
        ("intrinsic parameters", "内参", "Camera properties such as focal length and principal point.", "焦距、主点等描述相机自身的参数。", "Calibration estimates the intrinsic parameters."),
        ("extrinsic parameters", "外参", "The rotation and translation that place the camera in the world.", "确定相机在世界中位置和朝向的旋转与平移。", "The extrinsic parameters change as the robot moves."),
        ("camera calibration", "相机标定", "Estimating a camera's intrinsic parameters and distortion from known patterns.", "利用已知图案估计相机内参和畸变。", "A checkerboard is commonly used for camera calibration."),
        ("vanishing point", "消失点", "The image point where projections of parallel 3D lines converge.", "三维平行线在图像中投影后汇聚的点。", "The railway tracks meet at a vanishing point."),
        ("homography", "单应矩阵", "A 3 by 3 projective transform relating two views of a plane.", "联系平面两个视图的 3×3 射影变换。", "A homography straightens the photographed document."),
        ("feature matching", "特征匹配", "Finding corresponding keypoints between images by comparing descriptors.", "通过比较描述子在图像间寻找对应关键点。", "Feature matching is the first step of image stitching."),
        ("RANSAC", "随机采样一致性", "A robust fitting method that repeatedly fits models to random minimal samples and keeps the best-supported one.", "反复对随机最小样本拟合模型并保留支持最多者的鲁棒拟合方法。", "RANSAC ignored the thirty percent of wrong matches."),
        ("epipolar line", "对极线", "The line in one image on which the match of a point in the other image must lie.", "另一幅图中某点的对应点必须落在其上的直线。", "Stereo matching searches only along the epipolar line."),
        ("fundamental matrix", "基础矩阵", "A rank-two matrix encoding the epipolar geometry between two uncalibrated views.", "编码两幅未标定视图间对极几何的秩二矩阵。", "Eight point matches determine the fundamental matrix."),
        ("triangulation", "三角化", "Computing a 3D point from its projections in two or more views.", "由一个点在两幅或多幅视图中的投影计算其三维位置。", "Triangulation fails when the baseline is too short."),
        ("structure from motion", "运动恢复结构", "Recovering 3D structure and camera motion from a set of images.", "从一组图像中恢复三维结构和相机运动。", "Structure from motion rebuilt the cathedral in 3D."),
        ("bundle adjustment", "光束法平差", "Jointly refining all cameras and 3D points to minimize reprojection error.", "联合优化所有相机和三维点以最小化重投影误差。", "Bundle adjustment is the back end of most SLAM systems."),
    ],
    cards=[
        ("相机投影公式？", r"$\mathbf x\simeq K[R\mid\mathbf t]\mathbf X$，再做透视除法。"),
        ("单应 vs 基础矩阵？", "单应：平面或纯旋转，点到点映射，8 自由度；基础矩阵：一般场景，点到线约束，秩 2、7 自由度。"),
        ("RANSAC 迭代次数？", r"$N=\log(1-p)/\log(1-w^s)$"),
        ("双目深度？", r"$Z=fB/d$，误差随 $Z^2$ 增长。"),
    ],
)
