CHAPTER = dict(
    no=9,
    title="数字图像处理基础",
    title_en="Digital Image Processing Fundamentals",
    group="计算机视觉",
    summary="深度学习之前和之外的图像处理：采样与量化、灰度变换与直方图均衡化、空间滤波、傅里叶变换与频率域滤波、阈值分割与形态学。它们既是 CNN 的思想源头，也是今天数据预处理和工业视觉的日常工具。",
    sources=["Gonzalez & Woods《数字图像处理》第 2–4、9–10 章", "Torralba 等《计算机视觉基础》第 15–23 章", "Solem《Python 计算机视觉编程》第 1 章", "Szeliski 第 3 章", "Forsyth & Ponce 第 4 章"],
    learn=[
        dict(
            title="数字图像：采样、量化与颜色",
            title_en="Sampling, quantization and color",
            body=r"""
一幅数字图像就是一个数组：灰度图是 $H\times W$ 的矩阵，彩色图是 $H\times W\times3$ 的张量。把连续的场景变成数组需要两步（Gonzalez 第 2 章）：

- **采样**：在空间上取离散的点，决定**空间分辨率**（像素数）。采样太稀会产生**混叠**（aliasing），例如摩尔纹。奈奎斯特定理要求采样频率至少是信号最高频率的两倍。
- **量化**：把每个点的亮度映射到有限个灰度级，决定**灰度分辨率**。8 位图像有 $2^8=256$ 个灰度级；灰度级太少会出现假轮廓。

Gonzalez 按电磁波谱组织成像应用：伽马射线（核医学）、X 射线（CT）、紫外、可见光与红外（遥感、监控）、微波（雷达）、无线电波（核磁共振）——图像远不止我们眼睛看到的照片。

**颜色模型**：RGB 适合显示器；CMY/CMYK 适合印刷；HSI/HSV 把色调、饱和度与亮度分开，更符合人的感知，便于按颜色分割。像素之间的**邻域**与**连通性**（4 邻域、8 邻域）是后面滤波、形态学和连通区域分析的基础。
""",
            points=[
                "存储量 = 高 × 宽 × 通道数 × 每通道位数 / 8 字节。",
                "深度学习中图像张量的布局：PyTorch 常用 $C\\times H\\times W$，TensorFlow 常用 $H\\times W\\times C$。",
                "输入网络前通常把像素从 0–255 缩放到 0–1，再按通道标准化。",
                "下采样前先做低通（模糊）滤波，可以避免混叠（Torralba 第 20–21 章）。",
            ],
            code="ch09_ex_storage.py",
            code_note="计算不同分辨率、通道数和位深下的图像存储量。",
            src="Gonzalez 第 1–2、6 章；Torralba 第 20 章；Solem 第 1 章",
        ),
        dict(
            title="灰度变换与直方图均衡化",
            title_en="Intensity transformations and histogram equalization",
            body=r"""
**灰度变换**逐个像素地修改亮度 $s=T(r)$，与邻域无关：

- **图像反转** $s=L-1-r$，便于观察暗区域中的细节；
- **对数变换** $s=c\log(1+r)$，压缩高亮区域，常用于显示傅里叶频谱；
- **幂律（伽马）变换** $s=c\,r^\gamma$：$\gamma<1$ 提亮暗部，$\gamma>1$ 压暗，这就是显示器的伽马校正。

**直方图**统计每个灰度级的像素数。低对比度图像的直方图挤在一个窄区间里。**直方图均衡化**用累积分布函数作为映射：

$$s_k=(L-1)\sum_{j=0}^{k}p_r(r_j)$$

变换后的灰度大致均匀地铺满整个范围，从而自动增强对比度。它简单有效，但对噪声也会一起放大；**自适应直方图均衡化（CLAHE）**在局部窗口内做均衡，并限制对比度放大的幅度，在医学影像中很常用。
""",
            points=[
                "灰度变换是点运算：输出像素只依赖于同位置的输入像素。",
                "直方图均衡化不需要任何参数，映射表完全由图像自己的直方图决定。",
                "伽马校正是深度学习数据增强中“亮度/对比度抖动”的数学基础。",
                "直方图匹配（规定化）可以把一幅图的色调风格迁移到另一幅图上。",
            ],
            code="ch09_histogram.py",
            code_note="对一幅低对比度图像做反转、伽马变换和直方图均衡化，比较灰度范围和分布。",
            src="Gonzalez 第 3 章 3.1–3.3",
        ),
        dict(
            title="空间滤波与频率域滤波",
            title_en="Spatial and frequency-domain filtering",
            body=r"""
**空间滤波**用一个小模板（卷积核）在图像上滑动，对每个邻域做加权运算——这正是 CNN 卷积层的数学原型。

- **平滑（低通）**：均值滤波、高斯滤波，去除噪声但使边缘变模糊。高斯核可分离成两个一维核，计算更快。
- **非线性滤波**：**中值滤波**取邻域中值，擅长去除椒盐噪声，同时较好地保持边缘。
- **锐化（高通）**：一阶导数（梯度，Sobel 算子）与二阶导数（拉普拉斯算子）突出边缘和细节；“原图 + k × 高频”就是反锐化掩模。

**傅里叶变换**把图像分解成不同频率、方向的正弦波之和。低频对应缓慢变化的大片区域，高频对应边缘、纹理和噪声。在频率域里滤波非常直观：低通滤波器使图像平滑，高通滤波器使图像锐利；理想滤波器会产生振铃，巴特沃斯和高斯滤波器更平滑。周期性噪声在频谱上表现为几个孤立的亮点，用**陷波滤波器**把它们置零就能去掉。

**卷积定理**把两种视角联系起来：

$$f*h\ \Longleftrightarrow\ F\cdot H$$

空间域的卷积等于频率域的逐点相乘。大卷积核在频率域计算更快。
""",
            points=[
                "滤波核的系数之和为 1 时保持整体亮度；为 0 时（如拉普拉斯）只保留变化部分。",
                "PSNR（峰值信噪比）是衡量去噪/复原质量的常用指标，越高越好。",
                "频谱的中心是直流分量（图像平均亮度）。",
                "CNN 第一层学到的滤波器，往往很像边缘检测器和 Gabor 滤波器。",
            ],
            code="ch09_fourier.py",
            code_note="用 FFT 找到周期噪声的频谱峰值并用陷波滤波去除；演示理想低通的模糊效果；数值验证卷积定理。",
            src="Gonzalez 第 3–4 章；Torralba 第 15–18 章；Forsyth & Ponce 第 4 章",
        ),
        dict(
            title="图像复原、分割与形态学",
            title_en="Restoration, segmentation and morphology",
            body=r"""
**图像复原**是客观的：建立退化模型 $g=h*f+n$（模糊 + 噪声），再尽量恢复 $f$。方法有逆滤波、维纳滤波等。从投影重建图像的**滤波反投影**，就是 CT 成像的数学原理（Gonzalez 第 5 章）。

**分割**把图像分成有意义的区域（Gonzalez 第 10 章）：

- **阈值分割**：灰度大于阈值的为前景。**大津法**（Otsu）自动选择阈值，使前景与背景的**类间方差最大**。
- **边缘检测**：Canny 检测器（高斯平滑 → 梯度 → 非极大值抑制 → 双阈值滞后连接）；**霍夫变换**检测直线和圆。
- **区域方法**：区域生长、分裂合并、分水岭、聚类与超像素、图割。

**形态学**把二值图像看成集合，用一个小的**结构元**去“探测”它：

- **腐蚀**让物体变瘦，去掉小于结构元的细节；**膨胀**让物体变胖，填补小缝隙；
- **开运算** = 先腐蚀后膨胀，去除小噪点；**闭运算** = 先膨胀后腐蚀，填补小孔洞；
- 再加上边界提取、孔洞填充、**连通区域**标记、骨架化等算法。

这些方法不需要训练数据、可解释，在工业检测、医学影像和文档处理中至今被广泛使用，也常作为深度学习模型的预处理或后处理。
""",
            points=[
                "图像增强是主观的（看起来更好），图像复原是客观的（还原真实图像）。",
                "大津法在直方图呈双峰分布时效果最好。",
                "开运算去“岛”，闭运算补“洞”。",
                "分割网络（如 U-Net，第 10 章）的输出常用形态学操作做后处理。",
            ],
            code="ch09_otsu_morph.py",
            code_note="生成含噪声的“细胞”图像：大津法求阈值，开运算去除噪点，再统计连通区域数。",
            src="Gonzalez 第 5、9、10 章；Szeliski 3.3；Solem 第 1、9 章",
        ),
    ],
    examples=[
        dict(
            title="直方图均衡化的映射",
            problem=r"一幅 3 位图像（灰度级 0–7）共 64 个像素，灰度 0–7 的像素数依次为 0, 0, 16, 24, 16, 8, 0, 0。求均衡化后各灰度级的映射。",
            steps=[
                r"概率 $p_r$：$r=2$ 为 0.25，$r=3$ 为 0.375，$r=4$ 为 0.25，$r=5$ 为 0.125，其余为 0。",
                r"累积分布：$r=2$ 为 0.25，$r=3$ 为 0.625，$r=4$ 为 0.875，$r=5$ 为 1.0。",
                r"$s_k=7\times\text{CDF}$ 再取整：$2\to1.75\approx2$，$3\to4.375\approx4$，$4\to6.125\approx6$，$5\to7$。",
                r"原来只占 2–5 的灰度被拉伸到 2–7，相邻灰度的间隔变大，对比度增强。",
            ],
            answer=r"映射为 $2\to2,\ 3\to4,\ 4\to6,\ 5\to7$。",
            code="ch09_histogram.py",
        ),
        dict(
            title="为椒盐噪声选择滤波器",
            problem=r"一幅图像被 10% 的椒盐噪声污染。比较 3×3 均值滤波和 3×3 中值滤波：若某个 3×3 窗口的像素为 50, 50, 50, 50, 255, 50, 50, 0, 50，两种滤波的输出分别是多少？",
            steps=[
                r"均值滤波：$(50\times7+255+0)/9=605/9\approx67.2$，噪声被“摊”进了结果。",
                r"中值滤波：排序后为 0, 50, 50, 50, 50, 50, 50, 50, 255，中值为 50，完全去除了两个噪声点。",
                r"原因：椒盐噪声是极端离群值，中值对离群值不敏感，均值则会被拉偏。",
                r"代码中的实验也显示：椒盐噪声下中值滤波 PSNR 约 30 dB，远高于均值滤波的约 22 dB。",
            ],
            answer="均值 ≈ 67.2，中值 = 50；中值滤波更适合椒盐噪声。",
            code="ch09_filters.py",
        ),
        dict(
            title="开运算去除噪点",
            problem=r"阈值分割后，二值图中除了 4 个大的目标外，还有几十个只有 1 个像素的亮点。用 3×3 结构元做开运算后，为什么小亮点消失而大目标基本保持原样？",
            steps=[
                r"腐蚀：只有当 3×3 邻域全为前景时，中心才保留为前景。单像素的亮点邻域不全是前景，被腐蚀掉。",
                r"大目标被腐蚀后边界向内收缩一个像素，但主体还在。",
                r"膨胀：只要 3×3 邻域内有一个前景像素，中心就变成前景。大目标的边界重新向外扩张，大体恢复原形。",
                r"已经消失的小亮点不会被膨胀“复活”。所以开运算 = 去掉比结构元小的结构，保留大的结构。代码中连通区域数从 47 降到 4。",
            ],
            answer="腐蚀删除了小于结构元的结构，随后的膨胀只恢复幸存的大目标。",
            code="ch09_otsu_morph.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="一幅 8 位灰度图像有多少个灰度级？", options=["8", "128", "256", "1024"], answer=2, explain="$2^8=256$。"),
            dict(q="去除椒盐噪声效果最好的是：", options=["均值滤波", "中值滤波", "拉普拉斯滤波", "高通滤波"], answer=1, explain="中值对离群值不敏感。"),
            dict(q="伽马变换 $s=r^{\\gamma}$ 中 $\\gamma<1$ 的效果是：", options=["压暗图像", "提亮暗部", "二值化", "去噪"], answer=1, explain="把暗区的灰度拉伸。"),
            dict(q="卷积定理指出，空间域的卷积对应频率域的：", options=["加法", "逐点相乘", "求逆", "转置"], answer=1, explain="$f*h\\Leftrightarrow FH$。"),
            dict(q="大津法选择阈值的准则是：", options=["最小化类间方差", "最大化类间方差", "让前景像素最多", "取灰度中位数"], answer=1, explain="等价于最小化类内方差。"),
            dict(q="先腐蚀后膨胀的操作叫做：", options=["闭运算", "开运算", "顶帽变换", "骨架化"], answer=1, explain="开运算去除小亮点。"),
        ],
        tf=[
            dict(q="图像的高频分量主要对应边缘、细节和噪声。", answer=True, explain="快速变化 = 高频。"),
            dict(q="直方图均衡化需要人工指定一个参数 γ。", answer=False, explain="映射完全由图像的累积直方图决定。"),
            dict(q="采样频率过低会产生混叠现象。", answer=True, explain="违反奈奎斯特采样定理。"),
            dict(q="CT 图像的重建原理是从多个角度的投影重建截面图像。", answer=True, explain="滤波反投影。"),
            dict(q="闭运算主要用于去除背景中的小亮点。", answer=False, explain="闭运算填补前景中的小孔洞；去小亮点用开运算。"),
        ],
        fill=[
            dict(q="把连续坐标离散化的过程叫 ____，把连续亮度离散化的过程叫量化。", answer=["采样", "sampling"], explain="sampling。"),
            dict(q="常用于检测图像中直线和圆的变换是 ____ 变换。", answer=["霍夫", "Hough"], explain="Hough transform。"),
            dict(q="衡量去噪质量、单位为 dB 的指标缩写是 ____。", answer=["PSNR", "峰值信噪比"], explain="Peak Signal-to-Noise Ratio。"),
            dict(q="形态学中用来“探测”图像的小形状叫做 ____。", answer=["结构元", "结构元素", "structuring element"], explain="structuring element。"),
        ],
        reading=[
            dict(
                title="Before Neural Networks Learned to See",
                passage="""Long before convolutional networks learned their own filters, engineers designed them by hand. A smoothing filter averages each pixel with its neighbors to suppress noise. A Sobel filter responds strongly where brightness changes quickly, outlining the edges of objects. A median filter replaces each pixel with the middle value of its neighborhood, which removes isolated specks without blurring sharp boundaries.

The Fourier transform offers a second point of view. Any image can be written as a sum of waves of different frequencies. Smooth regions are made of low frequencies, while edges and fine textures need high ones. Seen this way, blurring is simply throwing away high frequencies, and the convolution theorem tells us that sliding a filter across an image is the same as multiplying two spectra.

These classical tools have not disappeared. They prepare images before they reach a network, they clean up a network's output, and they still power many industrial systems where training data is scarce and every decision must be explained. Understanding them also makes the first layers of a trained CNN much less mysterious: they often look remarkably like edge detectors.""",
                zh="""早在卷积网络自己学会滤波器之前，工程师们就在手工设计滤波器了。平滑滤波器把每个像素与邻居做平均以抑制噪声。Sobel 滤波器在亮度变化剧烈的地方响应强烈，勾勒出物体的边缘。中值滤波器用邻域的中间值替换每个像素，能去掉孤立的斑点而不模糊清晰的边界。

傅里叶变换提供了第二种视角。任何图像都可以写成不同频率波的叠加。平滑区域由低频组成，边缘和细密纹理则需要高频。这样看来，模糊就是丢掉高频；而卷积定理告诉我们，让滤波器在图像上滑动，等价于把两个频谱相乘。

这些经典工具并没有消失。它们在图像进入网络之前做预处理，对网络的输出做清理，并且仍然支撑着许多训练数据稀缺、每个决策都必须可解释的工业系统。理解它们还会让训练好的 CNN 的前几层不再神秘：那些层看起来常常与边缘检测器惊人地相似。""",
                questions=[
                    dict(q="Which filter is described as removing isolated specks without blurring sharp boundaries?", options=["Smoothing filter", "Sobel filter", "Median filter", "Low-pass filter"], answer=2, explain="中值滤波。"),
                    dict(q="According to the passage, blurring an image corresponds to:", options=["adding high frequencies", "throwing away high frequencies", "rotating the spectrum", "increasing contrast"], answer=1, explain="模糊 = 去掉高频。"),
                    dict(q="Why are classical tools still used, according to the passage?", options=["They are always more accurate than CNNs", "They help with pre- and post-processing and explainable, data-scarce systems", "They need large datasets", "They are required by law"], answer=1, explain="预处理/后处理与可解释、少数据场景。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("pixel", "像素", "The smallest element of a digital image, holding one intensity or color value.", "数字图像的最小单元，存储一个亮度或颜色值。", "The image is 640 pixels wide."),
        ("grayscale", "灰度", "An image representation that stores only brightness, without color.", "只存储亮度、不含颜色的图像表示。", "Converting to grayscale reduces computation."),
        ("sampling", "采样", "Measuring a continuous signal at discrete positions.", "在离散位置上测量连续信号。", "Insufficient sampling causes aliasing."),
        ("quantization", "量化", "Mapping continuous values to a finite set of levels.", "把连续值映射到有限个级别。", "Coarse quantization produces false contours."),
        ("aliasing", "混叠", "Artifacts caused by sampling a signal too sparsely.", "因采样过稀而产生的伪影。", "Blur before downsampling to avoid aliasing."),
        ("histogram equalization", "直方图均衡化", "Remapping intensities so that their distribution becomes roughly uniform.", "重新映射灰度，使其分布近似均匀。", "Histogram equalization revealed details in the dark X-ray."),
        ("gamma correction", "伽马校正", "A power-law intensity transform used to adjust brightness.", "用于调整亮度的幂律灰度变换。", "Gamma correction brightened the shadows."),
        ("Gaussian filter", "高斯滤波器", "A smoothing filter whose weights follow a Gaussian bell curve.", "权重服从高斯钟形曲线的平滑滤波器。", "A Gaussian filter reduces noise before edge detection."),
        ("median filter", "中值滤波器", "A nonlinear filter that replaces each pixel by the median of its neighborhood.", "用邻域中值替换每个像素的非线性滤波器。", "The median filter removed salt-and-pepper noise."),
        ("edge detection", "边缘检测", "Finding locations where image brightness changes sharply.", "寻找图像亮度剧烈变化的位置。", "Canny edge detection produces thin, connected edges."),
        ("Fourier transform", "傅里叶变换", "Decomposing a signal into sinusoidal components of different frequencies.", "把信号分解为不同频率的正弦分量。", "The Fourier transform exposed the periodic noise."),
        ("thresholding", "阈值分割", "Labeling pixels as foreground or background by comparing them with a threshold.", "把像素与阈值比较以标记前景或背景。", "Otsu's method chooses the threshold automatically."),
        ("morphological opening", "开运算", "Erosion followed by dilation, used to remove small bright specks.", "先腐蚀后膨胀，用于去除小亮点。", "A morphological opening cleaned up the binary mask."),
        ("connected component", "连通区域", "A maximal set of foreground pixels that are linked to each other.", "相互连通的前景像素的最大集合。", "Each connected component corresponds to one cell."),
    ],
    cards=[
        ("直方图均衡化的映射？", r"$s_k=(L-1)\sum_{j\le k}p_r(r_j)$，即用累积分布函数做映射。"),
        ("卷积定理？", r"$f*h\Leftrightarrow F\cdot H$：空间卷积 = 频域相乘。"),
        ("开运算与闭运算？", "开 = 先腐蚀后膨胀，去小亮点；闭 = 先膨胀后腐蚀，补小孔洞。"),
        ("大津法的准则？", "选择使前景与背景**类间方差最大**的阈值。"),
    ],
)
