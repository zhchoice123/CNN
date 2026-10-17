CHAPTER = dict(
    no=10,
    title="迁移学习、目标检测与图像分割",
    title_en="Transfer Learning, Object Detection and Segmentation",
    group="计算机视觉",
    summary="从“图里有什么”到“在哪里、是哪些像素”：先用迁移学习站在预训练模型的肩膀上，再学习目标检测的核心工具（IoU、NMS、mAP、锚框）与两阶段/单阶段检测器，最后是语义分割、实例分割与它们的评估指标。",
    sources=["Elgendy《视觉系统深度学习》第 6–7 章", "Ayyadevara & Reddy《PyTorch 现代计算机视觉》第 5–10 章", "Pointer《PyTorch 深度学习编程》第 4 章", "Chollet 第 8–9 章", "Géron 第 14 章"],
    learn=[
        dict(
            title="迁移学习：特征提取与微调",
            title_en="Transfer learning",
            body=r"""
从零训练一个深度 CNN 需要大量数据和算力。**迁移学习**把在大数据集（如 ImageNet，1000 类、120 多万张图）上预训练好的网络拿来解决自己的任务。它之所以有效，是因为 CNN 学到的特征是层次化的：浅层的边缘、颜色、纹理几乎对所有图像任务都通用，越深的层越专门化。

三种常见做法（Elgendy 第 6 章）：

1. **直接使用**：新任务的类别包含在原任务中，直接用预训练模型预测。
2. **特征提取**：冻结整个卷积基，只训练一个新的分类头。需要训练的参数极少，小数据也能做。
3. **微调**：先训练新分类头，再解冻最后几个卷积块，用**很小的学习率**一起训练。

选择策略取决于两点：新数据集的**大小**和它与原数据集的**相似度**。数据少且相似 → 特征提取；数据多且相似 → 微调更多层；数据少且不相似 → 只用较浅层的特征；数据多且不相似 → 可以大范围微调甚至从头训练。

Pointer 第 4 章介绍了两个实用技巧：**学习率查找器**（从很小的学习率逐步增大，选择损失下降最快的区间）和**差分学习率**（靠近输入的层学习率小，靠近输出的层学习率大）。
""",
            points=[
                "VGG16 的 1.38 亿参数中，约 1.24 亿在全连接层；用全局平均池化替换它们，新任务只需训练几千个参数。",
                "微调时学习率要比从头训练小一到两个数量级，否则会破坏预训练特征。",
                "输入预处理（尺寸、归一化的均值和方差）必须与预训练时一致。",
                "医学影像等与 ImageNet 差异很大的领域，迁移效果可能有限，需要领域内的预训练（第 16 章相关讨论）。",
            ],
            code="ch10_transfer_params.py",
            code_note="逐层计算 VGG16 的参数量，比较特征提取、部分微调与全部微调分别需要训练多少参数。",
            src="Elgendy 第 6 章；Ayyadevara & Reddy 第 5 章；Pointer 第 4 章；Chollet 8.3",
        ),
        dict(
            title="目标检测的基本工具：IoU、NMS 与 mAP",
            title_en="IoU, NMS and mAP",
            body=r"""
**目标检测**要同时回答“是什么”（分类）和“在哪里”（用边界框定位）。几个贯穿所有检测器的工具：

- **交并比（IoU）**：预测框与真实框的交集面积除以并集面积，$\text{IoU}=\dfrac{|A\cap B|}{|A\cup B|}$。通常 IoU ≥ 0.5 才算检测正确。
- **非极大值抑制（NMS）**：检测器往往对同一物体输出许多重叠的框。NMS 按置信度排序，保留最高的框，删除与它 IoU 超过阈值的其他框，再对剩下的框重复。
- **精确率—召回率曲线与 AP**：把所有检测按置信度从高到低排序，逐个计算累积的精确率和召回率，曲线下的面积就是**平均精度 AP**；所有类别的 AP 取平均就是 **mAP**。COCO 还会在 IoU 从 0.5 到 0.95 的多个阈值上平均（mAP@[.5:.95]），对定位精度要求更高。
- **速度**：用每秒帧数（FPS）衡量，实时应用通常需要 30 FPS 以上。

边界框的回归通常不直接预测坐标，而是预测相对于**锚框**（anchor）的偏移：$t_x=(x-x_a)/w_a$，$t_w=\log(w/w_a)$ 等，这样不同大小的物体都能用相近尺度的数值来学习。
""",
            points=[
                "IoU 同时惩罚“框太大”和“框太小”。",
                "NMS 的 IoU 阈值太低会误删紧挨在一起的不同物体，太高会留下重复框。",
                "AP 综合了不同阈值下的精确率和召回率，比单一阈值下的指标更全面。",
                "检测任务的评估要先按 IoU 把预测与真实框匹配，每个真实框只能被匹配一次。",
            ],
            code="ch10_iou_nms.py",
            code_note="实现 IoU 和 NMS：5 个候选框属于两个物体，NMS 后各保留一个。",
            src="Elgendy 7.1；Ayyadevara & Reddy 第 7 章",
        ),
        dict(
            title="检测器家族：R-CNN、YOLO 与 SSD",
            title_en="Two-stage and one-stage detectors",
            body=r"""
**两阶段检测器**（R-CNN 家族）先提出候选区域，再对每个区域分类和精修：

- **R-CNN（2014）**：用选择性搜索产生约 2000 个候选区域，每个区域缩放后单独送进 CNN 提特征，再用 SVM 分类。准确，但非常慢。
- **Fast R-CNN**：整张图只过一次 CNN，用 **RoI 池化**从共享的特征图上为每个候选区域截取固定大小的特征。
- **Faster R-CNN**：用**区域提议网络（RPN）**代替选择性搜索。RPN 在特征图的每个位置放置多个不同尺度和长宽比的**锚框**，预测每个锚框“是否含物体”以及坐标偏移。整个系统可以端到端训练。

**单阶段检测器**一步到位，速度快：

- **YOLO**（You Only Look Once）：把图像划分成网格，每个网格直接回归边界框和类别概率，把检测当作一个回归问题。
- **SSD**（单次多框检测器）：在多个不同分辨率的特征图上放置默认框并预测，浅层的高分辨率特征图负责小物体，深层负责大物体。

一般来说，两阶段方法精度更高，单阶段方法更快；后来的 RetinaNet 用 **Focal Loss** 缓解了单阶段检测中前景与背景极度不平衡的问题，缩小了精度差距。今天也有基于 Transformer 的 DETR 等方法，不再需要锚框和 NMS。
""",
            points=[
                "演进主线：减少重复计算（共享特征）、把手工组件换成可学习模块（RPN）、端到端训练。",
                "锚框的尺度和长宽比是重要的超参数，应与数据集中物体的形状相匹配。",
                "多尺度特征（SSD、特征金字塔 FPN）是检测小物体的关键。",
                "Elgendy 第 7 章的项目：在自动驾驶场景中训练 SSD。",
            ],
            src="Elgendy 第 7 章；Ayyadevara & Reddy 第 7–8 章；Géron 第 14 章“目标检测”",
        ),
        dict(
            title="图像分割：U-Net、Mask R-CNN 与评估",
            title_en="Image segmentation",
            body=r"""
**语义分割**给每个像素分配一个类别（不区分同类的不同个体）；**实例分割**还要区分同一类中的不同个体；**全景分割**两者兼顾。

分割网络需要“先压缩、再还原”：编码器通过卷积和下采样提取语义，解码器通过**上采样**恢复空间分辨率。上采样可以用插值，也可以用可学习的**转置卷积**（输出尺寸 $o=(n-1)s-2p+k$）。

- **U-Net**：编码器—解码器结构，并用**跳跃连接**把编码器每一层的高分辨率特征直接拼接到解码器对应层，弥补下采样丢失的细节。它最初为生物医学图像设计，在小数据集上也表现出色，是医学影像分割的标准基线。
- **Mask R-CNN**：在 Faster R-CNN 上增加一个预测掩码的分支，并用 **RoI Align**（双线性插值，不做取整）代替 RoI 池化，解决特征对齐问题，实现实例分割。

**评估**：前景很小时，像素准确率会严重误导——全部预测为背景也能得到 96% 的像素准确率。应该使用：

$$\text{IoU}=\frac{|P\cap G|}{|P\cup G|},\qquad \text{Dice}=\frac{2|P\cap G|}{|P|+|G|}$$

Dice 系数（等价于 F1）在医学影像中最常用，也常直接用作损失函数（Dice loss）来应对类别不平衡。
""",
            points=[
                "语义分割 = 逐像素分类；实例分割 = 检测 + 每个实例的掩码。",
                "U-Net 的跳跃连接是拼接（concatenate），ResNet 的残差连接是相加（add）。",
                "Dice ≥ IoU，二者单调相关：$\\text{Dice}=2\\,\\text{IoU}/(1+\\text{IoU})$。",
                "《PyTorch 现代计算机视觉》第 10 章还展示了人体姿态估计、人群计数、点云三维检测等应用。",
            ],
            code="ch10_seg_metrics.py",
            code_note="在前景只占 4% 的掩码上，对比像素准确率、IoU 和 Dice；并计算几种转置卷积的输出尺寸。",
            src="Ayyadevara & Reddy 第 9–10 章；Chollet 9.1；Géron 第 14 章“语义分割”；Zhou 等《医学图像分析中的深度学习》",
        ),
    ],
    examples=[
        dict(
            title="计算两个框的 IoU",
            problem=r"预测框 $A=(0,0,10,10)$，真实框 $B=(5,5,15,15)$（格式为左上角与右下角坐标）。求 IoU，这个检测在 0.5 的阈值下算正确吗？",
            steps=[
                r"交集：$x$ 方向 $[5,10]$，$y$ 方向 $[5,10]$，面积 $5\times5=25$。",
                r"两个框面积都是 100，并集 $=100+100-25=175$。",
                r"$\text{IoU}=25/175\approx0.143$。",
                r"小于 0.5，算作误检（FP）；同时这个真实框如果没有其他匹配，就是一次漏检（FN）。",
            ],
            answer="IoU ≈ 0.143，不算正确检测。",
            code="ch10_iou_nms.py",
        ),
        dict(
            title="手动执行非极大值抑制",
            problem=r"5 个候选框的得分为 0.90、0.85、0.75、0.70、0.60（编号 0、3、1、4、2）。框 0 与框 1、框 2 的 IoU 分别为 0.82、0.76，框 3 与框 4 的 IoU 为 0.80，两组框之间不重叠。NMS 阈值为 0.5，最终保留哪些框？",
            steps=[
                r"取最高分框 0 保留；删除与它 IoU > 0.5 的框 1（0.82）和框 2（0.76）。",
                r"剩下框 3、框 4，取最高分框 3 保留；框 4 与框 3 的 IoU 为 0.80，删除。",
                r"列表为空，结束。",
                r"保留框 0 和框 3，正好每个物体一个框。",
            ],
            answer="保留框 0（0.90）和框 3（0.85）。",
            code="ch10_iou_nms.py",
        ),
        dict(
            title="计算平均精度 AP",
            problem=r"一张图中有 6 个真实目标。按置信度排序的 10 个检测结果的正误为：✔✔✘✔✔✘✘✔✘✔。求精确率—召回率序列与 AP（所有点插值）。",
            steps=[
                r"逐个累计：第 1、2 个后 P=1.0、R=0.33；第 4 个后 P=0.75、R=0.50；第 5 个后 P=0.80、R=0.67；第 8 个后 P=0.625、R=0.83；第 10 个后 P=0.60、R=1.0。",
                r"插值：每个召回率处的精确率取其右侧（更高召回率）的最大精确率，使曲线单调不增。",
                r"在召回率 0→0.33 区间精确率为 1.0，0.33→0.67 为 0.80，0.67→0.83 为 0.625，0.83→1.0 为 0.60。",
                r"$\text{AP}=0.333\times1.0+0.333\times0.8+0.167\times0.625+0.167\times0.6\approx0.804$。",
            ],
            answer="AP ≈ 0.804。",
            code="ch10_average_precision.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="新任务数据很少，且与 ImageNet 图像相似，最合适的迁移策略是：", options=["从头训练", "冻结卷积基，只训练新分类头", "随机初始化后微调全部层", "只用最后一层特征做聚类"], answer=1, explain="特征提取最稳妥。"),
            dict(q="NMS 的作用是：", options=["生成候选框", "去除同一物体的重复检测框", "计算 mAP", "数据增强"], answer=1, explain="非极大值抑制。"),
            dict(q="Faster R-CNN 相比 Fast R-CNN 的关键改进是：", options=["使用 SVM 分类", "用区域提议网络 RPN 代替选择性搜索", "去掉了锚框", "使用了 Transformer"], answer=1, explain="RPN 让候选区域也可学习。"),
            dict(q="YOLO 属于：", options=["两阶段检测器", "单阶段检测器", "语义分割网络", "生成模型"], answer=1, explain="You Only Look Once。"),
            dict(q="U-Net 中跳跃连接的作用是：", options=["加快训练速度", "把编码器的高分辨率细节传给解码器", "减少参数量", "实现数据增强"], answer=1, explain="弥补下采样丢失的空间细节。"),
            dict(q="前景只占 2% 像素的分割任务，最不适合作为主要指标的是：", options=["Dice", "IoU", "像素准确率", "每类 IoU 的平均"], answer=2, explain="全预测背景也有 98% 准确率。"),
        ],
        tf=[
            dict(q="IoU 的取值范围是 0 到 1。", answer=True, explain="完全不重叠为 0，完全重合为 1。"),
            dict(q="微调预训练模型时应该使用比从头训练更大的学习率。", answer=False, explain="应使用更小的学习率，避免破坏预训练特征。"),
            dict(q="SSD 在多个不同分辨率的特征图上进行预测，以检测不同大小的物体。", answer=True, explain="多尺度默认框。"),
            dict(q="语义分割会区分同一类别的不同个体。", answer=False, explain="那是实例分割。"),
            dict(q="RoI Align 通过双线性插值避免了 RoI 池化中的坐标取整误差。", answer=True, explain="Mask R-CNN 的改进。"),
        ],
        fill=[
            dict(q="交集面积除以并集面积称为 ____（英文缩写）。", answer=["IoU", "交并比"], explain="Intersection over Union。"),
            dict(q="所有类别 AP 的平均值叫做 ____（英文缩写）。", answer=["mAP"], explain="mean Average Precision。"),
            dict(q="Dice 系数的公式是 $2|P\\cap G|/(|P|+$ ____ $)$。", answer=["|G|", "G"], explain="真实掩码的面积。"),
            dict(q="Faster R-CNN 中预先设定的不同尺度、长宽比的参考框叫做 ____ 框。", answer=["锚", "anchor"], explain="anchor box。"),
        ],
        reading=[
            dict(
                title="Standing on the Shoulders of ImageNet",
                passage="""Few practitioners train large vision models from scratch. Instead, they start from a network that has already been trained on millions of labeled photographs. The early layers of such a network have learned to detect edges, color blobs and simple textures, patterns that are useful for almost any visual task. Only the last layers are tied closely to the original thousand categories.

This observation leads to transfer learning. With a small dataset of, say, a few hundred images of flowers, one can freeze the pretrained layers and train only a new classification head. The model reaches good accuracy in minutes, because it is no longer learning to see from nothing; it is only learning how to combine features it already knows.

When more data is available, the practitioner can unfreeze some of the deeper layers and fine-tune them with a very small learning rate. The pretrained weights act as an excellent starting point that should be adjusted gently rather than overwritten. Choosing how many layers to unfreeze depends on how much data you have and how similar it is to the original images.""",
                zh="""很少有从业者会从零开始训练大型视觉模型。他们会从一个已经在数百万张有标注照片上训练过的网络出发。这种网络的浅层已经学会检测边缘、色块和简单纹理，这些模式几乎对任何视觉任务都有用。只有最后几层与原来的一千个类别紧密相关。

这一观察催生了迁移学习。假如只有几百张花卉图片这样的小数据集，可以冻结预训练层，只训练一个新的分类头。模型几分钟内就能达到不错的准确率，因为它不再需要从零开始学习“看”，只需学习如何组合它已经认识的特征。

当数据更多时，可以解冻一部分更深的层，用很小的学习率对它们进行微调。预训练权重是一个极好的起点，应当轻轻调整，而不是被覆盖掉。解冻多少层，取决于你有多少数据，以及这些数据与原始图像有多相似。""",
                questions=[
                    dict(q="According to the passage, which layers are tied most closely to the original categories?", options=["The first layers", "The middle layers", "The last layers", "All layers equally"], answer=2, explain="最后几层最专门化。"),
                    dict(q="Why does the frozen model train quickly on a small flower dataset?", options=["It uses a huge learning rate", "It only learns to combine features it already knows", "It ignores the images", "It uses more data than ImageNet"], answer=1, explain="只需学习组合已知特征。"),
                    dict(q="When fine-tuning deeper layers, the learning rate should be:", options=["very small", "as large as possible", "zero", "random"], answer=0, explain="轻轻调整预训练权重。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("transfer learning", "迁移学习", "Reusing a model trained on one task as the starting point for another.", "把在一个任务上训练好的模型作为另一个任务的起点。", "Transfer learning made our small dataset sufficient."),
        ("fine-tuning", "微调", "Continuing to train some pretrained layers on new data with a small learning rate.", "用小学习率在新数据上继续训练部分预训练层。", "Fine-tuning the last block improved accuracy by three points."),
        ("feature extraction", "特征提取", "Using a frozen pretrained network to produce features for a new classifier.", "用冻结的预训练网络为新分类器产生特征。", "Feature extraction only trains the new head."),
        ("object detection", "目标检测", "Finding objects in an image and drawing a bounding box around each.", "找出图像中的物体并为每个物体画出边界框。", "Object detection tells us both what and where."),
        ("bounding box", "边界框", "A rectangle that encloses an object, given by its coordinates.", "包围物体的矩形，由坐标给出。", "Each bounding box comes with a confidence score."),
        ("intersection over union", "交并比", "The overlap area of two regions divided by the area of their union.", "两区域重叠面积除以并集面积。", "A detection counts as correct when the intersection over union exceeds one half."),
        ("non-maximum suppression", "非极大值抑制", "Keeping the highest-scoring box and discarding boxes that overlap it heavily.", "保留得分最高的框，删除与之高度重叠的框。", "Non-maximum suppression removed the duplicate boxes."),
        ("anchor box", "锚框", "A predefined reference box of a given size and aspect ratio used to predict offsets.", "预先设定尺寸和长宽比、用于预测偏移的参考框。", "Each location in the feature map has nine anchor boxes."),
        ("region proposal network", "区域提议网络", "A small network that proposes candidate object regions from a feature map.", "从特征图中提出候选物体区域的小网络。", "The region proposal network replaced selective search."),
        ("mean average precision", "平均精度均值", "The average over classes of the area under each precision–recall curve.", "各类别精确率—召回率曲线下面积的平均。", "The model reached a mean average precision of 0.42 on COCO."),
        ("semantic segmentation", "语义分割", "Assigning a class label to every pixel of an image.", "为图像的每个像素分配一个类别标签。", "Semantic segmentation labels every road pixel."),
        ("instance segmentation", "实例分割", "Detecting each object and predicting a separate pixel mask for it.", "检测每个物体并为其预测独立的像素掩码。", "Instance segmentation separates two overlapping people."),
        ("transposed convolution", "转置卷积", "A learnable upsampling operation that increases spatial resolution.", "提高空间分辨率的可学习上采样运算。", "The decoder uses transposed convolutions to upsample."),
        ("Dice coefficient", "Dice 系数", "Twice the overlap of two masks divided by their total size.", "两掩码重叠的两倍除以它们的总大小。", "Medical segmentation papers usually report the Dice coefficient."),
    ],
    cards=[
        ("迁移学习的策略怎么选？", "看数据量和与原数据的相似度：少且相似 → 特征提取；多且相似 → 微调更多层；不相似 → 用浅层特征或大范围微调。"),
        ("NMS 的步骤？", "按得分排序 → 保留最高分框 → 删除与之 IoU 超过阈值的框 → 对剩余框重复。"),
        ("两阶段 vs 单阶段检测器？", "两阶段（Faster R-CNN）：先提议再分类，精度高；单阶段（YOLO、SSD）：一次预测，速度快。"),
        ("Dice 与 IoU 的关系？", r"$\mathrm{Dice}=\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}$"),
    ],
)
