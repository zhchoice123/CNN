CHAPTER = dict(
    no=13,
    title="生成模型：自编码器、VAE、GAN 与扩散模型",
    title_en="Generative Models: Autoencoders, VAEs, GANs and Diffusion",
    group="序列、生成与对齐",
    summary="“我不能创造的东西，我就不理解。”生成模型学习数据本身的分布并从中采样。本章依次讲自编码器与隐空间、变分自编码器、生成对抗网络及其训练难题，以及今天文生图系统背后的扩散模型与稳定扩散。",
    sources=["Foster《生成式深度学习》第 1、3–5、9 章", "Ayyadevara & Reddy《PyTorch 现代计算机视觉》第 11–17 章", "Elgendy 第 8–9 章", "Buduma 第 10 章", "邱锡鹏 第 13 章", "Torralba 等第 32–34 章"],
    learn=[
        dict(
            title="生成建模与自编码器",
            title_en="Generative modeling and autoencoders",
            body=r"""
**判别模型**学习 $p(y\mid\mathbf x)$：给定一张图，它是猫还是狗？**生成模型**学习数据本身的分布 $p(\mathbf x)$，然后从中**采样**出新的数据（Foster 第 1 章）。难点在于：图像这样的高维数据，像素之间存在极其复杂的依赖，直接在像素空间建模几乎不可能。解决思路是**表示学习**：先学一个低维的**隐空间**，再从隐空间生成数据。

**自编码器**由两部分组成：**编码器**把输入压缩成低维的隐向量 $\mathbf z$（瓶颈），**解码器**再从 $\mathbf z$ 重建输入，训练目标是让重建误差最小。Foster 用了一个寓言：两兄弟经营一家只有一面墙的画廊，哥哥把每幅画用墙上的一个坐标记下来然后把画扔掉，弟弟只凭坐标把画重新画出来——坐标就是隐向量。

一个线性自编码器学到的子空间与 PCA 完全相同；加入非线性层后，它能学到弯曲的低维流形。变体包括**去噪自编码器**（输入加噪、重建干净数据）和**稀疏自编码器**。

但普通自编码器不是一个好的生成模型：隐空间是不连续的，在墙上随便挑一个空白位置，解码出来的往往是毫无意义的图像。
""",
            points=[
                "自编码器是无监督的：标签就是输入本身。",
                "瓶颈迫使网络只保留最重要的信息，因此可用于降维、去噪和异常检测（重建误差大的就是异常）。",
                "《PyTorch 现代计算机视觉》第 11 章用卷积自编码器 + t-SNE 把相似图像聚在一起。",
                "生成模型的三条路线：隐变量模型（VAE）、对抗训练（GAN）、逐步去噪（扩散模型）；还有自回归模型（GPT、PixelCNN）。",
            ],
            code="ch13_autoencoder.py",
            code_note="训练一个 10→2→10 的线性自编码器，验证它学到的子空间与 PCA 相同。",
            src="Foster 第 1、3 章；Buduma 第 8 章；Ayyadevara & Reddy 第 11 章；邱锡鹏 9.1",
        ),
        dict(
            title="变分自编码器（VAE）",
            title_en="Variational autoencoders",
            body=r"""
**VAE** 让编码器不再输出一个点，而是输出一个分布 $q(\mathbf z\mid\mathbf x)=\mathcal N(\boldsymbol\mu,\boldsymbol\sigma^2)$，并要求所有这些分布都靠近标准正态先验 $p(\mathbf z)=\mathcal N(0,I)$。损失（负的证据下界 ELBO）由两项组成：

$$\mathcal L=\underbrace{\mathbb E_{q}\big[-\log p(\mathbf x\mid\mathbf z)\big]}_{\text{重建误差}}+\underbrace{D_{KL}\big(q(\mathbf z\mid\mathbf x)\,\|\,\mathcal N(0,I)\big)}_{\text{正则项}}$$

对高斯分布，KL 项有闭式解 $\tfrac12\sum(\mu^2+\sigma^2-1-\log\sigma^2)$。它把每个样本的“领地”拉向原点附近并撑开，使隐空间变得**连续而紧凑**：在隐空间里任取一点，都能解码出合理的样本。

采样操作本身不可导，VAE 用**重参数化技巧**解决：$\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon$，$\boldsymbol\epsilon\sim\mathcal N(0,I)$，随机性只来自 $\boldsymbol\epsilon$，梯度就能流过 $\boldsymbol\mu$ 和 $\boldsymbol\sigma$。这里用到的正是 MacKay、Bishop 讲过的**变分推断**。

隐空间的神奇之处（Foster 第 3 章）：**隐空间算术**——人脸向量加上“微笑”方向，脸就会笑起来；在两张脸的隐向量之间插值，可以得到平滑的变形过渡。VAE 生成的图像往往偏模糊，但它是稳定扩散的重要组件：稳定扩散在 VAE 的隐空间里做扩散。
""",
            points=[
                "重建项让输出像输入，KL 项让隐空间规整；二者需要权衡（β-VAE 加大 KL 权重以获得更解耦的表示）。",
                "KL 对 $\\mu=0,\\sigma=1$ 时为 0。",
                "生成新样本：从 $\\mathcal N(0,I)$ 采样 $\\mathbf z$，送入解码器。",
                "《医学图像分析中的深度学习》第 5、16 章用**解耦表示学习**把磁共振图像的解剖内容与扫描风格分开。",
            ],
            code="ch13_vae_kl.py",
            code_note="比较 KL 散度的闭式解与蒙特卡洛估计，并验证重参数化技巧给出的梯度。",
            src="Foster 第 3 章；Buduma 第 10 章；邱锡鹏 13.2；Bishop 第 10 章（变分推断）",
        ),
        dict(
            title="生成对抗网络（GAN）",
            title_en="Generative adversarial networks",
            body=r"""
**GAN** 由两个网络对抗训练（Goodfellow 等，2014）：**生成器** $G$ 把随机噪声 $\mathbf z$ 变成样本；**判别器** $D$ 判断样本是真实的还是生成的。二者玩一个极小极大博弈：

$$\min_G\max_D\ \mathbb E_{\mathbf x\sim p_{data}}[\log D(\mathbf x)]+\mathbb E_{\mathbf z}[\log(1-D(G(\mathbf z)))]$$

Foster 的寓言：一位从没见过神兽的摄影师（生成器）不断拍出越来越像的“照片”，一位收藏家（判别器）不断提高鉴别真伪的眼力，两人在对抗中共同进步。理论上，最优判别器是 $D^*(\mathbf x)=\dfrac{p_{data}(\mathbf x)}{p_{data}(\mathbf x)+p_g(\mathbf x)}$，博弈的均衡点是 $p_g=p_{data}$。

**训练难题**（Foster 第 4 章）：损失震荡、**模式崩溃**（生成器只会生成少数几种样本）、损失值无法反映生成质量、对超参数敏感。改进方法：

- **WGAN**：用瓦瑟斯坦距离（把一个分布“搬运”成另一个的最小代价）代替原损失，即使两个分布不重叠也有有用的梯度；要求判别器满足利普希茨约束，用权重裁剪或**梯度惩罚**（WGAN-GP）实现。
- **DCGAN**（卷积结构）、条件 GAN（指定生成类别）、**Pix2Pix**（成对图像翻译）、**CycleGAN**（无配对翻译，用循环一致性损失，如把照片变成莫奈风格）、**StyleGAN**（高质量人脸）、**SRGAN**（超分辨率）。

评估 GAN 常用 **Inception 分数**和 **FID**（真实与生成图像在特征空间中的分布距离，越低越好）。
""",
            points=[
                "生成器常用“非饱和”损失：最大化 $\\log D(G(\\mathbf z))$，训练初期梯度更强。",
                "代码中的一维 GAN：均值很快学到，方差却来回震荡——这就是训练不稳定的缩影。",
                "CycleGAN 的循环一致性：A→B→A 应该回到原图。",
                "Elgendy 第 8 章的项目、《PyTorch 现代计算机视觉》第 12–13 章都有完整的 GAN 实现。",
            ],
            code="ch13_gan_1d.py",
            code_note="一个极简的一维 GAN：线性生成器学习模仿 $\\mathcal N(4,1.25^2)$，观察均值与标准差的变化。",
            src="Foster 第 4–5 章；Elgendy 第 8 章；Ayyadevara & Reddy 第 12–13 章；邱锡鹏 13.3",
        ),
        dict(
            title="扩散模型与稳定扩散",
            title_en="Diffusion models and Stable Diffusion",
            body=r"""
**扩散模型**把生成拆成很多个小步骤（《PyTorch 现代计算机视觉》第 16 章）：

- **前向过程**：逐步往图像里加高斯噪声，$T$ 步（如 1000 步）后变成纯噪声。它有闭式解，任意 $t$ 步可以一步算出：$\mathbf x_t=\sqrt{\bar\alpha_t}\,\mathbf x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$。
- **反向过程**：训练一个网络（通常是带注意力的 U-Net）预测加入的噪声 $\boldsymbol\epsilon_\theta(\mathbf x_t,t)$，损失只是简单的均方误差 $\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\|^2$。生成时从纯噪声出发，一步步去噪。

预测噪声与学习数据分布的**分数**（对数密度的梯度）是等价的，这把扩散模型与 Buduma 第 10 章讲的**去噪自编码器和分数匹配**联系在一起。相比 GAN，扩散模型训练稳定、不易模式崩溃、样本多样性好，代价是采样需要很多步（DDIM、蒸馏等方法可以大幅减少步数，SDXL Turbo 只需 1–4 步）。

**稳定扩散**（Stable Diffusion）的三个组件：

1. **VAE**：把 $512\times512$ 的图像压缩到 $64\times64$ 的隐空间，扩散在隐空间进行，计算量大幅下降；
2. **U-Net 去噪器**：通过**交叉注意力**接收文本条件；
3. **文本编码器**（CLIP）：把提示词变成向量。

再加上**无分类器引导**（classifier-free guidance）增强与提示词的一致性。衍生应用有图像修复（in-painting）、**ControlNet**（用边缘图、姿态图等精确控制构图）、文本生成视频等。
""",
            points=[
                "前向加噪不需要学习；需要学习的只有去噪网络。",
                "$\\bar\\alpha_t$ 从接近 1 降到接近 0：信号逐渐被噪声淹没。",
                "代码中用解析的“理想去噪器”验证了采样算法：从纯噪声生成的样本均值和方差与数据分布一致。",
                "Foster 第一版（2019）还没有扩散模型，但它讲的 VAE、GAN 和世界模型正是今天生成式 AI 的根基。",
            ],
            code="ch13_diffusion.py",
            code_note="计算线性噪声调度下信号与噪声的比例；对高斯数据使用解析的最优噪声预测器，运行完整的 DDPM 逆向采样。",
            src="Ayyadevara & Reddy 第 16–17 章；Buduma 10.4；Torralba 第 32–34 章",
        ),
    ],
    examples=[
        dict(
            title="计算 VAE 的 KL 项",
            problem=r"一维隐变量，编码器输出 $\mu=2,\ \sigma=0.3$。求 $D_{KL}(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,1))$。如果 $\mu=0,\ \sigma=1$ 呢？",
            steps=[
                r"公式：$D_{KL}=\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)$。",
                r"代入：$\mu^2=4$，$\sigma^2=0.09$，$\log\sigma^2=\log0.09\approx-2.408$。",
                r"$D_{KL}=\tfrac12(4+0.09-1+2.408)=\tfrac12\times5.498\approx2.749$。",
                r"$\mu=0,\sigma=1$ 时四项相加为 0，KL 为 0：后验与先验完全一致。KL 项惩罚“偏离原点”和“方差过小”两种情况。",
            ],
            answer="约 2.749；第二种情况为 0。",
            code="ch13_vae_kl.py",
        ),
        dict(
            title="最优判别器",
            problem=r"在某个点 $\mathbf x$ 处，真实数据密度 $p_{data}(\mathbf x)=0.3$，生成器密度 $p_g(\mathbf x)=0.1$。最优判别器在这一点的输出是多少？当生成器完美时呢？",
            steps=[
                r"最优判别器 $D^*(\mathbf x)=\dfrac{p_{data}(\mathbf x)}{p_{data}(\mathbf x)+p_g(\mathbf x)}$。",
                r"代入：$D^*=0.3/(0.3+0.1)=0.75$，判别器有 75% 的把握认为该点来自真实数据。",
                r"生成器完美时 $p_g=p_{data}$，$D^*=0.5$ 处处成立，判别器只能瞎猜。",
                r"此时原始 GAN 的目标等于 $-\log4$，对应真实分布与生成分布的 JS 散度为 0。",
            ],
            answer="0.75；生成器完美时为 0.5。",
        ),
        dict(
            title="扩散过程中还剩多少信号",
            problem=r"线性噪声调度 $\beta_t$ 从 $10^{-4}$ 增加到 $0.02$，共 1000 步。在第 500 步时，$\mathbf x_t$ 中原图像的系数约为 0.279、噪声系数约为 0.960。这说明什么？第 999 步呢？",
            steps=[
                r"$\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$，系数的平方和为 1。",
                r"第 500 步：信号能量占 $0.279^2\approx7.8\%$，噪声占约 92%，图像已经很难辨认。",
                r"第 999 步：信号系数约 0.006，几乎是纯噪声——所以生成时可以直接从 $\mathcal N(0,I)$ 出发。",
                r"训练时随机抽取 $t$，用闭式解一步得到 $\mathbf x_t$，无需真的逐步加噪，训练非常高效。",
            ],
            answer="中途信号只剩约 8% 的能量；最后一步几乎是纯噪声。",
            code="ch13_diffusion.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="VAE 中重参数化技巧的作用是：", options=["加速推理", "让采样步骤可以反向传播梯度", "减少隐空间维度", "替代 KL 项"], answer=1, explain="$\\mathbf z=\\mu+\\sigma\\epsilon$。"),
            dict(q="GAN 中“生成器只生成少数几种样本”的现象叫：", options=["梯度消失", "模式崩溃", "过平滑", "后验坍塌"], answer=1, explain="mode collapse。"),
            dict(q="CycleGAN 能在没有成对数据时做图像翻译，关键是：", options=["交叉熵损失", "循环一致性损失", "L1 正则", "注意力机制"], answer=1, explain="A→B→A 应还原。"),
            dict(q="扩散模型的去噪网络通常被训练来预测：", options=["图像类别", "加入的噪声", "文本描述", "判别概率"], answer=1, explain="$\\|\\epsilon-\\epsilon_\\theta\\|^2$。"),
            dict(q="稳定扩散在哪里进行扩散过程？", options=["原始像素空间", "VAE 的隐空间", "文本空间", "频率域"], answer=1, explain="隐空间扩散大幅降低计算量。"),
            dict(q="FID 指标：", options=["越高越好", "越低越好", "只衡量多样性", "只用于文本"], answer=1, explain="衡量生成与真实分布的距离。"),
        ],
        tf=[
            dict(q="普通自编码器的隐空间通常是连续的，随便采样都能生成合理图像。", answer=False, explain="这正是 VAE 要解决的问题。"),
            dict(q="当生成器分布等于真实分布时，最优判别器处处输出 0.5。", answer=True, explain="$D^*=p_{data}/(p_{data}+p_g)$。"),
            dict(q="WGAN 用瓦瑟斯坦距离代替原始 GAN 的 JS 散度，使训练更稳定。", answer=True, explain="分布不重叠时仍有梯度。"),
            dict(q="扩散模型的前向加噪过程需要用神经网络学习。", answer=False, explain="前向过程是固定的，有闭式解。"),
            dict(q="线性自编码器学到的子空间与 PCA 相同。", answer=True, explain="代码中已验证。"),
        ],
        fill=[
            dict(q="VAE 的损失 = 重建误差 + ____ 散度。", answer=["KL"], explain="KL 正则项。"),
            dict(q="GAN 由生成器和 ____ 两个网络组成。", answer=["判别器", "discriminator"], explain="discriminator。"),
            dict(q="用边缘图、姿态图等条件精确控制稳定扩散构图的方法叫 ____。", answer=["ControlNet"], explain="ControlNet。"),
            dict(q="WGAN-GP 中的 GP 指梯度 ____。", answer=["惩罚", "penalty"], explain="gradient penalty。"),
        ],
        reading=[
            dict(
                title="What I Cannot Create",
                passage="""The physicist Richard Feynman once wrote that what he could not create, he did not understand. Generative models take this idea seriously. A classifier only needs to tell cats from dogs; a generative model must capture what cats and dogs look like well enough to draw new ones.

Early approaches compressed images into a small set of numbers and learned to rebuild them. Variational autoencoders added a twist: they forced the compressed codes to fill a smooth, well-behaved space, so that any point in it decodes into something sensible. Generative adversarial networks took a different route, pitting a forger against a detective until the forgeries became hard to tell apart from real photographs. Their results were sharp, but training them was notoriously fragile.

Diffusion models arrived with a patient strategy. They learn to undo noise one small step at a time. Starting from pure static, the model gradually sculpts an image, guided, if we wish, by a sentence describing what we want to see. Today's text-to-image systems are built on this idea, often combined with the compressed spaces that autoencoders first explored.""",
                zh="""物理学家理查德·费曼曾写道：我不能创造的东西，我就不理解。生成模型认真对待了这句话。分类器只需要区分猫和狗；生成模型却必须足够好地把握猫和狗长什么样，才能画出新的猫和狗。

早期方法把图像压缩成一小组数字，再学习把它们重建出来。变分自编码器加了一个转折：它强迫这些压缩编码填满一个平滑、规整的空间，使其中的任意一点都能解码成有意义的东西。生成对抗网络走了另一条路，让一个伪造者与一个侦探对抗，直到伪造品难以与真实照片区分。它们的结果很清晰，但训练出了名地脆弱。

扩散模型带来了一种耐心的策略：一次一小步地学习撤销噪声。从纯粹的雪花噪声出发，模型逐渐雕刻出一幅图像，如果我们愿意，还可以用一句描述想看内容的话来引导它。今天的文生图系统就建立在这个想法之上，并且常常结合了自编码器最先探索的压缩空间。""",
                questions=[
                    dict(q="What extra property do variational autoencoders enforce on their codes?", options=["They must be integers", "They must fill a smooth, well-behaved space", "They must be very large", "They must match the labels"], answer=1, explain="平滑规整的隐空间。"),
                    dict(q="How does the passage characterize GAN training?", options=["Very stable", "Notoriously fragile", "Impossible", "Identical to diffusion"], answer=1, explain="出了名地脆弱。"),
                    dict(q="What is the core strategy of diffusion models?", options=["Compress images into one number", "Undo noise one small step at a time", "Use two competing networks", "Copy images from the training set"], answer=1, explain="逐步去噪。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("generative model", "生成模型", "A model that learns the data distribution and can sample new examples.", "学习数据分布并能采样出新样本的模型。", "A generative model can draw faces that never existed."),
        ("latent space", "隐空间", "A lower-dimensional space of hidden variables from which data are generated.", "生成数据所依据的低维隐变量空间。", "Walking through the latent space morphs one face into another."),
        ("autoencoder", "自编码器", "A network trained to reconstruct its input through a narrow bottleneck.", "通过狭窄瓶颈重建自身输入的网络。", "The autoencoder compressed each image to sixteen numbers."),
        ("encoder", "编码器", "The part of a model that maps inputs to a compact representation.", "把输入映射为紧凑表示的模型部分。", "The encoder outputs a mean and a variance."),
        ("decoder", "解码器", "The part of a model that maps a representation back to data space.", "把表示映射回数据空间的模型部分。", "The decoder turns a latent vector into an image."),
        ("variational autoencoder", "变分自编码器", "An autoencoder whose encoder outputs a distribution regularized toward a standard normal prior.", "编码器输出分布、并被拉向标准正态先验的自编码器。", "A variational autoencoder can sample new digits."),
        ("reparameterization trick", "重参数化技巧", "Writing a random sample as mean plus sigma times noise so gradients can flow.", "把随机样本写成均值加 σ 乘噪声以便梯度传播。", "The reparameterization trick makes the VAE trainable by backpropagation."),
        ("generative adversarial network", "生成对抗网络", "A generator and a discriminator trained against each other.", "相互对抗训练的生成器与判别器。", "The generative adversarial network produced sharp images."),
        ("discriminator", "判别器", "The network in a GAN that judges whether a sample is real or generated.", "GAN 中判断样本真假的网络。", "The discriminator became too strong too early."),
        ("mode collapse", "模式崩溃", "A GAN failure in which the generator produces only a few kinds of outputs.", "生成器只产生少数几种输出的 GAN 失败模式。", "Mode collapse made every generated digit look like a seven."),
        ("cycle consistency", "循环一致性", "The requirement that translating to another domain and back recovers the original.", "翻译到另一个域再翻译回来应还原原图的要求。", "Cycle consistency lets CycleGAN learn without paired data."),
        ("diffusion model", "扩散模型", "A model that generates data by learning to reverse a gradual noising process.", "通过学习逆转逐步加噪过程来生成数据的模型。", "The diffusion model needs many denoising steps."),
        ("denoising", "去噪", "Removing noise to recover a clean signal.", "去除噪声以恢复干净信号。", "Each step of the reverse process performs a little denoising."),
        ("classifier-free guidance", "无分类器引导", "Mixing conditional and unconditional predictions to follow the prompt more closely.", "混合有条件和无条件预测，使生成更贴合提示词。", "Higher classifier-free guidance follows the prompt more strictly."),
    ],
    cards=[
        ("VAE 损失？", r"重建误差 + $D_{KL}(q(\mathbf z|\mathbf x)\|\mathcal N(0,I))$；高斯 KL $=\frac12(\mu^2+\sigma^2-1-\log\sigma^2)$。"),
        ("GAN 的目标函数？", r"$\min_G\max_D\ \mathbb E[\log D(\mathbf x)]+\mathbb E[\log(1-D(G(\mathbf z)))]$"),
        ("扩散前向过程闭式解？", r"$\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$"),
        ("稳定扩散三大组件？", "VAE（隐空间）、带交叉注意力的 U-Net 去噪器、CLIP 文本编码器。"),
    ],
)
