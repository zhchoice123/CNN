CHAPTER = dict(
    no=12,
    title="序列模型、注意力机制与 Transformer",
    title_en="Sequence Models, Attention and Transformers",
    group="序列、生成与对齐",
    summary="文本、语音和时间序列都是序列。本章从词嵌入讲起，经过循环神经网络与 LSTM、编码器—解码器与注意力机制，最终到自注意力、多头注意力和 Transformer——今天大语言模型和视觉 Transformer 的共同架构。",
    sources=["Chollet《Python 深度学习》第 10–11 章", "Buduma《深度学习基础》第 8–9 章", "邱锡鹏 第 6、8、15 章", "Géron 第 15–16 章", "Patterson & Gibson 第 4 章", "Torralba 等第 25–26 章"],
    learn=[
        dict(
            title="文本表示：分词、独热编码与词嵌入",
            title_en="Text representation and embeddings",
            body=r"""
神经网络只认识数字。处理文本的第一步是**分词**（切成词、子词或字符），建立**词表**，把每个词元映射成一个整数索引。

最直接的表示是**独热编码**：词表有 $V$ 个词，每个词就是一个 $V$ 维向量，只有一个位置为 1。它简单，但维度极高、极其稀疏，而且任意两个词都互相正交，“猫”和“狗”与“猫”和“汽车”一样不相似——这是**局部表示**。

**词嵌入**是**分布式表示**（邱锡鹏 1.3）：每个词用一个低维稠密向量（如 300 维）表示，每一维没有明确含义，但整体刻画了语义，相似的词向量彼此靠近。嵌入层本质上是一个查表操作，权重可以随任务一起训练，也可以预训练。

**Word2Vec**（Buduma 第 8 章）的思想来自一句话：“一个词的含义由它的上下文决定”。跳字模型用中心词预测周围的词，训练完成后隐藏层的权重就是词向量，著名的性质是向量运算 $\text{king}-\text{man}+\text{woman}\approx\text{queen}$。

Chollet 比较了两种文本建模思路：把文本看成**词袋**（忽略顺序），或者看成**序列**。他给出一条经验法则：训练样本数除以平均每个样本的词数，比值大于约 1500 时，序列模型更好；比值较小时，词袋（二元语法）模型往往更好。
""",
            points=[
                "词表外的词（OOV）可以用子词切分（如 BPE）缓解，现代大语言模型都使用子词分词器。",
                "嵌入层 = 一个 $V\\times d$ 的可学习矩阵，输入索引输出对应行。",
                "余弦相似度常用来衡量两个词向量的语义相似程度。",
                "TF-IDF 是一种考虑词频与文档频率的经典词袋加权方法。",
            ],
            src="Chollet 11.1–11.3；Buduma 第 8 章；邱锡鹏 1.3；Patterson & Gibson 第 8 章",
        ),
        dict(
            title="循环神经网络与 LSTM",
            title_en="Recurrent neural networks and LSTM",
            body=r"""
**循环神经网络（RNN）**按时间步依次读入序列，用一个**隐藏状态**记住之前看到的内容：

$$\mathbf h_t=\tanh(W\mathbf h_{t-1}+U\mathbf x_t+\mathbf b)$$

所有时间步共享同一组参数。训练时把网络沿时间展开，用**随时间反向传播（BPTT）**求梯度。问题在于：梯度要连乘 $T$ 次 $W^T\mathrm{diag}(1-\mathbf h^2)$，很快就会消失（或爆炸），于是普通 RNN 很难学到**长程依赖**。

**长短期记忆网络（LSTM）**引入一条**细胞状态** $\mathbf c_t$ 作为“信息传送带”，并用三个门控制它：

$$\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t,\qquad \mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t)$$

- **遗忘门** $\mathbf f_t$ 决定丢掉多少旧记忆；
- **输入门** $\mathbf i_t$ 决定写入多少新信息；
- **输出门** $\mathbf o_t$ 决定输出多少。

细胞状态的更新是**加法**，当遗忘门接近 1 时，信息和梯度都能几乎无损地穿越很多时间步——这和残差连接的思想如出一辙。**GRU** 是简化版，只有更新门和重置门。

此外还有**双向 RNN**（同时从前往后和从后往前读）、**堆叠 RNN**，以及用于序列的一维卷积。Chollet 第 10 章在温度预测任务上提醒：先建立一个常识性基线（明天温度 = 今天温度），很多复杂模型未必打得过它。
""",
            points=[
                "RNN 的参数在时间步之间共享，所以能处理任意长度的序列。",
                "梯度爆炸用梯度裁剪解决，梯度消失用 LSTM/GRU 解决。",
                "RNN 必须按时间顺序计算，难以并行，这是它被 Transformer 取代的重要原因之一。",
                "邱锡鹏第 6 章还把循环网络推广到树结构（递归神经网络）和图结构（图神经网络）。",
            ],
            code="ch12_rnn_memory.py",
            code_note="比较普通 RNN 的梯度随时间步的衰减，以及 LSTM 细胞状态在遗忘门≈1 时保留信息的能力。",
            src="Chollet 第 10 章；Buduma 第 9 章；邱锡鹏 6.1–6.6；Géron 第 15 章",
        ),
        dict(
            title="编码器—解码器与注意力机制",
            title_en="Encoder–decoder and attention",
            body=r"""
机器翻译等**序列到序列**任务的输入和输出长度不同。经典做法是**编码器—解码器**：编码器 RNN 读完整个源句，把它压缩成一个固定长度的向量；解码器 RNN 从这个向量出发，逐词生成目标句子。

问题在于：把一整句话（甚至一段话）压进一个固定向量是**信息瓶颈**，句子越长效果越差。**注意力机制**的想法是：解码器在生成每个词时，不只看那一个压缩向量，而是回头查看编码器在**所有**位置的输出，并按相关程度加权求和。权重本身由网络计算：

$$\alpha_{t,i}=\mathrm{softmax}_i\big(\mathrm{score}(\mathbf s_t,\mathbf h_i)\big),\qquad \mathbf c_t=\sum_i\alpha_{t,i}\mathbf h_i$$

可视化注意力权重，可以看到译文中的每个词“对齐”到原文中的哪些词——这也是一种可解释性。

推广到一般形式，注意力就是一种“软查表”：用**查询**（query）与每个**键**（key）计算相似度，再按相似度对对应的**值**（value）加权平均。邱锡鹏第 8 章把它与人脑的注意力和记忆联系起来；Buduma 第 12 章的神经图灵机和可微分神经计算机，也是用注意力来读写外部记忆。

生成序列时常用**束搜索**（beam search）代替逐步贪心，保留若干个最好的候选序列。
""",
            points=[
                "注意力打破了固定长度向量的信息瓶颈。",
                "注意力权重非负且和为 1（softmax），可以解释为“关注程度”。",
                "教师强制：训练解码器时输入真实的上一个词，推理时输入模型自己的输出（会带来曝光偏差）。",
                "注意力本身没有顺序概念，需要额外注入位置信息。",
            ],
            src="Buduma 9.6–9.7；Chollet 11.5；邱锡鹏 8.1–8.2、15.6；Géron 第 16 章",
        ),
        dict(
            title="自注意力、多头注意力与 Transformer",
            title_en="Self-attention and the Transformer",
            body=r"""
2017 年的 Transformer 完全抛弃了循环结构，只用注意力。核心是**缩放点积注意力**：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

在**自注意力**中，$Q,K,V$ 都由同一个序列的输入线性变换得到，每个词元都能直接关注序列中的任何其他词元，任意两个位置之间的“路径长度”都是 1，而且所有位置可以**并行**计算。除以 $\sqrt{d_k}$ 是因为点积的方差随维度增长，不缩放时 softmax 会接近 one-hot，梯度很小。

**多头注意力**并行地做 $h$ 组注意力，每组有自己的投影矩阵，捕捉不同类型的关系，最后拼接起来再投影。

一个**编码器层** = 多头自注意力 + 逐位置前馈网络，每个子层都有**残差连接**和**层归一化**。由于注意力不考虑顺序，需要加入**位置编码**（如正弦位置编码，或可学习的位置向量）。**解码器**使用**因果掩码**，保证预测第 $t$ 个词时只能看见前面的词。

Transformer 衍生出两大家族：只用编码器的 **BERT**（双向理解，掩码语言模型预训练）和只用解码器的 **GPT**（从左到右生成）——后者正是今天大语言模型的架构（第 14 章）。**视觉 Transformer（ViT）**把图像切成 16×16 的小块，像处理单词一样处理它们（Torralba 第 26 章，《PyTorch 现代计算机视觉》第 15 章）。自注意力的代价是计算量与序列长度的**平方**成正比。
""",
            points=[
                "Q 问“我在找什么”，K 答“我包含什么”，V 是“我实际提供的内容”。",
                "$d_{model}=512$、8 头时，每个头的维度 $d_k=64$。",
                "残差 + 层归一化 让很深的 Transformer 也能稳定训练。",
                "长序列的平方复杂度催生了稀疏注意力、线性注意力、FlashAttention 等改进。",
            ],
            code="ch12_attention.py",
            code_note="实现缩放点积自注意力和因果掩码；统计不同 $d_k$ 下点积的标准差，验证为什么要除以 $\\sqrt{d_k}$。",
            src="Chollet 11.4；Buduma 9.8；邱锡鹏 8.3；Géron 第 16 章；Foster 第 9 章；Torralba 第 26 章",
        ),
    ],
    examples=[
        dict(
            title="手算两个词元的注意力",
            problem=r"查询 $\mathbf q=(1,0)$，两个键 $\mathbf k_1=(1,0)$、$\mathbf k_2=(0,1)$，对应的值 $\mathbf v_1=(10,0)$、$\mathbf v_2=(0,10)$，$d_k=2$。求注意力权重和输出。",
            steps=[
                r"点积：$\mathbf q\cdot\mathbf k_1=1$，$\mathbf q\cdot\mathbf k_2=0$。",
                r"缩放：除以 $\sqrt2$，得到 $0.707$ 和 $0$。",
                r"softmax：$\alpha_1=e^{0.707}/(e^{0.707}+1)\approx2.028/3.028\approx0.670$，$\alpha_2\approx0.330$。",
                r"输出 $=0.670\times(10,0)+0.330\times(0,10)=(6.70,3.30)$。查询与 $\mathbf k_1$ 更相似，所以输出更偏向 $\mathbf v_1$。",
            ],
            answer=r"权重约 (0.670, 0.330)，输出约 (6.70, 3.30)。",
        ),
        dict(
            title="一个 Transformer 编码器层有多少参数",
            problem=r"$d_{model}=512$，8 个头，前馈层隐藏维度 $d_{ff}=2048$。计算一个编码器层的参数量（含偏置和层归一化）。",
            steps=[
                r"多头注意力：$W_Q,W_K,W_V$ 和输出投影 $W_O$ 都是 $512\times512$ 加偏置，共 $4\times(512\times512+512)=1{,}050{,}624$。",
                r"前馈网络：$512\to2048\to512$，$(512\times2048+2048)+(2048\times512+512)=2{,}099{,}712$。",
                r"两个层归一化各有 $\gamma,\beta$ 共 $2\times512$，合计 $2048$。",
                r"总计 $3{,}152{,}384$，约 315 万；多头并不增加参数，只是把 512 维切成 8 个 64 维的头。前馈网络占了约三分之二的参数。",
            ],
            answer="约 3.15M 参数（注意力 1.05M，前馈 2.10M）。",
            code="ch12_transformer_numbers.py",
        ),
        dict(
            title="该用词袋模型还是序列模型",
            problem=r"根据 Chollet 的经验法则判断：(A) 2 万条影评，平均每条 230 个词；(B) 50 万条短消息，平均每条 20 个词。",
            steps=[
                r"计算“样本数 / 平均词数”。(A) $20000/230\approx87$。",
                r"(B) $500000/20\approx25000$。",
                r"比值小于约 1500 时，词袋（加二元语法）模型通常更好、更省；比值大时，序列模型（如 Transformer）更能发挥作用。",
                r"所以 (A) 先试词袋模型，(B) 适合序列模型。这只是经验法则，最终仍要用验证集比较。",
            ],
            answer="(A) 词袋模型；(B) 序列模型。",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="LSTM 中决定丢弃多少旧记忆的是：", options=["输入门", "遗忘门", "输出门", "重置门"], answer=1, explain="遗忘门 $f_t$。"),
            dict(q="缩放点积注意力中除以 $\\sqrt{d_k}$ 的原因是：", options=["减少参数量", "防止点积过大导致 softmax 饱和", "实现因果掩码", "加入位置信息"], answer=1, explain="点积方差 ∝ $d_k$。"),
            dict(q="Transformer 需要位置编码，是因为：", options=["注意力本身不区分词元顺序", "为了减少计算量", "为了做归一化", "为了生成掩码"], answer=0, explain="自注意力对排列不敏感。"),
            dict(q="GPT 类模型使用的是 Transformer 的：", options=["只有编码器", "只有解码器（带因果掩码）", "编码器和解码器都有", "不使用 Transformer"], answer=1, explain="从左到右生成。"),
            dict(q="自注意力的计算量随序列长度 $n$ 的增长是：", options=["$O(\\log n)$", "$O(n)$", "$O(n^2)$", "$O(n^3)$"], answer=2, explain="每对位置都要计算分数。"),
            dict(q="词嵌入相对于独热编码的主要优势是：", options=["维度更高", "能表达词与词之间的语义相似性", "完全不需要训练", "只能表示 100 个词"], answer=1, explain="分布式表示。"),
        ],
        tf=[
            dict(q="普通 RNN 很难学到长程依赖，主要是因为梯度消失。", answer=True, explain="梯度随时间步连乘衰减。"),
            dict(q="RNN 的不同时间步使用不同的参数。", answer=False, explain="参数在时间步之间共享。"),
            dict(q="多头注意力会让参数量按头数成倍增加。", answer=False, explain="通常把 $d_{model}$ 切分到各头，总参数量不变。"),
            dict(q="Transformer 的每个子层都使用了残差连接和层归一化。", answer=True, explain="Add & Norm。"),
            dict(q="注意力机制缓解了编码器—解码器模型中固定长度向量的信息瓶颈。", answer=True, explain="解码时可以回看所有位置。"),
        ],
        fill=[
            dict(q="注意力公式：$\\mathrm{softmax}(QK^T/\\sqrt{d_k})$ 乘以 ____。", answer=["V", "value", "值"], explain="对值向量加权求和。"),
            dict(q="训练 RNN 时沿时间展开求梯度的方法缩写为 ____。", answer=["BPTT"], explain="Backpropagation Through Time。"),
            dict(q="只用编码器、以掩码语言模型预训练的代表模型是 ____。", answer=["BERT"], explain="Bidirectional Encoder Representations from Transformers。"),
            dict(q="把图像切成小块再用 Transformer 处理的模型缩写为 ____。", answer=["ViT", "视觉Transformer"], explain="Vision Transformer。"),
        ],
        reading=[
            dict(
                title="Attention Is All You Need",
                passage="""Before 2017, the best translation systems read a sentence one word at a time with recurrent networks. This sequential reading had two costs. Information from early words had to survive a long chain of updates to influence later ones, and the computation could not be spread across many processors because each step waited for the previous one.

The Transformer removed recurrence entirely. In a self-attention layer, every word looks directly at every other word in the sentence and decides how much to borrow from each. Because all of these comparisons happen at once, training runs efficiently on modern hardware. Since attention by itself ignores word order, the model adds a positional signal to each word's representation.

Stacking many such layers, each wrapped with residual connections and normalization, produced better translations at a fraction of the training cost. The same architecture soon spread far beyond translation. Encoder-only versions learned to understand text, decoder-only versions learned to generate it, and vision researchers cut images into patches and treated them as words. A single idea had become the backbone of modern AI.""",
                zh="""2017 年以前，最好的翻译系统用循环网络一个词一个词地读句子。这种顺序读取有两个代价：前面词语的信息必须经过一长串更新才能影响后面的词；而且计算无法分摊到许多处理器上，因为每一步都要等上一步完成。

Transformer 完全去掉了循环。在自注意力层中，每个词直接查看句子中的其他每个词，并决定从每个词那里借用多少信息。由于所有这些比较同时进行，训练能在现代硬件上高效运行。因为注意力本身忽略词序，模型会给每个词的表示加上位置信号。

把许多这样的层堆叠起来，每层都包裹着残差连接和归一化，就以远低于以往的训练成本得到了更好的翻译。同样的架构很快扩散到翻译之外：只有编码器的版本学会了理解文本，只有解码器的版本学会了生成文本，视觉研究者则把图像切成小块，把它们当作单词来处理。一个想法成为了现代人工智能的骨干。""",
                questions=[
                    dict(q="What were the two costs of sequential reading with RNNs?", options=["Too many parameters and too little data", "Long information paths and poor parallelism", "No attention and no embeddings", "High memory and low accuracy only"], answer=1, explain="信息路径长、难以并行。"),
                    dict(q="Why does the Transformer add a positional signal?", options=["Attention alone ignores word order", "To reduce memory", "To translate faster", "To replace normalization"], answer=0, explain="注意力不区分顺序。"),
                    dict(q="How did vision researchers apply the architecture?", options=["By using recurrent layers", "By treating image patches as words", "By removing attention", "By translating images into text first"], answer=1, explain="图像块当作单词。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("tokenization", "分词", "Splitting text into units such as words, subwords or characters.", "把文本切分成词、子词或字符等单元。", "Subword tokenization handles rare words gracefully."),
        ("one-hot encoding", "独热编码", "A vector with a single one at the index of the item and zeros elsewhere.", "只在该项索引处为 1、其余为 0 的向量。", "One-hot encoding treats every pair of words as equally different."),
        ("word embedding", "词嵌入", "A dense learned vector that represents a word's meaning.", "表示词义的稠密可学习向量。", "Similar words have nearby word embeddings."),
        ("recurrent neural network", "循环神经网络", "A network that processes sequences step by step while keeping a hidden state.", "逐步处理序列并维持隐藏状态的网络。", "A recurrent neural network reads the sentence word by word."),
        ("hidden state", "隐藏状态", "The internal vector an RNN carries from one time step to the next.", "循环网络在时间步之间传递的内部向量。", "The final hidden state summarizes the whole input."),
        ("long short-term memory", "长短期记忆网络", "An RNN with gated cell state that can retain information over long spans.", "带门控细胞状态、能长期保留信息的循环网络。", "Long short-term memory networks power many speech systems."),
        ("gate", "门", "A sigmoid-controlled multiplier deciding how much information passes through.", "由 sigmoid 控制、决定信息通过多少的乘法因子。", "The forget gate decides what to erase from memory."),
        ("sequence-to-sequence", "序列到序列", "Mapping an input sequence to an output sequence of possibly different length.", "把输入序列映射为长度可能不同的输出序列。", "Machine translation is a sequence-to-sequence task."),
        ("attention mechanism", "注意力机制", "Weighting different input positions by relevance when producing each output.", "生成每个输出时按相关性对不同输入位置加权。", "The attention mechanism aligns target words with source words."),
        ("self-attention", "自注意力", "Attention in which queries, keys and values all come from the same sequence.", "查询、键、值都来自同一序列的注意力。", "Self-attention lets every token see every other token."),
        ("query, key and value", "查询、键与值", "The three projections used to compute and apply attention weights.", "计算并应用注意力权重的三种投影。", "Each head has its own query, key and value matrices."),
        ("multi-head attention", "多头注意力", "Running several attention operations in parallel and concatenating their outputs.", "并行运行多组注意力并拼接其输出。", "Multi-head attention captures different kinds of relations."),
        ("positional encoding", "位置编码", "Information added to token representations to indicate their order.", "加到词元表示上、指示其顺序的信息。", "Sinusoidal positional encoding needs no training."),
        ("Transformer", "Transformer", "An architecture built from stacked self-attention and feed-forward layers.", "由堆叠的自注意力层和前馈层构成的架构。", "The Transformer replaced recurrence in machine translation."),
    ],
    cards=[
        ("LSTM 细胞状态更新？", r"$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$，加法更新让信息和梯度能长距离传递。"),
        ("缩放点积注意力？", r"$\mathrm{softmax}(QK^T/\sqrt{d_k})V$"),
        ("BERT vs GPT？", "BERT：仅编码器、双向、掩码语言模型；GPT：仅解码器、因果掩码、从左到右生成。"),
        ("自注意力相比 RNN 的优势与代价？", "优势：任意位置一步可达、可并行；代价：计算量随序列长度平方增长。"),
    ],
)
