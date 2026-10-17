CHAPTER = dict(
    no=14,
    title="强化学习与基于人类反馈的强化学习（RLHF）",
    title_en="Reinforcement Learning and RLHF",
    group="序列、生成与对齐",
    summary="智能体在与环境的交互中通过奖励学习。本章讲马尔可夫决策过程、探索与利用、Q 学习与深度 Q 网络、策略梯度与 PPO，再走进大语言模型的后训练：指令微调、奖励模型、RLHF、DPO 与可验证奖励的强化学习。",
    sources=["Lambert《基于人类反馈的强化学习》（2026 年 3 月版）第 1–17 章", "Buduma《深度学习基础》第 13 章", "Géron《机器学习实战》第 18 章", "邱锡鹏 第 14 章", "Thrun 等《概率机器人》第 15–16 章"],
    learn=[
        dict(
            title="马尔可夫决策过程与探索—利用",
            title_en="MDPs and exploration versus exploitation",
            body=r"""
强化学习的设定：**智能体**在**状态** $s_t$ 下选择**动作** $a_t$，环境返回**奖励** $r_t$ 和新状态 $s_{t+1}$。目标是学到一个**策略** $\pi(a\mid s)$，最大化**折扣回报**：

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\dots=\sum_{k\ge0}\gamma^k r_{t+k}$$

折扣因子 $\gamma\in[0,1)$ 让近期奖励比远期奖励更重要，也保证无穷和收敛。这一形式化框架叫**马尔可夫决策过程（MDP）**：下一状态只依赖当前状态和动作。

两个核心函数：**状态价值** $V^\pi(s)$ 是从 $s$ 出发、按 $\pi$ 行动的期望回报；**动作价值** $Q^\pi(s,a)$ 是先做动作 $a$、再按 $\pi$ 行动的期望回报。它们满足递归的**贝尔曼方程**，例如最优动作价值：

$$Q^*(s,a)=\mathbb E\big[r+\gamma\max_{a'}Q^*(s',a')\big]$$

与监督学习不同，强化学习没有“正确答案”，只有延迟的、稀疏的奖励，还必须面对**探索与利用**的权衡：一直选当前看起来最好的动作（利用），可能永远发现不了更好的选择；不断尝试新动作（探索）又会损失眼前的奖励。最简单的办法是 **ε-贪心**：以概率 $\varepsilon$ 随机探索，否则选择当前最优，且 $\varepsilon$ 随训练逐渐减小（Buduma 13.2）。
""",
            points=[
                "奖励是对“好坏”的标量信号，设计不当的奖励会被智能体“钻空子”。",
                "信用分配问题：一盘棋输了，到底是哪一步走错了？",
                "《概率机器人》第 15–16 章用 MDP 和 POMDP 处理机器人在不确定性下的决策。",
                "OpenAI Gym（现为 Gymnasium）提供了倒立摆、雅达利等标准环境。",
            ],
            code="ch14_bandit.py",
            code_note="在 10 臂老虎机上比较纯贪心与两种 ε-贪心策略的平均奖励和选中最优臂的比例。",
            src="Buduma 13.2；Géron 18 章；邱锡鹏 14.1；Thrun 第 15 章",
        ),
        dict(
            title="基于价值的方法：Q 学习与深度 Q 网络",
            title_en="Value-based methods: Q-learning and DQN",
            body=r"""
**Q 学习**直接逼近最优动作价值，每走一步就用**时序差分（TD）**目标更新：

$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[\underbrace{r+\gamma\max_{a'}Q(s',a')}_{\text{TD 目标}}-Q(s,a)\big]$$

它是**离策略**的：行动时可以用 ε-贪心探索，学习的却是贪心策略的价值。状态很少时可以用一张 Q 表；状态巨大（例如游戏画面）时，就用神经网络近似 $Q(s,a;\theta)$——这就是**深度 Q 网络（DQN）**，2015 年 DeepMind 用它在许多雅达利游戏上达到人类水平。

让 DQN 稳定训练的两个关键技巧：

- **经验回放**：把交互经验存进缓冲区，训练时随机抽取小批量，打破样本之间的时间相关性；
- **目标网络**：计算 TD 目标时使用一份定期同步的旧参数，避免“追逐移动的目标”。

后续改进包括 Double DQN（减少过高估计）、Dueling DQN 等。《PyTorch 现代计算机视觉》第 14 章用带固定目标网络的 DQN 训练会玩乒乓球的智能体，并在 CARLA 模拟器里训练自动驾驶智能体。
""",
            points=[
                "TD 学习用“当前估计 + 一步真实奖励”来更新估计，不必等到回合结束。",
                "Q 学习只适用于离散动作；连续动作需要策略梯度或演员—评论员方法。",
                "每步加一个小的负奖励，会鼓励智能体走最短路径。",
                "函数逼近 + 自举 + 离策略三者结合可能导致不稳定（“致命三角”）。",
            ],
            code="ch14_qlearning.py",
            code_note="在带陷阱的 4×4 网格世界里用表格 Q 学习训练 2000 个回合，打印学到的策略和贪心路径。",
            src="Buduma 13.4；Géron 18 章；Ayyadevara & Reddy 第 14 章；邱锡鹏 14.2",
        ),
        dict(
            title="策略梯度、演员—评论员与 PPO",
            title_en="Policy gradients and PPO",
            body=r"""
**策略梯度**方法直接把策略参数化为 $\pi_\theta(a\mid s)$，沿着期望回报的梯度上升。**REINFORCE** 的梯度估计为：

$$\nabla_\theta J=\mathbb E\big[\nabla_\theta\log\pi_\theta(a\mid s)\,(G-b)\big]$$

直观地说：回报高的动作，提高它的概率；回报低的，降低它的概率。减去一个**基线** $b$（如回报的平均值）不改变梯度的期望，却能大幅**降低方差**。用一个学到的价值函数作基线，就得到**演员—评论员**方法：演员是策略，评论员估计价值，二者的差 $G-V(s)$ 叫**优势**。

策略梯度的问题是步子迈大了策略可能突然崩溃。**TRPO** 用 KL 约束限制每次更新的幅度；**PPO**（近端策略优化）用一个更简单的“裁剪”目标达到类似效果：

$$L=\mathbb E\big[\min\big(\rho\,\hat A,\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\hat A\big)\big],\qquad \rho=\frac{\pi_\theta(a\mid s)}{\pi_{old}(a\mid s)}$$

PPO 稳定、易实现，是 InstructGPT 等 RLHF 系统最初使用的强化学习算法。Lambert 第 6 章还介绍了用于语言模型的 RLOO 和 **GRPO**：GRPO 对同一个问题采样一组回答，用组内的相对好坏估计优势，不需要单独训练价值网络。
""",
            points=[
                "策略梯度可以直接处理连续动作和随机策略。",
                "基线降低方差、不引入偏差；代码中有基线时梯度方差约减小 40%。",
                "PPO 的裁剪让新旧策略的概率比保持在 $[1-\\epsilon,1+\\epsilon]$ 附近。",
                "Buduma 第 13 章用倒立摆任务实现了策略梯度智能体，并介绍了 TRPO 与 PPO。",
            ],
            code="ch14_reinforce.py",
            code_note="在三个动作的老虎机上实现带/不带基线的 REINFORCE，比较最终策略和梯度估计的方差。",
            src="Buduma 13.3；Lambert 第 6 章；邱锡鹏 14.3–14.4；Géron 18 章“策略梯度”",
        ),
        dict(
            title="RLHF 与大语言模型的后训练",
            title_en="RLHF and post-training of language models",
            body=r"""
只经过预训练的语言模型只会**续写**文本。Lambert 举例：问基座模型“2006 年的美国总统是”，它会接着写出一堆像网页一样的条目；经过后训练的模型则会简洁地回答问题。**后训练**主要包括三类方法（Lambert 第 1 章）：

1. **指令微调（SFT）**：用“指令—回答”对做监督学习，教会模型问答格式。
2. **偏好微调**：对齐人类偏好，主要关乎风格、语气和难以量化的细微偏好——RLHF 主导这一部分。
3. **基于可验证奖励的强化学习（RLVR）**：在数学、代码等答案可以自动验证的领域，用更多的强化学习提升能力，推理模型（如 o1、DeepSeek R1）由此而来。

**经典 RLHF 流程**：先指令微调；再收集人类对同一问题两个回答的偏好，训练**奖励模型**——通常用 **Bradley–Terry 模型** $P(y_w\succ y_l)=\sigma\big(r(y_w)-r(y_l)\big)$；最后用 PPO 等算法优化策略，最大化奖励，同时加 **KL 惩罚**防止偏离参考模型太远。

**DPO**（直接偏好优化）通过推导绕过了显式的奖励模型和强化学习，直接用成对偏好数据优化：

$$\mathcal L_{DPO}=-\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\Big)$$

Lambert 特别强调几个现实问题：奖励模型只是人类偏好的**替代品**，优化太狠会被“钻空子”（**过度优化**，即古德哈特定律），表现为回答越来越长、越来越谄媚；所以需要 KL 等正则化。偏好数据也越来越多由 AI 产生（如宪法 AI、用大模型当评委）。他还提出后训练的“激发理论”：基座模型决定了潜力上限，后训练负责把潜力激发出来。
""",
            points=[
                "RLHF 在整段回答的层面上优化，并且能利用负反馈，因此比逐词的指令微调泛化得更好。",
                "KL 惩罚让模型保持在参考模型附近，防止奖励被钻空子。",
                "DPO 是离线方法；PPO、GRPO 是在线方法，需要不断从当前策略采样。",
                "RLHF 结果噪声大、难复现，要多读几篇论文再下结论（Lambert 1.4.3）。",
            ],
            code="ch14_preferences.py",
            code_note="用 2000 条带噪声的成对偏好训练 Bradley–Terry 奖励模型并恢复隐藏的偏好权重；再手算一次 DPO 损失。",
            src="Lambert 第 1、3–5、8、14–15 章",
        ),
    ],
    examples=[
        dict(
            title="折扣回报与一次 Q 学习更新",
            problem=r"(1) 一个回合依次获得奖励 1, 0, 0, 5，$\gamma=0.9$，求起点的折扣回报。(2) 已知 $Q(s,a)=2.0$，执行后得到 $r=1$，下一状态的最大 Q 值为 3.0，$\alpha=0.1$，求更新后的 $Q(s,a)$。",
            steps=[
                r"(1) $G=1+0.9\times0+0.9^2\times0+0.9^3\times5=1+0.729\times5=4.645$。",
                r"(2) TD 目标 $=r+\gamma\max Q(s',\cdot)=1+0.9\times3=3.7$。",
                r"TD 误差 $=3.7-2.0=1.7$。",
                r"新值 $Q=2.0+0.1\times1.7=2.17$：只朝目标移动一小步，因为目标本身也是一个估计。",
            ],
            answer="G = 4.645；Q 更新为 2.17。",
            code="ch14_ex_return.py",
        ),
        dict(
            title="Bradley–Terry 偏好概率",
            problem=r"奖励模型给回答 A 打 2.5 分，给回答 B 打 1.0 分。按 Bradley–Terry 模型，人类更喜欢 A 的概率是多少？若人类实际选择了 A，这条数据的损失是多少？",
            steps=[
                r"$P(A\succ B)=\sigma(r_A-r_B)=\sigma(1.5)=1/(1+e^{-1.5})\approx0.818$。",
                r"损失为负对数似然：$-\log0.818\approx0.201$。",
                r"若人类选了 B，损失为 $-\log(1-0.818)=-\log0.182\approx1.70$，模型会被强烈纠正。",
                r"只有奖励之**差**有意义，给所有回答加同一个常数不改变任何概率。",
            ],
            answer="约 0.818；损失约 0.201。",
            code="ch14_preferences.py",
        ),
        dict(
            title="手算 DPO 损失",
            problem=r"$\beta=0.1$。对偏好回答 $y_w$：策略对数概率 −12，参考模型 −14；对被拒回答 $y_l$：策略 −15，参考模型 −13。求 DPO 损失，并与策略等于参考模型时比较。",
            steps=[
                r"偏好回答的对数概率比：$-12-(-14)=+2$；被拒回答：$-15-(-13)=-2$。",
                r"隐式奖励差 $=\beta(2-(-2))=0.1\times4=0.4$。",
                r"损失 $=-\log\sigma(0.4)=-\log0.599\approx0.513$。",
                r"若策略与参考模型相同，两个比值都是 0，损失为 $-\log\sigma(0)=\log2\approx0.693$。损失下降，说明策略已经朝人类偏好的方向移动。",
            ],
            answer="DPO 损失约 0.513（初始为 0.693）。",
            code="ch14_preferences.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="折扣因子 $\\gamma$ 越接近 0，智能体越：", options=["重视长远奖励", "重视眼前奖励", "倾向于探索", "倾向于随机行动"], answer=1, explain="远期奖励被大幅折扣。"),
            dict(q="DQN 中经验回放的主要作用是：", options=["增加奖励", "打破样本间的时间相关性", "减少网络层数", "实现探索"], answer=1, explain="随机抽取历史经验。"),
            dict(q="在策略梯度中减去基线的作用是：", options=["改变梯度的期望", "降低梯度估计的方差", "增加探索", "减少参数"], answer=1, explain="无偏且降方差。"),
            dict(q="经典 RLHF 流程的三个阶段依次是：", options=["预训练、剪枝、量化", "指令微调、奖励模型训练、强化学习优化", "数据增强、蒸馏、部署", "分词、嵌入、解码"], answer=1, explain="InstructGPT 式流程。"),
            dict(q="DPO 相比经典 RLHF 的主要区别是：", options=["需要更多人类标注", "不需要显式训练奖励模型和运行强化学习", "只能用于图像", "不使用参考模型"], answer=1, explain="直接从偏好数据优化。"),
            dict(q="RLHF 中加入对参考模型的 KL 惩罚，主要是为了：", options=["加快采样", "防止过度优化奖励模型、偏离太远", "增加词表", "减少显存"], answer=1, explain="正则化。"),
        ],
        tf=[
            dict(q="Q 学习是一种离策略方法。", answer=True, explain="行为策略可以探索，目标是贪心策略。"),
            dict(q="纯贪心策略一定能找到最优动作。", answer=False, explain="可能过早锁定次优动作。"),
            dict(q="PPO 通过裁剪新旧策略的概率比来限制每次更新的幅度。", answer=True, explain="clip 目标。"),
            dict(q="在 Bradley–Terry 模型中，给所有回答的奖励同时加一个常数会改变偏好概率。", answer=False, explain="只有奖励差有意义。"),
            dict(q="GRPO 用同一问题一组回答的相对好坏来估计优势，不需要单独的价值网络。", answer=True, explain="Lambert 第 6 章。"),
        ],
        fill=[
            dict(q="以概率 ε 随机探索、否则选当前最优动作的策略叫 ε-____ 策略。", answer=["贪心", "greedy"], explain="ε-greedy。"),
            dict(q="Q 学习的更新目标 $r+\\gamma\\max_{a'}Q(s',a')$ 叫做 ____ 目标（英文缩写）。", answer=["TD", "时序差分"], explain="Temporal Difference。"),
            dict(q="奖励模型被过度优化、“钻空子”的现象可以用 ____ 定律概括。", answer=["古德哈特", "Goodhart"], explain="指标一旦成为目标就不再是好指标。"),
            dict(q="在数学、代码等可自动验证答案的领域使用的强化学习缩写为 ____。", answer=["RLVR"], explain="Reinforcement Learning with Verifiable Rewards。"),
        ],
        reading=[
            dict(
                title="From Autocomplete to Assistant",
                passage="""A language model fresh from pretraining is an extraordinary autocomplete engine. Ask it who was president in 2006 and it may continue the sentence and then drift into a list of unrelated facts copied from the style of web pages. It knows a great deal, but it does not yet behave like a helpful assistant.

Post-training changes that behavior. Instruction tuning teaches the model the basic format of answering questions. Preference tuning goes further: people compare pairs of answers, a reward model learns to predict which one they would prefer, and the language model is optimized to produce answers that score well. A penalty keeps it close to its original behavior so that it does not discover strange tricks to please the reward model.

Much of what users notice is style: a warm tone, a clear structure, an appropriate length. These qualities are hard to write down as rules, which is exactly why learning from human preferences works so well. The same reward signal can also be exploited, however. Push too hard and answers grow longer, flatter and more eager to agree, a reminder that a measure turned into a target stops being a good measure.""",
                zh="""一个刚完成预训练的语言模型是一台非凡的自动补全引擎。问它 2006 年谁是总统，它可能会把句子续写下去，然后漂移到一串模仿网页风格、互不相关的事实里。它知道很多东西，但还不像一个乐于助人的助手那样行事。

后训练改变了这种行为。指令微调教会模型回答问题的基本格式。偏好微调更进一步：人们比较成对的回答，奖励模型学习预测他们会更喜欢哪一个，语言模型则被优化去产生得分高的回答。一个惩罚项让它保持在原有行为附近，以免它找到一些讨好奖励模型的奇怪花招。

用户注意到的很多东西是风格：温暖的语气、清晰的结构、恰当的篇幅。这些品质很难写成规则，这正是从人类偏好中学习效果如此好的原因。然而，同样的奖励信号也可能被钻空子。用力过猛，回答就会变得更长、更奉承、更急于附和——这提醒我们，一个指标一旦变成目标，就不再是好指标。""",
                questions=[
                    dict(q="What does instruction tuning mainly teach the model?", options=["New facts about history", "The basic format of answering questions", "How to compress text", "How to play games"], answer=1, explain="问答的基本格式。"),
                    dict(q="Why is a penalty used during preference optimization?", options=["To make answers longer", "To keep the model close to its original behavior and avoid exploiting the reward model", "To speed up pretraining", "To remove the reward model"], answer=1, explain="KL 惩罚防止钻空子。"),
                    dict(q="Which symptom of over-optimization does the passage mention?", options=["Answers become shorter and ruder", "Answers become longer, flatter and more eager to agree", "The model stops answering", "The vocabulary shrinks"], answer=1, explain="更长、更奉承。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("reinforcement learning", "强化学习", "Learning to act by trial and error from rewards given by an environment.", "通过试错、从环境给出的奖励中学习如何行动。", "Reinforcement learning taught the agent to balance the pole."),
        ("agent", "智能体", "The learner that observes states and chooses actions.", "观察状态并选择动作的学习者。", "The agent receives a reward after every move."),
        ("reward", "奖励", "A scalar signal telling the agent how good its last action was.", "告诉智能体上一个动作有多好的标量信号。", "A poorly designed reward can be gamed."),
        ("policy", "策略", "A rule or distribution mapping states to actions.", "把状态映射到动作的规则或分布。", "The policy network outputs a probability for each action."),
        ("discount factor", "折扣因子", "A number below one that reduces the weight of future rewards.", "小于一、用来降低未来奖励权重的数。", "A discount factor of 0.99 makes the agent far-sighted."),
        ("Markov decision process", "马尔可夫决策过程", "A formal model of sequential decisions where the next state depends only on the current state and action.", "下一状态只依赖当前状态和动作的序贯决策形式化模型。", "Most RL problems are framed as a Markov decision process."),
        ("exploration", "探索", "Trying actions whose value is uncertain to gather information.", "尝试价值不确定的动作以获取信息。", "Without exploration, the agent got stuck with a mediocre strategy."),
        ("Q-learning", "Q 学习", "An off-policy method that learns action values from temporal-difference targets.", "用时序差分目标学习动作价值的离策略方法。", "Q-learning found the shortest path in the grid."),
        ("experience replay", "经验回放", "Storing past transitions and sampling them randomly for training.", "存储历史交互并随机采样用于训练。", "Experience replay stabilized the deep Q-network."),
        ("policy gradient", "策略梯度", "Optimizing a parameterized policy by following the gradient of expected return.", "沿期望回报的梯度优化参数化策略。", "Policy gradient methods handle continuous actions naturally."),
        ("advantage", "优势", "How much better an action is than the average at that state.", "某动作比该状态下平均水平好多少。", "Positive advantage increases the action's probability."),
        ("reward model", "奖励模型", "A model trained on human preferences to score responses.", "在人类偏好上训练、用于给回答打分的模型。", "The reward model preferred concise, polite answers."),
        ("preference data", "偏好数据", "Pairs of responses labeled with which one people prefer.", "标注了人们更喜欢哪一个的成对回答。", "Collecting preference data is expensive and noisy."),
        ("direct preference optimization", "直接偏好优化", "Training a policy directly on preference pairs without a separate reward model.", "不训练单独奖励模型、直接在偏好对上训练策略。", "Direct preference optimization is simpler to implement than PPO."),
    ],
    cards=[
        ("贝尔曼最优方程（Q）？", r"$Q^*(s,a)=\mathbb E[r+\gamma\max_{a'}Q^*(s',a')]$"),
        ("REINFORCE 梯度？", r"$\nabla J=\mathbb E[\nabla\log\pi_\theta(a|s)(G-b)]$"),
        ("Bradley–Terry 模型？", r"$P(y_w\succ y_l)=\sigma(r(y_w)-r(y_l))$"),
        ("后训练的三类方法（Lambert）？", "指令微调（SFT）、偏好微调（RLHF/DPO）、可验证奖励的强化学习（RLVR）。"),
    ],
)
