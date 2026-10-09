CHAPTER = dict(
    no=16,
    title="工程基础：Python、PyTorch 原理、操作系统与部署",
    title_en="Engineering Foundations: Python, PyTorch Internals, Operating Systems and Deployment",
    group="机器人与工程",
    summary="书单里几乎所有代码都用 Python 写成、运行在操作系统之上，最终要部署到真实环境。本章讲 Python 对象模型与常见陷阱，用类和运算符重载读懂 PyTorch 的自动微分，从操作系统角度理解进程、线程与内存，最后讨论部署、监控与可靠的模型评估。",
    sources=["Lutz《Python 学习手册》第 4、6、14–20、25–31、38 章", "Pointer《PyTorch 深度学习编程》第 1–2、7–8 章", "Tanenbaum & Bos《现代操作系统》第 2–3、5、7–8 章", "Ayyadevara & Reddy 第 18 章", "Zhou 等《医学图像分析中的深度学习》第 17 章", "Patterson & Gibson 第 9 章"],
    learn=[
        dict(
            title="Python 对象模型：动态类型、引用与作用域",
            title_en="The Python object model",
            body=r"""
Lutz 说：在 Python 中，**一切都是对象**。理解 Python 最重要的一章是“动态类型插曲”（第 6 章）：

- **类型属于对象，而不属于变量**。变量只是一个名字，赋值是让名字**指向**一个对象，而不是把值复制进去。
- **共享引用**：`b = a` 之后，两个名字指向同一个列表，通过 `b` 修改，`a` 也会“变”。需要独立副本时用 `copy()` 或 `copy.deepcopy()`。
- **可变与不可变**：列表、字典、集合可以原地修改；数字、字符串、元组不可变，`+=` 会创建新对象。
- **垃圾回收**：Python 通过引用计数自动回收不再被引用的对象。

两个经典陷阱：

1. **可变默认参数**：默认值只在函数定义时求值一次，`def f(x, bucket=[])` 的所有调用共享同一个列表。正确写法是默认 `None`，在函数内部再创建。
2. **作用域 LEGB 规则**：查找名字时依次搜索 Local（局部）→ Enclosing（外层函数）→ Global（全局）→ Built-in（内置）。在内层函数里给外层变量赋值，需要 `nonlocal`（全局变量用 `global`），否则会被当作新的局部变量。

在深度学习中，这些规则直接影响代码的正确性：比如 PyTorch 的很多操作是**原地**修改张量的（名字带下划线，如 `add_()`），共享引用的张量会一起改变；而把列表作为默认参数传给数据处理函数，可能导致数据在不同调用之间“串”在一起。
""",
            points=[
                "`is` 比较身份（是不是同一个对象），`==` 比较值。",
                "函数参数按“对象引用”传递：在函数内原地修改可变对象，调用者看得到。",
                "闭包能“记住”外层函数的变量，是装饰器的基础。",
                "Lutz 第 1 章：Python 的两大优势是软件质量（可读性）和开发效率，代码通常只有 C++/Java 的三分之一到五分之一长。",
            ],
            code="ch16_python_objects.py",
            code_note="演示共享引用、可变默认参数、闭包与 `nonlocal`，以及不可变对象的 `+=`。",
            src="Lutz 第 6、17–18 章",
        ),
        dict(
            title="迭代器、生成器、类与运算符重载：读懂 PyTorch",
            title_en="Iterators, generators, classes and operator overloading",
            body=r"""
**迭代协议**：任何可迭代对象都可以用 `for` 遍历，背后是 `iter()` 和 `next()`。**生成器**（含 `yield` 的函数或生成器表达式）按需逐个产生值，而不必一次把所有结果放进内存——处理大数据集时至关重要，也只能遍历一次（Lutz 第 14、20 章）。

**类与运算符重载**（Lutz 第 25–31 章）：双下划线方法让你的类像内置类型一样工作——`__init__` 初始化，`__len__` 与 `__getitem__` 支持 `len()` 和索引，`__call__` 让实例可以像函数一样调用，`__add__`、`__mul__` 重载 `+` 和 `*`，`__repr__` 控制打印。**装饰器**（第 38 章）可以在不修改函数的情况下给它增加功能，例如计时、缓存、日志。

有了这些知识，PyTorch 的设计就一目了然（Pointer 第 1–2 章）：

- 每个网络都是继承 `nn.Module` 的**类**，在 `__init__` 中定义层，在 `forward` 中定义计算；调用 `model(x)` 时触发的是 `__call__`。
- `Dataset` 是实现了 `__len__` 和 `__getitem__` 的类；`DataLoader` 是一个按小批量产生数据的**迭代器**。
- **自动微分**：张量运算通过运算符重载记录下计算图，`loss.backward()` 按拓扑逆序应用链式法则。代码中用 40 行 Python 实现的标量自动微分引擎，就是这一机制的缩影（与第 6 章的反向传播对照着看）。
""",
            points=[
                "生成器节省内存，适合流式读取大文件和批量产生数据。",
                "`forward` 定义计算，`backward` 由自动微分完成，用户几乎不需要手写梯度。",
                "每次 `backward` 前要把梯度清零（`optimizer.zero_grad()`），因为梯度是累加的——代码中的 `grad +=` 正是原因。",
                "Pointer 第 7 章：用 TensorBoard、钩子（hook）和类激活映射调试模型。",
            ],
            code="ch16_micrograd.py",
            code_note="用类与运算符重载实现一个最小的自动微分引擎，对一个 tanh 神经元求梯度并与数值梯度对比。",
            src="Lutz 第 14、20、25–31、38 章；Pointer 第 1–2、7 章",
        ),
        dict(
            title="操作系统视角：进程、线程、内存与 I/O",
            title_en="An operating-systems view of training",
            body=r"""
Tanenbaum 给操作系统下了两个定义：它是一台**扩展的机器**，把难用的硬件包装成文件、进程、地址空间等优雅的抽象；它也是一个**资源管理者**，在多个程序之间分配处理器、内存和磁盘。训练模型时遇到的许多问题，本质上都是操作系统问题：

- **进程与线程**（第 2 章）：进程有独立的地址空间，线程共享地址空间。多线程会带来**竞争条件**，需要互斥锁、信号量等同步机制；不当的加锁顺序会导致**死锁**（第 6 章）。CPython 的 **GIL** 让同一时刻只有一个线程执行 Python 字节码：I/O 密集任务用多线程即可，CPU 密集的纯 Python 计算要用**多进程**——这正是 PyTorch `DataLoader` 用 `num_workers` 开多个进程做数据预处理的原因。
- **虚拟内存**（第 3 章）：每个进程都以为自己独占一大块连续内存，操作系统通过**分页**映射到物理内存，不常用的页可以换到磁盘。内存不足时频繁换页（抖动）会让训练慢到几乎停止；`np.memmap` 等**内存映射文件**可以只把需要的部分读进内存。
- **I/O 与 DMA**（第 5 章）：数据从磁盘到内存、再到显卡，常常是训练的瓶颈；固定页内存（pinned memory）让 DMA 传输更快。
- **虚拟化与云**（第 7 章）：虚拟机与容器（Docker）让训练和部署环境可复现。
- **多处理器**（第 8 章）：多核、缓存一致性，与多卡并行训练密切相关。
""",
            points=[
                "GPU 利用率低时，先检查数据加载是否成为瓶颈（CPU 预处理、磁盘 I/O）。",
                "死锁的四个必要条件：互斥、占有并等待、不可抢占、循环等待。",
                "多进程之间不共享内存，传递大数组要考虑序列化开销或共享内存。",
                "Tanenbaum 引用圣埃克苏佩里：完美不是无可增加，而是无可删减——设计越简单越好。",
            ],
            code="ch16_concurrency.py",
            code_note="比较 I/O 密集与 CPU 密集任务在串行、多线程、多进程下的耗时（受 GIL 影响）。注意：浏览器中的 Python 不支持多进程，这段代码请在本机运行。",
            src="Tanenbaum 第 1–3、5–8 章；Pointer 第 2 章 DataLoader",
        ),
        dict(
            title="从实验到生产：部署、监控与可靠的评估",
            title_en="From experiment to production",
            body=r"""
一个模型在笔记本里跑出好成绩，只是开始。**部署**（Pointer 第 8 章，《PyTorch 现代计算机视觉》第 18 章）通常包括：

- 把模型包装成 **Web 服务**（Flask、FastAPI），提供预测接口；
- 用 **Docker** 打包代码与依赖，在 Kubernetes 或云上扩展；
- 用 **TorchScript**、ONNX 导出模型，脱离 Python 运行，或用**量化**、剪枝压缩模型以部署到移动端；
- 大数据平台上的分布式训练与推理（Patterson & Gibson 第 9 章的 Spark）。

**上线之后**更要持续监控。现实数据的分布会随时间变化——换了相机、换了医院、用户行为改变——这叫**数据漂移**，模型性能会悄悄下降。可以用 PSI、KS 检验等统计量比较线上数据与训练数据的分布，发现漂移就要排查原因或重新训练。还可以用**向量数据库**存储图像的嵌入，支持相似检索与异常发现。

**可靠的评估**是这一切的前提。《医学图像分析中的深度学习》第 17 章（作者来自美国 FDA 的研究团队）强调：研究阶段的目标是找到最好的方法，而要部署到临床的算法需要严格得多的评估——训练集与测试集必须相互独立（例如同一病人的图像不能同时出现在两边）；参考标准要可靠；数据采集要考虑公平性；还要检验**可迁移性**：在一家医院验证过的模型，在另一家医院是否仍然有效。如果没有达到预先设定的成功标准，就应当停止部署。
""",
            points=[
                "推理时记得 `model.eval()` 和 `torch.no_grad()`。",
                "按“病人/用户”而不是按“图像/记录”划分数据集，防止数据泄露。",
                "PSI < 0.1 通常视为稳定，> 0.25 视为显著漂移（经验阈值）。",
                "预先确定评估指标和成功标准，而不是看到结果后再挑指标。",
            ],
            code="ch16_drift.py",
            code_note="模拟“换了新相机”导致的输入分布变化，用群体稳定性指数（PSI）检测数据漂移。",
            src="Pointer 第 8 章；Ayyadevara & Reddy 第 18 章；Zhou 等第 17 章；Patterson & Gibson 第 9 章",
        ),
    ],
    examples=[
        dict(
            title="预测输出：作用域与生成器",
            problem=r"""阅读以下代码，预测输出：

<pre><code>x = 10
def f():
    x = 20
    def g():
        return x
    return g()
print(f(), x)
nums = [1, 2, 3]
squares = (n * n for n in nums)
nums.append(4)
print(list(squares))
print(list(squares))</code></pre>""",
            steps=[
                r"`g` 中的 `x` 按 LEGB 规则先找局部（没有），再找外层函数 `f`，找到 `x = 20`，所以 `f()` 返回 20；全局的 `x` 仍是 10。",
                r"生成器表达式是**惰性**的：创建时没有计算，直到 `list()` 迭代时才遍历 `nums`，此时列表已经变成 `[1, 2, 3, 4]`，所以得到 `[1, 4, 9, 16]`。",
                r"生成器只能遍历一次，第二次 `list(squares)` 得到空列表 `[]`。",
                r"在数据管道中，这意味着要小心：把生成器当成可重复遍历的列表来用，第二个 epoch 会“没有数据”。",
            ],
            answer="`20 10`，然后 `[1, 4, 9, 16]`，最后 `[]`。",
            code="ch16_ex_scope.py",
        ),
        dict(
            title="可变默认参数",
            problem=r"`def add_item(x, bucket=[]): bucket.append(x); return bucket`。依次调用 `add_item(1)` 和 `add_item(2)`，两次的返回值分别是什么？为什么代码中打印出来两次都是 `[1, 2]`？",
            steps=[
                r"默认参数 `[]` 只在 `def` 语句执行时创建一次，之后每次调用都复用同一个列表对象。",
                r"第一次调用返回这个列表，此时内容是 `[1]`；第二次调用往同一个列表追加 2，内容变为 `[1, 2]`。",
                r"`print(add_item(1), add_item(2))` 先完成两次调用再打印，两个返回值是**同一个对象**，所以都显示 `[1, 2]`。",
                r"修正：默认值设为 `None`，在函数体内 `bucket = [] if bucket is None else bucket`。",
            ],
            answer="两次返回同一个列表对象，最终内容为 [1, 2]；应改用 None 作默认值。",
            code="ch16_python_objects.py",
        ),
        dict(
            title="为 DataLoader 选择并发方式",
            problem=r"训练时 GPU 利用率只有 30%。数据预处理是纯 Python 实现的图像解码与增强，每张图约 20 毫秒；另一个任务是从对象存储下载图片，每张需等待 200 毫秒网络延迟。分别应该用多线程还是多进程来加速？",
            steps=[
                r"GPU 利用率低，说明 GPU 在等数据：瓶颈在数据加载。",
                r"图像解码与增强是 **CPU 密集**的纯 Python 计算，受 GIL 限制，多线程几乎没有加速；应该用**多进程**（如 `DataLoader(num_workers=8)`）。",
                r"下载图片是 **I/O 密集**任务，大部分时间在等待网络，等待时会释放 GIL，多线程（或异步 I/O）就能大幅加速。",
                r"代码实验也显示：8 个 I/O 任务用 8 线程从 1.65 秒降到 0.21 秒；CPU 任务用 4 线程几乎不变，用 4 进程明显加速。",
            ],
            answer="CPU 密集的预处理用多进程；I/O 密集的下载用多线程或异步 I/O。",
            code="ch16_concurrency.py",
        ),
    ],
    quiz=dict(
        mc=[
            dict(q="执行 `b = a` 后修改 `b` 中的元素，`a` 也会改变，前提是：", options=["a 是整数", "a 是可变对象（如列表）", "a 是字符串", "a 是元组"], answer=1, explain="共享引用 + 原地修改。"),
            dict(q="Python 查找变量名的顺序（LEGB）是：", options=["全局→局部→内置→外层", "局部→外层函数→全局→内置", "内置→全局→局部", "随机"], answer=1, explain="Local → Enclosing → Global → Built-in。"),
            dict(q="PyTorch 中让 `model(x)` 能被调用的特殊方法是：", options=["__init__", "__call__", "__len__", "__repr__"], answer=1, explain="`nn.Module.__call__` 内部调用 `forward`。"),
            dict(q="自定义 PyTorch Dataset 至少需要实现：", options=["__add__ 和 __mul__", "__len__ 和 __getitem__", "__iter__ 和 __next__", "forward 和 backward"], answer=1, explain="长度与按索引取样本。"),
            dict(q="CPU 密集的纯 Python 数据预处理，最有效的并行方式是：", options=["多线程", "多进程", "增大批大小", "使用生成器"], answer=1, explain="绕开 GIL。"),
            dict(q="模型上线后输入数据的分布逐渐偏离训练数据，这叫：", options=["过拟合", "数据漂移", "梯度消失", "模式崩溃"], answer=1, explain="data drift。"),
        ],
        tf=[
            dict(q="在 Python 中，变量本身有类型，对象没有类型。", answer=False, explain="类型属于对象，变量只是名字。"),
            dict(q="生成器可以被反复遍历多次。", answer=False, explain="生成器只能遍历一次。"),
            dict(q="PyTorch 中的梯度默认是累加的，所以每次反向传播前要清零。", answer=True, explain="`optimizer.zero_grad()`。"),
            dict(q="同一个进程中的多个线程共享地址空间。", answer=True, explain="进程才有独立地址空间。"),
            dict(q="医学影像数据集划分时，同一病人的图像可以同时出现在训练集和测试集中。", answer=False, explain="会造成数据泄露，评估结果过于乐观。"),
        ],
        fill=[
            dict(q="在内层函数中修改外层函数的变量，需要使用 ____ 关键字。", answer=["nonlocal"], explain="全局变量用 `global`。"),
            dict(q="包含 `yield` 的函数叫做 ____ 函数。", answer=["生成器", "generator"], explain="generator。"),
            dict(q="CPython 中限制同一时刻只有一个线程执行字节码的锁叫 ____（英文缩写）。", answer=["GIL"], explain="Global Interpreter Lock。"),
            dict(q="把代码和依赖打包成可移植镜像的常用工具是 ____。", answer=["Docker", "docker"], explain="容器化。"),
        ],
        reading=[
            dict(
                title="The Last Mile",
                passage="""Many machine learning projects die in the gap between a promising notebook and a working product. A model that scores well on a carefully prepared test set still has to be packaged, served, monitored and maintained. Each of these steps brings its own surprises.

Serving means wrapping the model behind an interface that other software can call, often a small web service running inside a container. The container fixes the exact versions of every library, so the model behaves the same on a laptop and in the cloud. For mobile phones or embedded devices, the model may be exported to a format that does not need Python at all and compressed so that it fits in limited memory.

Once the system is live, the real world starts to drift away from the training data. A hospital buys a new scanner, a shop changes its lighting, customers adopt new habits. Performance can decline quietly, without any error message. Good teams therefore measure how the incoming data compare with the training data, define in advance what level of performance is acceptable, and are prepared to retrain or even withdraw a model when it no longer meets that standard.""",
                zh="""许多机器学习项目都死在“有希望的笔记本”与“能用的产品”之间的鸿沟里。一个在精心准备的测试集上得分很高的模型，仍然需要被打包、提供服务、被监控和维护。每一步都会带来意想不到的问题。

提供服务意味着把模型包装在其他软件可以调用的接口后面，通常是运行在容器中的一个小型 Web 服务。容器固定了每个库的确切版本，使模型在笔记本电脑上和云端表现一致。对于手机或嵌入式设备，模型可能会被导出成完全不需要 Python 的格式，并经过压缩以装进有限的内存。

系统一旦上线，真实世界就开始偏离训练数据。医院买了新的扫描仪，商店换了照明，顾客养成了新习惯。性能可能悄无声息地下降，不会出现任何错误信息。因此，优秀的团队会衡量新流入的数据与训练数据有何不同，预先定义什么样的性能是可以接受的，并准备好在模型不再达标时重新训练，甚至将其撤下。""",
                questions=[
                    dict(q="Why are containers useful according to the passage?", options=["They make models more accurate", "They fix library versions so behavior is consistent across machines", "They replace the training data", "They remove the need for monitoring"], answer=1, explain="固定依赖版本，保证一致。"),
                    dict(q="What makes performance decline after deployment dangerous?", options=["It always causes a crash", "It can happen quietly without error messages", "It only affects mobile phones", "It improves the model"], answer=1, explain="悄无声息地下降。"),
                    dict(q="Which practice does the passage recommend?", options=["Choosing metrics after seeing results", "Defining acceptable performance in advance", "Never retraining a model", "Ignoring incoming data"], answer=1, explain="预先定义可接受的性能标准。"),
                ],
            ),
        ],
    ),
    vocab=[
        ("dynamic typing", "动态类型", "Types belong to objects at runtime rather than to variable names.", "类型在运行时属于对象而非变量名。", "Dynamic typing lets one name refer to a list and later to a string."),
        ("reference", "引用", "A name that points to an object rather than containing it.", "指向对象而非包含对象的名字。", "Two references to the same list see the same changes."),
        ("mutable", "可变的", "Able to be changed in place after creation.", "创建后可以原地修改的。", "Lists are mutable but tuples are not."),
        ("scope", "作用域", "The region of code in which a name is visible.", "名字可见的代码区域。", "Python resolves names using the LEGB scope rule."),
        ("closure", "闭包", "A function that remembers variables from the scope where it was created.", "记住其创建时所在作用域变量的函数。", "A closure keeps the counter alive between calls."),
        ("generator", "生成器", "A function or expression that produces values lazily, one at a time.", "惰性地逐个产生值的函数或表达式。", "A generator streams batches without loading the whole dataset."),
        ("operator overloading", "运算符重载", "Defining special methods so that a class works with operators such as plus.", "定义特殊方法使类能使用加号等运算符。", "Operator overloading lets tensors be added with a plus sign."),
        ("decorator", "装饰器", "A function that wraps another function to add behavior.", "包装另一个函数以增加功能的函数。", "A timing decorator measures how long each call takes."),
        ("process", "进程", "A running program with its own address space.", "拥有独立地址空间的运行中的程序。", "Each data loader worker runs in a separate process."),
        ("thread", "线程", "A flow of execution that shares memory with other threads in the same process.", "与同一进程中其他线程共享内存的执行流。", "Threads are ideal for waiting on network requests."),
        ("virtual memory", "虚拟内存", "An abstraction giving each process its own large address space backed by paging.", "通过分页为每个进程提供独立大地址空间的抽象。", "Virtual memory lets a program use more memory than physically installed."),
        ("deadlock", "死锁", "A state in which processes wait for each other forever.", "进程互相永远等待的状态。", "Acquiring locks in a fixed order prevents deadlock."),
        ("container", "容器", "A lightweight package bundling an application with its dependencies.", "把应用及其依赖打包在一起的轻量级运行单元。", "The model is served from a container in the cloud."),
        ("data drift", "数据漂移", "A change in the input data distribution after deployment.", "部署后输入数据分布发生的变化。", "Data drift appeared after the clinic switched scanners."),
    ],
    cards=[
        ("LEGB 规则？", "Local → Enclosing → Global → Built-in；修改外层变量用 `nonlocal`，修改全局用 `global`。"),
        ("可变默认参数陷阱？", "默认值只在定义时求值一次，所有调用共享；应使用 `None` 再在函数内创建。"),
        ("多线程 vs 多进程？", "I/O 密集用线程（等待时释放 GIL）；CPU 密集的纯 Python 用进程。"),
        ("上线后要监控什么？", "数据漂移、性能指标、预先设定的成功标准；不达标就排查、重训或撤下。"),
    ],
)
