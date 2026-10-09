"""训练配置的唯一来源：默认值、可选项、说明。
train.py 用它校验配置，server.py 通过 /api/schema 把它发给网页生成表单（纯 Python，不依赖 torch）。"""

DATASETS = ["mnist", "fmnist", "cifar10", "cifar100", "kmnist", "svhn"]

# 每个字段：default / type / choices 或 min,max / label / help / group
FIELDS = [
    dict(key="dataset", type="choice", default="cifar10", choices=DATASETS, label="数据集", group="数据",
         help="已预下载的数据集；未下载的需要先在“数据集”面板下载。"),
    dict(key="train_subset", type="int", default=0, min=0, max=60000, label="训练样本数", group="数据",
         help="0 表示用全部。想快速试错可以先设成 5000，几秒就能跑完一轮。"),
    dict(key="augment", type="choice", default="none", choices=["none", "flip", "crop", "crop_flip"], label="数据增强", group="数据",
         help="flip=随机左右翻转；crop=随机平移裁剪（pad 4）。数字类数据集不要用翻转。"),

    dict(key="model", type="choice", default="simple_cnn", choices=["mlp", "lenet", "simple_cnn", "vgg_mini", "resnet_mini", "resnet18"], label="模型", group="模型",
         help="mlp 全连接基线；lenet 经典小 CNN；simple_cnn 可调深度/宽度；vgg_mini 堆叠 3×3 卷积；resnet_mini 带残差连接（CIFAR 版）；resnet18 标准 18 层（从零训练）。"),
    dict(key="width", type="int", default=32, min=4, max=256, label="基础宽度", group="模型",
         help="第一层的通道数，后面的层按倍数增长。越大越强也越慢。"),
    dict(key="depth", type="int", default=3, min=1, max=8, label="深度", group="模型",
         help="simple_cnn 的卷积块数；resnet_mini 每个阶段的残差块数（3 即 ResNet-20）。其他模型忽略。"),
    dict(key="norm", type="choice", default="bn", choices=["bn", "none"], label="批量归一化", group="模型",
         help="bn=使用 BatchNorm；none=不用，用来观察它对收敛速度的影响。"),
    dict(key="dropout", type="float", default=0.0, min=0.0, max=0.9, label="Dropout", group="模型", help="分类头前的丢弃概率。"),

    dict(key="optimizer", type="choice", default="adam", choices=["sgd", "sgd_momentum", "nesterov", "adam", "adamw", "rmsprop"], label="优化器", group="优化",
         help="sgd 最朴素；sgd_momentum 加动量；adam/adamw 自适应学习率，通常收敛更快。"),
    dict(key="lr", type="float", default=0.001, min=0.00001, max=5.0, label="学习率", group="优化", help="最重要的超参数。Adam 常用 1e-3；SGD 常用 0.01–0.1。"),
    dict(key="momentum", type="float", default=0.9, min=0.0, max=0.999, label="动量", group="优化", help="仅 sgd_momentum / nesterov / rmsprop 使用。"),
    dict(key="weight_decay", type="float", default=0.0, min=0.0, max=0.1, label="权重衰减", group="优化", help="L2 正则强度，常用 5e-4。"),
    dict(key="scheduler", type="choice", default="none", choices=["none", "step", "cosine", "onecycle"], label="学习率调度", group="优化",
         help="none 恒定；step 每 1/3 总轮数乘 0.1；cosine 余弦退火；onecycle 先升后降（通常最快收敛）。"),
    dict(key="batch_size", type="int", default=128, min=8, max=2048, label="批大小", group="优化", help="越大每步越稳、吞吐越高，但每轮的更新次数变少。"),
    dict(key="epochs", type="int", default=5, min=1, max=200, label="训练轮数", group="优化", help="完整遍历训练集的次数。"),
    dict(key="label_smoothing", type="float", default=0.0, min=0.0, max=0.5, label="标签平滑", group="优化", help="常用 0.1，抑制过度自信。"),
    dict(key="grad_clip", type="float", default=0.0, min=0.0, max=100.0, label="梯度裁剪", group="优化", help="0 表示不裁剪。"),
    dict(key="seed", type="int", default=0, min=0, max=999999, label="随机种子", group="其他", help="固定后结果可复现；对比实验请保持一致。"),
    dict(key="device", type="choice", default="auto", choices=["auto", "cuda", "mps", "cpu"], label="设备", group="其他",
         help="auto 自动选择：NVIDIA 显卡用 cuda，Apple 芯片用 mps，否则 cpu。选 cpu 可以对比 GPU 加速比。"),
]

DEFAULTS = {f["key"]: f["default"] for f in FIELDS}
BY_KEY = {f["key"]: f for f in FIELDS}


def normalize(cfg):
    """补默认值、转换类型、校验取值。返回新字典；出错抛 ValueError。"""
    out = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if k not in BY_KEY:
            continue
        f = BY_KEY[k]
        try:
            if f["type"] == "int":
                v = int(float(v))
            elif f["type"] == "float":
                v = float(v)
            else:
                v = str(v)
        except (TypeError, ValueError):
            raise ValueError(f"参数 {k} 的值无效：{v!r}")
        if f["type"] == "choice" and v not in f["choices"]:
            raise ValueError(f"参数 {k} 只能是 {f['choices']}，收到 {v!r}")
        if f["type"] in ("int", "float"):
            if v < f["min"] or v > f["max"]:
                raise ValueError(f"参数 {k} 应在 [{f['min']}, {f['max']}] 之间，收到 {v}")
        out[k] = v
    return out
