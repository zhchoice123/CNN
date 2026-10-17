#!/usr/bin/env python3
"""把项目里散落的教材代码整理成可搜索的“代码库”索引。

    python3 tools/build_codelib.py          # 在 study_site/ 下运行；输出 assets/js/codelib-data.js

来源（路径相对项目根目录）：
  d2l       动手学深度学习（PyTorch 版笔记本）
  raschka   Raschka《Machine Learning with PyTorch and Scikit-Learn》
  elgendy   Elgendy《Deep Learning for Vision Systems》（Keras）
  numpy     本站每章的 NumPy 验证代码（study_site/code）
  lab       实验室模板（lab/templates，可直接在本机训练）

做法：把每个 notebook 按“小标题”切成片段（标题 + 说明 + 该小节全部代码），给每个片段自动打中文标签、
识别数据集 / 模型 / 框架，供前端搜索。只读 .ipynb 的源码，不读输出（避免体积膨胀）。
"""
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
ROOT = SITE.parent
OUT = SITE / "assets" / "js" / "codelib-data.js"
MAX_CHARS = 5500       # 单个片段最大长度，超过按单元格边界拆成多段
MIN_CHARS = 60

D2L_CHAPTERS = ["chapter_03", "chapter_04", "chapter_05", "chapter_06", "chapter_07", "chapter_08", "chapter_09", "chapter_10",
                "chapter_11", "chapter_12", "chapter_14", "chapter_20"]

SOURCES = [
    dict(id="d2l", name="动手学深度学习 D2L（PyTorch）", short="D2L", root="book_code/d2l_动手学深度学习/pytorch", fw="pytorch",
         note="大量代码依赖 d2l 工具库；原理讲得最透，适合读"),
    dict(id="raschka", name="Raschka《ML with PyTorch & scikit-learn》", short="Raschka", root="book_code/raschka_machine-learning-book", fw="pytorch",
         note="第 11–19 章为 PyTorch；之前为 scikit-learn"),
    dict(id="elgendy", name="Elgendy《Deep Learning for Vision Systems》", short="Elgendy", root="book_code/elgendy_deep-learning-for-vision-systems", fw="keras",
         note="Keras / TensorFlow 写法，覆盖分类、迁移学习、检测、GAN、DeepDream"),
    dict(id="numpy", name="本站 NumPy 从零实现", short="NumPy", root="study_site/code", fw="numpy",
         note="不依赖框架，直接看清算法本身；每个都对应学习站的一节"),
    dict(id="practice", name="代码练习的完整答案（分步讲解）", short="练习", root="study_site/exercises", fw="pytorch",
         note="每段都能在“代码练习”里分步练：选择 → 填空 → 写整块 → 默写"),
    dict(id="lab", name="实验室模板（可本机直接训练）", short="实验室", root="lab/templates", fw="pytorch",
         note="用 labkit 读本地数据集，一键运行并参与对比"),
]

# ── 标签规则：(中文标签, 正则)。对 “标题 + 说明 + 代码” 的小写文本匹配 ──
TAGS = [
    ("卷积", r"conv(2d|olution|3d|1d)|\bconv\b|nn\.conv|convolutional"),
    ("池化", r"pool"),
    ("批量归一化", r"batch.?norm|batchnormalization|layernorm|group.?norm"),
    ("Dropout", r"dropout"),
    ("激活函数", r"\brelu\b|sigmoid|\btanh\b|softmax|\bgelu\b|activation|leaky"),
    ("全连接层", r"nn\.linear|\bdense\(|fully.connected|lazylinear"),
    ("残差连接", r"residual|resnet|skip.connection|shortcut"),
    ("数据增强", r"augment|randomcrop|randomhorizontalflip|imagedatagenerator|transforms\.random|randomflip|colorjitter"),
    ("迁移学习", r"pretrained|fine.?tun|transfer.learn|include_top|freeze|requires_grad.?=.?false"),
    ("优化器", r"optim\.|optimizer|\bsgd\b|\badam\b|rmsprop|momentum|adagrad"),
    ("学习率调度", r"lr_scheduler|scheduler|learning.rate.(decay|schedule)|reducelronplateau|onecycle|cosineannealing"),
    ("损失函数", r"cross.?entropy|\bloss\b|nn\.\w*loss|mseloss"),
    ("训练循环", r"for epoch|\.fit\(|loss\.backward|def train|train_epoch|num_epochs"),
    ("评估指标", r"accuracy|confusion|precision|recall|f1.score|\broc\b|evaluate"),
    ("数据加载", r"dataloader|torch\.utils\.data|datasets\.|flow_from_directory|tf\.data|\bdataset\b"),
    ("可视化", r"plt\.|imshow|matplotlib|seaborn|\.plot\("),
    ("目标检测", r"bounding.?box|\banchor|\biou\b|\bnms\b|yolo|\bssd\b|r-?cnn|object.detection"),
    ("图像分割", r"segmentation|u-?net|\bfcn\b|transposed|deconv"),
    ("生成模型", r"\bgan\b|generator|discriminator|\bvae\b|autoencoder|diffusion|dcgan"),
    ("序列模型", r"\blstm\b|\bgru\b|\brnn\b|recurrent"),
    ("注意力/Transformer", r"attention|transformer"),
    ("权重初始化", r"xavier|kaiming|he_normal|init\.|initializ"),
    ("梯度下降", r"gradient.descent|minibatch|mini-batch"),
    ("模型保存", r"torch\.save|state_dict|model\.save|load_model|checkpoint"),
    ("GPU 加速", r"cuda|\bmps\b|\.to\(device|get_device"),
    ("NumPy 实现", r"import numpy"),
]
MODELS = [("LeNet", r"lenet"), ("AlexNet", r"alexnet"), ("VGG", r"\bvgg"), ("NiN", r"\bnin\b|network in network"), ("GoogLeNet/Inception", r"googlenet|inception"),
          ("ResNet", r"resnet|residual"), ("DenseNet", r"densenet"), ("MobileNet", r"mobilenet"), ("EfficientNet", r"efficientnet"), ("U-Net", r"u-?net"),
          ("FCN", r"\bfcn\b"), ("YOLO", r"yolo"), ("SSD", r"\bssd\b"), ("R-CNN", r"r-?cnn"), ("GAN", r"\bgan\b|dcgan"), ("VAE", r"\bvae\b|variational"),
          ("LSTM/GRU", r"\blstm\b|\bgru\b"), ("Transformer", r"transformer"), ("MLP", r"\bmlp\b|multilayer.perceptron"), ("Softmax 回归", r"softmax.regression"),
          ("线性回归", r"linear.regression")]
DATASETS = [("MNIST", r"(?<!fashion.)(?<!fashion_)mnist"), ("Fashion-MNIST", r"fashion.?mnist"), ("CIFAR-10", r"cifar.?10\b"), ("CIFAR-100", r"cifar.?100"),
            ("ImageNet", r"imagenet"), ("VOC", r"\bvoc\b|pascal"), ("COCO", r"\bcoco\b"), ("SVHN", r"svhn"), ("猫狗", r"cats.?(vs|and).?dogs|dogs.?vs.?cats|kaggle.?dog"),
            ("CelebA", r"celeba"), ("IMDB", r"imdb")]
LOCAL_DS = {"MNIST": "mnist", "Fashion-MNIST": "fmnist", "CIFAR-10": "cifar10", "CIFAR-100": "cifar100", "SVHN": "svhn"}


def detect(rules, text):
    return [name for name, rx in rules if re.search(rx, text, re.I)]


def framework(code, default):
    if re.search(r"^\s*(import|from)\s+torch|torch\.", code, re.M):
        return "pytorch"
    if re.search(r"tensorflow|keras", code, re.I):
        return "keras"
    if re.search(r"mxnet|mx\.", code):
        return "mxnet"
    if re.search(r"^\s*(import|from)\s+sklearn|sklearn\.", code, re.M):
        return "sklearn"
    if re.search(r"^\s*import numpy|^\s*from numpy", code, re.M):
        return "numpy"
    return default


# ── notebook 解析 ──
RX_HEAD = re.compile(r"^(#{1,4})\s+(.*)$", re.M)


def clean_title(t):
    t = re.sub(r":label:`[^`]*`|:numref:`[^`]*`|:eqref:`[^`]*`|:eqlabel:`[^`]*`|:begin_tab:.*|:end_tab:", "", t)
    t = t.replace("[**", "").replace("**]", "").replace("**", "").replace("`", "")
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    return re.sub(r"\s+", " ", t).strip(" #:-")


def clean_md(t):
    t = re.sub(r":(label|numref|eqref|eqlabel|cite|ref|width|height):`[^`]*`", "", t)
    t = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", "", t, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t).replace("[**", "").replace("**]", "").replace("**", "").replace("`", "")
    t = re.sub(r"^\s*#+.*$", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


def clean_code(src):
    keep = []
    for ln in src.splitlines():
        if re.match(r"\s*(%|!pip|!conda|!wget|!curl|!unzip|!ls|!mkdir)", ln):
            continue
        keep.append(ln.rstrip())
    code = "\n".join(keep).strip("\n")
    return code


def meaningful(code):
    return any(l.strip() and not l.strip().startswith("#") for l in code.splitlines())


def notebook_snippets(path, src_id, rel):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    nb_title = None
    sections = []                       # [{title, intro, codes:[...]}]
    cur = None
    for c in cells:
        s = c.get("source", "")
        s = "".join(s) if isinstance(s, list) else s
        if c.get("cell_type") == "markdown":
            heads = list(RX_HEAD.finditer(s))
            if heads:
                title = clean_title(heads[0].group(2))
                nb_title = nb_title or title
                cur = dict(title=title, intro="", codes=[])
                sections.append(cur)
                s = s[heads[0].end():]
            body = clean_md(s)
            if cur is not None and not cur["codes"] and body and len(cur["intro"]) < 300:
                cur["intro"] = (cur["intro"] + " " + body).strip()[:320]
        elif c.get("cell_type") == "code":
            code = clean_code(s)
            if not meaningful(code):
                continue
            if cur is None:
                cur = dict(title=nb_title or path.stem, intro="", codes=[])
                sections.append(cur)
            cur["codes"].append(code)
    out = []
    for sec in sections:
        codes = sec["codes"]
        if not codes:
            continue
        # 按长度打包成多段
        parts, buf = [], ""
        for code in codes:
            if buf and len(buf) + len(code) > MAX_CHARS:
                parts.append(buf); buf = ""
            buf = (buf + "\n\n" + code) if buf else code
        if buf:
            parts.append(buf)
        for i, code in enumerate(parts):
            if len(code) < MIN_CHARS:
                continue
            title = sec["title"] + (f"（{i + 1}/{len(parts)}）" if len(parts) > 1 else "")
            out.append(dict(title=title, nb=nb_title or path.stem, intro=sec["intro"], code=code, file=rel))
    return out


def numpy_snippets():
    out = []
    for p in sorted((ROOT / "study_site" / "code").glob("ch*.py")):
        code = p.read_text(encoding="utf-8")
        try:
            doc = (ast.get_docstring(ast.parse(code)) or "").strip().splitlines()
        except SyntaxError:
            doc = []
        m = re.match(r"ch(\d+)_", p.name)
        title = (doc[0] if doc else p.stem).strip().rstrip("。.")
        out.append(dict(title=title, nb=p.name, intro=" ".join(doc[1:3])[:300], code=code, file="study_site/code/" + p.name,
                        chapter=("ch" + m.group(1)) if m else None))
    return out


def lab_snippets():
    out = []
    for p in sorted((ROOT / "lab" / "templates").glob("*.py")):
        code = p.read_text(encoding="utf-8")
        m = re.match(r'"""(.*?)"""', code, re.S)
        doc = m.group(1).strip().splitlines() if m else []
        out.append(dict(title=doc[0].strip() if doc else p.stem, nb=p.name, intro=" ".join(doc[1:3])[:300], code=code,
                        file="lab/templates/" + p.name, template=p.stem))
    return out


def practice_snippets():
    sys.path.insert(0, str(ROOT / "lab"))
    import exercise_core as xc
    out = []
    for ex in xc.load_all():
        m = ex["meta"]
        out.append(dict(title=m.get("title", ex["id"]), nb=ex["file"], intro=m.get("goal", "")[:300], code=xc.answer_script(ex),
                        file="study_site/exercises/" + ex["file"], exercise=ex["id"]))
    return out


def build():
    snippets, seen = [], set()
    stats = {}

    def add(src, raw, default_fw):
        code = raw["code"].strip("\n")
        h = hashlib.md5(re.sub(r"\s+", "", code).encode()).hexdigest()
        if h in seen:
            return
        seen.add(h)
        text = (raw["title"] + " " + raw["nb"] + " " + raw["intro"] + " " + code).lower()
        head = (raw["title"] + " " + raw["nb"] + " " + raw["file"]).lower()
        fw = framework(code, default_fw)
        ds = detect(DATASETS, text)
        sn = dict(id=f"{src['id']}-{len(snippets)}", src=src["id"], title=raw["title"], nb=raw["nb"], intro=raw["intro"], code=code, file=raw["file"],
                  fw=fw, tags=detect(TAGS, text), models=detect(MODELS, head + " " + raw["intro"].lower() + " " + " ".join(re.findall(r"class\s+(\w+)", code)).lower()),
                  ds=ds, d2l=bool(re.search(r"\bd2l\b", code)), n=len(code.splitlines()))
        loc = [LOCAL_DS[d] for d in ds if d in LOCAL_DS]
        if loc and fw == "pytorch":
            sn["local_ds"] = loc
        for k in ("chapter", "template", "exercise"):
            if raw.get(k):
                sn[k] = raw[k]
        snippets.append(sn)
        stats[src["id"]] = stats.get(src["id"], 0) + 1

    for src in SOURCES:
        root = ROOT / src["root"]
        if src["id"] == "numpy":
            raws = numpy_snippets()
        elif src["id"] == "practice":
            raws = practice_snippets()
        elif src["id"] == "lab":
            raws = lab_snippets()
        else:
            raws = []
            if src["id"] == "d2l":
                files = [p for d in D2L_CHAPTERS for cd in sorted(root.glob(d + "*")) for p in sorted(cd.glob("*.ipynb")) if p.stem != "index"]
            elif src["id"] == "raschka":
                files = [p for cd in sorted(root.glob("ch*")) for p in sorted(cd.glob("*.ipynb"))]
            else:
                files = [p for cd in sorted(root.glob("chapter_*")) for p in sorted(cd.glob("*.ipynb"))]
            for p in files:
                try:
                    raws += notebook_snippets(p, src["id"], str(p.relative_to(ROOT)))
                except (ValueError, KeyError) as e:
                    print("跳过", p, e, file=sys.stderr)
        for r in raws:
            add(src, r, src["fw"])

    # 实验室模板排最前，其余保持来源顺序
    snippets.sort(key=lambda s: 0 if s["src"] in ("lab", "practice") else 1)
    for i, s in enumerate(snippets):
        s["id"] = f"s{i}"
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), sources=[{k: v for k, v in s.items() if k != "root"} | {"count": stats.get(s["id"], 0)} for s in SOURCES],
                local_datasets={v: k for k, v in LOCAL_DS.items()}, snippets=snippets)
    OUT.write_text("window.CODELIB=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"代码库：{len(snippets)} 个片段，{kb:.0f} KB → {OUT.relative_to(ROOT)}")
    for k, v in stats.items():
        print(f"  {k:8s} {v}")
    tagc = {}
    for s in snippets:
        for t in s["tags"]:
            tagc[t] = tagc.get(t, 0) + 1
    print("  标签：", ", ".join(f"{k}{v}" for k, v in sorted(tagc.items(), key=lambda x: -x[1])[:14]))


if __name__ == "__main__":
    build()
