/* 代码库搜索：用中文/英文提问，从教材代码里找到可用的训练代码，并一键放进训练实验室 */
(function () {
  "use strict";
  const UI = window.StudyUI;
  const esc = UI.esc;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const LAB_API = location.port === "8766" ? "" : "http://127.0.0.1:8766";
  const D = window.CODELIB;

  /* ---------- 中文 → 检索意图 ---------- */
  // tag: 中文标签（命中加分）；en: 英文关键词（在标题/说明/代码里匹配）；model / ds / fw: 结构化条件
  const RULES = [
    [/卷积|conv/i, { tag: ["卷积"], en: ["conv"] }],
    [/池化|pool/i, { tag: ["池化"], en: ["pool"] }],
    [/批量?归一化|batch ?norm|\bbn\b/i, { tag: ["批量归一化"], en: ["batchnorm", "batch_norm", "batch norm"] }],
    [/dropout|丢弃/i, { tag: ["Dropout"], en: ["dropout"] }],
    [/残差|resnet|shortcut/i, { tag: ["残差连接"], model: ["ResNet"], en: ["resnet", "residual"] }],
    [/数据增强|增广|augment|翻转|裁剪/i, { tag: ["数据增强"], en: ["augment", "randomcrop", "flip"] }],
    [/迁移|微调|预训练|fine.?tun|pretrain|transfer/i, { tag: ["迁移学习"], en: ["fine-tun", "pretrained", "transfer"] }],
    [/学习率|lr\b|scheduler|衰减|退火|warm.?up|onecycle/i, { tag: ["学习率调度", "优化器"], en: ["learning rate", "scheduler", "lr_"] }],
    [/优化器|adam|sgd|动量|momentum|rmsprop/i, { tag: ["优化器"], en: ["optim", "adam", "sgd", "momentum"] }],
    [/过拟合|正则|overfit|weight.?decay|权重衰减/i, { tag: ["Dropout"], en: ["overfit", "regulariz", "weight_decay", "dropout"] }],
    [/损失|loss|交叉熵/i, { tag: ["损失函数"], en: ["loss", "cross_entropy", "crossentropy"] }],
    [/准确率|评估|混淆矩阵|精确率|召回|roc|指标|accuracy|metric/i, { tag: ["评估指标"], en: ["accuracy", "confusion", "precision", "recall", "roc"] }],
    [/加载数据|读取数据|数据加载|dataloader|数据集类/i, { tag: ["数据加载"], en: ["dataloader", "dataset"] }],
    [/可视化|画图|画出|plot|绘制/i, { tag: ["可视化"], en: ["plt.", "imshow", "plot"] }],
    [/初始化|init/i, { tag: ["权重初始化"], en: ["init", "xavier", "kaiming"] }],
    [/梯度下降|minibatch|小批量/i, { tag: ["梯度下降"], en: ["gradient descent", "minibatch"] }],
    [/保存模型|加载模型|checkpoint|state_dict|save/i, { tag: ["模型保存"], en: ["torch.save", "state_dict", "checkpoint", "load_model"] }],
    [/gpu|显卡|cuda|mps|加速/i, { tag: ["GPU 加速"], en: ["cuda", "mps", "device"] }],
    [/目标检测|检测|锚框|anchor|iou|nms/i, { tag: ["目标检测"], en: ["detection", "anchor", "iou", "nms", "bounding"] }],
    [/分割|segment/i, { tag: ["图像分割"], en: ["segmentation", "mask"] }],
    [/生成|gan|对抗|扩散|diffusion/i, { tag: ["生成模型"], en: ["generator", "discriminator", "gan"] }],
    [/循环|lstm|gru|rnn|序列/i, { tag: ["序列模型"], en: ["lstm", "gru", "rnn"] }],
    [/注意力|transformer|attention/i, { tag: ["注意力/Transformer"], en: ["attention", "transformer"] }],
    [/训练|怎么练|跑起来|train|fit\b/i, { tag: ["训练循环"], en: ["train", "epoch"], intent: "train" }],
    [/从零|手写|不用框架|numpy/i, { fw: "numpy", en: ["numpy"] }],
    [/pytorch|torch/i, { fw: "pytorch" }],
    [/keras|tensorflow|\btf\b/i, { fw: "keras" }],
    [/sklearn|scikit/i, { fw: "sklearn" }],
    [/全连接|线性层|mlp|多层感知机/i, { tag: ["全连接层"], model: ["MLP"], en: ["linear", "mlp"] }],
    [/激活|relu|sigmoid|softmax/i, { tag: ["激活函数"], en: ["relu", "sigmoid", "softmax"] }],
  ];
  const MODELS = [
    [/lenet/i, "LeNet"], [/alexnet/i, "AlexNet"], [/vgg/i, "VGG"], [/\bnin\b/i, "NiN"], [/googlenet|inception/i, "GoogLeNet/Inception"], [/resnet|残差网络/i, "ResNet"],
    [/densenet/i, "DenseNet"], [/mobilenet/i, "MobileNet"], [/efficientnet/i, "EfficientNet"], [/u-?net/i, "U-Net"], [/\bfcn\b/i, "FCN"], [/yolo/i, "YOLO"], [/\bssd\b/i, "SSD"],
    [/r-?cnn/i, "R-CNN"], [/\bgan\b|dcgan/i, "GAN"], [/\bvae\b|变分自编码/i, "VAE"], [/lstm|gru/i, "LSTM/GRU"], [/transformer/i, "Transformer"], [/\bmlp\b|多层感知机/i, "MLP"],
    [/线性回归/i, "线性回归"], [/softmax ?回归/i, "Softmax 回归"],
  ];
  const DSETS = [
    [/fashion|服饰|衣服/i, "Fashion-MNIST"], [/mnist|手写数字/i, "MNIST"], [/cifar.?100/i, "CIFAR-100"], [/cifar.?10\b|cifar/i, "CIFAR-10"], [/imagenet/i, "ImageNet"],
    [/\bvoc\b/i, "VOC"], [/coco/i, "COCO"], [/svhn|门牌/i, "SVHN"], [/猫狗|cats?.?(vs|and).?dogs?/i, "猫狗"], [/celeba|人脸/i, "CelebA"],
  ];
  const LOCAL_ID = { "MNIST": "mnist", "Fashion-MNIST": "fmnist", "CIFAR-10": "cifar10", "CIFAR-100": "cifar100", "SVHN": "svhn" };
  const LAB_MODEL = { "ResNet": "resnet_mini", "LeNet": "lenet", "VGG": "vgg_mini", "MLP": "mlp" };

  function parse(q) {
    const out = { tags: new Set(), en: new Set(), models: new Set(), ds: new Set(), fw: null, intent: null, words: [], lab: {} };
    const ql = q.toLowerCase();
    RULES.forEach(([rx, sp]) => {
      if (!rx.test(q)) return;
      (sp.tag || []).forEach(t => out.tags.add(t)); (sp.en || []).forEach(t => out.en.add(t));
      (sp.model || []).forEach(t => out.models.add(t)); if (sp.fw) out.fw = sp.fw; if (sp.intent) out.intent = sp.intent;
    });
    MODELS.forEach(([rx, m]) => { if (rx.test(q)) out.models.add(m); });
    const seenDs = new Set();
    DSETS.forEach(([rx, d]) => { if (rx.test(q) && !(d === "MNIST" && out.ds.has("Fashion-MNIST")) && !(d === "CIFAR-10" && out.ds.has("CIFAR-100"))) { out.ds.add(d); seenDs.add(d); } });
    out.words = (ql.match(/[a-z][a-z0-9_.+-]{2,}/g) || []).filter(w => !["pytorch", "torch", "keras", "tensorflow", "numpy", "sklearn", "the", "how", "and", "for", "with", "using"].includes(w));
    // 能直接交给训练实验室的参数
    const L = out.lab;
    const ds = [...out.ds].map(d => LOCAL_ID[d]).find(Boolean); if (ds) L.dataset = ds;
    const md = [...out.models].map(m => LAB_MODEL[m]).find(Boolean);
    if (md) L.model = /resnet.?18/i.test(q) ? "resnet18" : md; else if (/cnn|卷积/i.test(q) && (L.dataset || out.intent === "train")) L.model = "simple_cnn";
    let m;
    if ((m = q.match(/(?:学习率|lr)\s*[=:：为是]?\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)/i))) L.lr = +m[1];
    if ((m = q.match(/(\d+)\s*(?:轮|个?\s*epochs?)/i))) L.epochs = +m[1];
    if ((m = q.match(/(?:batch.?size|批大小|批量大小|批次大小)\s*[=:：为是]?\s*(\d+)/i))) L.batch_size = +m[1];
    if ((m = q.match(/dropout\s*[=:：为是]?\s*(0?\.\d+)/i))) L.dropout = +m[1];
    if (/adamw/i.test(q)) L.optimizer = "adamw"; else if (/adam/i.test(q)) L.optimizer = "adam"; else if (/rmsprop/i.test(q)) L.optimizer = "rmsprop";
    else if (/nesterov/i.test(q)) L.optimizer = "nesterov"; else if (/动量|momentum/i.test(q)) L.optimizer = "sgd_momentum"; else if (/\bsgd\b/i.test(q)) L.optimizer = "sgd";
    if (/onecycle|one.?cycle/i.test(q)) L.scheduler = "onecycle"; else if (/余弦|cosine/i.test(q)) L.scheduler = "cosine"; else if (/衰减|step.?lr|阶梯/i.test(q)) L.scheduler = "step";
    if (/数据增强|增广|augment|翻转|裁剪/i.test(q)) L.augment = (L.dataset === "mnist" || L.dataset === "fmnist") ? "crop" : "crop_flip";
    if (/(不用|去掉|没有|无|no).{0,4}(批量?归一化|bn|batch ?norm)/i.test(q)) L.norm = "none";
    return out;
  }

  /* ---------- 打分 ---------- */
  function score(s, p, qRaw) {
    let sc = 0, hits = 0;
    const head = (s.title + " " + s.nb + " " + s.file).toLowerCase(), intro = s.intro.toLowerCase(), codeL = s._c || (s._c = s.code.toLowerCase());
    p.ds.forEach(d => { if (s.ds.includes(d)) { sc += 8; hits++; } });
    p.models.forEach(m => { if (s.models.includes(m)) { sc += 8; hits++; } });
    p.tags.forEach(t => { if (s.tags.includes(t)) { sc += 5; hits++; } });
    if (p.fw) { if (s.fw === p.fw) { sc += 4; hits++; } else sc -= 6; }
    p.en.forEach(w => { if (head.includes(w)) { sc += 4; hits++; } else if (intro.includes(w)) { sc += 2; } else if (codeL.includes(w)) sc += 0.8; });
    p.words.forEach(w => { if (head.includes(w)) { sc += 6; hits++; } else if (intro.includes(w)) sc += 3; else if (codeL.includes(w)) { sc += 1.5; hits++; } });
    if (!hits) return 0;
    if (s.src === "lab" || s.src === "practice") sc += p.intent === "train" || p.ds.size ? 5 : 2;
    if (p.fw === "numpy" && s.src === "numpy") sc += 6;
    if (s.local_ds && p.ds.size) sc += 1.5;
    if (s.n < 6) sc -= 3; else if (s.n >= 15 && s.n <= 220) sc += 1;
    return sc;
  }

  /* ---------- 复制 / 运行：把下载路径改成本地数据集 ---------- */
  let dataRoot = null;
  async function getRoot() {
    if (dataRoot) return dataRoot;
    try {
      const c = new AbortController(); setTimeout(() => c.abort(), 700);
      dataRoot = (await (await fetch(LAB_API + "/api/health", { signal: c.signal })).json()).data_root;
    } catch (e) { dataRoot = null; }
    return dataRoot;
  }
  function localize(code, root) {
    let c = code.replace(/download\s*=\s*True/g, "download=False");
    // 1) 直接写字符串路径：datasets.X('./data', ...) / root='./data'
    c = c.replace(/(torchvision\.datasets\.\w+|\bdatasets\.\w+)\(\s*(?:root\s*=\s*)?(['"])[^'"]*\2/g, (m, fn) => `${fn}(root=DATA_ROOT`);
    // 2) 路径放在变量里：image_path = './'  →  image_path = DATA_ROOT
    const vars = new Set();
    c.replace(/(?:torchvision\.datasets\.\w+|\bdatasets\.\w+)\(\s*(?:root\s*=\s*)?([A-Za-z_]\w*)\s*[,)]/g, (m, v) => { if (v !== "DATA_ROOT") vars.add(v); return m; });
    vars.forEach(v => { c = c.replace(new RegExp("^(\\s*)" + v + "\\s*=\\s*(['\"])[^'\"]*\\2", "m"), `$1${v} = DATA_ROOT`); });
    const head = `import os\nDATA_ROOT = os.environ.get("LAB_DATA", ${JSON.stringify(root || "datasets")})   # 本地已下载的数据集目录（学习站「实验室 → 数据集」）\n\n`;
    return head + c;
  }
  const copy = (t, msg) => navigator.clipboard.writeText(t).then(() => UI.toast(msg || "已复制"), () => UI.toast("复制失败"));

  /* ---------- 页面 ---------- */
  const EXAMPLES = ["用 PyTorch 在 CIFAR-10 上训练 ResNet", "怎么做数据增强", "批量归一化怎么写", "迁移学习 微调 预训练模型", "学习率衰减 余弦退火", "手写卷积 numpy 从零实现", "MNIST LeNet 学习率0.002 训练5轮", "目标检测 IoU NMS", "GAN 生成图片", "过拟合 Dropout 权重衰减"];
  const st = { q: "", src: new Set(), fw: new Set(), localOnly: false, limit: 12, open: new Set() };

  function render(main, query) {
    st.q = query || ""; st.limit = 12;
    const topTags = (() => { const c = {}; D.snippets.forEach(s => s.tags.forEach(t => { c[t] = (c[t] || 0) + 1; })); return Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, 18); })();
    main.innerHTML = `<div class="ch-head"><div class="kicker">Code Library</div><h1>代码库</h1>
      <p class="muted">项目里 ${D.sources.filter(s => s.count).length} 套教材/脚本，共 <b>${D.snippets.length}</b> 个代码片段，已按主题切好并打了标签。直接用中文提问，找到后可以<b>复制</b>、改成<b>本地数据集</b>，或<b>一键放进训练实验室</b>运行并对比。</p></div>
      <div class="card cl-search"><input id="clQ" type="search" placeholder="例如：用 PyTorch 在 CIFAR-10 上训练 ResNet，学习率0.1，训练15轮" value="${esc(st.q)}" autocomplete="off">
        <div class="cl-ex">${EXAMPLES.map(e => `<button class="chip-b" data-ex="${esc(e)}">${esc(e)}</button>`).join("")}</div>
        <div class="cl-filters"><span class="small muted">来源</span>${D.sources.filter(s => s.count).map(s => `<label class="fchip"><input type="checkbox" data-src="${s.id}" ${st.src.has(s.id) ? "checked" : ""}>${esc(s.short)} <small>${s.count}</small></label>`).join("")}
          <span class="small muted" style="margin-left:10px">框架</span>${["pytorch", "keras", "numpy", "sklearn"].map(f => `<label class="fchip"><input type="checkbox" data-fw="${f}" ${st.fw.has(f) ? "checked" : ""}>${f}</label>`).join("")}
          <label class="fchip"><input type="checkbox" id="clLocal" ${st.localOnly ? "checked" : ""}>用到本地数据集</label></div></div>
      <div id="clIntent"></div><div id="clRes"></div>`;
    const q = $("#clQ", main);
    let t; q.addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { st.q = q.value; st.limit = 12; history.replaceState(null, "", "#/code" + (st.q ? "?q=" + encodeURIComponent(st.q) : "")); draw(main, topTags); }, 180); });
    $$("[data-ex]", main).forEach(b => b.addEventListener("click", () => { q.value = st.q = b.dataset.ex; st.limit = 12; history.replaceState(null, "", "#/code?q=" + encodeURIComponent(st.q)); draw(main, topTags); }));
    $$("[data-src]", main).forEach(c => c.addEventListener("change", () => { c.checked ? st.src.add(c.dataset.src) : st.src.delete(c.dataset.src); st.limit = 12; draw(main, topTags); }));
    $$("[data-fw]", main).forEach(c => c.addEventListener("change", () => { c.checked ? st.fw.add(c.dataset.fw) : st.fw.delete(c.dataset.fw); st.limit = 12; draw(main, topTags); }));
    $("#clLocal", main).addEventListener("change", e => { st.localOnly = e.target.checked; st.limit = 12; draw(main, topTags); });
    draw(main, topTags);
    if (!st.q) q.focus();
  }

  function draw(main, topTags) {
    const q = st.q.trim(), res = $("#clRes", main), intent = $("#clIntent", main);
    const pass = s => (!st.src.size || st.src.has(s.src)) && (!st.fw.size || st.fw.has(s.fw)) && (!st.localOnly || s.local_ds);
    if (!q) {
      intent.innerHTML = "";
      const labs = D.snippets.filter(s => s.src === "lab" && pass(s));
      res.innerHTML = `<div class="card"><h3>按主题浏览</h3><div class="cl-ex">${topTags.map(([t, n]) => `<button class="chip-b" data-ex="${esc(t)}">${esc(t)} <small>${n}</small></button>`).join("")}</div></div>
        <h3 class="res-h">🧪 可直接在本机训练的模板</h3>${labs.map(s => card(s, 0)).join("")}
        <h3 class="res-h">📚 来源</h3><div class="res-grid">${D.sources.filter(s => s.count).map(s => `<div class="res"><div class="res-t">${esc(s.name)}</div><div class="small muted">${esc(s.note)}</div><div class="res-meta"><span class="tag">${s.count} 个片段</span></div></div>`).join("")}</div>`;
      bind(main, topTags); return;
    }
    const p = parse(q);
    let scored = D.snippets.filter(pass).map(s => [s, score(s, p, q)]).filter(x => x[1] > 0).sort((a, b) => b[1] - a[1]);
    if (scored.length) { const top = scored[0][1]; scored = scored.filter(x => x[1] >= top * 0.35); }     // 只保留和最佳结果量级相当的
    const chips = [...[...p.ds].map(d => `数据集 <b>${esc(d)}</b>`), ...[...p.models].map(m => `模型 <b>${esc(m)}</b>`), ...(p.fw ? [`框架 <b>${esc(p.fw)}</b>`] : []), ...[...p.tags].slice(0, 5).map(t => `主题 <b>${esc(t)}</b>`),
      ...p.words.filter(w => ![...p.ds, ...p.models].some(x => x.toLowerCase().replace(/[^a-z0-9]/g, "") === w.replace(/[^a-z0-9]/g, ""))).slice(0, 4).map(w => `关键词 <b>${esc(w)}</b>`)];
    const L = p.lab, canLab = L.dataset && (L.model || p.intent === "train");
    const labCfg = Object.assign({ model: "simple_cnn" }, L);
    const qs = Object.entries(labCfg).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
    intent.innerHTML = `<div class="card cl-intent"><div class="small"><b>我的理解：</b>${chips.length ? chips.map(c => `<span class="ichip">${c}</span>`).join("") : '<span class="muted">没有识别出明确主题，按全文搜索</span>'}</div>
      ${canLab ? `<div class="cl-lab"><span>🧪 想直接练一遍？按你的描述预填了训练参数：<code>${esc(Object.entries(labCfg).map(([k, v]) => k + "=" + v).join(" · "))}</code></span>
        <a class="btn primary sm" href="#/lab/new?${qs}">在实验室里训练 →</a></div>` : ""}</div>`;
    const shown = scored.slice(0, st.limit);
    res.innerHTML = scored.length ? `<p class="small muted">${scored.length} 个相关片段${scored.length > shown.length ? "，先显示前 " + shown.length + " 个" : ""}</p>` + shown.map(([s, sc]) => card(s, sc)).join("") +
      (scored.length > shown.length ? `<div style="text-align:center"><button class="btn" id="clMore">再显示 12 个</button></div>` : "") :
      `<div class="card"><h3>没找到合适的片段</h3><p class="muted">换个说法试试：用“卷积、池化、残差、数据增强、迁移学习、学习率、过拟合、目标检测”这类主题词，或直接写数据集和模型（如 CIFAR-10 ResNet）。也可以取消上面的来源/框架过滤。</p></div>`;
    bind(main, topTags);
    const more = $("#clMore", main); if (more) more.addEventListener("click", () => { st.limit += 12; draw(main, topTags); });
  }

  function preview(code, open) {
    const lines = code.split("\n"), show = open ? code : lines.slice(0, 9).join("\n");
    let html = esc(show);
    if (window.hljs) { try { html = hljs.highlight(show, { language: "python" }).value; } catch (e) { /* 用转义文本 */ } }
    return `<pre class="cl-code"><code class="hljs">${html}</code></pre>${lines.length > 9 ? `<button class="btn sm ghost" data-toggle="1">${open ? "收起" : `展开全部 ${lines.length} 行`}</button>` : ""}`;
  }
  const SRC_NAME = id => (D.sources.find(s => s.id === id) || {}).short || id;
  function card(s, sc) {
    const open = st.open.has(s.id);
    const canRun = s.fw === "pytorch" || s.fw === "numpy" || s.src === "lab";
    return `<article class="card cl-card" data-id="${s.id}">
      <div class="row cl-h"><span class="badge-src b-${s.src}">${esc(SRC_NAME(s.src))}</span><span class="badge-fw">${esc(s.fw)}</span>
        ${s.src === "lab" ? '<span class="tag">可直接运行</span>' : ""}${s.d2l ? '<span class="tag warn" title="依赖 d2l 工具库，且常是上文的延续">依赖 d2l</span>' : ""}</div>
      <h3 class="cl-t">${esc(s.title)}</h3><div class="small muted cl-path">${esc(s.nb)} · ${esc(s.file)}</div>
      ${s.intro ? `<p class="small cl-intro">${esc(s.intro)}</p>` : ""}
      <div class="cl-tags">${[...s.ds, ...s.models, ...s.tags.slice(0, 6)].map(t => `<span class="ichip sm">${esc(t)}</span>`).join("")}</div>
      <div class="cl-codebox">${preview(s.code, open)}</div>
      <div class="row cl-actions">
        <button class="btn sm" data-act="copy">复制代码</button>
        ${s.local_ds ? `<button class="btn sm" data-act="copy-local" title="把 download=True 和路径改成本地已下载的数据集">复制（改用本地数据集）</button>` : ""}
        ${canRun ? `<button class="btn sm primary" data-act="run" title="载入到“实验室 → 自定义脚本”，在本机运行">${s.src === "lab" ? "▶ 在实验室运行" : "载入实验室脚本编辑器"}</button>` : `<span class="small muted">Keras/TensorFlow 代码：本机未装 TF，仅供阅读</span>`}
        ${s.chapter ? `<a class="btn sm ghost" href="#/ch/${s.chapter}/learn">对应学习小节 →</a>` : ""}
        ${s.exercise ? `<a class="btn sm" href="#/practice/${s.exercise}">✍️ 分步练习这段代码 →</a>` : ""}
        <span class="spacer"></span>${sc ? `<span class="small muted" title="匹配度">匹配 ${sc.toFixed(0)}</span>` : ""}
      </div></article>`;
  }

  function bind(main, topTags) {
    $$("[data-ex]", main).forEach(b => b.onclick = () => { st.q = b.dataset.ex; $("#clQ", main).value = st.q; history.replaceState(null, "", "#/code?q=" + encodeURIComponent(st.q)); st.limit = 12; draw(main, topTags); });
    $$(".cl-card", main).forEach(el => {
      const s = D.snippets.find(x => x.id === el.dataset.id);
      el.addEventListener("click", async e => {
        const t = e.target.closest("[data-act],[data-toggle]"); if (!t) return;
        if (t.dataset.toggle) { st.open.has(s.id) ? st.open.delete(s.id) : st.open.add(s.id); $(".cl-codebox", el).innerHTML = preview(s.code, st.open.has(s.id)); return; }
        const root = await getRoot();
        if (t.dataset.act === "copy") copy(s.code);
        else if (t.dataset.act === "copy-local") copy(localize(s.code, root), "已复制：数据集路径已改为本地");
        else if (t.dataset.act === "run") {
          const code = s.src === "lab" ? s.code : (s.local_ds ? localize(s.code, root) : s.code);
          sessionStorage.setItem("labScriptDraft", JSON.stringify({ title: s.title.slice(0, 60), code }));
          if (s.d2l) UI.toast("提示：这段代码依赖 d2l 库/上文变量，可能需要补全后才能运行");
          location.hash = "#/lab/script";
        }
      });
    });
  }

  window.CodeLib = { render, parse };
})();
