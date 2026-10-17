/* 训练实验室：本地训练、参数扫描、多次运行对比（需要启动 lab/server.py） */
(function () {
  "use strict";
  const UI = window.StudyUI;
  const esc = UI.esc;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const API = location.port === "8766" ? "" : "http://127.0.0.1:8766";

  async function api(path, opt = {}) {
    const o = { method: opt.method || "GET", headers: {} };
    if (opt.body !== undefined) { o.method = opt.method || "POST"; o.headers["Content-Type"] = "application/json"; o.body = JSON.stringify(opt.body); }
    if (o.method !== "GET") o.headers["X-Lab"] = "1";
    const r = await fetch(API + path, o);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  }

  /* ---------- 状态 ---------- */
  const FORM_KEY = "dlcv-lab-form";
  const S = {
    schema: null, health: null, runs: [], detail: {}, sel: new Map(), hidden: new Set(),
    metric: "test_acc", xaxis: "epoch", table: false, timer: null, view: 0, root: null, sig: "", sweep: { param: "lr", values: "0.0003, 0.001, 0.003, 0.01" },
    form: null, mode: "single",
  };
  const loadForm = () => { try { return JSON.parse(localStorage.getItem(FORM_KEY)) || {}; } catch (e) { return {}; } };
  const saveForm = () => { try { localStorage.setItem(FORM_KEY, JSON.stringify(S.form)); } catch (e) { /* 隐私模式 */ } };

  const GROUPS = ["数据", "模型", "优化", "其他"];
  const PRESETS = [
    { name: "MNIST · LeNet（几秒钟体验）", cfg: { dataset: "mnist", model: "lenet", optimizer: "adam", lr: 0.002, epochs: 5, batch_size: 128, augment: "none", scheduler: "none" } },
    { name: "Fashion-MNIST · 简单 CNN", cfg: { dataset: "fmnist", model: "simple_cnn", width: 32, depth: 3, optimizer: "adam", lr: 0.001, epochs: 8, augment: "none" } },
    { name: "CIFAR-10 · 简单 CNN + 增强", cfg: { dataset: "cifar10", model: "simple_cnn", width: 32, depth: 3, optimizer: "adam", lr: 0.001, epochs: 15, augment: "crop_flip", dropout: 0.2 } },
    { name: "CIFAR-10 · ResNet-20（OneCycle）", cfg: { dataset: "cifar10", model: "resnet_mini", width: 16, depth: 3, optimizer: "nesterov", lr: 0.1, momentum: 0.9, weight_decay: 0.0005, scheduler: "onecycle", augment: "crop_flip", epochs: 15, batch_size: 128 } },
  ];
  const SWEEPS = [
    ["lr", "学习率", "0.0003, 0.001, 0.003, 0.01"], ["batch_size", "批大小", "32, 128, 512"], ["optimizer", "优化器", "sgd, sgd_momentum, adam"],
    ["norm", "批量归一化", "bn, none"], ["augment", "数据增强", "none, crop_flip"], ["model", "模型", "mlp, lenet, simple_cnn, resnet_mini"],
    ["scheduler", "学习率调度", "none, cosine, onecycle"], ["width", "网络宽度", "8, 16, 32, 64"], ["dropout", "Dropout", "0, 0.2, 0.5"], ["device", "设备（CPU vs GPU）", "mps, cpu"],
  ];
  const STATUS = { queued: ["⏳", "排队中"], running: ["▶", "运行中"], done: ["✔", "已完成"], failed: ["✖", "失败"], stopped: ["■", "已停止"], diverged: ["⚠", "发散"], interrupted: ["⚠", "已中断"] };
  const METRICS = { test_acc: ["测试准确率", "acc"], train_loss: ["训练损失", "loss"], test_loss: ["测试损失", "loss"], train_acc: ["训练准确率", "acc"] };

  const pct = v => v == null ? "—" : (v * 100).toFixed(2) + "%";
  const fsec = s => s == null ? "—" : s < 60 ? s.toFixed(1) + " s" : Math.floor(s / 60) + ":" + String(Math.round(s % 60)).padStart(2, "0");
  const fnum = v => v == null ? "—" : v >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(0);
  const fieldOf = k => S.schema.fields.find(f => f.key === k);
  const fmtVal = v => typeof v === "number" ? String(+v.toPrecision(6)) : String(v);

  /* ---------- 入口 ---------- */
  async function render(main, hash) {
    S.root = main; S.view++;
    clearInterval(S.timer);
    const [path, qs] = hash.replace(/^#\/?/, "").split("?");
    const tab = path.split("/")[1] || "new";
    const q = Object.fromEntries(new URLSearchParams(qs || ""));
    main.innerHTML = `<div class="loading">连接训练实验室…</div>`;
    try {
      S.health = await api("/api/health");
      if (!S.schema) S.schema = await api("/api/schema");
    } catch (e) { return offline(main, e); }
    if (!S.form) S.form = Object.assign({}, S.schema.defaults, loadForm());
    if (tab === "new" && Object.keys(q).length) {            // 来自代码库的“一键训练”：#/lab/new?dataset=...&lr=...
      const cfg = {};
      Object.keys(q).forEach(k => { if (fieldOf(k)) cfg[k] = fieldOf(k).type === "choice" ? q[k] : +q[k]; });
      S.form = Object.assign({}, S.schema.defaults, cfg); saveForm();
    }
    main.innerHTML = shell(tab);
    ({ new: tabNew, runs: tabRuns, data: tabData, script: tabScript }[tab] || tabNew)($("#labBody", main));
  }

  function offline(main, err) {
    main.innerHTML = `<div class="ch-head"><div class="kicker">Training Lab</div><h1>训练实验室</h1></div>
      <div class="card"><h3>🔌 还没有连接到本地训练服务</h3>
        <p>训练需要在你的电脑上运行 PyTorch，所以要先启动一个本地小服务器（只监听本机 127.0.0.1，数据不会上传）。在项目根目录运行：</p>
        <pre class="cmd"><code>python3 lab/server.py</code></pre>
        <p>然后用 <a href="http://localhost:8766/#/lab">http://localhost:8766/#/lab</a> 打开本页面（这个地址同时提供整个学习站）。</p>
        <div class="row"><button class="btn primary" id="labRetry">重新连接</button><span class="small muted">错误信息：${esc(err.message || err)}</span></div>
        <p class="small muted" style="margin:12px 0 0">首次使用还需要一次性准备：<code>lab/.venv</code> 虚拟环境（含 PyTorch）和数据集。详见 <code>lab/README.md</code>。从 file:// 直接打开页面时无法连接。</p></div>`;
    $("#labRetry", main).addEventListener("click", () => render(main, location.hash));
  }

  function shell(tab) {
    const h = S.health, env = h.env || {};
    const dev = env.error ? `<span class="chip-s bad">⚠ ${esc(env.error)}</span>` :
      `<span class="chip-s ok">✔ PyTorch ${esc(env.torch || "…")}</span><span class="chip-s ${env.mps ? "ok" : ""}">${env.mps ? "✔ Apple GPU（MPS）可用" : "仅 CPU"}</span>`;
    const tabs = [["new", "新建训练"], ["runs", "运行记录与对比"], ["data", "数据集"], ["script", "自定义脚本"]];
    return `<div class="ch-head"><div class="kicker">Training Lab</div><h1>训练实验室</h1>
      <p class="muted">在本机训练 CNN，换参数跑几次，直接对比准确率和训练效率。${dev}</p></div>
      <nav class="tabs lab-tabs">${tabs.map(([k, t]) => `<a class="tab ${k === tab ? "active" : ""}" href="#/lab/${k}">${t}${k === "runs" && h.busy ? ` <span class="badge">${h.busy}</span>` : ""}</a>`).join("")}</nav>
      <div id="labBody"></div>`;
  }

  /* ---------- 新建训练 ---------- */
  function control(f, v) {
    const id = "f_" + f.key;
    if (f.type === "choice") return `<select id="${id}" data-k="${f.key}">${f.choices.map(c => `<option ${c === v ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
    const step = f.type === "int" ? 1 : "any";
    return `<input id="${id}" data-k="${f.key}" type="number" step="${step}" min="${f.min}" max="${f.max}" value="${esc(fmtVal(v))}">`;
  }
  function readForm(root) {
    const cfg = {};
    $$("[data-k]", root).forEach(el => { const f = fieldOf(el.dataset.k); cfg[f.key] = f.type === "choice" ? el.value : Number(el.value); });
    return cfg;
  }
  function tabNew(el) {
    const ready = {};
    api("/api/datasets").then(ds => { ds.forEach(d => { ready[d.name] = d.ready; }); mark(); }).catch(() => { });
    el.innerHTML = `
      <div class="card"><div class="row"><b>快速预设</b>${PRESETS.map((p, i) => `<button class="btn sm" data-preset="${i}">${esc(p.name)}</button>`).join("")}</div></div>
      <div class="seg lab-mode"><button data-mode="single" class="${S.mode === "single" ? "on" : ""}">单次训练</button><button data-mode="sweep" class="${S.mode === "sweep" ? "on" : ""}">参数对比（扫描）</button></div>
      <div id="sweepBox" class="card ${S.mode === "sweep" ? "" : "hidden"}"></div>
      <div id="formBox">${GROUPS.map(g => `<div class="card"><h3>${g}</h3><div class="fgrid">${S.schema.fields.filter(f => f.group === g).map(f => `
        <label class="fld" for="f_${f.key}"><span class="fl">${esc(f.label)}<code>${f.key}</code></span>${control(f, S.form[f.key])}<span class="fh">${esc(f.help)}</span></label>`).join("")}</div></div>`).join("")}</div>
      <div class="card row"><span id="dsWarn" class="small"></span><span class="spacer"></span>
        <input id="runName" class="name-in" placeholder="实验名称（可留空，自动生成）">
        <button class="btn primary" id="goRun">▶ 开始训练</button></div>`;

    const mark = () => {
      const d = $("#f_dataset", el); if (!d) return;
      $$("option", d).forEach(o => { if (ready[o.value] === false) o.textContent = o.value + "（未下载）"; else o.textContent = o.value; });
      const w = $("#dsWarn", el); const cur = d.value;
      w.innerHTML = ready[cur] === false ? `⚠ 数据集 <b>${esc(cur)}</b> 还没下载，请先到 <a href="#/lab/data">数据集</a> 面板下载。` : "";
    };
    $("#f_dataset", el).addEventListener("change", mark);
    const sync = () => { S.form = readForm(el); saveForm(); };
    el.addEventListener("change", sync);
    $$("[data-preset]", el).forEach(b => b.addEventListener("click", () => {
      S.form = Object.assign({}, S.schema.defaults, PRESETS[+b.dataset.preset].cfg); saveForm(); tabNew(el);
    }));
    $$("[data-mode]", el).forEach(b => b.addEventListener("click", () => { S.mode = b.dataset.mode; tabNew(el); }));
    if (S.mode === "sweep") drawSweep($("#sweepBox", el), el);
    $("#goRun", el).addEventListener("click", () => submit(el));
  }

  function parseValues(f, text) {
    const parts = text.split(/[,，\s]+/).filter(Boolean);
    if (f.type === "choice") return parts.filter(p => f.choices.includes(p));
    return parts.map(Number).filter(n => Number.isFinite(n));
  }
  function drawSweep(box, el) {
    const sw = S.sweep, f = fieldOf(sw.param);
    box.innerHTML = `<h3>参数对比：只改一个参数，其余保持不变</h3>
      <p class="small muted">这是理解超参数最有效的办法——控制变量。下面每个取值会各训练一次（<b>依次</b>运行，保证计时公平），然后在“运行记录与对比”里看曲线和效率。</p>
      <div class="row">${SWEEPS.map(([k, n, v]) => `<button class="btn sm ${sw.param === k ? "primary" : ""}" data-sw="${k}" data-v="${v}">${n}</button>`).join("")}</div>
      <div class="fgrid" style="margin-top:12px"><label class="fld"><span class="fl">要对比的参数</span>
        <select id="swParam">${S.schema.fields.filter(x => x.key !== "dataset").map(x => `<option value="${x.key}" ${x.key === sw.param ? "selected" : ""}>${esc(x.label)}（${x.key}）</option>`).join("")}</select></label>
        <label class="fld"><span class="fl">取值（逗号分隔）${f.type === "choice" ? `<code>可选：${f.choices.join(" / ")}</code>` : ""}</span><input id="swVals" value="${esc(sw.values)}"></label></div>
      <div id="swInfo" class="small" style="margin-top:8px"></div>`;
    const info = () => {
      const vals = parseValues(fieldOf($("#swParam", box).value), $("#swVals", box).value);
      $("#swInfo", box).innerHTML = vals.length ? `将运行 <b>${vals.length}</b> 次：${vals.map(v => `<code>${esc(fmtVal(v))}</code>`).join(" ")}${$("#swParam", box).value === "device" ? " · ⚠ CPU 比 GPU 慢约 5–6 倍" : ""}` : `<span style="color:var(--bad)">没有有效取值</span>`;
      S.sweep = { param: $("#swParam", box).value, values: $("#swVals", box).value };
    };
    $$("[data-sw]", box).forEach(b => b.addEventListener("click", () => { S.sweep = { param: b.dataset.sw, values: b.dataset.v }; drawSweep(box, el); }));
    $("#swParam", box).addEventListener("change", e => { const x = SWEEPS.find(s => s[0] === e.target.value); S.sweep = { param: e.target.value, values: x ? x[2] : "" }; drawSweep(box, el); });
    $("#swVals", box).addEventListener("input", info);
    info();
  }

  async function submit(el) {
    const cfg = readForm(el); S.form = cfg; saveForm();
    const prefix = $("#runName", el).value.trim();
    let runs;
    if (S.mode === "sweep") {
      const f = fieldOf(S.sweep.param), vals = parseValues(f, S.sweep.values);
      if (!vals.length) { UI.toast("请先填写有效的对比取值"); return; }
      runs = vals.map(v => ({ name: `${prefix ? prefix + " · " : ""}${S.sweep.param}=${fmtVal(v)}`, config: Object.assign({}, cfg, { [S.sweep.param]: v }) }));
    } else runs = [{ name: prefix, config: cfg }];
    try {
      const r = await api("/api/runs", { body: { runs, group: S.mode === "sweep" ? `${S.sweep.param} 对比 · ${new Date().toLocaleTimeString()}` : undefined } });
      S.sel.clear(); r.runs.forEach(id => assignSlot(id));
      UI.toast(`已提交 ${r.runs.length} 个训练任务`);
      location.hash = "#/lab/runs";
    } catch (e) { UI.toast("提交失败：" + e.message); }
  }

  /* ---------- 颜色槽位：颜色跟随运行本身，不随排名变化 ---------- */
  function assignSlot(id) {
    if (S.sel.has(id)) return;
    const used = new Set(S.sel.values());
    for (let i = 0; i < 8; i++) if (!used.has(i)) { S.sel.set(id, i); return; }
  }

  /* ---------- 运行记录与对比 ---------- */
  function tabRuns(el) {
    el.innerHTML = `<div class="runs-layout"><div id="runList"></div><div id="cmpPanel"></div></div>`;
    const view = S.view;
    const tick = async () => {
      if (view !== S.view || !document.body.contains(el)) { clearInterval(S.timer); return; }
      try { S.runs = await api("/api/runs"); } catch (e) { return; }
      if (!S.sel.size && !S.userCleared && S.runs.length) {      // 默认选中最近一组运行
        const g = S.runs[0].group;
        S.runs.filter(r => r.group === g).slice(0, 8).forEach(r => assignSlot(r.id));
      }
      const ids = [...S.sel.keys()].filter(id => S.runs.some(r => r.id === id));
      [...S.sel.keys()].forEach(id => { if (!ids.includes(id)) S.sel.delete(id); });
      const sig = JSON.stringify([S.runs.map(r => [r.id, r.status, r.epochs_done, r.name]), ids, S.metric, S.xaxis, S.table, [...S.hidden]]);
      if (sig === S.sig) return;
      S.sig = sig;
      if (ids.length) { try { (await api("/api/compare?ids=" + ids.join(","))).forEach(r => { S.detail[r.id] = r; }); } catch (e) { /* 下次再试 */ } }
      drawList(el); drawCompare(el);
    };
    S.sig = ""; S.forceTick = tick;
    tick(); S.timer = setInterval(tick, 1500);
  }

  function runLine(r) {
    const c = r.config || {};
    const hw = hwLabel(r), tail = (r.imported ? " · 导入" : "") + (hw ? " · " + hw : "");
    if (r.kind === "script") return "自定义脚本" + tail;
    return `${c.model} · ${c.dataset} · ${c.optimizer} lr=${fmtVal(c.lr)} · bs ${c.batch_size}${tail}`;
  }
  function drawList(el) {
    const box = $("#runList", el);
    if (!S.runs.length) { box.innerHTML = `<div class="card"><h3>还没有运行记录</h3><p class="muted">去 <a href="#/lab/new">新建训练</a> 跑一次吧。几秒钟就能看到第一条曲线。</p></div>`; return; }
    const groups = [];
    S.runs.forEach(r => { let g = groups.find(x => x.name === r.group); if (!g) groups.push(g = { name: r.group, runs: [] }); g.runs.push(r); });
    box.innerHTML = `<div class="card runs-card"><div class="row"><h3 style="margin:0">运行记录</h3><span class="spacer"></span><button class="btn sm ghost" id="clrSel">取消全选</button></div>
      <div class="row" style="margin:6px 0 2px"><button class="btn sm" id="expSel" title="把勾选的运行导出成文件，拿到另一台电脑上导入对比">导出所选</button>
        <label class="btn sm ghost" title="导入另一台电脑导出的运行记录">导入<input type="file" id="impRuns" accept=".json,application/json" hidden></label></div>
      <p class="small muted" style="margin:4px 0 8px">勾选最多 8 条进行对比。Mac 和 Windows 上的结果可以互相导出/导入，放进同一张图里比较。</p>` +
      groups.map(g => `<div class="rgroup"><div class="rg-h"><b>${esc(g.name)}</b><span class="small muted">${g.runs.length} 条</span><span class="spacer"></span>
          <button class="btn sm ghost" data-grp="${esc(g.name)}">选择整组</button></div>` +
        g.runs.map(r => {
          const [ic, st] = STATUS[r.status] || ["", r.status];
          const total = (r.config || {}).epochs, prog = r.progress && total ? Math.min(100, ((r.epochs_done + (r.progress.step / r.progress.steps)) / total) * 100) : (r.epochs_done && total ? 100 * r.epochs_done / total : 0);
          const slot = S.sel.get(r.id);
          return `<div class="rrow ${S.sel.has(r.id) ? "on" : ""}"><label class="rchk"><input type="checkbox" data-sel="${r.id}" ${S.sel.has(r.id) ? "checked" : ""}>
              <span class="rdot" style="${slot != null ? `background:var(--s${slot + 1})` : ""}"></span></label>
            <div class="rmain"><div class="rname">${esc(r.name)}</div><div class="small muted">${esc(runLine(r))}</div>
              ${r.status === "running" || (r.status === "done" && total) ? `<div class="progress" style="margin-top:4px"><i style="width:${prog}%"></i></div>` : ""}</div>
            <div class="rside"><span class="st st-${r.status}">${ic} ${st}</span><span class="small">${r.epochs_done}${total ? "/" + total : ""} 轮 · ${pct(r.best_test_acc)}</span>
              <span>${r.status === "running" || r.status === "queued" ? `<button class="btn sm" data-stop="${r.id}">停止</button>` : `<button class="btn sm ghost" data-rerun="${r.id}" title="用这组参数重新配置">再来</button><button class="btn sm ghost" data-del="${r.id}" title="删除">🗑</button>`}</span></div></div>`;
        }).join("") + `</div>`).join("") + `</div>`;
    $$("[data-sel]", box).forEach(c => c.addEventListener("change", () => {
      if (c.checked) { if (S.sel.size >= 8) { c.checked = false; UI.toast("最多同时对比 8 条"); return; } assignSlot(c.dataset.sel); S.userCleared = false; } else { S.sel.delete(c.dataset.sel); if (!S.sel.size) S.userCleared = true; }
      S.sig = ""; refreshNow();
    }));
    $$("[data-grp]", box).forEach(b => b.addEventListener("click", () => {
      S.runs.filter(r => r.group === b.dataset.grp).slice(0, 8).forEach(r => assignSlot(r.id)); S.sig = ""; refreshNow();
    }));
    $("#expSel", box).addEventListener("click", async () => {
      const ids = [...S.sel.keys()];
      if (!ids.length) { UI.toast("先勾选要导出的运行"); return; }
      try {
        const b = await api("/api/runs/export?ids=" + ids.join(","));
        const host = ((b.host || {}).host || "lab").replace(/[^\w.-]+/g, "_");
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([JSON.stringify(b)], { type: "application/json" }));
        a.download = `cnn-lab-runs-${host}-${new Date().toISOString().slice(0, 10)}.json`; document.body.appendChild(a); a.click(); a.remove();
        UI.toast(`已导出 ${b.runs.length} 条`);
      } catch (e) { UI.toast("导出失败：" + e.message); }
    });
    $("#impRuns", box).addEventListener("change", async e => {
      const f = e.target.files[0]; if (!f) return;
      try {
        const r = await api("/api/runs/import", { body: { bundle: JSON.parse(await f.text()) } });
        S.sel.clear(); r.imported.slice(0, 8).forEach(id => assignSlot(id));
        UI.toast(`导入 ${r.imported.length} 条${r.skipped ? `，跳过 ${r.skipped} 条已存在的` : ""}`); refreshNow();
      } catch (err) { UI.toast("导入失败：" + err.message); }
    });
    $("#clrSel", box).addEventListener("click", () => { S.sel.clear(); S.userCleared = true; S.sig = ""; refreshNow(); });
    $$("[data-stop]", box).forEach(b => b.addEventListener("click", async () => { await api(`/api/runs/${b.dataset.stop}/stop`, { body: {} }).catch(e => UI.toast(e.message)); S.sig = ""; }));
    $$("[data-del]", box).forEach(b => b.addEventListener("click", async () => {
      if (!confirm("删除这条运行记录？")) return;
      await api(`/api/runs/${b.dataset.del}`, { method: "DELETE" }).catch(e => UI.toast(e.message)); S.sel.delete(b.dataset.del); S.sig = ""; refreshNow();
    }));
    $$("[data-rerun]", box).forEach(b => b.addEventListener("click", () => {
      const r = S.runs.find(x => x.id === b.dataset.rerun); if (!r || r.kind === "script") { UI.toast("脚本运行请到“自定义脚本”里修改后重跑"); return; }
      S.form = Object.assign({}, S.schema.defaults, r.config); saveForm(); S.mode = "single"; location.hash = "#/lab/new";
    }));
  }
  function refreshNow() { S.sig = ""; if (S.forceTick) S.forceTick(); }

  function hwLabel(r) {        // 哪台电脑跑的：GPU 名称（去掉冗长前缀）
    const info = r.info || {}, hw = info.hw || {};
    const name = String(hw.gpu || "").replace(/^NVIDIA GeForce /, "").replace(/ GPU \(MPS\)$/, "");
    return name || (info.device ? String(info.device).toUpperCase() : "");     // 早期的运行只记录了设备类型
  }

  /* ---------- 对比面板 ---------- */
  function timeToAcc(r, acc) {
    const row = (r.metrics || []).find(m => m.test_acc >= acc - 1e-9);
    return row ? { t: row.train_elapsed, epoch: row.epoch } : null;
  }
  function drawCompare(el) {
    const box = $("#cmpPanel", el);
    const sel = [...S.sel.keys()].map(id => S.detail[id]).filter(Boolean);
    if (!sel.length) { box.innerHTML = `<div class="card"><h3>对比面板</h3><p class="muted">在左侧勾选一条或多条运行，这里会画出曲线并比较训练效率。</p></div>`; return; }
    const withData = sel.filter(r => r.metrics && r.metrics.length);
    const keys = S.schema.fields.map(f => f.key);
    const cfgs = sel.filter(r => r.kind === "train").map(r => r.config);
    const diff = keys.filter(k => cfgs.length > 1 && new Set(cfgs.map(c => JSON.stringify(c[k]))).size > 1);
    const KEY_SET = ["dataset", "model", "optimizer", "lr", "batch_size", "epochs", "scheduler", "augment", "width", "depth"];
    const common = keys.filter(k => cfgs.length && !diff.includes(k) && KEY_SET.includes(k)).map(k => `${k}=${fmtVal(cfgs[0][k])}`);
    const slotOf = id => S.sel.get(id);
    const series = withData.map(r => ({
      key: r.id, name: r.name, color: `var(--s${slotOf(r.id) + 1})`, hidden: S.hidden.has(r.id),
      pts: r.metrics.map(m => [S.xaxis === "epoch" ? m.epoch : m.train_elapsed, m[S.metric]]).filter(p => p[1] != null),
    }));
    box.innerHTML = `<div class="card viz-root">
      <div class="row cmp-bar"><div class="seg" id="mSeg">${Object.entries(METRICS).map(([k, [n]]) => `<button data-m="${k}" class="${S.metric === k ? "on" : ""}">${n}</button>`).join("")}</div>
        <div class="seg" id="xSeg"><button data-x="epoch" class="${S.xaxis === "epoch" ? "on" : ""}">横轴：轮数</button><button data-x="time" class="${S.xaxis === "time" ? "on" : ""}">横轴：训练用时</button></div>
        <span class="spacer"></span><button class="btn sm ${S.table ? "primary" : ""}" id="tblBtn">表格视图</button></div>
      ${diff.length ? `<div class="small muted" style="margin:8px 0 4px">对比的参数：${diff.map(k => `<code>${k}</code>`).join(" ")}　不变的设置：${esc(common.join(" · "))}</div>` : (common.length ? `<div class="small muted" style="margin:8px 0 4px">设置：${esc(common.join(" · "))}</div>` : "")}
      <div id="chartHost" class="chart-host"></div>
      <div class="legend">${sel.map(r => `<button class="lg ${S.hidden.has(r.id) ? "off" : ""}" data-lg="${r.id}" title="点击显示/隐藏"><i style="background:var(--s${slotOf(r.id) + 1})"></i>${esc(r.name)}</button>`).join("")}</div>
      <div id="tblHost" class="${S.table ? "" : "hidden"}"></div></div>
      ${withData.length > 1 ? effTable(withData, diff) : (withData.length ? effTable(withData, diff) : "")}
      ${insights(withData)}
      ${sel.length === 1 ? runDetail(sel[0]) : ""}`;
    $$("[data-m]", box).forEach(b => b.addEventListener("click", () => { S.metric = b.dataset.m; S.sig = ""; refreshNow(); }));
    $$("[data-x]", box).forEach(b => b.addEventListener("click", () => { S.xaxis = b.dataset.x; S.sig = ""; refreshNow(); }));
    $("#tblBtn", box).addEventListener("click", () => { S.table = !S.table; S.sig = ""; refreshNow(); });
    $$("[data-lg]", box).forEach(b => b.addEventListener("click", () => { const id = b.dataset.lg; S.hidden.has(id) ? S.hidden.delete(id) : S.hidden.add(id); S.sig = ""; refreshNow(); }));
    const host = $("#chartHost", box);
    const kind = METRICS[S.metric][1];
    const draw = () => lineChart(host, { series, kind, xTitle: S.xaxis === "epoch" ? "训练轮数 epoch" : "累计训练用时（秒，不含评估）", yTitle: METRICS[S.metric][0], integerX: S.xaxis === "epoch" });
    draw();
    if (window.ResizeObserver) { const ro = new ResizeObserver(() => draw()); ro.observe(host); host._ro = ro; }
    if (S.table) $("#tblHost", box).innerHTML = dataTable(series);
  }

  function dataTable(series) {
    const xs = [...new Set(series.flatMap(s => s.pts.map(p => p[0])))].sort((a, b) => a - b);
    const fmt = S.metric.includes("acc") ? pct : v => v == null ? "—" : v.toFixed(4);
    return `<div class="table-wrap"><table class="dt"><thead><tr><th>${S.xaxis === "epoch" ? "轮数" : "用时 (s)"}</th>${series.map(s => `<th>${esc(s.name)}</th>`).join("")}</tr></thead><tbody>${
      xs.map(x => `<tr><td>${S.xaxis === "epoch" ? x : x.toFixed(1)}</td>${series.map(s => { const p = s.pts.find(q => q[0] === x); return `<td>${p ? fmt(p[1]) : "—"}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function effTable(rs, diff) {
    const done = rs.filter(r => r.best_test_acc != null);
    const target = Math.min(...done.map(r => r.best_test_acc));
    const times = done.map(r => timeToAcc(r, target));
    const valid = times.filter(Boolean).map(x => x.t);
    const fastest = valid.length > 1 ? Math.min(...valid) : null;
    const bestAcc = Math.max(...done.map(r => r.best_test_acc));
    return `<div class="card"><h3 style="margin-bottom:4px">效率对比</h3>
      <p class="small muted" style="margin:0 0 8px">“达到共同目标”：所有运行里最低的那个最佳准确率（<b>${pct(target)}</b>）——谁先到达它，谁就更高效。吞吐已排除第 1 轮（包含 GPU 预热）。</p>
      <div class="table-wrap"><table class="dt eff"><thead><tr><th>运行</th>${diff.map(k => `<th>${k}</th>`).join("")}<th>最佳测试准确率</th><th>末轮训练−测试</th><th>机器</th><th>参数量</th><th>稳态吞吐</th><th>每轮用时</th><th>总训练用时</th><th>达到 ${pct(target)} 用时</th></tr></thead><tbody>${
        rs.map(r => {
          const c = r.config || {}, t = done.includes(r) ? timeToAcc(r, target) : null;
          const n = (r.info || {}).n_train, per = n && r.steady_img_per_s ? n / r.steady_img_per_s : null;
          const last = r.last || {};
          const isBest = r.best_test_acc === bestAcc, isFast = t && fastest != null && t.t === fastest;
          return `<tr><td><i class="sw" style="background:var(--s${S.sel.get(r.id) + 1})"></i>${esc(r.name)}</td>${diff.map(k => `<td>${esc(fmtVal(c[k]))}</td>`).join("")}
            <td>${pct(r.best_test_acc)}${isBest && done.length > 1 ? ' <b class="star" title="最高准确率">★ 最高</b>' : ""}<span class="small muted"> 第 ${r.best_epoch ?? "—"} 轮</span></td>
            <td>${last.train_acc != null ? ((last.train_acc - last.test_acc) * 100).toFixed(1) + " 个百分点" : "—"}</td>
            <td>${esc(hwLabel(r) || "—")}</td>
            <td>${(r.info || {}).params ? Number(r.info.params).toLocaleString() : "—"}</td>
            <td>${fnum(r.steady_img_per_s)} 图/秒</td><td>${fsec(per)}</td><td>${fsec(r.train_time)}</td>
            <td>${t ? fsec(t.t) + `<span class="small muted"> · ${t.epoch} 轮</span>` + (isFast ? ' <b class="star" title="最快到达">★ 最快</b>' : "") : "—"}</td></tr>`;
        }).join("")}</tbody></table></div></div>`;
  }

  function insights(rs) {
    const done = rs.filter(r => r.best_test_acc != null && r.metrics.length);
    if (done.length < 2) return "";
    const L = [];
    const best = done.reduce((a, b) => b.best_test_acc > a.best_test_acc ? b : a);
    L.push(`最终准确率最高：<b>${esc(best.name)}</b>（${pct(best.best_test_acc)}，第 ${best.best_epoch} 轮）。`);
    const target = Math.min(...done.map(r => r.best_test_acc));
    const tt = done.map(r => ({ r, x: timeToAcc(r, target) })).filter(o => o.x).sort((a, b) => a.x.t - b.x.t);
    if (tt.length > 1) {
      const a = tt[0], z = tt[tt.length - 1];
      L.push(`达到 ${pct(target)} 最快的是 <b>${esc(a.r.name)}</b>（${fsec(a.x.t)}，${a.x.epoch} 轮），最慢的是 <b>${esc(z.r.name)}</b>（${fsec(z.x.t)}，${z.x.epoch} 轮）${a.x.t > 0 ? `，相差约 ${(z.x.t / a.x.t).toFixed(1)} 倍` : ""}。`);
    }
    const fastThr = done.reduce((a, b) => (b.steady_img_per_s || 0) > (a.steady_img_per_s || 0) ? b : a);
    const slowThr = done.reduce((a, b) => (b.steady_img_per_s || 1e18) < (a.steady_img_per_s || 1e18) ? b : a);
    if (fastThr !== slowThr && slowThr.steady_img_per_s) L.push(`单位时间处理图片最多的是 <b>${esc(fastThr.name)}</b>（${fnum(fastThr.steady_img_per_s)} 图/秒），是最慢的 ${esc(slowThr.name)} 的 ${(fastThr.steady_img_per_s / slowThr.steady_img_per_s).toFixed(1)} 倍。注意：吞吐高不等于收敛快，要结合上面“达到目标用时”一起看。`);
    done.forEach(r => { const l = r.last; if (l && l.train_acc - l.test_acc > 0.08) L.push(`<b>${esc(r.name)}</b> 训练准确率比测试高 ${((l.train_acc - l.test_acc) * 100).toFixed(1)} 个百分点，有过拟合迹象——可以试试数据增强、Dropout 或权重衰减。`); });
    const hws = [...new Set(done.map(hwLabel).filter(Boolean))];
    if (hws.length > 1) L.push(`这些运行来自不同的电脑（${hws.map(esc).join("、")}）：<b>准确率</b>可以直接比较；<b>速度</b>主要取决于硬件，想比较参数对效率的影响，请用同一台电脑上的运行。`);
    rs.forEach(r => { if (r.status === "diverged") L.push(`<b>${esc(r.name)}</b> 训练发散（损失变成 NaN/Inf），学习率通常过大。`); });
    return `<div class="card"><h3>读图要点</h3><ul class="insights">${L.map(x => `<li>${x}</li>`).join("")}</ul></div>`;
  }

  function runDetail(r) {
    const i = r.info || {};
    return `<div class="card"><h3>运行详情</h3><div class="small muted">${esc(r.id)} · 设备 ${esc(i.device || "—")} · 参数量 ${i.params ? Number(i.params).toLocaleString() : "—"} · 训练样本 ${i.n_train || "—"}</div>
      <details ${r.status === "failed" ? "open" : ""} style="margin-top:8px"><summary>训练日志</summary><pre class="logbox">${esc(r.log || "（暂无输出）")}</pre></details>
      ${r.script ? `<details style="margin-top:8px"><summary>脚本源码</summary><pre class="logbox">${esc(r.script)}</pre></details>` : `<details style="margin-top:8px"><summary>完整配置</summary><pre class="logbox">${esc(JSON.stringify(r.config, null, 2))}</pre></details>`}</div>`;
  }

  /* ---------- 折线图（SVG，无依赖） ---------- */
  function niceTicks(lo, hi, n) {
    if (hi <= lo) hi = lo + 1;
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), nr = raw / mag;
    const step = (nr < 1.5 ? 1 : nr < 3 ? 2 : nr < 7 ? 5 : 10) * mag, t = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-6; v += step) t.push(+v.toFixed(10));
    return { ticks: t, step };
  }
  function lineChart(host, o) {
    const vis = o.series.filter(s => !s.hidden && s.pts.length);
    const W = Math.max(320, host.clientWidth || 640), H = 320;
    if (!vis.length) { host.innerHTML = `<div class="muted" style="padding:40px 0;text-align:center">所选运行还没有数据，稍等一下…</div>`; return; }
    const direct = vis.length <= 4 && W > 560;
    const m = { l: 56, r: direct ? 118 : 18, t: 14, b: 44 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = vis.flatMap(s => s.pts.map(p => p[0])), ys = vis.flatMap(s => s.pts.map(p => p[1]));
    let x0 = o.integerX ? 1 : 0, x1 = Math.max(...xs, x0 + 1);
    if (o.integerX && x1 === x0) x1 = x0 + 1;
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (o.kind === "acc") { y0 = Math.max(0, y0 - (y1 - y0) * 0.1 - 0.005); y1 = Math.min(1, y1 + 0.005); } else { y0 = 0; y1 = y1 * 1.05 || 1; }
    const yt = niceTicks(y0, y1, 5), xt = niceTicks(x0, x1, o.integerX ? Math.min(8, x1 - x0) : 6);
    if (o.kind === "acc") { y0 = Math.min(y0, yt.ticks[0]); y1 = Math.max(y1, yt.ticks[yt.ticks.length - 1]); }
    const X = v => m.l + ((v - x0) / (x1 - x0)) * iw, Y = v => m.t + (1 - (v - y0) / (y1 - y0)) * ih;
    const yf = v => o.kind === "acc" ? (v * 100).toFixed(yt.step < 0.01 ? 1 : 0) + "%" : (+v.toPrecision(3)).toString();
    const xf = v => o.integerX ? String(Math.round(v)) : (v < 100 ? v.toFixed(v % 1 ? 1 : 0) : Math.round(v)) + "";
    const grid = yt.ticks.filter(v => v >= y0 - 1e-9 && v <= y1 + 1e-9).map(v => `<line class="gl-l" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${yf(v)}</text>`).join("");
    const xax = xt.ticks.filter(v => v >= x0 - 1e-9 && v <= x1 + 1e-9).map(v => `<text class="ax" x="${X(v)}" y="${H - m.b + 18}" text-anchor="middle">${xf(v)}</text>`).join("");
    const paths = vis.map(s => {
      const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
      const dots = s.pts.length <= 40 ? s.pts.map(p => `<circle class="dot" cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="4" fill="${s.color}"/>`).join("") : "";
      return `<path class="ln" d="${d}" stroke="${s.color}"/>${dots}`;
    }).join("");
    let labels = "";
    if (direct) {                                    // 直接标注；彼此太近就放弃，交给图例和提示框
      const ends = vis.map(s => ({ s, y: Y(s.pts[s.pts.length - 1][1]) })).sort((a, b) => a.y - b.y);
      if (ends.every((e, i) => !i || e.y - ends[i - 1].y >= 14)) labels = ends.map(e => `<text class="end" x="${W - m.r + 8}" y="${e.y + 4}">${esc(e.s.name.length > 16 ? e.s.name.slice(0, 15) + "…" : e.s.name)}</text>`).join("");
    }
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.yTitle)}对比曲线">
      ${grid}<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>${xax}
      <text class="axt" x="${m.l}" y="10">${esc(o.yTitle)}</text><text class="axt" x="${m.l + iw / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xTitle)}</text>
      ${paths}${labels}<line class="cross hidden" y1="${m.t}" y2="${H - m.b}"/><g class="hl"></g>
      <rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/></svg><div class="tip hidden"></div>`;
    const svg = $("svg", host), tip = $(".tip", host), cross = $(".cross", svg), hl = $(".hl", svg);
    const move = ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (W / r.width);
      let best = null;
      vis.forEach(s => s.pts.forEach(p => { const d = Math.abs(X(p[0]) - px); if (!best || d < best.d) best = { d, x: p[0] }; }));
      if (!best) return;
      const rows = vis.map(s => { const p = s.pts.reduce((a, q) => Math.abs(q[0] - best.x) < Math.abs(a[0] - best.x) ? q : a); return Math.abs(p[0] - best.x) < 1e-9 ? { s, v: p[1] } : null; }).filter(Boolean);
      cross.classList.remove("hidden"); cross.setAttribute("x1", X(best.x)); cross.setAttribute("x2", X(best.x));
      hl.innerHTML = rows.map(q => `<circle cx="${X(best.x)}" cy="${Y(q.v)}" r="5" fill="${q.s.color}" class="dot"/>`).join("");
      tip.classList.remove("hidden");
      tip.innerHTML = `<div class="tt">${o.integerX ? "第 " + Math.round(best.x) + " 轮" : "训练 " + best.x.toFixed(1) + " s"}</div>` +
        rows.map(q => `<div class="tr"><i style="background:${q.s.color}"></i><span>${esc(q.s.name)}</span><b>${o.kind === "acc" ? pct(q.v) : q.v.toFixed(4)}</b></div>`).join("");
      const hr = host.getBoundingClientRect(), left = (X(best.x) / W) * r.width + 12;
      tip.style.left = Math.min(left, hr.width - tip.offsetWidth - 4) + "px"; tip.style.top = "30px";
      if (left + tip.offsetWidth > hr.width - 4) tip.style.left = Math.max(4, (X(best.x) / W) * r.width - tip.offsetWidth - 12) + "px";
    };
    const leave = () => { cross.classList.add("hidden"); hl.innerHTML = ""; tip.classList.add("hidden"); };
    const hit = $(".hit", svg);
    hit.addEventListener("pointermove", move); hit.addEventListener("pointerdown", move); hit.addEventListener("pointerleave", leave);
  }

  /* ---------- 数据集 ---------- */
  function tabData(el) {
    const view = S.view;
    const draw = async () => {
      if (view !== S.view || !document.body.contains(el)) { clearInterval(S.timer); return; }
      let ds; try { ds = await api("/api/datasets"); } catch (e) { return; }
      const root = S.health.data_root;
      const sig = JSON.stringify(ds.map(d => [d.name, d.ready, d.job && d.job.state, d.job && d.job.percent, d.partial && d.partial.percent]));
      if (sig === S.dsig) return; S.dsig = sig;
      const ready = ds.filter(d => d.ready).length;
      el.innerHTML = `<div class="card"><h3>本地数据集（已就绪 ${ready} / ${ds.length}）</h3>
        <p class="small muted" style="margin:0">所有数据集放在 <code>${esc(root)}</code>，采用 torchvision 原生目录结构。训练实验室、教材里的 PyTorch 代码、你自己的脚本都可以直接引用，不必再下载：</p>
        <pre class="cmd"><code>import torchvision
ds = torchvision.datasets.CIFAR10(root="${esc(root)}", train=True, download=False)</code></pre>
        <button class="btn sm" id="cpRoot">复制数据目录路径</button></div>
      <div class="grid">${ds.map(d => {
        const ext = d.partial && d.partial.active ? { state: "running", percent: d.partial.percent } : null;   // 终端里手动启动的下载
        const job = d.job && d.job.state === "running" ? d.job : (ext || d.job), running = job && job.state === "running";
        return `<div class="card ds-card"><div class="row"><h3 style="margin:0">${esc(d.title)}</h3><span class="spacer"></span>
          <span class="st ${d.ready ? "st-done" : ""}">${d.ready ? "✔ 已下载" : (running ? "⬇ 下载中" : "未下载")}</span></div>
          <p class="small muted" style="margin:6px 0">${esc(d.desc)}</p>
          <div class="small">${d.shape.join("×")} · ${d.classes} 类 · 训练 ${d.n_train.toLocaleString()} / 测试 ${d.n_test.toLocaleString()}</div>
          <div class="small muted">${d.ready ? `占用 ${d.size_mb} MB` : `下载约 ${d.approx_mb} MB`}${d.core ? " · 核心数据集" : " · 按需"}</div>
          ${running ? `<div class="progress" style="margin-top:8px"><i style="width:${job.percent || 2}%"></i></div><div class="small muted">${job.percent != null ? job.percent.toFixed(0) + "%" : "连接中…"}（官方服务器较慢，可离开本页）</div>` : ""}
          ${job && job.state === "failed" ? `<div class="small" style="color:var(--bad)">下载失败，请检查网络后重试</div>` : ""}
          ${!d.ready && !running ? `<button class="btn primary sm" data-dl="${d.name}" style="margin-top:8px">下载</button>` : ""}</div>`;
      }).join("")}</div>`;
      $("#cpRoot", el).addEventListener("click", () => navigator.clipboard.writeText(root).then(() => UI.toast("已复制"), () => UI.toast("复制失败")));
      $$("[data-dl]", el).forEach(b => b.addEventListener("click", async () => { await api(`/api/datasets/${b.dataset.dl}/download`, { body: {} }).catch(e => UI.toast(e.message)); S.dsig = ""; draw(); }));
    };
    S.dsig = ""; draw(); S.timer = setInterval(draw, 2000);
  }

  /* ---------- 自定义脚本 ---------- */
  async function tabScript(el) {
    const tpls = await api("/api/templates").catch(() => []);
    let draft = null; try { draft = JSON.parse(sessionStorage.getItem("labScriptDraft")); sessionStorage.removeItem("labScriptDraft"); } catch (e) { /* 无草稿 */ }
    const first = draft ? draft.code : (tpls[0] ? tpls[0].code : "");
    el.innerHTML = `<div class="card"><h3>运行任意 PyTorch 脚本，并和内置训练一起对比</h3>
      <p class="small muted" style="margin:0">脚本在本机的 <code>lab/.venv</code> 里运行，工作目录是一个独立的新文件夹。只要在脚本里用 <code>labkit</code> 读本地数据、每轮调用一次 <code>log_epoch()</code>，它的曲线就会出现在“运行记录与对比”里。没有 <code>log_epoch</code> 也能运行，只是看不到曲线（可看日志）。</p>
      <pre class="cmd"><code>from labkit import DATA_ROOT, get_device, load_tensors, log_epoch
xtr, ytr, xte, yte = load_tensors("cifar10", get_device())     # 已归一化，直接用
...
log_epoch(epoch, train_loss=..., train_acc=..., test_loss=..., test_acc=..., epoch_time=...)</code></pre></div>
      <div class="card"><div class="row"><b>起点</b><select id="tplSel" class="sel">${draft ? `<option value="-1">${esc(draft.title || "来自代码库")}</option>` : ""}${tpls.map((t, i) => `<option value="${i}">${esc(t.file)} — ${esc(t.doc.split("\n")[0])}</option>`).join("")}</select>
        <span class="spacer"></span><input id="scName" class="name-in" placeholder="名称" value="${esc(draft ? draft.title || "" : "")}"><button class="btn primary" id="scRun">▶ 运行脚本</button></div>
        <textarea id="scCode" class="codearea" spellcheck="false">${esc(first)}</textarea>
        <p class="small muted" style="margin:6px 0 0">提示：同一时刻只运行一个任务（避免抢 GPU 影响计时）；脚本最长运行 3 小时。</p></div>`;
    $("#tplSel", el).addEventListener("change", e => { const t = tpls[+e.target.value]; if (t) { $("#scCode", el).value = t.code; $("#scName", el).value = ""; } });
    $("#scRun", el).addEventListener("click", async () => {
      const code = $("#scCode", el).value, sel = tpls[+$("#tplSel", el).value];
      const name = $("#scName", el).value.trim() || (sel ? "脚本 · " + sel.file : "自定义脚本");
      try { const r = await api("/api/script", { body: { name, code } }); S.sel.clear(); assignSlot(r.runs[0]); UI.toast("已提交"); location.hash = "#/lab/runs"; }
      catch (e) { UI.toast("提交失败：" + e.message); }
    });
  }

  window.Lab = { render };
})();
