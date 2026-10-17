/* 分步代码练习：L1 选择 → L2 填空 → L3 写整块 → L4 默写完整训练脚本（L3/L4 在本机运行评测） */
(function () {
  "use strict";
  const UI = window.StudyUI;
  const esc = UI.esc;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const API = location.port === "8766" ? "" : "http://127.0.0.1:8766";
  const DATA = () => (window.EXERCISES || { items: [] }).items;
  const LEVELS = [
    { n: 1, name: "选择", en: "Choose", desc: "在挖空处从选项里选出正确的代码，适合第一遍熟悉流程。" },
    { n: 2, name: "填空", en: "Fill in", desc: "自己输入挖空处的关键代码（函数名、参数），支持等价写法。" },
    { n: 3, name: "写整块", en: "Write blocks", desc: "挖空的整行/整块代码由你来写，在本机运行并按步骤检查。" },
    { n: 4, name: "默写完整代码", en: "From scratch", desc: "只剩步骤标题和提示，从导入到训练、保存全部自己写。" },
  ];

  /* ---------- 进度存储（与学习站同一个 localStorage） ---------- */
  const P = () => UI.store.get("practice", {});
  const exState = id => { const p = P(); return p[id] || (p[id] = { L1: {}, L2: {}, L3: {}, L4: {} }); };
  const save = () => UI.store.save();

  function levelDone(ex, st, n) {
    const blanks = ex.steps.flatMap(s => s.blanks);
    if (n <= 2) return blanks.length > 0 && blanks.every(b => (st["L" + n][b.id] || {}).ok);
    return !!(st["L" + n] && st["L" + n].done);
  }
  function levelPct(ex, st, n) {
    const blanks = ex.steps.flatMap(s => s.blanks);
    if (n <= 2) return Math.round(100 * blanks.filter(b => (st["L" + n][b.id] || {}).ok).length / Math.max(1, blanks.length));
    const L = st["L" + n] || {};
    return L.done ? 100 : Math.round(100 * (L.passed || []).length / ex.steps.length);
  }
  /* 错题：答错（或直接看答案）记入学习站的错题本（id 前缀 px:）；之后在错题本里点“重做”才移出 */
  function recordMistake(ex, b, ok, val) {
    if (ok) return;
    UI.answers[`px:${ex.id}:${b.id}`] = { ok: false, val, t: Date.now() };
    save();
  }

  /* ---------- 小工具 ---------- */
  function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
  function shuffled(arr, seed) {        // 固定种子的打乱：刷新后选项顺序不变
    const a = arr.slice(); let s = seed || 1;
    for (let i = a.length - 1; i > 0; i--) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; const j = s % (i + 1); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  const norm = s => String(s).replace(/\s+/g, "").replace(/"/g, "'")
    .replace(/\d*\.?\d+(?:e-?\d+)?/gi, m => { const v = Number(m); return Number.isFinite(v) ? String(v) : m; });
  const matches = (b, v) => [b.mask, ...(b.accept || [])].some(x => norm(x) === norm(v));
  function hl(code) {
    if (!window.hljs) return esc(code);
    try { return hljs.highlight(code, { language: "python" }).value; } catch (e) { return esc(code); }
  }
  /* 讲解：轻量 Markdown（段落、列表、引用、行内代码、粗体、代码块、$公式$） */
  function md(src) {
    const parts = String(src || "").split(/```(?:python)?\n?/);
    return parts.map((chunk, i) => {
      if (i % 2) return `<pre class="px-pre"><code class="hljs">${hl(chunk.replace(/\n$/, ""))}</code></pre>`;
      const inline = t => esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
      let html = "", list = null;
      const close = () => { if (list) { html += `</${list}>`; list = null; } };
      chunk.split("\n").forEach(line => {
        let m;
        if ((m = line.match(/^\s*[-*]\s+(.*)/))) { if (list !== "ul") { close(); html += "<ul>"; list = "ul"; } html += `<li>${inline(m[1])}</li>`; }
        else if ((m = line.match(/^\s*\d+\.\s+(.*)/))) { if (list !== "ol") { close(); html += "<ol>"; list = "ol"; } html += `<li>${inline(m[1])}</li>`; }
        else if ((m = line.match(/^>\s?(.*)/))) { close(); html += `<div class="note">${inline(m[1])}</div>`; }
        else if (!line.trim()) { close(); html += "\n"; }
        else { close(); html += `<p>${inline(line)}</p>`; }
      });
      close();
      return html;
    }).join("");
  }

  /* ---------- 服务器 ---------- */
  async function api(path, body) {
    const o = body ? { method: "POST", headers: { "Content-Type": "application/json", "X-Lab": "1" }, body: JSON.stringify(body) } : {};
    const r = await fetch(API + path, o);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || "HTTP " + r.status);
    return j;
  }
  let serverOK = null;
  async function probe() {
    try { const c = new AbortController(); setTimeout(() => c.abort(), 1500); const h = await (await fetch(API + "/api/health", { signal: c.signal })).json(); serverOK = h; }
    catch (e) { serverOK = false; }
    return serverOK;
  }

  /* ---------- 入口 ---------- */
  function render(main, hash) {
    const [path, qs] = hash.replace(/^#\/?/, "").split("?");
    const id = path.split("/")[1];
    const q = new URLSearchParams(qs || "");
    if (!id) return renderList(main);
    const ex = DATA().find(e => e.id === id);
    if (!ex) { main.innerHTML = `<div class="card">没有找到练习 ${esc(id)}。<a href="#/practice">返回练习列表</a></div>`; return; }
    const st = exState(ex.id);
    const level = +(q.get("level") || st.level || 1);
    st.level = level; save();
    renderExercise(main, ex, st, level);
  }

  function renderList(main) {
    const items = DATA();
    main.innerHTML = `<div class="ch-head"><div class="kicker">Practice</div><h1>代码练习</h1>
      <p class="muted">把训练代码拆成固定环节——加载数据、定义模型、损失与优化器、前向传播、反向传播、step、评估、保存——在关键位置自己写。每篇 4 级难度，从选择题一直到默写完整训练脚本。</p></div>
      <div class="card"><div class="px-levels">${LEVELS.map(l => `<div><b>L${l.n} ${l.name}</b><span>${l.desc}</span></div>`).join("")}</div>
        <p class="small muted" style="margin:10px 0 0">L1/L2 不需要任何安装；L3/L4 会在你的电脑上真正运行代码检查（需要启动训练实验室服务：<code>python3 lab/server.py</code>），支持 Apple 芯片（MPS）和 NVIDIA 显卡（CUDA）。</p></div>
      <div class="grid">${items.map(ex => {
        const st = exState(ex.id), m = ex.meta, nb = ex.steps.reduce((s, x) => s + x.blanks.length, 0);
        return `<a class="ch-card px-card" href="#/practice/${ex.id}"><div class="row"><span class="tag">${esc(m.level || "")}</span><span class="small muted">${esc(m.dataset || "")} · 约 ${esc(m.minutes || "?")} 分钟</span></div>
          <h3>${esc(m.title)}</h3><p>${esc(m.goal || "")}</p><div class="small muted">${ex.steps.length} 步 · ${nb} 个空</div>
          <div class="px-lv">${LEVELS.map(l => `<span class="${levelDone(ex, st, l.n) ? "done" : ""}" title="L${l.n} ${l.name}：${levelPct(ex, st, l.n)}%">L${l.n}${levelDone(ex, st, l.n) ? " ✔" : ""}</span>`).join("")}</div></a>`;
      }).join("")}</div>`;
  }

  function renderExercise(main, ex, st, level) {
    const m = ex.meta;
    main.innerHTML = `<div class="ch-head"><div class="kicker"><a href="#/practice">代码练习</a> · ${esc(m.level || "")}</div><h1>${esc(m.title)}</h1>
      <div class="summary prose"><p>${esc(m.goal || "")}</p></div>
      <div class="sources"><span>🎯 ${esc(m.task || "")}</span><span>📦 数据集 ${esc(m.dataset || "—")}</span><span>⏱ 约 ${esc(m.minutes || "?")} 分钟</span>
        ${m.chapter ? `<span><a href="#/ch/${esc(m.chapter)}/learn">📖 对应章节</a></span>` : ""}</div></div>
      <nav class="tabs">${LEVELS.map(l => `<a class="tab ${l.n === level ? "active" : ""}" href="#/practice/${ex.id}?level=${l.n}">L${l.n} ${l.name}${levelDone(ex, st, l.n) ? " ✔" : ""}<small>${l.en}</small></a>`).join("")}</nav>
      <p class="small muted" style="margin:-4px 0 14px">${LEVELS[level - 1].desc}</p>
      <div id="pxBody"></div>`;
    const body = $("#pxBody", main);
    if (level <= 2) renderBlanks(body, ex, st, level); else renderEditor(body, ex, st, level);
  }

  /* ---------- L1 / L2：挖空选择、填空 ---------- */
  function renderBlanks(el, ex, st, level) {
    const L = st["L" + level];
    const total = ex.steps.flatMap(s => s.blanks).length;
    const draw = () => {
      const done = ex.steps.flatMap(s => s.blanks).filter(b => (L[b.id] || {}).ok).length;
      const first = ex.steps.flatMap(s => s.blanks).filter(b => (L[b.id] || {}).first).length;
      el.innerHTML = `<div class="card px-score"><div><div class="small muted">完成 Done</div><b>${done} / ${total}</b></div>
          <div><div class="small muted">一次答对 First try</div><b>${first} / ${total}</b></div>
          <div class="spacer"></div><button class="btn" id="pxReset">重做这一级</button>
          ${done === total ? `<a class="btn primary" href="#/practice/${ex.id}?level=${level + 1}">进入 L${level + 1} ${LEVELS[level].name} →</a>` : ""}</div>` +
        ex.steps.map(s => stepCard(ex, s, L, level)).join("") + finishCard(ex, done === total, level);
      bind();
      UI.typeset(el);
    };
    const bind = () => {
      $("#pxReset", el).onclick = () => { if (!confirm("清空这一级的作答记录？")) return; st["L" + level] = {}; save(); renderBlanks(el, ex, st, level); };
      $$("[data-pick]", el).forEach(btn => btn.onclick = () => {
        const b = findBlank(ex, btn.dataset.b), v = btn.dataset.pick;
        answer(ex, b, L, v);
        draw();
      });
      $$("input[data-b]", el).forEach(inp => {
        inp.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); submitStep(inp.closest(".px-step")); } });
      });
      $$("[data-check]", el).forEach(btn => btn.onclick = () => submitStep(btn.closest(".px-step")));
      $$("[data-hint]", el).forEach(btn => btn.onclick = () => { const b = findBlank(ex, btn.dataset.hint); (L[b.id] = L[b.id] || {}).hinted = true; save(); draw(); });
      $$("[data-reveal]", el).forEach(btn => btn.onclick = () => {
        const b = findBlank(ex, btn.dataset.reveal);
        const s = L[b.id] = L[b.id] || {}; s.revealed = true; s.ok = true; s.val = b.mask; s.first = false; save();
        recordMistake(ex, b, false, "（看了答案）"); draw();
      });
      $$("[data-runans]", el).forEach(btn => btn.onclick = () => runAnswerInLab(ex));
    };
    const submitStep = card => {
      $$("input[data-b]", card).forEach(inp => { const b = findBlank(ex, inp.dataset.b); if (inp.value.trim() && !(L[b.id] || {}).ok) answer(ex, b, L, inp.value.trim()); });
      const sid = card.id;
      draw();
      const cards = $$(".px-step", el), from = Math.max(0, cards.findIndex(c => c.id === sid));   // 从当前步骤往后找下一个没答对的空
      const nx = cards.slice(from).concat(cards.slice(0, from)).map(c => c.querySelector("input.blank-in:not(.ok)")).find(Boolean);
      if (nx) nx.focus();
    };
    draw();
  }
  function findBlank(ex, id) { for (const s of ex.steps) for (const b of s.blanks) if (b.id === id) return b; return null; }
  function answer(ex, b, L, v) {
    const s = L[b.id] = L[b.id] || {};
    if (s.ok) return;
    const ok = matches(b, v);
    s.tries = (s.tries || 0) + 1;
    if (s.tries === 1) s.first = ok;
    s.val = v; s.ok = ok; s.last = v;
    recordMistake(ex, b, ok, v);
    save();
  }
  function stepCard(ex, s, L, level) {
    const done = s.blanks.every(b => (L[b.id] || {}).ok);
    // 先用占位标识符做语法高亮，再把占位换成挖空控件
    let code = s.view;
    s.blanks.forEach((b, i) => { code = code.split(`⟦${b.id}⟧`).join(`__BLANK${i}__`); });
    let html = hl(code);
    s.blanks.forEach((b, i) => {
      const st = L[b.id] || {};
      const w = level === 1
        ? `<span class="blank-slot ${st.ok ? "ok" : st.last ? "bad" : ""}" title="${esc(b.hint)}">${st.ok ? esc(b.mask) : (st.last ? esc(st.last) : "____")}</span>`
        : `<input class="blank-in ${st.ok ? "ok" : st.last ? "bad" : ""}" data-b="${b.id}" value="${esc(st.ok ? b.mask : (st.last || ""))}" ${st.ok ? "disabled" : ""} size="${Math.max(6, b.mask.length + 2)}" spellcheck="false" autocomplete="off" placeholder="?">`;
      html = html.split(`__BLANK${i}__`).join(w);
    });
    const qs = s.blanks.map((b, i) => {
      const st = L[b.id] || {};
      const opts = shuffled(b.options, hash(ex.id + b.id));
      const fb = st.ok ? `<div class="feedback ok"><b>✔ ${st.revealed ? "答案" : "正确"}</b> <code>${esc(b.mask)}</code> ${esc(b.why)}</div>`
        : st.last ? `<div class="feedback bad"><b>✖ 不对</b>你的答案 <code>${esc(st.last)}</code>。${st.hinted ? "" : "可以先看看提示。"}</div>` : "";
      return `<div class="px-q"><div class="px-qh"><span class="qn">空 ${i + 1}</span><span class="spacer"></span>
          ${!st.ok ? `<button class="btn sm ghost" data-hint="${b.id}">💡 提示</button><button class="btn sm ghost" data-reveal="${b.id}">看答案</button>` : ""}</div>
        ${st.hinted && !st.ok ? `<div class="small px-hint">💡 ${esc(b.hint)}</div>` : ""}
        ${level === 1 ? `<div class="opts">${opts.map((o, k) => `<button class="opt ${st.ok && o === b.mask ? "correct" : (!st.ok && st.last === o ? "wrong" : "")}" data-b="${b.id}" data-pick="${esc(o)}" ${st.ok ? "disabled" : ""}><span class="letter">${"ABCDEF"[k]}</span><code>${esc(o)}</code></button>`).join("")}</div>` : ""}
        ${fb}</div>`;
    }).join("");
    return `<article class="card px-step ${done ? "done" : ""}" id="st-${s.id}">
      <h2><span class="num">${s.n}</span>${esc(s.title)}${done ? '<span class="lvl core">✔ 完成</span>' : ""}</h2>
      ${s.explain ? `<details class="px-ex" ${done ? "" : "open"}><summary>讲解</summary><div class="prose">${md(s.explain)}</div></details>` : ""}
      <pre class="px-code"><code class="hljs">${html}</code></pre>
      ${s.blanks.length ? qs + (level === 2 && !done ? `<div class="row"><span class="spacer"></span><button class="btn primary" data-check="1">检查本步 ⏎</button></div>` : "") : `<p class="small muted">这一步没有挖空，读懂代码即可。</p>`}
    </article>`;
  }
  function finishCard(ex, done, level) {
    return `<div class="card px-finish ${done ? "on" : ""}"><h3>${done ? "🎉 这一级完成了" : "完成所有挖空后"}</h3>
      <p class="muted">${done ? `下一步：进入 L${level + 1}，${LEVELS[level].desc}` : "可以随时把完整答案放进训练实验室，真的训练一遍、看曲线。"}</p>
      <div class="row"><button class="btn" data-runans="1">▶ 在实验室运行完整答案</button>${done ? `<a class="btn primary" href="#/practice/${ex.id}?level=${level + 1}">进入 L${level + 1} →</a>` : ""}</div></div>`;
  }

  async function runAnswerInLab(ex, code, name) {
    if (!(await probe())) { UI.toast("需要先启动训练实验室：python3 lab/server.py"); return; }
    try {
      await api("/api/script", { name: name || `练习答案 · ${ex.meta.title}`.slice(0, 70), code: code || ex.answer, source: "practice:" + ex.id });
      UI.toast("已提交到训练实验室"); location.hash = "#/lab/runs";
    } catch (e) { UI.toast("提交失败：" + e.message); }
  }

  /* ---------- L3 / L4：编辑器 + 本机评测 ---------- */
  function renderEditor(el, ex, st, level) {
    const L = st["L" + level] = st["L" + level] || {};
    const tpl = level === 3 ? ex.l3 : ex.l4;
    const code = L.code || tpl;
    el.innerHTML = `<div class="px-ed-layout">
      <div class="card px-ed-card">
        <div class="row px-ed-bar"><b>${level === 3 ? "把 ✏️ 标记的地方补全" : "从零写出完整训练脚本"}</b><span class="spacer"></span>
          <button class="btn sm ghost" id="edReset" title="恢复成初始模板">重置</button>
          <button class="btn sm" id="edUpto" title="只运行到光标所在的步骤">运行到当前步骤</button>
          <button class="btn primary sm" id="edRun" title="Ctrl/⌘ + Enter">▶ 运行全部检查</button></div>
        <div class="editor"><pre class="ed-gutter" aria-hidden="true"></pre><div class="ed-hl hidden"></div><textarea class="ed-ta" spellcheck="false" wrap="off" autocapitalize="off" autocomplete="off">${esc(code)}</textarea></div>
        <p class="small muted" style="margin:6px 0 0">Tab 缩进 · Shift+Tab 反缩进 · 回车自动缩进 · ⌘/Ctrl+Enter 运行。草稿自动保存在本浏览器。快速评测会自动缩小数据和轮数，几秒钟出结果。</p>
      </div>
      <div class="px-side">
        <div class="card" id="edStatus"></div>
        <div class="card"><h3>按步骤提示</h3>${ex.steps.map(s => `<details class="px-tip"><summary>${s.n}. ${esc(s.title)}</summary>
          ${s.explain ? `<div class="prose small">${md(s.explain)}</div>` : ""}
          ${s.checks.length ? `<div class="small muted">评测会检查：</div><ul class="small px-checks">${s.checks.map(c => `<li>${esc(checkText(c))}</li>`).join("")}</ul>` : ""}
          <button class="btn sm ghost" data-peek="${s.id}">看这一步的参考答案</button><pre class="px-pre hidden" id="peek-${s.id}"><code class="hljs">${hl(s.answer)}</code></pre></details>`).join("")}</div>
      </div></div>`;
    const ta = $(".ed-ta", el), gut = $(".ed-gutter", el), hlBox = $(".ed-hl", el);
    const LH = 20;
    const lines = () => ta.value.split("\n").length;
    const paintGutter = () => {
      const n = lines(), err = L.lastFail && L.lastFail.line;
      gut.innerHTML = Array.from({ length: n }, (_, i) => `<span class="${err === i + 1 ? "err" : ""}">${i + 1}</span>`).join("\n");
      gut.scrollTop = ta.scrollTop;
      if (err) { hlBox.classList.remove("hidden"); hlBox.style.top = (10 + (err - 1) * LH - ta.scrollTop) + "px"; } else hlBox.classList.add("hidden");
    };
    let saveT;
    ta.addEventListener("input", () => { paintGutter(); clearTimeout(saveT); saveT = setTimeout(() => { L.code = ta.value; save(); }, 400); });
    ta.addEventListener("scroll", paintGutter);
    ta.addEventListener("keydown", e => editorKeys(e, ta, () => { paintGutter(); L.code = ta.value; save(); }, () => run()));
    paintGutter();
    UI.typeset(el);

    const status = $("#edStatus", el);
    const drawStatus = (res, busy) => {
      const passed = new Set((res && res.passed) || L.passed || []);
      const f = res ? res.failed : L.lastFail;
      status.innerHTML = `<h3>评测结果${busy ? ' <span class="small muted">运行中…</span>' : ""}</h3>
        ${serverOK === false ? `<div class="feedback bad">没有连接到本地评测服务。L3/L4 需要在本机运行代码：在项目根目录执行 <code>python3 lab/server.py</code>，并用 <a href="http://localhost:8766/${location.hash}">localhost:8766</a> 打开本页。</div>` : ""}
        <ol class="px-steps">${ex.steps.map(s => {
          const isF = f && f.step === s.id;
          return `<li class="${passed.has(s.id) ? "ok" : isF ? "bad" : ""}" data-jump="${s.id}"><span>${passed.has(s.id) ? "✔" : isF ? "✖" : "○"}</span>${esc(s.title)}</li>`;
        }).join("")}</ol>
        ${f ? `<div class="feedback bad"><b>${f.kind === "check" ? "检查没通过" : f.kind === "timeout" ? "超时" : f.kind === "structure" ? "结构有问题" : "运行出错"}</b>${f.title ? `（步骤：${esc(f.title)}${f.line ? `，第 ${f.line} 行` : ""}）` : ""}<div class="px-msg">${esc(f.message || "")}</div>${explainError(f.message)}</div>` : ""}
        ${L.done ? `<div class="feedback ok"><b>🎉 全部通过！</b>${level === 3 ? "可以挑战 L4：从零默写。" : "你已经能独立写出完整的训练脚本了。"}</div>
          <div class="row"><button class="btn primary" id="toLab">▶ 用你的代码在实验室完整训练</button>${level === 3 ? `<a class="btn" href="#/practice/${ex.id}?level=4">进入 L4 →</a>` : ""}</div>` : ""}
        ${res && res.output ? `<details style="margin-top:8px"><summary>运行输出（${res.seconds} s）</summary><pre class="logbox">${esc(res.output)}</pre></details>` : ""}`;
      $$("[data-jump]", status).forEach(li => li.onclick = () => jumpTo(li.dataset.jump));
      const tl = $("#toLab", status); if (tl) tl.onclick = () => runAnswerInLab(ex, ta.value, `练习 L${level} · ${ex.meta.title}`.slice(0, 70));
    };
    const jumpTo = sid => {
      const i = ta.value.split("\n").findIndex(l => l.includes(`[${sid}]`) && l.startsWith("# ="));
      if (i >= 0) { ta.focus(); const pos = ta.value.split("\n").slice(0, i).join("\n").length + (i ? 1 : 0); ta.setSelectionRange(pos, pos); ta.scrollTop = Math.max(0, i * LH - 40); paintGutter(); }
    };
    const stepAtCursor = () => {
      const before = ta.value.slice(0, ta.selectionStart).split("\n");
      for (let i = before.length - 1; i >= 0; i--) { const m = before[i].match(/^# =+ 步骤 \d+[：:].*\[(\w+)\]/); if (m) return m[1]; }
      return ex.steps[0].id;
    };
    const run = async upto => {
      if (!(await probe())) { drawStatus(null, false); return; }
      L.code = ta.value; save();
      drawStatus(null, true);
      $$("#edRun, #edUpto", el).forEach(b => b.disabled = true);
      try {
        const res = await api("/api/exercise/check", { id: ex.id, code: ta.value, upto: upto || undefined });
        L.passed = res.passed; L.lastFail = res.failed || null; L.tries = (L.tries || 0) + 1;
        if (res.ok && !upto) { L.done = true; L.doneAt = Date.now(); UI.toast("🎉 全部检查通过"); }
        save(); drawStatus(res, false); paintGutter();
        if (res.failed && res.failed.line) { const i = res.failed.line - 1; ta.scrollTop = Math.max(0, i * LH - 120); paintGutter(); }
      } catch (e) { UI.toast("评测失败：" + e.message); drawStatus(null, false); }
      finally { $$("#edRun, #edUpto", el).forEach(b => b.disabled = false); }
    };
    $("#edRun", el).onclick = () => run();
    $("#edUpto", el).onclick = () => run(stepAtCursor());
    $("#edReset", el).onclick = () => { if (!confirm("恢复成初始模板？你写的代码会被清空。")) return; ta.value = tpl; L.code = tpl; L.passed = []; L.lastFail = null; save(); paintGutter(); drawStatus(null, false); };
    $$("[data-peek]", el).forEach(b => b.onclick = () => { $("#peek-" + b.dataset.peek, el).classList.toggle("hidden"); (L.peeked = L.peeked || {})[b.dataset.peek] = true; save(); });
    probe().then(() => drawStatus(null, false));
    drawStatus(null, false);
  }

  function checkText(c) {      // 把检查代码变成一句人话：优先用 assert 的提示信息
    const m = c.match(/assert[^"]*?,\s*f?"([^"]+)"/);
    return m ? m[1].replace(/\{[^}]*\}/g, "…") : c;
  }
  function explainError(msg) {   // 常见报错的中文解释
    const M = [
      [/NameError: name '(\w+)' is not defined/, (m) => `变量或函数 <code>${m[1]}</code> 还没定义：检查拼写，或是否在前面的步骤里创建了它（变量名要和任务里给的一致）。`],
      [/AttributeError: .*has no attribute '(\w+)'/, (m) => `对象没有 <code>${m[1]}</code> 这个属性/方法：多半是拼写错误。`],
      [/shapes cannot be multiplied|size mismatch|mat1 and mat2/, () => "矩阵形状对不上：检查全连接层的输入大小，或前面是否忘了把图片拉平（flatten）。"],
      [/Expected all tensors to be on the same device/, () => "数据和模型不在同一个设备上：两边都要 <code>.to(device)</code>。"],
      [/IndentationError|unexpected indent|expected an indented block/, () => "缩进不对：Python 用缩进表示代码块，同一层的代码要对齐（4 个空格）。"],
      [/SyntaxError/, () => "语法错误：检查括号是否配对、冒号是否漏写。"],
      [/element 0 of tensors does not require grad/, () => "损失不需要梯度：通常是在 <code>torch.no_grad()</code> 里计算了损失，或者对 <code>.item()</code> 的结果调用了 backward。"],
      [/ModuleNotFoundError: No module named '(\w+)'/, (m) => `找不到模块 <code>${m[1]}</code>：检查 import 拼写。`],
    ];
    for (const [rx, f] of M) { const m = String(msg || "").match(rx); if (m) return `<div class="small px-why">💬 ${f(m)}</div>`; }
    return "";
  }

  function editorKeys(e, ta, after, runFn) {
    const v = ta.value, s = ta.selectionStart, en = ta.selectionEnd;
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); runFn(); return; }
    if (e.key === "Tab") {
      e.preventDefault();
      const ls = v.lastIndexOf("\n", s - 1) + 1;
      if (e.shiftKey) {
        const lineEnd = v.indexOf("\n", ls) === -1 ? v.length : v.indexOf("\n", ls);
        const line = v.slice(ls, lineEnd), n = Math.min(4, line.length - line.trimStart().length);
        ta.value = v.slice(0, ls) + line.slice(n) + v.slice(lineEnd);
        ta.selectionStart = ta.selectionEnd = Math.max(ls, s - n);
      } else if (s !== en) {                                  // 选中多行：整体缩进
        const block = v.slice(ls, en).split("\n").map(l => "    " + l).join("\n");
        ta.value = v.slice(0, ls) + block + v.slice(en);
        ta.selectionStart = ls; ta.selectionEnd = ls + block.length;
      } else { ta.value = v.slice(0, s) + "    " + v.slice(en); ta.selectionStart = ta.selectionEnd = s + 4; }
      after(); return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      const ls = v.lastIndexOf("\n", s - 1) + 1, line = v.slice(ls, s);
      let ind = line.match(/^\s*/)[0];
      if (/:\s*(#.*)?$/.test(line)) ind += "    ";
      e.preventDefault();
      ta.value = v.slice(0, s) + "\n" + ind + v.slice(en);
      ta.selectionStart = ta.selectionEnd = s + 1 + ind.length;
      after();
    }
  }

  window.Practice = { render, exState, levelDone };
})();
