/* 深度学习与视觉 · 双语学习站 — single-page app (works from file://) */
(function () {
  "use strict";
  const C = window.COURSE || { chapters: [] };
  const chapters = C.chapters;
  const byId = Object.fromEntries(chapters.map(c => [c.id, c]));
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
  const main = $("#main");
  const LETTERS = "ABCDEFGH";
  const DAY = 86400000;
  const INTERVALS = [0, 1, 3, 7, 14, 30]; // review interval (days) after reaching box 1..5

  /* ---------------- storage ---------------- */
  const KEY = "dlcv-study-v1";
  const store = {
    d: (() => {
      try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; }
    })(),
    save() { try { localStorage.setItem(KEY, JSON.stringify(this.d)); } catch (e) { /* private mode */ } },
    get(k, def) { if (!(k in this.d)) this.d[k] = def; return this.d[k]; },
  };
  const read = store.get("read", {});
  const answers = store.get("answers", {});
  const cards = store.get("cards", {});
  const prefs = store.get("prefs", { cardDir: "en", vocabMask: "none" });
  const notes = store.get("notes", {});          // 每节的个人笔记（费曼学习法）
  const activity = store.get("activity", {});    // 日期 -> 当天学习动作次数，用于连续天数与热力图
  const R = window.RESOURCES || { tracks: [], chapters: {}, tools: [], methods: [] };

  const dayKey = (t = Date.now()) => { const d = new Date(t); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
  function touch(n = 1) { const k = dayKey(); activity[k] = (activity[k] || 0) + n; store.save(); }
  function streak() {
    let n = 0, t = Date.now();
    if (!activity[dayKey(t)]) t -= 86400000;
    while (activity[dayKey(t)]) { n++; t -= 86400000; }
    return n;
  }
  const meta = ch => R.chapters[ch.id] || {};
  const isCore = (ch, i) => !meta(ch).core || meta(ch).core.includes(i);
  const applyFont = () => document.documentElement.style.setProperty("--fs", prefs.fontScale || 1);

  /* ---------------- helpers ---------------- */
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const strip = s => String(s || "").replace(/<[^>]+>/g, "");
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 1600);
  }
  function shuffle(a) { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
  function typeset(el) {
    if (window.renderMathInElement) {
      try {
        renderMathInElement(el, {
          delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
          throwOnError: false, ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "input"],
        });
      } catch (e) { console.warn(e); }
    }
    if (window.hljs) $$("pre code.language-python", el).forEach(b => { if (!b.dataset.hl) { hljs.highlightElement(b); b.dataset.hl = 1; } });
  }

  /* ---------------- 与 lab.js / codelib.js 共享的小工具 + 按需加载 ---------------- */
  const loaded = {};
  function loadScript(src) {
    if (!loaded[src]) loaded[src] = new Promise((res, rej) => {
      const el = document.createElement("script");
      el.src = src + "?v=" + SCRIPT_V; el.onload = res; el.onerror = () => { delete loaded[src]; rej(new Error("加载失败：" + src)); };
      document.head.appendChild(el);
    });
    return loaded[src];
  }
  const SCRIPT_V = "20261010d";
  window.StudyUI = { toast, esc, store, answers, typeset, loadScript };

  /* ---------------- audio ---------------- */
  const player = new Audio();
  let playingBtn = null;
  function play(src, btn) {
    if (!src) { toast("该条目暂无音频"); return; }
    if (playingBtn) playingBtn.classList.remove("playing");
    player.pause(); player.src = src; player.currentTime = 0;
    player.play().catch(() => toast("音频无法播放"));
    playingBtn = btn || null;
    if (btn) btn.classList.add("playing");
  }
  player.addEventListener("ended", () => { if (playingBtn) playingBtn.classList.remove("playing"); playingBtn = null; });
  document.addEventListener("click", e => {
    const b = e.target.closest("[data-audio]");
    if (b) { e.preventDefault(); e.stopPropagation(); play(b.dataset.audio, b); }
  });

  /* ---------------- ids & progress ---------------- */
  const qids = ch => [
    ...ch.quiz.mc.map((_, i) => `${ch.id}-mc-${i}`),
    ...ch.quiz.tf.map((_, i) => `${ch.id}-tf-${i}`),
    ...ch.quiz.fill.map((_, i) => `${ch.id}-fill-${i}`),
    ...ch.quiz.reading.flatMap((r, ri) => r.questions.map((_, j) => `${ch.id}-rd-${ri}-${j}`)),
  ];
  function deck(ch) {
    return [
      ...ch.vocab.map(v => ({ id: `${ch.id}-v-${v.id}`, type: "vocab", v, ch })),
      ...ch.cards.map((c, i) => ({ id: `${ch.id}-c-${i}`, type: "concept", c, ch })),
    ];
  }
  function progress(ch) {
    const r = ch.learn.filter(s => read[s.id]).length / Math.max(1, ch.learn.length);
    const ids = qids(ch);
    const q = ids.filter(id => answers[id] && answers[id].ok).length / Math.max(1, ids.length);
    const d = deck(ch);
    const m = d.filter(c => (cards[c.id] || {}).box >= 3).length / Math.max(1, d.length);
    return Math.round((0.4 * r + 0.4 * q + 0.2 * m) * 100);
  }
  function dueCards() {
    const now = Date.now();
    return chapters.flatMap(deck).filter(c => cards[c.id] && cards[c.id].due <= now);
  }
  function updateBadge() { const n = dueCards().length; $("#dueBadge").textContent = n ? n : ""; }

  /* ---------------- sidebar ---------------- */
  function groups() {
    const g = [];
    chapters.forEach(c => { let x = g.find(y => y.name === c.group); if (!x) g.push(x = { name: c.group, list: [] }); x.list.push(c); });
    return g;
  }
  function renderSidebar(activeId, page) {
    const q = (href, no, text, key) => `<a class="side-link ${page === key ? "active" : ""}" href="${href}"><span class="side-no">${no}</span>${text}</a>`;
    const trackIds = (R.tracks.find(t => t.id === prefs.track) || {}).chapters || [];
    $("#sidebar").innerHTML =
      `<div class="side-group"><a class="side-link ${!activeId && !page ? "active" : ""}" href="#/"><span class="side-no">⌂</span>课程首页</a>` +
      q("#/start", "🧭", "新手入门 · 学习路线", "start") + q("#/practice", "✍️", "代码练习 · 边学边写", "practice") + q("#/code", "🔎", "代码库 · 搜索训练代码", "code") +
      q("#/lab", "🧪", "训练实验室 · 本地训练对比", "lab") + q("#/tools", "🧰", "工具箱 · 笔记 · 番茄钟", "tools") + `</div>` +
      groups().map(g => `<div class="side-group"><div class="side-group-title">${esc(g.name)}</div>` +
        g.list.map(c => {
          const k = trackIds.indexOf(c.id);
          return `<a class="side-link ${c.id === activeId ? "active" : ""}" href="#/ch/${c.id}/learn">
          <span class="side-no">${String(c.no).padStart(2, "0")}</span><span>${esc(c.title)}</span>
          ${k >= 0 ? `<span class="side-step" title="我的路线中的第 ${k + 1} 步">${k + 1}</span>` : ""}
          <span class="side-prog"><i style="width:${progress(c)}%"></i></span></a>`;
        }).join("") + `</div>`).join("");
  }

  /* ---------------- home ---------------- */
  function renderHome() {
    const totals = {
      sections: chapters.reduce((s, c) => s + c.learn.length, 0),
      examples: chapters.reduce((s, c) => s + c.examples.length, 0),
      questions: chapters.reduce((s, c) => s + qids(c).length, 0),
      vocab: chapters.reduce((s, c) => s + c.vocab.length, 0),
      code: chapters.reduce((s, c) => s + c.learn.filter(x => x.code).length + c.examples.filter(x => x.code).length, 0),
    };
    const last = store.d.last && byId[store.d.last.ch] ? store.d.last : null;
    main.innerHTML = `
      <section class="hero">
        <h1>深度学习与计算机视觉 · 双语学习站</h1>
        <p>把 30 本经典教材合并成 ${chapters.length} 个主题章节：学习讲解、代码验证、例题解析、四类测验，以及带真人发音的中英术语与闪卡。</p>
        <div class="stats">
          <div class="stat"><b>${chapters.length}</b><span>章节 Chapters</span></div>
          <div class="stat"><b>${totals.sections}</b><span>知识小节 Sections</span></div>
          <div class="stat"><b>${totals.examples}</b><span>例题 Examples</span></div>
          <div class="stat"><b>${totals.questions}</b><span>测验题 Questions</span></div>
          <div class="stat"><b>${totals.vocab}</b><span>核心术语 Terms</span></div>
          <div class="stat"><b>${totals.code}</b><span>验证代码 Code</span></div>
        </div>
        <div class="row" style="margin-top:20px">
          ${last ? `<a class="btn btn-light" href="#/ch/${last.ch}/${last.tab}">继续学习：${esc(byId[last.ch].title)} →</a>` : ""}
          <a class="btn btn-ghost" href="#/review">今日闪卡复习（${dueCards().length}）</a>
          <a class="btn btn-ghost" href="#/start">🧭 新手入门：该从哪开始？</a>
        </div>
      </section>
      ${todayCard()}
      <div class="card">
        <h3>怎么用</h3>
        <div class="howto">
          <div><b>① 学习</b>每节先读讲解，再看要点细化，最后看代码和真实运行输出。可点“在浏览器运行”自己改代码试试。</div>
          <div><b>② 例题</b>先自己想，再逐步展开解析。</div>
          <div><b>③ 测验</b>选择、判断、填空和英文阅读理解，答完立即判分，错题自动进入错题本。</div>
          <div><b>④ 词汇与闪卡</b>中英互译、点按发音、遮挡自测，闪卡按记忆曲线安排复习。</div>
          <div><b>⑤ 卡住了？</b>正文里带虚线的术语可以悬停查看释义；每节有“求助”面板（AI 助教、视频搜索）和“资源”页的精选课程。</div>
          <div><b>⑥ 笔记与工具</b>每节可写笔记；工具箱里有番茄钟、字号调节、Anki 导出和进度备份。</div>
          <div><b>⑦ 找代码</b><a href="#/code">代码库</a>收录 700 多段教材代码：用中文提问（如“CIFAR-10 上训练 ResNet”），复制或一键放进实验室。</div>
          <div><b>⑧ 自己练</b><a href="#/lab">训练实验室</a>在本机训练 CNN：改参数跑几次，对比准确率和训练效率。数据集已预下载。</div>
          <div><b>⑨ 写代码</b><a href="#/practice">代码练习</a>把训练代码拆成“加载数据 → 模型 → 优化器 → 前向 → 反向 → step”等环节，从选择题练到默写完整脚本。</div>
        </div>
      </div>
      ${groups().map(g => `<h2 class="group-title">${esc(g.name)}</h2><div class="grid">` + g.list.map(c => `
        <a class="ch-card" href="#/ch/${c.id}/learn">
          <div class="row"><span class="tag">第 ${c.no} 章</span><span class="spacer"></span><span class="small muted">${progress(c)}%</span></div>
          <h3>${esc(c.title)}</h3><div class="en">${esc(c.titleEn)}</div>
          <p>${strip(c.summary)}</p>
          <div class="progress"><i style="width:${progress(c)}%"></i></div>
        </a>`).join("") + `</div>`).join("")}
      <p class="small muted" style="margin-top:30px">数据构建于 ${esc(C.builtAt || "")} · 学习进度保存在本浏览器的 localStorage 中。</p>`;
  }

  /* ---------------- chapter shell ---------------- */
  const TABS = [["learn", "学习", "Learn"], ["examples", "例题", "Examples"], ["quiz", "测验", "Quiz"], ["vocab", "词汇", "Vocabulary"], ["cards", "闪卡", "Flashcards"], ["res", "资源", "Resources"]];
  function metaBar(ch) {
    const m = meta(ch);
    if (!m.level) return "";
    const pre = (m.prereq || []).filter(id => byId[id]).map(id => `<a href="#/ch/${id}/learn">第 ${byId[id].no} 章</a>`).join("、") || "无硬性先修";
    const nCore = ch.learn.filter((_, i) => isCore(ch, i)).length;
    return `<div class="meta-bar">
      <div><b>难度</b><span class="stars" title="1 最易 – 5 最难">${"★".repeat(m.level)}${"☆".repeat(5 - m.level)}</span></div>
      <div><b>预计用时</b>约 ${m.hours} 小时</div>
      <div><b>建议先修</b>${pre}</div>
      <div><b>入门必读</b>${nCore} / ${ch.learn.length} 节</div>
      <div class="goal"><b>学完能做到</b>${esc(m.goal || "")}</div></div>`;
  }
  function renderChapter(ch, tab, sub) {
    store.d.last = { ch: ch.id, tab }; store.save();
    const idx = chapters.indexOf(ch);
    const prev = chapters[idx - 1], next = chapters[idx + 1];
    main.innerHTML = `
      <header class="ch-head">
        <div class="kicker">第 ${ch.no} 章 · ${esc(ch.group)}</div>
        <h1>${esc(ch.title)}</h1>
        <div class="en">${esc(ch.titleEn)}</div>
        <div class="summary prose">${ch.summary}</div>
        <div class="sources">${ch.sources.map(s => `<span>📖 ${esc(s)}</span>`).join("")}</div>
        ${metaBar(ch)}
      </header>
      <nav class="tabs">${TABS.map(([k, zh, en]) => `<a class="tab ${k === tab ? "active" : ""}" href="#/ch/${ch.id}/${k}">${zh}<small>${en}</small></a>`).join("")}</nav>
      <div id="tabBody"></div>
      <div class="pager">
        ${prev ? `<a class="btn" href="#/ch/${prev.id}/${tab}">← ${esc(prev.title)}</a>` : "<span></span>"}
        ${next ? `<a class="btn" href="#/ch/${next.id}/${tab}">${esc(next.title)} →</a>` : "<span></span>"}
      </div>`;
    const body = $("#tabBody");
    ({ learn: renderLearn, examples: renderExamples, quiz: renderQuiz, vocab: renderVocab, cards: renderCards, res: renderRes }[tab] || renderLearn)(ch, body);
    typeset(main);
    if (tab === "learn" || tab === "examples") glossify(body);
    if (sub) {
      const el = document.getElementById(sub);
      if (el) { el.classList.remove("collapsed"); setTimeout(() => el.scrollIntoView({ behavior: "smooth", block: "start" }), 50); }
    }
  }

  /* ---------------- learn ---------------- */
  function codeBox(code, key) {
    return `<div class="code-box" data-key="${key}">
      <div class="code-head"><span class="fname">${esc(code.name)}</span><span class="spacer"></span>
        <button class="btn sm ghost" data-copy="${key}">复制</button>
        <button class="btn sm" data-run="${key}">▶ 在浏览器运行</button></div>
      <pre><code class="language-python" contenteditable="true" spellcheck="false">${esc(code.source)}</code></pre>
      <div class="code-out"><div class="label">✔ 构建时已运行验证的输出 Verified output</div><pre>${esc(code.output || "(无输出)")}</pre></div>
    </div>`;
  }
  const codeIndex = {};
  const AI_TARGETS = { claude: "https://claude.ai/new?q=", gpt: "https://chatgpt.com/?q=" };
  const aiPrompts = {};
  function buildPrompts(ch, s, i) {
    const ctx = `《${ch.title}》第 ${i + 1} 节「${s.title} / ${s.titleEn}」`;
    const mine = (notes[s.id] || "").trim();
    return [
      { label: "大白话讲一遍", text: `我是机器学习初学者，正在学习${ctx}。请用不超过 300 字的大白话和一个生活中的类比，解释这一节最核心的思想，先不要堆公式；最后告诉我学这一节之前我需要先懂哪些东西。` },
      { label: "出题考考我", text: `请围绕${ctx}出 5 道由易到难的练习题（其中 2 道需要动手计算或写代码）。先只给题目，等我作答后再逐题批改并讲解。` },
      { label: "检查我的理解", text: `我正在学习${ctx}。下面是我用自己的话写的总结，请指出其中的错误或遗漏，并给出更准确的说法：\n\n${mine || "（在这里写下你的总结）"}` },
      { label: "写个最小代码示例", text: `请用 NumPy（不使用深度学习框架）为${ctx}写一个不超过 30 行、可直接运行的最小示例，逐行加注释，并告诉我改哪些参数可以观察到什么现象。` },
    ];
  }
  function helperBox(ch, s, i) {
    aiPrompts[s.id] = { ch, s, i };
    const q = encodeURIComponent;
    return `<details class="helper"><summary>🆘 没看懂？让 AI 助教或视频来帮你</summary>
      <div class="helper-body">
        <div class="small muted">点击后会在新标签页打开对应网站并带上提示词（只包含章节名称，不含你的个人数据，除非你写了笔记并点“检查我的理解”）。也可以复制提示词，粘贴给任意 AI 工具。</div>
        ${buildPrompts(ch, s, i).map((p, k) => `<div class="ai-row"><span>${p.label}</span><span class="spacer"></span>
          <button class="btn sm" data-ai="claude" data-sec="${s.id}" data-k="${k}">Claude</button>
          <button class="btn sm" data-ai="gpt" data-sec="${s.id}" data-k="${k}">ChatGPT</button>
          <button class="btn sm ghost" data-ai="copy" data-sec="${s.id}" data-k="${k}">复制</button></div>`).join("")}
        <div class="ai-row"><span>找视频 / 文章</span><span class="spacer"></span>
          <a class="btn sm" target="_blank" rel="noopener" href="https://www.youtube.com/results?search_query=${q(s.titleEn)}">YouTube</a>
          <a class="btn sm" target="_blank" rel="noopener" href="https://search.bilibili.com/all?keyword=${q(s.title)}">B 站</a>
          <a class="btn sm" target="_blank" rel="noopener" href="https://www.google.com/search?q=${q(s.titleEn + " explained tutorial")}">Google</a>
          <a class="btn sm" target="_blank" rel="noopener" href="https://zh.wikipedia.org/w/index.php?search=${q(s.title)}">维基</a></div>
      </div></details>`;
  }
  function notesBox(s) {
    const v = notes[s.id] || "";
    return `<details class="notes-box" ${v ? "open" : ""}><summary>✍️ 我的笔记${v ? "（已写）" : ""} <span class="small muted">用自己的话讲一遍，写不出来就是还没懂</span></summary>
      <textarea data-note="${s.id}" rows="4" placeholder="例如：这一节想解决什么问题？核心公式的含义？我最容易搞混的地方？">${esc(v)}</textarea></details>`;
  }
  function renderLearn(ch, el) {
    const coreOnly = !!prefs.coreOnly;
    el.innerHTML =
      `<div class="learn-tools card">
        <label class="read-toggle"><input type="checkbox" id="coreOnly" ${coreOnly ? "checked" : ""}>只看“入门必读”（进阶小节折叠）</label>
        <span class="spacer"></span><span class="small muted">第一遍建议只读必读小节，建立整体印象后再回头读进阶部分。</span></div>` +
      `<div class="toc">${ch.learn.map((s, i) => `<a href="#/ch/${ch.id}/learn/${s.id}" class="${read[s.id] ? "done" : ""} ${isCore(ch, i) ? "" : "adv"}">${i + 1}. ${esc(s.title)}${isCore(ch, i) ? "" : " · 进阶"}</a>`).join("")}</div>` +
      ch.learn.map((s, i) => {
        if (s.code) codeIndex[s.id] = s.code;
        const core = isCore(ch, i);
        return `<article class="card section ${!core && coreOnly ? "collapsed" : ""}" id="${s.id}">
          <h2><span class="num">${ch.no}.${i + 1}</span>${esc(s.title)}<span class="en">${esc(s.titleEn)}</span>
            <span class="lvl ${core ? "core" : "adv"}">${core ? "入门必读" : "进阶选读"}</span></h2>
          <button class="btn sm expand-btn" data-expand="${s.id}">进阶内容已折叠，点击展开 ▾</button>
          <div class="sec-body">
          <div class="block-title">讲解 Explanation</div>
          <div class="prose">${s.body}</div>
          ${s.points.length ? `<div class="block-title">要点细化 Key points</div><ul class="points">${s.points.map(p => `<li>${p}</li>`).join("")}</ul>` : ""}
          ${s.code ? `<div class="block-title">代码验证 Code check</div>${s.codeNote ? `<p class="small muted">${s.codeNote}</p>` : ""}${codeBox(s.code, s.id)}` : ""}
          ${helperBox(ch, s, i)}
          ${notesBox(s)}
          <div class="row" style="margin-top:14px">
            ${s.src ? `<div class="src-line">📖 出处：${esc(s.src)}</div>` : ""}<span class="spacer"></span>
            <label class="read-toggle"><input type="checkbox" data-read="${s.id}" ${read[s.id] ? "checked" : ""}>已掌握本节</label>
          </div></div>
        </article>`;
      }).join("") +
      `<div class="card row"><span>学完了？</span><span class="spacer"></span><a class="btn" href="#/ch/${ch.id}/examples">去做例题 →</a><a class="btn" href="#/ch/${ch.id}/res">更多学习资源 →</a><a class="btn primary" href="#/ch/${ch.id}/quiz">开始测验 →</a></div>`;
    $("#coreOnly", el).addEventListener("change", e => {
      prefs.coreOnly = e.target.checked; store.save();
      const y = window.scrollY; renderLearn(ch, el); typeset(el); glossify(el); window.scrollTo(0, y);
    });
    $$("[data-expand]", el).forEach(b => b.addEventListener("click", () => b.closest("article").classList.remove("collapsed")));
    $$("[data-read]", el).forEach(cb => cb.addEventListener("change", () => {
      read[cb.dataset.read] = cb.checked; store.save(); if (cb.checked) touch(); renderSidebar(ch.id);
      const a = $(`.toc a[href$="${cb.dataset.read}"]`, el); if (a) a.classList.toggle("done", cb.checked);
    }));
    $$("[data-ai]", el).forEach(b => b.addEventListener("click", () => {
      const x = aiPrompts[b.dataset.sec];
      const p = buildPrompts(x.ch, x.s, x.i)[+b.dataset.k];   // 每次点击重新生成，带上最新笔记
      if (b.dataset.ai === "copy") { navigator.clipboard.writeText(p.text).then(() => toast("提示词已复制"), () => toast("复制失败")); return; }
      window.open(AI_TARGETS[b.dataset.ai] + encodeURIComponent(p.text), "_blank", "noopener");
      touch();
    }));
    $$("[data-note]", el).forEach(t => {
      let timer;
      t.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(() => { const v = t.value.trim(); if (v) notes[t.dataset.note] = t.value; else delete notes[t.dataset.note]; store.save(); }, 400);
      });
      t.addEventListener("change", () => { if (t.value.trim()) touch(); });
    });
    bindCode(el);
    loadScript("assets/js/exercises-data.js").then(() => {      // 这一章有配套代码练习就显示入口
      const list = (window.EXERCISES || { items: [] }).items.filter(e => e.meta.chapter === ch.id);
      if (!list.length || !document.body.contains(el) || $(".px-cta", el)) return;
      const box = document.createElement("div");
      box.className = "card px-cta";
      box.innerHTML = `<h3>✍️ 动手练习</h3><p class="small muted" style="margin:0 0 8px">把这一章的训练代码拆成固定环节，自己在关键位置写代码：选择 → 填空 → 写整块 → 默写完整脚本。</p>` +
        list.map(e => `<a class="result" href="#/practice/${e.id}"><b>${esc(e.meta.title)}</b><div class="small muted">${esc(e.meta.goal || "")}</div></a>`).join("");
      el.appendChild(box);
    }, () => { });
  }

  /* ---------------- code: copy & run (Pyodide, lazy) ---------------- */
  let pyodideP = null;
  function loadPyodide_() {
    if (pyodideP) return pyodideP;
    pyodideP = new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
      s.onload = () => window.loadPyodide().then(res, rej);
      s.onerror = () => { pyodideP = null; rej(new Error("无法加载 Pyodide（需要联网）")); };
      document.head.appendChild(s);
    });
    return pyodideP;
  }
  function bindCode(el) {
    $$("[data-copy]", el).forEach(b => b.addEventListener("click", () => {
      const code = $("code", b.closest(".code-box")).innerText;
      navigator.clipboard.writeText(code).then(() => toast("已复制代码"), () => toast("复制失败"));
    }));
    $$("[data-run]", el).forEach(b => b.addEventListener("click", async () => {
      const box = b.closest(".code-box");
      const src = $("code", box).innerText;
      const out = $(".code-out", box);
      out.innerHTML = `<div class="label live">⏳ 正在浏览器中加载 Python（首次约需 10–20 秒）…</div><pre></pre>`;
      b.disabled = true;
      try {
        const py = await loadPyodide_();
        if (/import numpy|from numpy/.test(src)) await py.loadPackage("numpy");
        let buf = "";
        py.setStdout({ batched: s => { buf += s + "\n"; } });
        py.setStderr({ batched: s => { buf += s + "\n"; } });
        await py.runPythonAsync(src);
        out.innerHTML = `<div class="label live">▶ 浏览器实时运行结果 Live output</div><pre>${esc(buf.trimEnd() || "(无输出)")}</pre>`;
      } catch (e) {
        out.innerHTML = `<div class="label" style="color:var(--bad)">✖ 运行出错</div><pre>${esc(String(e.message || e))}</pre>`;
      } finally { b.disabled = false; }
    }));
  }

  /* ---------------- examples ---------------- */
  function renderExamples(ch, el) {
    el.innerHTML = ch.examples.map((e, i) => `
      <article class="card example" id="${e.id}">
        <h2>例题 ${i + 1}　${esc(e.title)}</h2>
        <div class="problem prose">${e.problem}</div>
        <div class="row"><button class="btn primary" data-step="${e.id}">展开下一步</button><button class="btn" data-all="${e.id}">显示全部解析</button><span class="spacer"></span><span class="small muted">${e.steps.length} 步</span></div>
        <div class="steps" style="margin-top:12px">${e.steps.map(s => `<div class="step prose hidden">${s}</div>`).join("")}</div>
        ${e.answer ? `<div class="answer-box prose hidden"><b>答案：</b>${e.answer}</div>` : ""}
        ${e.code ? `<div class="hidden code-wrap"><div class="block-title">代码验证 Code check</div>${codeBox(e.code, e.id)}</div>` : ""}
      </article>`).join("");
    const reveal = (art, all) => {
      const hid = $$(".step.hidden", art);
      (all ? hid : hid.slice(0, 1)).forEach(x => x.classList.remove("hidden"));
      if ($$(".step.hidden", art).length === 0) {
        $$(".answer-box, .code-wrap", art).forEach(x => x.classList.remove("hidden"));
        const b = $("[data-step]", art); b.textContent = "解析已全部展开"; b.disabled = true;
      }
    };
    $$("[data-step]", el).forEach(b => b.addEventListener("click", () => reveal(b.closest("article"), false)));
    $$("[data-all]", el).forEach(b => b.addEventListener("click", () => reveal(b.closest("article"), true)));
    bindCode(el);
  }

  /* ---------------- quiz ---------------- */
  const norm = s => String(s).toLowerCase().replace(/[\s　]+/g, "").replace(/[.,;:!?。，；：！？'"“”‘’()（）-]/g, "");
  function feedback(ok, explain, correctText) {
    return `<div class="feedback ${ok ? "ok" : "bad"}"><b>${ok ? "✔ 回答正确" : "✖ 回答错误"}</b> ${!ok && correctText ? `正确答案：${correctText}。` : ""}${explain || ""}</div>`;
  }
  function optButtons(qid, opts, ans) {
    const a = answers[qid];
    return `<div class="opts">${opts.map((o, i) => {
      let cls = "";
      if (a) { if (i === ans) cls = "correct"; else if (i === a.val) cls = "wrong"; }
      return `<button class="opt ${cls}" data-q="${qid}" data-i="${i}" ${a ? "disabled" : ""}><span class="letter">${LETTERS[i]}</span><span>${o}</span></button>`;
    }).join("")}</div>`;
  }
  function renderQuiz(ch, el) {
    const Q = ch.quiz; let n = 0;
    const mcHtml = Q.mc.map((m, i) => {
      const id = `${ch.id}-mc-${i}`; const a = answers[id]; n++;
      return `<div class="q" id="${id}"><div class="q-text"><span class="qn">${n}.</span>${m.q}</div>${optButtons(id, m.options, m.answer)}
        ${a ? feedback(a.ok, m.explain, LETTERS[m.answer]) : ""}</div>`;
    }).join("");
    const tfHtml = Q.tf.map((t, i) => {
      const id = `${ch.id}-tf-${i}`; const a = answers[id]; n++;
      const cls = v => !a ? "" : (v === t.answer ? "correct" : (a.val === v ? "wrong" : ""));
      return `<div class="q" id="${id}"><div class="q-text"><span class="qn">${n}.</span>${t.q}</div>
        <div class="tf-row"><button class="opt ${cls(true)}" data-tf="${id}" data-v="1" ${a ? "disabled" : ""}>✔ 正确 True</button>
        <button class="opt ${cls(false)}" data-tf="${id}" data-v="0" ${a ? "disabled" : ""}>✖ 错误 False</button></div>
        ${a ? feedback(a.ok, t.explain, t.answer ? "正确" : "错误") : ""}</div>`;
    }).join("");
    const fillHtml = Q.fill.map((f, i) => {
      const id = `${ch.id}-fill-${i}`; const a = answers[id]; n++;
      return `<div class="q" id="${id}"><div class="q-text"><span class="qn">${n}.</span>${f.q.replace("____", '<span class="blank"></span>')}</div>
        <div class="fill-row"><input data-fill="${id}" placeholder="输入答案（中英文均可）" value="${a ? esc(a.val) : ""}" class="${a ? (a.ok ? "correct" : "wrong") : ""}" ${a ? "disabled" : ""}>
        <button class="btn primary" data-fillbtn="${id}" ${a ? "disabled" : ""}>提交</button></div>
        ${a ? feedback(a.ok, f.explain, f.answer.join(" / ")) : ""}</div>`;
    }).join("");
    const rdHtml = Q.reading.map((r, ri) => `
      <div class="card"><h3>📰 ${esc(r.title)}</h3>
        <div class="row" style="margin-bottom:10px">${r.audio ? `<button class="audio-btn" data-audio="${r.audio}">🔊 朗读全文 Listen</button>` : ""}
          <button class="btn sm" data-zh="${ch.id}-rd-${ri}">显示中文译文</button></div>
        <div class="passage">${r.passage}</div>
        <div class="passage-zh prose hidden" id="zh-${ch.id}-rd-${ri}">${r.zh}</div>
        <div style="margin-top:14px">${r.questions.map((m, j) => {
          const id = `${ch.id}-rd-${ri}-${j}`; const a = answers[id]; n++;
          return `<div class="q" id="${id}"><div class="q-text"><span class="qn">${n}.</span>${m.q}</div>${optButtons(id, m.options, m.answer)}
            ${a ? feedback(a.ok, m.explain, LETTERS[m.answer]) : ""}</div>`;
        }).join("")}</div></div>`).join("");
    const ids = qids(ch);
    const done = ids.filter(id => answers[id]).length, right = ids.filter(id => answers[id] && answers[id].ok).length;
    el.innerHTML = `
      <div class="card score-bar"><div><div class="small muted">得分 Score</div><div class="big">${right} / ${ids.length}</div></div>
        <div class="spacer"></div><div class="small muted">已作答 ${done} 题${done === ids.length ? ` · 正确率 ${Math.round(100 * right / ids.length)}%` : ""}</div>
        <button class="btn" id="resetQuiz">重做本章测验</button></div>
      <section class="q-section"><h2>一、选择题 Multiple choice</h2>${mcHtml}</section>
      <section class="q-section"><h2>二、判断题 True or false</h2>${tfHtml}</section>
      <section class="q-section"><h2>三、填空题 Fill in the blank</h2>${fillHtml}</section>
      <section class="q-section"><h2>四、阅读理解 Reading comprehension</h2>${rdHtml}</section>`;
    const record = (id, ok, val) => { answers[id] = { ok, val, t: Date.now() }; store.save(); touch(); };
    const rerender = (id) => { const y = window.scrollY; renderQuiz(ch, el); typeset(el); window.scrollTo(0, y); renderSidebar(ch.id); };
    $$("[data-q]", el).forEach(b => b.addEventListener("click", () => {
      const id = b.dataset.q, i = +b.dataset.i;
      const parts = id.split("-");
      let ans;
      if (parts[1] === "mc") ans = Q.mc[+parts[2]].answer; else ans = Q.reading[+parts[2]].questions[+parts[3]].answer;
      record(id, i === ans, i); rerender(id);
    }));
    $$("[data-tf]", el).forEach(b => b.addEventListener("click", () => {
      const id = b.dataset.tf, v = b.dataset.v === "1";
      record(id, v === Q.tf[+id.split("-")[2]].answer, v); rerender(id);
    }));
    const submitFill = id => {
      const inp = $(`[data-fill="${id}"]`, el); const v = inp.value.trim(); if (!v) { inp.focus(); return; }
      const f = Q.fill[+id.split("-")[2]];
      record(id, f.answer.some(x => norm(x) === norm(v)), v); rerender(id);
    };
    $$("[data-fillbtn]", el).forEach(b => b.addEventListener("click", () => submitFill(b.dataset.fillbtn)));
    $$("[data-fill]", el).forEach(i => i.addEventListener("keydown", e => { if (e.key === "Enter") submitFill(i.dataset.fill); }));
    $$("[data-zh]", el).forEach(b => b.addEventListener("click", () => {
      const z = $("#zh-" + b.dataset.zh); z.classList.toggle("hidden"); b.textContent = z.classList.contains("hidden") ? "显示中文译文" : "隐藏中文译文";
    }));
    $("#resetQuiz", el).addEventListener("click", () => {
      if (!confirm("清空本章所有作答记录并重做？")) return;
      ids.forEach(id => delete answers[id]); store.save(); rerender();
    });
  }

  /* ---------------- vocab ---------------- */
  function audioBtns(v) {
    const a = v.audio || {};
    return `<div class="audio-btns">${a.en ? `<button class="audio-btn" data-audio="${a.en}">🔊 EN</button>` : ""}${a.zh ? `<button class="audio-btn" data-audio="${a.zh}">🔊 中文</button>` : ""}${a.ex ? `<button class="audio-btn" data-audio="${a.ex}">🔊 例句</button>` : ""}</div>`;
  }
  function renderVocab(ch, el) {
    const mask = prefs.vocabMask;
    el.innerHTML = `
      <div class="vocab-tools"><span class="small muted">遮挡自测：</span>
        <div class="seg" id="maskSeg">${[["none", "全部显示"], ["zh", "遮住中文"], ["en", "遮住英文"]].map(([k, t]) => `<button data-mask="${k}" class="${mask === k ? "on" : ""}">${t}</button>`).join("")}</div>
        <span class="spacer"></span><a class="btn" href="#/ch/${ch.id}/cards">用闪卡记忆 →</a></div>
      <div class="vocab-grid">${ch.vocab.map(v => `
        <div class="v-card" id="v-${v.id}">
          <div class="v-top"><span class="v-en ${mask === "en" ? "masked" : ""}">${esc(v.en)}</span><span class="v-zh ${mask === "zh" ? "masked" : ""}">${esc(v.zh)}</span></div>
          <div class="v-def">${esc(v.defEn)}</div>
          <div class="v-def zh ${mask === "zh" ? "masked" : ""}">${esc(v.defZh)}</div>
          <div class="v-ex">“${esc(v.example)}”</div>
          ${audioBtns(v)}
        </div>`).join("")}</div>
      <div class="card drill" id="drill"></div>`;
    $$("[data-mask]", el).forEach(b => b.addEventListener("click", () => { prefs.vocabMask = b.dataset.mask; store.save(); renderVocab(ch, el); }));
    $$(".masked", el).forEach(m => m.addEventListener("click", () => m.classList.remove("masked")));
    renderDrill(ch, $("#drill", el));
  }
  function renderDrill(ch, el, state) {
    state = state || { dir: "en2zh", score: 0, total: 0 };
    const pool = ch.vocab;
    if (pool.length < 4) { el.innerHTML = ""; return; }
    const target = pool[Math.floor(Math.random() * pool.length)];
    const opts = shuffle([target, ...shuffle(pool.filter(v => v !== target)).slice(0, 3)]);
    const en2zh = state.dir === "en2zh";
    el.innerHTML = `<div class="row"><h3 style="margin:0">中英互译练习 Translation drill</h3><span class="spacer"></span>
        <div class="seg"><button data-dir="en2zh" class="${en2zh ? "on" : ""}">英 → 中</button><button data-dir="zh2en" class="${!en2zh ? "on" : ""}">中 → 英</button></div>
        <span class="small muted">本轮 ${state.score}/${state.total}</span></div>
      <div class="prompt">${esc(en2zh ? target.en : target.zh)} ${target.audio && target.audio[en2zh ? "en" : "zh"] ? `<button class="audio-btn" data-audio="${target.audio[en2zh ? "en" : "zh"]}">🔊</button>` : ""}</div>
      <div class="opts">${opts.map((o, i) => `<button class="opt" data-pick="${i}"><span class="letter">${LETTERS[i]}</span><span>${esc(en2zh ? o.zh : o.en)}</span></button>`).join("")}</div>
      <div id="drillFb"></div>`;
    $$("[data-dir]", el).forEach(b => b.addEventListener("click", () => renderDrill(ch, el, { dir: b.dataset.dir, score: 0, total: 0 })));
    $$("[data-pick]", el).forEach(b => b.addEventListener("click", () => {
      const ok = opts[+b.dataset.pick] === target;
      $$("[data-pick]", el).forEach((x, i) => { x.disabled = true; if (opts[i] === target) x.classList.add("correct"); else if (x === b) x.classList.add("wrong"); });
      state.total++; if (ok) state.score++;
      if (target.audio && target.audio.en) play(target.audio.en);
      $("#drillFb", el).innerHTML = `<div class="feedback ${ok ? "ok" : "bad"}"><b>${ok ? "✔ 正确" : "✖ 再记一下"}</b>${esc(target.en)} = ${esc(target.zh)}。${esc(target.defZh)}</div>
        <div class="row" style="margin-top:10px"><span class="spacer"></span><button class="btn primary" id="nextDrill">下一题 →</button></div>`;
      $("#nextDrill", el).addEventListener("click", () => renderDrill(ch, el, state));
    }));
  }

  /* ---------------- flashcards ---------------- */
  function faceHtml(card, back) {
    if (card.type === "vocab") {
      const v = card.v, enFirst = prefs.cardDir === "en";
      const showEn = back ? !enFirst : enFirst;
      if (!back) return `<span class="kind tag">术语 Term</span><div class="main-term">${esc(showEn ? v.en : v.zh)}</div>
          ${audioBtns({ audio: showEn ? { en: v.audio.en } : { zh: v.audio.zh } })}<div class="hint">点击卡片或按空格翻面</div>`;
      return `<div class="main-term">${esc(showEn ? v.en : v.zh)}</div><div class="sub" style="font-size:16px">${esc(showEn ? v.zh : v.en)}</div><div class="sub">${esc(v.defZh)}</div><div class="sub">${esc(v.defEn)}</div>
          <div class="sub" style="font-style:italic">“${esc(v.example)}”</div>${audioBtns(v)}`;
    }
    return back ? `<div class="prose" style="font-size:17px">${card.c.back}</div>`
      : `<span class="kind tag">概念 Concept</span><div class="main-term" style="font-size:22px">${card.c.front}</div><div class="hint">点击卡片或按空格翻面</div>`;
  }
  function flashSession(list, el, title, onExit) {
    let queue = list.slice(), flipped = false, doneN = 0;
    const total = queue.length;
    function grade(card, good) {
      const s = cards[card.id] || { box: 0, due: 0 };
      s.box = good ? Math.min(5, (s.box || 0) + 1) : 1;
      s.due = good ? Date.now() + INTERVALS[s.box] * DAY : Date.now();   // 没记住：立即到期，本轮稍后再出现
      cards[card.id] = s; store.save(); touch();
      queue.shift();
      if (good) doneN++; else queue.push(card);
      flipped = false; draw(); updateBadge();
    }
    function draw() {
      if (!queue.length) {
        el.innerHTML = `<div class="card" style="text-align:center"><h2>🎉 本轮完成</h2><p class="muted">共 ${total} 张卡片已全部记住。系统会按记忆曲线安排下次复习。</p>
          <div class="row" style="justify-content:center"><button class="btn primary" id="again">再来一轮</button>${onExit ? `<button class="btn" id="exit">返回</button>` : ""}</div></div>`;
        $("#again", el).addEventListener("click", () => flashSession(shuffle(list), el, title, onExit));
        if (onExit) $("#exit", el).addEventListener("click", onExit);
        return;
      }
      const card = queue[0];
      const box = (cards[card.id] || {}).box || 0;
      el.innerHTML = `<div class="fc-wrap">
        <div class="fc-meta"><b>${esc(title)}</b><span class="tag">已记住 ${doneN} / ${total}</span><span class="small muted">${card.ch.title} · 盒子 ${box || "新卡"}</span>
          <div class="seg"><button data-cd="en" class="${prefs.cardDir === "en" ? "on" : ""}">英 → 中</button><button data-cd="zh" class="${prefs.cardDir === "zh" ? "on" : ""}">中 → 英</button></div></div>
        <div class="flashcard ${flipped ? "flipped" : ""}" id="fc"><div class="fc-inner">
          <div class="fc-face fc-front">${faceHtml(card, false)}</div><div class="fc-face fc-back">${faceHtml(card, true)}</div></div></div>
        <div class="fc-actions"><button class="btn again" id="bAgain">✖ 没记住 <span class="kbd">1</span></button>
          <button class="btn" id="bFlip">翻面 <span class="kbd">空格</span></button>
          <button class="btn good" id="bGood">✔ 记住了 <span class="kbd">2</span></button>
          <button class="btn ghost" id="bSkip">跳过 <span class="kbd">→</span></button></div>
        <div class="boxes">${[1, 2, 3, 4, 5].map(b => `<span>盒${b}: ${list.filter(c => (cards[c.id] || {}).box === b).length}</span>`).join("")}</div></div>`;
      const fc = $("#fc", el);
      const flip = () => { flipped = !flipped; fc.classList.toggle("flipped", flipped); };
      fc.addEventListener("click", e => { if (!e.target.closest("[data-audio]")) flip(); });
      $("#bFlip", el).addEventListener("click", flip);
      $("#bAgain", el).addEventListener("click", () => grade(card, false));
      $("#bGood", el).addEventListener("click", () => grade(card, true));
      $("#bSkip", el).addEventListener("click", () => { queue.push(queue.shift()); flipped = false; draw(); });
      $$("[data-cd]", el).forEach(b => b.addEventListener("click", () => { prefs.cardDir = b.dataset.cd; store.save(); flipped = false; draw(); }));
      typeset(el);
    }
    flashSession.key = e => {
      if (!document.body.contains(el) || !queue.length || /INPUT|TEXTAREA/.test(document.activeElement.tagName)) return;
      if (e.code === "Space") { e.preventDefault(); $("#bFlip", el).click(); }
      else if (e.key === "1") $("#bAgain", el).click();
      else if (e.key === "2") $("#bGood", el).click();
      else if (e.key === "ArrowRight") $("#bSkip", el).click();
    };
    draw();
  }
  document.addEventListener("keydown", e => { if (flashSession.key) flashSession.key(e); });

  function renderCards(ch, el) {
    const d = deck(ch);
    const now = Date.now();
    const ordered = d.slice().sort((a, b) => {
      const sa = cards[a.id], sb = cards[b.id];
      const ka = !sa ? 1 : (sa.due <= now ? 0 : 2), kb = !sb ? 1 : (sb.due <= now ? 0 : 2);
      return ka - kb || ((sa || {}).box || 0) - ((sb || {}).box || 0);
    });
    flashSession(ordered, el, `${ch.title} · ${d.length} 张`);
  }
  function renderReview() {
    renderSidebar(null, "other");
    const due = dueCards();
    const fresh = chapters.flatMap(deck).filter(c => !cards[c.id]);
    main.innerHTML = `<div class="ch-head"><div class="kicker">间隔重复 Spaced repetition</div><h1>闪卡复习</h1>
      <p class="muted">到期卡片 ${due.length} 张 · 未学新卡 ${fresh.length} 张。“记住了”会让卡片进入下一个盒子，盒子 1–5 的复习间隔依次为 1、3、7、14、30 天；“没记住”会让卡片回到盒子 1 并立即到期。</p></div>
      <div class="row" style="margin-bottom:16px"><button class="btn primary" id="rvDue" ${due.length ? "" : "disabled"}>复习到期卡片（${due.length}）</button>
      <button class="btn" id="rvNew" ${fresh.length ? "" : "disabled"}>学习 20 张新卡</button><button class="btn" id="rvAll">全部卡片随机复习</button></div><div id="rvBody"></div>`;
    const body = $("#rvBody");
    const back = () => renderReview();
    $("#rvDue").addEventListener("click", () => flashSession(shuffle(due), body, "到期复习", back));
    $("#rvNew").addEventListener("click", () => flashSession(shuffle(fresh).slice(0, 20), body, "新卡学习", back));
    $("#rvAll").addEventListener("click", () => flashSession(shuffle(chapters.flatMap(deck)).slice(0, 40), body, "随机复习 40 张", back));
  }

  /* ---------------- mistakes ---------------- */
  function findQ(id) {
    if (id.startsWith("px:")) {
      const [, exId, bid] = id.split(":");
      const ex = window.EXERCISES && window.EXERCISES.items.find(e => e.id === exId);
      if (!ex) return null;
      const st = ex.steps.find(s => s.blanks.some(b => b.id === bid)); const b = st && st.blanks.find(x => x.id === bid);
      if (!b) return null;
      const ch = byId[ex.meta.chapter] || { no: "练", title: "代码练习" };
      return { ch, q: `代码练习《${esc(ex.meta.title)}》· ${esc(st.title)}：${esc(b.hint)}`, ans: `<code>${esc(b.mask)}</code>`, explain: esc(b.why), redo: `#/practice/${ex.id}?level=1#st-${st.id}` };
    }
    const p = id.split("-"); const ch = byId[p[0]]; if (!ch) return null;
    const Q = ch.quiz;
    if (p[1] === "mc") { const m = Q.mc[+p[2]]; return m && { ch, q: m.q, ans: m.options[m.answer], explain: m.explain }; }
    if (p[1] === "tf") { const t = Q.tf[+p[2]]; return t && { ch, q: t.q, ans: t.answer ? "正确" : "错误", explain: t.explain }; }
    if (p[1] === "fill") { const f = Q.fill[+p[2]]; return f && { ch, q: f.q, ans: f.answer.join(" / "), explain: f.explain }; }
    if (p[1] === "rd") { const r = Q.reading[+p[2]]; const m = r && r.questions[+p[3]]; return m && { ch, q: m.q, ans: m.options[m.answer], explain: m.explain }; }
    return null;
  }
  function renderMistakes() {
    renderSidebar(null, "other");
    if (!window.EXERCISES && Object.keys(answers).some(k => k.startsWith("px:") && !answers[k].ok)) {
      loadScript("assets/js/exercises-data.js").then(renderMistakes, () => { });
    }
    const wrong = Object.entries(answers).filter(([, a]) => !a.ok).sort((a, b) => b[1].t - a[1].t).map(([id]) => [id, findQ(id)]).filter(x => x[1]);
    main.innerHTML = `<div class="ch-head"><div class="kicker">Review your mistakes</div><h1>错题本</h1><p class="muted">共 ${wrong.length} 道错题。点“重做”会清除该题记录并跳到对应测验。</p></div>` +
      (wrong.length ? wrong.map(([id, x]) => `<div class="card"><div class="small muted">第 ${x.ch.no} 章 · ${esc(x.ch.title)}</div>
        <div class="q-text" style="margin-top:6px">${x.q}</div><div class="feedback ok"><b>正确答案：</b>${x.ans}。${x.explain || ""}</div>
        <div class="row" style="margin-top:10px"><span class="spacer"></span><button class="btn" data-redo="${id}">重做</button></div></div>`).join("")
        : `<div class="card">还没有错题，继续保持！</div>`);
    $$("[data-redo]").forEach(b => b.addEventListener("click", () => {
      const id = b.dataset.redo; const x = findQ(id); delete answers[id]; store.save();
      location.hash = x && x.redo ? x.redo.split("#st-")[0] : `#/ch/${id.split("-")[0]}/quiz/${id}`;
    }));
    typeset(main);
  }

  /* ---------------- search ---------------- */
  function renderSearch(q) {
    renderSidebar(null, "other");
    const k = q.trim().toLowerCase();
    const hits = [];
    if (k) chapters.forEach(ch => {
      ch.vocab.forEach(v => { if ((v.en + " " + v.zh).toLowerCase().includes(k)) hits.push({ href: `#/ch/${ch.id}/vocab/v-${v.id}`, title: `${v.en} · ${v.zh}`, where: `第 ${ch.no} 章 词汇`, text: v.defZh }); });
      ch.learn.forEach(s => {
        const t = strip(s.body);
        if ((s.title + s.titleEn).toLowerCase().includes(k) || t.toLowerCase().includes(k)) {
          const i = t.toLowerCase().indexOf(k);
          hits.push({ href: `#/ch/${ch.id}/learn/${s.id}`, title: s.title, where: `第 ${ch.no} 章 学习`, text: i >= 0 ? "…" + t.slice(Math.max(0, i - 40), i + 80) + "…" : s.titleEn });
        }
      });
      ch.examples.forEach(e => { if ((e.title + strip(e.problem)).toLowerCase().includes(k)) hits.push({ href: `#/ch/${ch.id}/examples/${e.id}`, title: e.title, where: `第 ${ch.no} 章 例题`, text: strip(e.problem).slice(0, 100) }); });
    });
    const hl = s => k ? esc(s).replace(new RegExp(k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig"), m => `<mark>${m}</mark>`) : esc(s);
    main.innerHTML = `<div class="ch-head"><h1>搜索：${esc(q)}</h1><p class="muted">${hits.length} 条结果</p></div>` +
      `<a class="result" href="#/code?q=${encodeURIComponent(q)}"><div class="where">代码库</div><b>🔎 在 700 多个教材代码片段里搜索「${esc(q)}」→</b><div class="small muted">找训练代码、模型写法，可直接放进训练实验室运行</div></a>` +
      hits.slice(0, 80).map(h => `<a class="result" href="${h.href}"><div class="where">${h.where}</div><b>${hl(h.title)}</b><div class="small muted">${hl(h.text)}</div></a>`).join("");
  }
  let searchT;
  $("#searchInput").addEventListener("input", e => {
    clearTimeout(searchT);
    const v = e.target.value;
    searchT = setTimeout(() => { if (v.trim()) location.hash = "#/search/" + encodeURIComponent(v); }, 250);
  });

  /* ---------------- glossary: 正文术语悬停释义 ---------------- */
  const GL_SKIP = ".katex, code, pre, a, button, textarea, input, h2, h3, summary, .block-title, .gl, .code-box, .helper, .notes-box, .lvl";
  let glossIndex = null;
  function buildGloss() {
    const map = new Map();
    chapters.forEach(ch => ch.vocab.forEach(v => {
      const add = k => { k = k.trim().toLowerCase(); if (!map.has(k)) map.set(k, { v, ch }); };
      if (/^[\x20-\x7e]+$/.test(v.en) && v.en.length >= 3) add(v.en);
      if (v.zh.length >= 2 && !/[、，,（(]/.test(v.zh)) { add(v.zh); if (v.zh.endsWith("的") && v.zh.length > 2) add(v.zh.slice(0, -1)); }
    }));
    const keys = [...map.keys()].sort((a, b) => b.length - a.length);
    return { map, re: new RegExp(keys.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "gi") };
  }
  function glossify(root) {
    if (!glossIndex) glossIndex = buildGloss();
    const { map, re } = glossIndex;
    const nodes = new Set();
    $$(".prose, .points li", root).forEach(host => {
      const w = document.createTreeWalker(host, NodeFilter.SHOW_TEXT, {
        acceptNode: n => (!n.nodeValue.trim() || n.parentElement.closest(GL_SKIP)) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
      });
      let n; while ((n = w.nextNode())) nodes.add(n);
    });
    const seenByScope = new Map();
    const isWord = c => /[a-z0-9]/i.test(c || "");
    nodes.forEach(n => {
      const scope = n.parentElement.closest("article") || root;
      let seen = seenByScope.get(scope); if (!seen) seenByScope.set(scope, seen = new Set());
      const text = n.nodeValue; let m, last = 0, frag = null;
      re.lastIndex = 0;
      while ((m = re.exec(text))) {
        const hit = map.get(m[0].toLowerCase());
        if (!hit) continue;
        const id = hit.ch.id + "|" + hit.v.id;
        if (seen.has(id)) continue;
        if (isWord(m[0][0]) && isWord(text[m.index - 1])) continue;
        if (isWord(m[0][m[0].length - 1]) && isWord(text[m.index + m[0].length])) continue;
        seen.add(id);
        frag = frag || document.createDocumentFragment();
        frag.append(text.slice(last, m.index));
        const s = document.createElement("span");
        s.className = "gl"; s.dataset.gl = id; s.tabIndex = 0; s.textContent = m[0];
        frag.append(s); last = m.index + m[0].length;
      }
      if (frag) { frag.append(text.slice(last)); n.replaceWith(frag); }
    });
  }
  const tip = document.createElement("div");
  tip.className = "gl-tip"; tip.hidden = true; document.body.appendChild(tip);
  let tipTimer;
  function showTip(span) {
    clearTimeout(tipTimer);
    const [cid, vid] = span.dataset.gl.split("|");
    const ch = byId[cid], v = ch && ch.vocab.find(x => x.id === vid);
    if (!v) return;
    tip.innerHTML = `<div class="gl-h"><b>${esc(v.en)}</b> · ${esc(v.zh)} ${v.audio && v.audio.en ? `<button class="audio-btn" data-audio="${v.audio.en}">🔊</button>` : ""}</div>
      <div>${esc(v.defZh)}</div><div class="gl-en">${esc(v.defEn)}</div><a href="#/ch/${cid}/vocab/v-${vid}">查看词条与例句 →</a>`;
    tip.hidden = false;
    const r = span.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    const x = Math.min(Math.max(8, r.left), innerWidth - w - 8);
    let y = r.bottom + 8; if (y + h > innerHeight - 8) y = Math.max(8, r.top - h - 8);
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tipTimer = setTimeout(() => { tip.hidden = true; }, 220); };
  document.addEventListener("mouseover", e => {
    const g = e.target.closest && e.target.closest(".gl");
    if (g) showTip(g); else if (e.target.closest && e.target.closest(".gl-tip")) clearTimeout(tipTimer);
  });
  document.addEventListener("mouseout", e => { if (e.target.closest && (e.target.closest(".gl") || e.target.closest(".gl-tip"))) hideTip(); });
  document.addEventListener("focusin", e => { const g = e.target.closest && e.target.closest(".gl"); if (g) showTip(g); });
  document.addEventListener("click", e => { const g = e.target.closest(".gl"); if (g) showTip(g); else if (!e.target.closest(".gl-tip")) tip.hidden = true; });
  window.addEventListener("scroll", () => { tip.hidden = true; }, { passive: true });

  /* ---------------- 学习路线与“今日任务” ---------------- */
  function trackChapters() {
    const t = R.tracks.find(x => x.id === prefs.track);
    return (t ? t.chapters : chapters.map(c => c.id)).map(id => byId[id]).filter(Boolean);
  }
  const coreOnlyTrack = () => !!prefs.track && prefs.track !== "full";
  function nextSection() {
    for (const ch of trackChapters()) {
      const x = ch.learn.map((s, i) => [s, i]).find(([s, i]) => !read[s.id] && (!coreOnlyTrack() || isCore(ch, i)));
      if (x) return { ch, s: x[0], i: x[1] };
    }
    return null;
  }
  function todayCard() {
    const nx = nextSection(), due = dueCards().length;
    const wrong = Object.values(answers).filter(a => !a.ok).length;
    const tr = R.tracks.find(t => t.id === prefs.track);
    const lv = n => n === 0 ? 0 : n < 3 ? 1 : n < 8 ? 2 : 3;
    const days = Array.from({ length: 14 }, (_, i) => { const k = dayKey(Date.now() - (13 - i) * DAY); return [k, activity[k] || 0]; });
    let route = `<div class="small muted">还没选学习路线？<a href="#/start">做个 1 分钟自测，获得推荐 →</a></div>`;
    if (tr) {
      const secs = trackChapters().flatMap(ch => ch.learn.map((s, i) => [ch, s, i]).filter(([c, , i]) => !coreOnlyTrack() || isCore(c, i)));
      const done = secs.filter(([, s]) => read[s.id]).length;
      route = `<div class="small muted">当前路线：<b>${tr.icon} ${esc(tr.name)}</b> · 已掌握 ${done} / ${secs.length} 节 · <a href="#/start">更换路线</a></div>
        <div class="progress" style="margin-top:6px"><i style="width:${Math.round(100 * done / Math.max(1, secs.length))}%"></i></div>`;
    }
    return `<div class="card today"><div class="today-main"><h3>📅 今日任务</h3>
      <ul class="todo">
        <li>${nx ? `<a href="#/ch/${nx.ch.id}/learn/${nx.s.id}"><b>继续学习</b>　第 ${nx.ch.no} 章 · ${esc(nx.s.title)} →</a>` : `<span>🎉 当前路线的必读小节已全部掌握，<a href="#/start">选下一条路线</a>或去做测验</span>`}</li>
        <li><a href="#/review"><b>闪卡复习</b>　${due ? `${due} 张到期` : "今天没有到期的卡片"} →</a></li>
        <li><a href="#/mistakes"><b>错题本</b>　${wrong ? `${wrong} 道待复盘` : "暂无错题"} →</a></li>
      </ul>${route}</div>
      <div class="today-side"><div class="streak"><b>${streak()}</b><span>连续学习天数</span></div>
        <div class="heat" title="最近 14 天的学习动作次数">${days.map(([k, n]) => `<i class="h${lv(n)}" title="${k}：${n} 次"></i>`).join("")}</div></div></div>`;
  }

  /* ---------------- 章节资源页 ---------------- */
  const RES_TYPE = { video: ["🎬", "视频课程"], interactive: ["🧪", "交互演示（边玩边懂）"], read: ["📖", "文章与教材"], practice: ["💻", "动手实践"] };
  const hostOf = u => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch (e) { return ""; } };
  function searchLinks(s) {
    const q = encodeURIComponent;
    return `<a class="btn sm" target="_blank" rel="noopener" href="https://www.youtube.com/results?search_query=${q(s.titleEn)}">YouTube</a>
      <a class="btn sm" target="_blank" rel="noopener" href="https://search.bilibili.com/all?keyword=${q(s.title)}">B 站</a>
      <a class="btn sm" target="_blank" rel="noopener" href="https://www.google.com/search?q=${q(s.titleEn + " explained tutorial")}">Google</a>`;
  }
  function renderRes(ch, el) {
    const links = meta(ch).links || [];
    const html = Object.keys(RES_TYPE).map(t => {
      const list = links.filter(l => l.type === t);
      return list.length ? `<h3 class="res-h">${RES_TYPE[t][0]} ${RES_TYPE[t][1]}</h3><div class="res-grid">${list.map(l => `
        <a class="res" href="${esc(l.url)}" target="_blank" rel="noopener">
          <div class="res-t">${esc(l.title)}</div><div class="small muted">${esc(l.note || "")}</div>
          <div class="res-meta"><span class="tag">${l.lang === "zh" ? "中文" : "English"}</span><span class="small muted">${esc(hostOf(l.url))} ↗</span></div></a>`).join("")}</div>` : "";
    }).join("");
    el.innerHTML = `<div class="card"><h3>本章精选资源</h3>
        <p class="small muted" style="margin:0">优先选择免费、权威、可交互的资源，链接在新标签页打开（需要联网）。想补充或替换资源，直接编辑 <code>assets/js/resources.js</code>。英文资源可配合浏览器的“翻译此页”阅读。</p></div>
      ${html || `<div class="card">暂无精选资源。</div>`}
      <div class="card"><h3>按小节搜索更多</h3>${ch.learn.map(s => `<div class="ai-row"><span>${esc(s.title)}</span><span class="spacer"></span>${searchLinks(s)}</div>`).join("")}</div>`;
  }

  /* ---------------- 新手入门页 ---------------- */
  const CHECKS = [
    ["py", "我会写简单的 Python（变量、循环、函数）"],
    ["la", "我知道矩阵乘法、向量点积是什么"],
    ["calc", "我知道导数和链式法则是什么"],
    ["ml", "我听说过过拟合、梯度下降，并跑过简单的机器学习例子"],
    ["cnn", "我已经学过卷积神经网络（CNN）"],
  ];
  function recommend(a) { return a.cnn ? "vision" : (a.py && (a.ml || (a.la && a.calc))) ? "coder" : "zero"; }
  function trackHours(t) { return t.chapters.reduce((s, id) => s + ((R.chapters[id] || {}).hours || 0), 0); }
  function renderStart() {
    const chk = prefs.check || (prefs.check = {});
    const rec = () => {
      const id = recommend(chk), t = R.tracks.find(x => x.id === id);
      const py = chk.py ? "" : `<p class="small">⚠️ 你还不熟悉 Python：建议同时花 1–2 周学 <a target="_blank" rel="noopener" href="https://cs50.harvard.edu/python/">CS50P</a> 或 <a target="_blank" rel="noopener" href="https://docs.python.org/zh-cn/3/tutorial/">官方中文教程</a>，每章的代码都能直接在浏览器里改着跑。</p>`;
      return t ? `<div class="note"><b>推荐你从「${t.icon} ${esc(t.name)}」开始</b>（约 ${t.weeks}）<br>${esc(t.tip)}${py}
        <div class="row" style="margin-top:8px"><button class="btn primary sm" data-pick="${t.id}">设为我的路线</button></div></div>` : "";
    };
    main.innerHTML = `<div class="ch-head"><div class="kicker">Start here</div><h1>新手入门</h1>
      <p class="muted">不知道从哪下手？先做一个 1 分钟自测，再选一条路线。路线只是建议顺序，随时可以换。</p></div>
      <div class="card"><h3>① 1 分钟自测：勾选符合你的描述</h3>
        <div class="checks">${CHECKS.map(([k, t]) => `<label class="read-toggle"><input type="checkbox" data-chk="${k}" ${chk[k] ? "checked" : ""}>${t}</label>`).join("")}</div>
        <div id="recBox" style="margin-top:12px">${rec()}</div></div>
      <h2 class="group-title">② 选一条学习路线</h2>
      <div class="grid tracks">${R.tracks.map(t => `<div class="card track ${prefs.track === t.id ? "on" : ""}">
        <div class="row"><h3 style="margin:0">${t.icon} ${esc(t.name)}</h3><span class="spacer"></span><span class="tag">${t.weeks} · 约 ${trackHours(t)} 小时</span></div>
        <p class="small muted" style="margin:6px 0">适合：${esc(t.who)}</p><p class="small" style="margin:0 0 8px">目标：${esc(t.goal)}</p>
        <div class="chips">${t.chapters.map((id, k) => byId[id] ? `<a class="chip" href="#/ch/${id}/learn"><i>${k + 1}</i>${esc(byId[id].title.split("：")[0])}</a>` : "").join("")}</div>
        <p class="small muted" style="margin:8px 0">💡 ${esc(t.tip)}</p>
        <button class="btn ${prefs.track === t.id ? "" : "primary"}" data-pick="${t.id}">${prefs.track === t.id ? "✔ 当前路线" : "设为我的路线"}</button></div>`).join("")}</div>
      <h2 class="group-title">③ 高效学习的 6 个方法</h2>
      <div class="howto">${R.methods.map(m => `<div><b>${esc(m.t)}</b>${esc(m.d)}</div>`).join("")}</div>
      <h2 class="group-title">④ 推荐工具</h2>
      <div class="res-grid">${R.tools.map(t => `<a class="res" href="${esc(t.url)}" target="_blank" rel="noopener"><div class="res-t">${esc(t.name)}</div>
        <div class="small muted">${esc(t.note)}</div><div class="res-meta"><span class="small muted">${esc(hostOf(t.url))} ↗</span></div></a>`).join("")}</div>
      <p class="small muted" style="margin-top:20px">更多：<a href="#/tools">工具箱</a>里有番茄钟、字号调节、Anki 导出、笔记导出和进度备份。</p>`;
    const bind = () => $$("[data-pick]", main).forEach(b => b.onclick = () => {
      prefs.track = b.dataset.pick; store.save(); toast("已设为我的路线"); renderSidebar(null, "start"); renderStart(); window.scrollTo(0, 0);
    });
    $$("[data-chk]", main).forEach(c => c.addEventListener("change", () => {
      chk[c.dataset.chk] = c.checked; store.save(); $("#recBox").innerHTML = rec(); bind();
    }));
    bind();
  }

  /* ---------------- 工具箱：字号、番茄钟、导出与备份 ---------------- */
  function download(name, text, type) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: (type || "text/plain") + ";charset=utf-8" }));
    a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  const mathAnki = s => String(s).replace(/\$\$([\s\S]+?)\$\$/g, "\\[$1\\]").replace(/\$([^$\n]+?)\$/g, "\\($1\\)");
  const cell = s => String(s).replace(/\t/g, " ").replace(/\r?\n/g, "<br>");
  function ankiTSV(list) {
    const rows = ["#separator:tab", "#html:true", "#tags column:3"];
    list.forEach(ch => {
      ch.vocab.forEach(v => rows.push([cell(v.en), cell(`<b>${v.zh}</b><br>${v.defZh}<br><br>${v.defEn}<br><i>${v.example}</i>`), `dlcv ${ch.id} vocab`].join("\t")));
      ch.cards.forEach(c => rows.push([cell(mathAnki(c.front)), cell(mathAnki(c.back)), `dlcv ${ch.id} concept`].join("\t")));
    });
    return rows.join("\n");
  }
  function notesMD() {
    const out = ["# 我的学习笔记\n"];
    chapters.forEach(ch => {
      const items = ch.learn.filter(s => notes[s.id]);
      if (!items.length) return;
      out.push(`\n## 第 ${ch.no} 章 ${ch.title}\n`);
      items.forEach(s => out.push(`### ${s.title}\n\n${notes[s.id]}\n`));
    });
    return out.join("\n");
  }
  /* 番茄钟 */
  const baseTitle = document.title;
  let pomoEl = null, pomoTimer = null;
  function beep() {
    try {
      const ac = new (window.AudioContext || window.webkitAudioContext)();
      [0, 0.35].forEach(t => { const o = ac.createOscillator(), g = ac.createGain(); o.frequency.value = 880; g.gain.value = 0.15; o.connect(g); g.connect(ac.destination); o.start(ac.currentTime + t); o.stop(ac.currentTime + t + 0.25); });
    } catch (e) { /* 无音频权限时忽略 */ }
  }
  function startPomo(focus, brk) { store.d.pomo = { phase: "focus", end: Date.now() + focus * 60000, focus, brk }; store.save(); pomoTick(); toast(`🍅 开始专注 ${focus} 分钟`); }
  function stopPomo() { delete store.d.pomo; store.save(); pomoTick(); }
  function pomoTick() {
    clearInterval(pomoTimer);
    const p = store.d.pomo;
    if (!pomoEl) {
      pomoEl = document.createElement("div"); pomoEl.className = "pomo";
      pomoEl.innerHTML = `<span class="pomo-ph"></span><b class="pomo-t"></b><button class="btn sm ghost">结束</button>`;
      document.body.appendChild(pomoEl); $("button", pomoEl).addEventListener("click", stopPomo);
    }
    if (!p) { pomoEl.hidden = true; document.title = baseTitle; return; }
    pomoEl.hidden = false;
    const upd = () => {
      let left = p.end - Date.now();
      if (left <= 0) {
        beep();
        if (p.phase === "focus") { toast("🍅 专注结束，休息一下"); touch(5); p.phase = "break"; p.end = Date.now() + p.brk * 60000; }
        else { toast("休息结束，开始下一个专注"); p.phase = "focus"; p.end = Date.now() + p.focus * 60000; }
        store.save(); left = p.end - Date.now();
      }
      const s = Math.max(0, Math.round(left / 1000)), t = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
      $(".pomo-ph", pomoEl).textContent = p.phase === "focus" ? "🍅 专注" : "☕ 休息";
      $(".pomo-t", pomoEl).textContent = t;
      document.title = `${p.phase === "focus" ? "🍅" : "☕"} ${t} · ${baseTitle}`;
    };
    upd(); pomoTimer = setInterval(upd, 1000);
  }
  function renderTools() {
    const secs = chapters.reduce((s, c) => s + c.learn.length, 0), doneSecs = Object.values(read).filter(Boolean).length;
    const allQ = chapters.reduce((s, c) => s + qids(c).length, 0), ans = Object.values(answers), okQ = ans.filter(a => a.ok).length;
    const allCards = chapters.flatMap(deck).length, learned = Object.values(cards).filter(c => c.box >= 1).length;
    const fs = Math.round((prefs.fontScale || 1) * 100);
    main.innerHTML = `<div class="ch-head"><div class="kicker">Toolbox</div><h1>工具箱</h1><p class="muted">阅读设置、专注计时、导出备份。所有数据只保存在你的浏览器里。</p></div>
      <div class="card"><h3>📊 我的学习数据</h3><div class="stats-lite">
        <div><b>${doneSecs} / ${secs}</b><span>已掌握小节</span></div>
        <div><b>${ans.length ? Math.round(100 * okQ / ans.length) + "%" : "—"}</b><span>测验正确率（已答 ${ans.length} / ${allQ}）</span></div>
        <div><b>${learned} / ${allCards}</b><span>已开始记忆的闪卡</span></div>
        <div><b>${Object.keys(notes).length}</b><span>篇笔记</span></div>
        <div><b>${streak()}</b><span>连续学习天数</span></div></div></div>
      <div class="card"><h3>🔠 阅读设置</h3><div class="row">
        <button class="btn" data-fs="-0.1">A −</button><b id="fsVal">${fs}%</b><button class="btn" data-fs="0.1">A +</button><button class="btn ghost" data-fs="0">还原</button>
        <span class="spacer"></span><label class="read-toggle"><input type="checkbox" id="coreOnly2" ${prefs.coreOnly ? "checked" : ""}>只看“入门必读”小节</label></div></div>
      <div class="card"><h3>🍅 番茄钟</h3><p class="small muted">专注一段时间后休息，防止长时间硬啃。计时器显示在页面右下角，换页面也不会中断；完成一个番茄会计入学习天数。</p>
        <div class="row"><button class="btn primary" data-pomo="25,5">专注 25 分钟 / 休息 5 分钟</button><button class="btn" data-pomo="50,10">专注 50 分钟 / 休息 10 分钟</button><button class="btn ghost" data-pomo="0">结束计时</button></div></div>
      <div class="card"><h3>🃏 导出闪卡到 Anki</h3>
        <p class="small muted">Anki 是最流行的间隔重复软件，手机和电脑都能用。导出为带标签的制表符文本：Anki → 文件 → 导入，选择“Basic（基本）”，如想中英双向都背请选“Basic (and reversed card)”。公式使用 Anki 内置的 MathJax 渲染。</p>
        <div class="row"><select id="ankiSel" class="sel"><option value="all">全部 ${chapters.length} 章（${allCards} 张）</option>${chapters.map(c => `<option value="${c.id}">第 ${c.no} 章 ${esc(c.title)}（${deck(c).length} 张）</option>`).join("")}</select>
        <button class="btn primary" id="ankiBtn">下载 Anki 导入文件</button></div></div>
      <div class="card"><h3>💾 笔记与进度</h3><div class="row">
        <button class="btn" id="notesBtn">导出笔记（Markdown）</button><button class="btn" id="bakBtn">备份全部进度（JSON）</button>
        <label class="btn">导入备份<input type="file" id="impFile" accept="application/json,.json" hidden></label>
        <span class="spacer"></span><button class="btn ghost" id="resetAll" style="color:var(--bad)">清除全部学习数据</button></div>
        <p class="small muted" style="margin-bottom:0">换浏览器或清理浏览器数据之前，请先备份。备份包含进度、答题记录、闪卡状态、笔记和设置。</p></div>`;
    $$("[data-fs]", main).forEach(b => b.addEventListener("click", () => {
      const d = +b.dataset.fs; prefs.fontScale = d === 0 ? 1 : Math.min(1.4, Math.max(0.85, Math.round(((prefs.fontScale || 1) + d) * 100) / 100));
      store.save(); applyFont(); $("#fsVal").textContent = Math.round(prefs.fontScale * 100) + "%";
    }));
    $("#coreOnly2").addEventListener("change", e => { prefs.coreOnly = e.target.checked; store.save(); });
    $$("[data-pomo]", main).forEach(b => b.addEventListener("click", () => {
      if (b.dataset.pomo === "0") { stopPomo(); toast("已结束计时"); return; }
      const [f, k] = b.dataset.pomo.split(",").map(Number); startPomo(f, k);
    }));
    $("#ankiBtn").addEventListener("click", () => {
      const v = $("#ankiSel").value, list = v === "all" ? chapters : [byId[v]];
      download(`dlcv-anki-${v}.txt`, ankiTSV(list)); toast("已下载，请在 Anki 中导入");
    });
    $("#notesBtn").addEventListener("click", () => { if (!Object.keys(notes).length) { toast("还没有笔记"); return; } download("dlcv-notes.md", notesMD(), "text/markdown"); });
    $("#bakBtn").addEventListener("click", () => download(`dlcv-backup-${dayKey()}.json`, JSON.stringify(store.d, null, 1), "application/json"));
    $("#impFile").addEventListener("change", e => {
      const f = e.target.files[0]; if (!f) return;
      f.text().then(t => {
        const o = JSON.parse(t);
        if (!o || typeof o !== "object" || Array.isArray(o)) throw new Error("格式不对");
        if (!confirm("导入会覆盖当前浏览器中的全部学习数据，确定继续？")) return;
        localStorage.setItem(KEY, JSON.stringify(o)); location.reload();
      }).catch(() => toast("导入失败：不是有效的备份文件"));
    });
    $("#resetAll").addEventListener("click", () => {
      if (!confirm("将清除本浏览器中的全部进度、答题记录、闪卡状态和笔记，且无法恢复。确定？")) return;
      if (!confirm("再确认一次：真的要清除吗？建议先点“备份全部进度”。")) return;
      localStorage.removeItem(KEY); location.reload();
    });
  }

  /* ---------------- router ---------------- */
  function route() {
    document.body.classList.remove("nav-open");
    const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
    if (parts[0] === "ch" && byId[parts[1]]) {
      const ch = byId[parts[1]];
      renderSidebar(ch.id);
      renderChapter(ch, parts[2] || "learn", parts[3]);
      if (!parts[3]) window.scrollTo(0, 0);
    } else if (parts[0] === "review") { renderReview(); window.scrollTo(0, 0); }
    else if (parts[0] === "mistakes") { renderMistakes(); window.scrollTo(0, 0); }
    else if (parts[0] === "search") { renderSearch(parts.slice(1).join("/")); }
    else if (/^#\/lab(\/|\?|$)/.test(location.hash)) {
      renderSidebar(null, "lab"); main.innerHTML = `<div class="loading">加载训练实验室…</div>`;
      loadScript("assets/js/lab.js").then(() => window.Lab.render(main, location.hash)).catch(e => { main.innerHTML = `<div class="card">${esc(e.message)}</div>`; });
      window.scrollTo(0, 0);
    }
    else if (/^#\/practice(\/|\?|$)/.test(location.hash)) {
      renderSidebar(null, "practice"); main.innerHTML = `<div class="loading">加载代码练习…</div>`;
      loadScript("assets/js/exercises-data.js").then(() => loadScript("assets/js/practice.js")).then(() => { window.Practice.render(main, location.hash); typeset(main); })
        .catch(e => { main.innerHTML = `<div class="card">${esc(e.message)}。请先运行 <code>python3 tools/build_exercises.py</code>。</div>`; });
      window.scrollTo(0, 0);
    }
    else if (/^#\/code(\/|\?|$)/.test(location.hash)) {
      renderSidebar(null, "code"); main.innerHTML = `<div class="loading">加载代码库…</div>`;
      const q = new URLSearchParams(location.hash.split("?")[1] || "").get("q") || "";
      loadScript("assets/js/codelib-data.js").then(() => loadScript("assets/js/codelib.js")).then(() => window.CodeLib.render(main, q))
        .catch(e => { main.innerHTML = `<div class="card">${esc(e.message)}。请先运行 <code>python3 tools/build_codelib.py</code> 生成代码库索引。</div>`; });
      window.scrollTo(0, 0);
    }
    else if (parts[0] === "start") { renderSidebar(null, "start"); renderStart(); window.scrollTo(0, 0); }
    else if (parts[0] === "tools") { renderSidebar(null, "tools"); renderTools(); window.scrollTo(0, 0); }
    else { renderSidebar(null); renderHome(); typeset(main); window.scrollTo(0, 0); }
    tip.hidden = true;
    updateBadge();
  }
  window.addEventListener("hashchange", route);

  /* ---------------- theme & menu ---------------- */
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    $("#hljs-light").disabled = t === "dark"; $("#hljs-dark").disabled = t !== "dark";
  }
  applyTheme(prefs.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("#themeBtn").addEventListener("click", () => { prefs.theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; store.save(); applyTheme(prefs.theme); });
  $("#menuBtn").addEventListener("click", () => document.body.classList.toggle("nav-open"));
  $("#scrim").addEventListener("click", () => document.body.classList.remove("nav-open"));

  applyFont(); pomoTick();
  if (!chapters.length) { main.innerHTML = `<div class="card">没有课程数据。请先运行 <code>python3 tools/build.py</code>。</div>`; return; }
  route();
})();
