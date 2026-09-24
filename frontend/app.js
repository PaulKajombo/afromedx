const q = document.getElementById("q");
const go = document.getElementById("go");
const statusEl = document.getElementById("status");
const res = document.getElementById("result");
const stepsEl = document.getElementById("steps");
const stepNote = document.getElementById("stepnote");

// ---- splash: logo on white, auto-dismisses shortly after load ----
(function splash() {
  const el = document.getElementById("splash");
  el.classList.remove("hidden");
  setTimeout(() => {
    el.classList.add("fade");
    setTimeout(() => el.classList.add("hidden"), 400);
  }, 900);
})();

// ---- mobile drawer ----
const sidebar = document.querySelector(".sidebar");
const scrim = document.getElementById("scrim");
function setDrawer(open) {
  sidebar.classList.toggle("open", open);
  scrim.classList.toggle("hidden", !open);
}
document.getElementById("burger").addEventListener("click", () =>
  setDrawer(!sidebar.classList.contains("open")));
scrim.addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", e => {
  if (e.key === "Escape") setDrawer(false);
});

// ---- consistent thin-stroke icon system (no mixed emoji styles) ----
const PATHS = {
  home: '<path d="M3 11l9-8 9 8"/><path d="M5 10v10h5v-6h4v6h5V10"/>',
  history: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/>',
  library: '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
  sources: '<path d="M7 3h8l4 4v14H7z"/><path d="M15 3v4h4"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9L7 7M17 17l2.1 2.1M19.1 4.9L17 7M7 17l-2.1 2.1"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
  doc: '<path d="M7 3h8l4 4v14H7z"/><path d="M15 3v4h4"/><path d="M10 12h6M10 16h6"/>',
  warn: '<path d="M12 3L2 21h20L12 3z"/><path d="M12 10v5"/><circle cx="12" cy="18" r="0.5"/>',
  check: '<path d="M4 12l5 5L20 6"/>',
  chevR: '<path d="M9 5l7 7-7 7"/>',
  back: '<path d="M19 12H5"/><path d="M11 6l-6 6 6 6"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  prev: '<path d="M15 5l-7 7 7 7"/>',
  next: '<path d="M9 5l7 7-7 7"/>',
  full: '<path d="M9 4H4v5M15 20h5v-5M20 9V4h-5M4 15v5h5"/>',
};
function icon(n, cls) {
  return `<svg class="ic ${cls || ""}" viewBox="0 0 24 24" aria-hidden="true">${PATHS[n] || ""}</svg>`;
}
(function initIcons() {
  const navIcons = { home: "home", library: "library", history: "history", sources: "sources", settings: "settings" };
  const navLabels = { home: "Home", library: "Guideline Library", history: "History",
                      sources: "Sources", settings: "Settings" };
  document.querySelectorAll(".nav button").forEach(b => {
    b.innerHTML = icon(navIcons[b.dataset.view] || "doc")
      + `<span>${navLabels[b.dataset.view] || ""}</span>`;
  });
  document.getElementById("backSearch").innerHTML = icon("back", "sm") + "<span>Back to search</span>";
  document.getElementById("vBack").innerHTML = icon("back", "sm") + "<span>Back</span>";
  document.getElementById("vClose").innerHTML = icon("close", "sm");
  document.getElementById("vPrev").innerHTML = icon("prev", "sm");
  document.getElementById("vNext").innerHTML = icon("next", "sm");
  document.getElementById("vFull").innerHTML = icon("full", "sm");
  // Mock home shows an example query in the box.
  if (!q.value) q.value = "What is the recommended treatment for severe malaria in an adult?";
})();
document.getElementById("backSearch").addEventListener("click", () => {
  res.classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
  q.focus();
});

// ---- view routing (sidebar) ----
document.querySelectorAll(".nav button").forEach(b =>
  b.addEventListener("click", () => showView(b.dataset.view)));
function showView(name) {
  document.querySelectorAll(".nav button").forEach(b =>
    b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
  document.getElementById("view-" + name).classList.remove("hidden");
  setDrawer(false);
  if (name === "library") loadLibrary();
  if (name === "history") renderHistory();
  if (name === "sources") renderSources();
  if (name === "settings") renderSettings();
}
document.getElementById("browseBtn").addEventListener("click", () => showView("library"));

// Show checklist progress for the actual request lifecycle. The client only
// marks source review and answer preparation complete once the API responds.
let stepsHideTimer = null;
function stepsReset() {
  clearTimeout(stepsHideTimer);
  const labels = ["Question received", "Searching indexed guidelines",
    "Checking source passages", "Preparing answer and citations"];
  stepsEl.innerHTML = labels.map((label, i) =>
    `<li class="${i === 0 ? "done" : i === 1 ? "active" : ""}">${label}</li>`).join("");
  stepsEl.classList.remove("hidden");
  stepNote.classList.remove("hidden");
}
function stepsDone() {
  clearTimeout(stepsHideTimer);
  stepsEl.querySelectorAll("li").forEach(li => li.className = "done");
  stepNote.classList.add("hidden");
  stepsHideTimer = setTimeout(() => stepsEl.classList.add("hidden"), 1400);
}
function stepsHide() {
  clearTimeout(stepsHideTimer);
  stepsEl.classList.add("hidden"); stepNote.classList.add("hidden");
}

// ---- search ----
document.querySelectorAll(".examples button").forEach(b =>
  b.addEventListener("click", () => { q.value = b.dataset.q; doSearch(); }));
go.addEventListener("click", doSearch);
q.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

// ---- preferences (localStorage only) ----
function getTopK() {
  const v = parseInt(localStorage.getItem("afromedx.topk") || "6", 10);
  return [3, 4, 6].includes(v) ? v : 6;
}
function getMode() { return localStorage.getItem("afromedx.mode") || "full"; }
function setMode(m) {
  try { localStorage.setItem("afromedx.mode", m); } catch (e) {}
  document.querySelectorAll("#modetoggle button").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === m));
  applyMode();
}
document.querySelectorAll("#modetoggle button").forEach(b =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));

let lastAnswer = null;
function applyMode() {
  if (!lastAnswer) return;
  const dose = getMode() === "dose" && !lastAnswer.abstained;
  document.getElementById("rBody").style.display = dose ? "none" : "";
  document.getElementById("rPointsH").style.display =
    (!dose && (lastAnswer.key_points || []).length) ? "" : "none";
  document.getElementById("rPoints").style.display = (!dose) ? "" : "none";
  const d = document.getElementById("rDose");
  if (dose) { d.classList.remove("hidden"); d.innerHTML = doseCard(lastAnswer); }
  else { d.classList.add("hidden"); d.innerHTML = ""; }
}

// Dose-card mode (mock panel 5): each source line is consumed by the FIRST
// slot it matches (Dose → Route → Frequency → Duration), so fields never
// repeat each other. No extra API call; unmatched slots show "—" and the full
// answer stays one tap away.
function doseCard(a) {
  const lines = [a.body || "", ...(a.key_points || [])].join("\n")
    .split("\n").map(l => l.trim()).filter(Boolean);
  const patterns = {
    Dose: [/\d+(?:\.\d+)?\s*(?:mg\/kg|mg|g|ml|mcg)\b/i],
    Route: [/\b(IV|IM|intravenous|intramuscular|oral|PO|subcutaneous|topical)\b/i],
    Frequency: [/daily|BD|TDS|QID|\bOD\b|weekly|every\s+\d+\s*\w+|at\s+0\b/i],
    Duration: [/up to|maximum|for \d+|days|weeks|months|full course|until/i],
  };
  const slots = {};
  const used = new Set();
  for (const [slot, res] of Object.entries(patterns)) {
    slots[slot] = "—";
    for (let i = 0; i < lines.length; i++) {
      if (used.has(i)) continue;
      if (res.some(re => re.test(lines[i]))) { slots[slot] = lines[i]; used.add(i); break; }
    }
  }
  const src = (a.citations || [])[0] || {};
  const srcLine = src.document
    ? `${src.document}${src.edition ? ", " + src.edition : ""}${src.page != null ? " · p." + src.page : ""}` : "—";
  const row = (k, v) => `<div class="doserow"><span>${k}</span><span>${mdInline(v)}</span></div>`;
  return `<div class="dosecard"><div class="dosetitle">${escapeHtml(a.title || "Dose card")}</div>`
    + row("Dose", slots.Dose) + row("Route", slots.Route)
    + row("Frequency", slots.Frequency) + row("Duration", slots.Duration)
    + row("Source", srcLine) + `</div>`;
}

async function doSearch() {
  const query = q.value.trim();
  if (query.length < 2) return;
  statusEl.textContent = "Searching indexed guidelines…";
  res.classList.add("hidden");
  stepsReset();
  try {
    const r = await fetch("/api/search", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ query, top_k: getTopK() }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    render(data, query);
    saveHistory(query, data.answer);
    stepsDone();
    statusEl.textContent = "";
  } catch (e) {
    stepsHide();
    statusEl.textContent = "Search failed: " + e.message;
  }
}

// Primary source card (mock panel 3): the top citation, prominent, with
// Open-guideline action. Remaining citations render compactly below.
function renderPrimarySource(a) {
  const el = document.getElementById("rSrc");
  const c = (a.citations || [])[0];
  if (a.abstained || !c) { el.innerHTML = ""; return; }
  const docId = docIdOf(c);
  const meta = [c.edition, c.year].filter(Boolean).join(", ");
  el.innerHTML = `<div class="srccard">
    <div class="srclabel">${icon("doc", "sm")} Source</div>
    <div class="srcname">${escapeHtml(c.document || "")}</div>
    <div class="srcmeta">${escapeHtml(meta)}${c.page != null ? (meta ? " · " : "") + "Page " + c.page : ""}</div>
    ${(docId && c.page != null) ? `<button class="openbtn" id="srcOpen">Open guideline →</button>` : ""}
  </div>`;
  const btn = document.getElementById("srcOpen");
  if (btn) btn.addEventListener("click", () =>
    openGuideline(docId, c.page, (c.document || "") + (c.edition ? ", " + c.edition : "")));
}

function docIdOf(c) {
  const id = (c.chunk_id || "").split("::")[0];
  return id || null;
}
function render(data, query) {
  const a = data.answer;
  lastAnswer = a;
  document.querySelectorAll("#modetoggle button").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === getMode()));
  document.getElementById("rTitle").textContent = a.title || "Answer";
  const ab = document.getElementById("rAbstain");
  const why = document.getElementById("rWhy");
  if (a.abstained) {
    ab.innerHTML = icon("warn") + `<span>${escapeHtml(a.body)}</span>`;
    ab.classList.remove("hidden");
    why.classList.remove("hidden");
    renderRefineHint(data);
  } else { ab.classList.add("hidden"); why.classList.add("hidden"); }
  document.getElementById("rBody").innerHTML = a.abstained ? "" : mdBlock(a.body);
  renderPrimarySource(a);
  const hasPoints = (a.key_points || []).length > 0;
  document.getElementById("rPointsH").style.display = hasPoints ? "" : "none";
  document.getElementById("rPoints").innerHTML =
    (a.key_points || []).map(p => `<li>${mdInline(p)}</li>`).join("") || "";
  document.getElementById("rCite").innerHTML = (a.citations || []).map(c => {
    const docId = docIdOf(c);
    const openBtn = (docId && c.page != null)
      ? `<br/><button class="openbtn" data-doc="${escapeHtml(docId)}" data-page="${c.page}"
         data-title="${escapeHtml((c.document || "") + (c.edition ? ", " + c.edition : ""))}">Open guideline →</button>` : "";
    return `
    <div class="cite"><strong>${escapeHtml(c.document || "")}</strong> ${escapeHtml(c.edition || "")}<br/>
    Section: ${escapeHtml(c.section || "")}${c.subsection ? " / " + escapeHtml(c.subsection) : ""} · Page: ${c.page ?? "—"}
    ${c.excerpt ? `<br/><em>${escapeHtml(c.excerpt.slice(0, 220))}…</em>` : ""}${openBtn}</div>`;
  }).join("") || "—";
  document.querySelectorAll("#rCite .openbtn").forEach(b =>
    b.addEventListener("click", () => openGuideline(b.dataset.doc, parseInt(b.dataset.page, 10), b.dataset.title)));
  applyMode();
  res.classList.remove("hidden");
  res.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- guideline viewer (mock panel 4): served PDF at the cited page ----
let viewerDoc = null, viewerPage = 1, viewerTitle = "", viewerSub = "";
function openGuideline(docId, page, title) {
  viewerDoc = docId; viewerPage = page || 1;
  const parts = String(title || "").split(",");
  viewerTitle = (parts[0] || "Guideline").trim();
  viewerSub = parts.slice(1).join(",").trim();
  document.getElementById("viewer").classList.remove("hidden");
  document.body.style.overflow = "hidden";
  showViewerPage();
}
function showViewerPage() {
  document.getElementById("vTitle").textContent = viewerTitle;
  document.getElementById("vSub").textContent = viewerSub;
  document.getElementById("vPage").textContent = "Page " + viewerPage;
  document.getElementById("vFrame").src =
    `/api/guideline/${encodeURIComponent(viewerDoc)}/pdf#page=${viewerPage}`;
  renderThumbs();
}
// Numeric thumbnail strip: neighbouring pages around the cited one, current
// highlighted (real PDF thumbnails would need a client PDF renderer).
function renderThumbs() {
  const el = document.getElementById("thumbs");
  el.innerHTML = "";
  for (let p = Math.max(1, viewerPage - 2); p <= viewerPage + 2; p++) {
    const b = document.createElement("button");
    b.className = "thumb" + (p === viewerPage ? " cur" : "");
    b.textContent = p;
    b.setAttribute("aria-label", "Go to page " + p);
    const target = p;
    b.addEventListener("click", () => { viewerPage = target; showViewerPage(); });
    el.appendChild(b);
  }
}
function closeGuideline() {
  document.getElementById("viewer").classList.add("hidden");
  document.getElementById("vFrame").src = "about:blank";
  document.body.style.overflow = "";
}
document.getElementById("vClose").addEventListener("click", closeGuideline);
document.getElementById("vPrev").addEventListener("click", () => {
  if (viewerPage > 1) { viewerPage -= 1; showViewerPage(); }
});
document.getElementById("vNext").addEventListener("click", () => {
  viewerPage += 1; showViewerPage();
});
document.getElementById("vFull").addEventListener("click", () => {
  const v = document.getElementById("viewer");
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else if (v.requestFullscreen) v.requestFullscreen().catch(() => {});
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !document.getElementById("viewer").classList.contains("hidden"))
    closeGuideline();
});

// ---- sources view: provenance grouped by publisher ----
async function renderSources() {
  const el = document.getElementById("srclist");
  try {
    if (!libCache) {
      const r = await fetch("/api/documents");
      if (!r.ok) throw new Error("HTTP " + r.status);
      libCache = await r.json();
    }
    const bySource = {};
    libCache.documents.forEach(d => {
      const s = d.source || "Unknown source";
      (bySource[s] = bySource[s] || []).push(d);
    });
    el.innerHTML = Object.keys(bySource).sort().map(s =>
      `<h3>${escapeHtml(s)} (${bySource[s].length})</h3>` + bySource[s].map(d =>
        `<div class="guide"><span class="t">${escapeHtml(d.title || d.id)}</span>`
        + `<div class="m">${escapeHtml([d.edition, d.publication_year].filter(Boolean).join(" · "))}</div></div>`
      ).join("")).join("")
      + "<p class='hint'>Every answer cites one of these editions with section and page. Verify against the cited page.</p>";
  } catch (e) {
    el.innerHTML = `<p class="hint">Could not load sources: ${escapeHtml(e.message)}</p>`;
  }
}

// ---- settings view: server info (read-only) + local preferences ----
async function renderSettings() {
  const el = document.getElementById("setinfo");
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const h = await r.json();
    el.textContent = `Provider: ${h.provider} · ${h.chunks} chunks indexed · embedder: ${h.embedder}`;
  } catch (e) {
    el.textContent = "Server info unavailable: " + e.message;
  }
  document.getElementById("topk").value = String(getTopK());
}
document.getElementById("topk").addEventListener("change", e => {
  try { localStorage.setItem("afromedx.topk", e.target.value); } catch (err) {}
});

// Refinement hint: when abstaining WITH retrieved passages, point at what the
// index actually holds so a too-general query becomes answerable on retry.
// Purely presentational — the safety gate itself is untouched.
function renderRefineHint(data) {
  const seen = new Set(), topics = [];
  for (const p of data.passages || []) {
    const label = [p.document_id, p.section].filter(Boolean).join(" · ");
    const key = (p.document_id || "") + "|" + (p.section || "");
    if (label && !seen.has(key) && topics.length < 3) { seen.add(key); topics.push(label); }
  }
  let el = document.getElementById("rRefine");
  if (!topics.length) { if (el) el.innerHTML = ""; return; }
  if (!el) return;
  el.innerHTML = `<p class="why">The closest indexed material covers: `
    + topics.map(t => `<strong>${escapeHtml(t)}</strong>`).join("; ")
    + `. Try asking about diagnosis or treatment of one of these specifically.</p>`;
}

// ---- history (localStorage only: queries never leave the browser) ----
const HIST_KEY = "afromedx.history.v1";
function loadHist() {
  try { return JSON.parse(localStorage.getItem(HIST_KEY)) || []; }
  catch (e) { return []; }
}
function saveHistory(query, ans) {
  const h = loadHist();
  const top = (ans.citations || [])[0] || {};
  h.unshift({ q: query, title: ans.title || "", doc: top.document || "",
              page: top.page ?? null, abstained: !!ans.abstained, t: Date.now() });
  try { localStorage.setItem(HIST_KEY, JSON.stringify(h.slice(0, 30))); } catch (e) {}
}
function relTime(t) {
  const d = new Date(t), now = new Date();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const day = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const diff = Math.round((today - day) / 86400000);
  if (diff <= 0) return "Today, " + time;
  if (diff === 1) return "Yesterday, " + time;
  return d.toLocaleDateString();
}
function renderHistory() {
  const h = loadHist();
  const el = document.getElementById("histlist");
  if (!h.length) { el.innerHTML = "<p class='hint'>No searches yet.</p>"; return; }
  el.innerHTML = "";
  h.forEach(item => {
    const b = document.createElement("button");
    b.className = "histrow";
    const ref = item.doc ? ` · ${item.doc}${item.page != null ? " (p." + item.page + ")" : ""}` : "";
    b.innerHTML = `${icon("search", "sm")}<span>${escapeHtml(item.q)}${escapeHtml(ref)}</span>`
      + `<span class="when">${escapeHtml(relTime(item.t))}</span><span>${icon("chevR", "sm")}</span>`;
    b.addEventListener("click", () => { showView("home"); q.value = item.q; doSearch(); });
    el.appendChild(b);
  });
}
document.getElementById("clearHist").addEventListener("click", () => {
  try { localStorage.removeItem(HIST_KEY); } catch (e) {}
  renderHistory();
});

// ---- guideline library (served by GET /api/documents) ----
let libCache = null;
let libTab = "all";
// Core set mirrors the targeted MVP: the guidelines clinicians reach for first.
const CORE_DOCS = new Set(["mstg", "malaria-treatment", "malaria-2020", "hiv-2022",
                           "blue-book", "sti-2025", "tb"]);
async function loadLibrary() {
  const el = document.getElementById("liblist");
  try {
    if (!libCache) {
      const r = await fetch("/api/documents");
      if (!r.ok) throw new Error("HTTP " + r.status);
      libCache = await r.json();
    }
    document.getElementById("libcount").textContent =
      `${libCache.documents.length} guidelines · ${libCache.chunks} chunks indexed`;
    renderLibrary();
  } catch (e) {
    el.innerHTML = `<p class="hint">Could not load library: ${escapeHtml(e.message)}</p>`;
  }
}
function renderLibrary() {
  if (!libCache) return;
  const f = document.getElementById("libq").value.trim().toLowerCase();
  const el = document.getElementById("liblist");
  const rows = libCache.documents
    .filter(d => (libTab === "all") || (libTab === "core") === CORE_DOCS.has(d.id))
    .filter(d => !f || ((d.title || "") + " " + (d.edition || "")).toLowerCase().includes(f))
    .map(d => {
      const core = CORE_DOCS.has(d.id);
      const meta = [d.edition || "", d.publication_year || "", d.source || ""]
        .filter(Boolean).join(" · ");
      const added = d.date_added ? " · indexed " + escapeHtml(String(d.date_added).slice(0, 7)) : "";
      return `<div class="guide"><span class="docicon">${icon("doc")}</span><span>`
        + `<span class="t">${escapeHtml(d.title || d.id)}</span>`
        + `<span class="badge ${core ? "core" : "other"}">${core ? "Core" : "Other"}</span>`
        + (meta ? `<div class="m">${escapeHtml(meta)}${added}</div>` : "") + `</span></div>`;
    });
  el.innerHTML = rows.join("") || "<p class='hint'>No guidelines match.</p>";
}
document.getElementById("libq").addEventListener("input", renderLibrary);
document.querySelectorAll("#libtabs button").forEach(b =>
  b.addEventListener("click", () => {
    libTab = b.dataset.tab;
    document.querySelectorAll("#libtabs button").forEach(x =>
      x.classList.toggle("active", x === b));
    renderLibrary();
  }));

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

// ---- PWA: register shell service worker (scope "/" needs /sw.js route) ----
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}

// Minimal markdown renderer (dependency-free, escape-first so model text can
// never inject HTML): **bold**, *italic*, ### headings, - bullets, paragraphs.
function mdInline(s) {
  return escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
}

function mdBlock(s) {
  let html = "", inList = false, listTag = "";
  const closeList = () => { if (inList) { html += `</${listTag}>`; inList = false; listTag = ""; } };
  const openList = (tag) => {
    if (!inList) { html += `<${tag}>`; inList = true; listTag = tag; }
    else if (listTag !== tag) { closeList(); html += `<${tag}>`; inList = true; listTag = tag; }
  };
  for (const raw of String(s || "").split("\n")) {
    const line = raw.trim();
    const h = line.match(/^(#{1,4})\s+(.*)/);
    const b = line.match(/^[-•*]\s+(.*)/);
    const o = line.match(/^\d+[.)]\s+(.*)/);
    if (h) { closeList(); const lvl = Math.min(4, h[1].length); html += `<h${lvl}>${mdInline(h[2])}</h${lvl}>`; }
    else if (b) { openList("ul"); html += `<li>${mdInline(b[1])}</li>`; }
    else if (o) { openList("ol"); html += `<li>${mdInline(o[1])}</li>`; }
    else if (!line) { closeList(); }
    else { closeList(); html += `<p>${mdInline(line)}</p>`; }
  }
  closeList();
  return html;
}
