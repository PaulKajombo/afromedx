const q = document.getElementById("q");
const go = document.getElementById("go");
const statusEl = document.getElementById("status");
const res = document.getElementById("result");

document.querySelectorAll(".examples button").forEach(b =>
  b.addEventListener("click", () => { q.value = b.dataset.q; doSearch(); }));
go.addEventListener("click", doSearch);
q.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

async function doSearch() {
  const query = q.value.trim();
  if (query.length < 2) return;
  statusEl.textContent = "Searching indexed guidelines…";
  res.classList.add("hidden");
  try {
    const r = await fetch("/api/search", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ query, top_k: 6 }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    render(data);
    statusEl.textContent = `Done · ${data.meta.chunks_indexed} chunks indexed · provider: ${data.answer.provider}`;
  } catch (e) {
    statusEl.textContent = "Search failed: " + e.message;
  }
}

function render(data) {
  const a = data.answer;
  document.getElementById("rTitle").textContent = a.title || "Answer";
  const ab = document.getElementById("rAbstain");
  if (a.abstained) { ab.textContent = a.body; ab.classList.remove("hidden"); }
  else { ab.classList.add("hidden"); }
  document.getElementById("rBody").innerHTML = a.abstained ? "" : mdBlock(a.body);
  document.getElementById("rPoints").innerHTML =
    (a.key_points || []).map(p => `<li>${mdInline(p)}</li>`).join("") || "<li>—</li>";
  document.getElementById("rCite").innerHTML = (a.citations || []).map(c => `
    <div class="cite"><strong>${escapeHtml(c.document || "")}</strong> ${escapeHtml(c.edition || "")}<br/>
    Section: ${escapeHtml(c.section || "")}${c.subsection ? " / " + escapeHtml(c.subsection) : ""} · Page: ${c.page ?? "—"}
    ${c.excerpt ? `<br/><em>${escapeHtml(c.excerpt.slice(0, 220))}…</em>` : ""}</div>`).join("") || "—";
  document.getElementById("rPassages").innerHTML = (data.passages || []).map(p => `
    <div class="passage"><div class="meta">${escapeHtml(p.document_id)} · ${escapeHtml(p.section)} · p.${p.page} · score ${p.score}</div>
    ${escapeHtml(p.text.slice(0, 500))}…</div>`).join("");
  res.classList.remove("hidden");
  res.scrollIntoView({ behavior: "smooth", block: "start" });
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

// Minimal markdown renderer (dependency-free, escape-first so model text can
// never inject HTML): **bold**, *italic*, ### headings, - bullets, paragraphs.
function mdInline(s) {
  return escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
}

function mdBlock(s) {
  let html = "", inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of String(s || "").split("\n")) {
    const line = raw.trim();
    const h = line.match(/^(#{1,4})\s+(.*)/);
    const b = line.match(/^[-•*]\s+(.*)/);
    if (h) { closeList(); const lvl = Math.min(4, h[1].length); html += `<h${lvl}>${mdInline(h[2])}</h${lvl}>`; }
    else if (b) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${mdInline(b[1])}</li>`; }
    else if (!line) { closeList(); }
    else { closeList(); html += `<p>${mdInline(line)}</p>`; }
  }
  closeList();
  return html;
}
