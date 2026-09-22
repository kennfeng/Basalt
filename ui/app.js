async function fetchHealth() {
  const started = performance.now();
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const latency = Math.round(performance.now() - started);
    const setText = (id, value) => {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    };
    setText("h-status", data.status);
    const badge = document.getElementById("health-status");
    if (badge) {
      badge.textContent = data.status;
      badge.style.background = data.status === "ok" ? "#dcfce7" : "#fef9c3";
    }
    setText("h-db", data.db_ready ? "ready" : "not ready");
    setText("h-count", String(data.db_count ?? "-"));
    setText("h-ollama", data.ollama_reachable ? "connected" : "not connected");
    setText("h-pipeline", data.pipeline_initialized ? "ready" : "starting");
    const legacy = {
      "h-hybrid": data.hybrid_enabled ? "enabled" : "disabled",
      "h-rrf": String(data.rrf_k ?? "-"),
      "h-bm25": data.bm25_ready ? "ready" : "missing",
      "h-models": data.models ? `${data.models.llm} / ${data.models.reranker}` : "-",
      "h-vdb": data.vector_db || "ChromaDB HNSW",
      "h-pipe": data.retrieval_pipeline || "-",
    };
    for (const [k, v] of Object.entries(legacy)) setText(k, v);
    setText("corpus-count", String(data.db_count || 250));
    const cfgEl = document.getElementById("h-config");
    if (cfgEl) cfgEl.textContent = JSON.stringify({ hybrid: data.hybrid_config, hnsw: data.hnsw_config, models: data.models, pipeline: data.retrieval_pipeline, latency_ms: latency }, null, 2);
  } catch (e) {
    const badge = document.getElementById("health-status");
    if (badge) badge.textContent = "error";
    const cfgEl = document.getElementById("h-config");
    if (cfgEl) cfgEl.textContent = String(e);
  }
}

function humanBytes(b) {
  if (b == null || isNaN(b)) return "-";
  const gb = b / (1024 * 1024 * 1024);
  if (gb >= 1) return gb.toFixed(2) + " GB";
  const mb = b / (1024 * 1024);
  return mb.toFixed(1) + " MB";
}

async function fetchScaleReport() {
  const tbody = document.getElementById("scale-tbody");
  const modeEl = document.getElementById("scale-mode");
  const overallEl = document.getElementById("scale-overall");
  const metaEl = document.getElementById("scale-meta");
  const thrEl = document.getElementById("scale-thresholds");
  if (!tbody) return;
  try {
    const res = await fetch("/scale_report");
    if (res.status === 404) {
      tbody.innerHTML = `<tr><td colspan="9">Scale report not available - run <code>bench_scale --real</code> to generate. <a href="/scale_report">/scale_report</a> 404</td></tr>`;
      if (modeEl) { modeEl.textContent = "missing"; modeEl.style.background = "#fee2e2"; }
      if (overallEl) { overallEl.textContent = "-"; }
      if (metaEl) metaEl.textContent = "No report yet";
      if (thrEl) thrEl.textContent = "no scale_report.json";
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const cfg = data.config || {};
    const results = data.results || [];
    if (modeEl) {
      modeEl.textContent = cfg.mode || "-";
      modeEl.style.background = cfg.mode === "real" ? "#dcfce7" : "#fef9c3";
    }
    const overall = data.overall_slo_pass;
    if (overallEl) {
      overallEl.textContent = overall ? "overall SLO pass" : "overall SLO fail";
      overallEl.style.background = overall ? "#dcfce7" : "#fee2e2";
      overallEl.style.color = overall ? "#16a34a" : "#dc2626";
    }
    if (metaEl) metaEl.textContent = `mode=${cfg.mode} search size=${cfg.retrieve_n} sizes=${(cfg.n_values || []).join(",")} results=${results.length}`;
    if (thrEl) thrEl.textContent = JSON.stringify(cfg.thresholds || {}, null, 2);
    if (results.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9">No results (overall_slo_pass=${overall})</td></tr>`;
      return;
    }
    tbody.innerHTML = "";
    results.forEach((r) => {
      const tr = document.createElement("tr");
      const slo = r.slo_pass ? "pass" : "fail";
      const sloColor = r.slo_pass ? "#16a34a" : "#dc2626";
      const reason = escapeHtml(r.slo_reason || "");
      tr.innerHTML = `<td>${r.n}</td><td>${Number(r.docs_per_sec).toFixed(1)}</td><td>${Number(r.p50_ms).toFixed(1)}</td><td>${Number(r.p95_ms).toFixed(1)}</td><td>${Number(r.p99_ms).toFixed(1)}</td><td>${humanBytes(r.rss_bytes)}</td><td>${humanBytes(r.disk_bytes)}</td><td>${r.retrieve_n}</td><td style="color:${sloColor};font-weight:600">${slo}<span title="${reason}"> ${r.slo_pass ? "✓" : "✗"}</span></td>`;
      tbody.appendChild(tr);
    });
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="9">error loading scale report: ${escapeHtml(String(e))}</td></tr>`;
    if (thrEl) thrEl.textContent = String(e);
  }
}

function renderSources(docs) {
  const wrap = document.getElementById("sources");
  wrap.innerHTML = "";
  docs.forEach((src, idx) => {
    const el = document.createElement("div");
    el.className = "source";
    const score = typeof src.score === "number" ? src.score.toFixed(4) : "-";
    const doc = src.document || src.text || "";
    el.innerHTML = `<div class="meta"><span>#${idx + 1} ${src.id || ""}</span><span>score ${score}</span></div><div class="doc">${escapeHtml(doc.slice(0, 900))}</div><div class="actions"><button>Copy citation</button></div>`;
    el.querySelector("button").addEventListener("click", () => {
      const text = `${src.id || "doc"} - ${doc.slice(0, 600)}`;
      navigator.clipboard.writeText(text);
    });
    wrap.appendChild(el);
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function postAsk(query, useStream) {
  const btn = document.getElementById("ask-btn");
  const errEl = document.getElementById("error");
  const answerWrap = document.getElementById("answer-wrap");
  const answerEl = document.getElementById("answer");
  const latencyEl = document.getElementById("latency");
  btn.disabled = true;
  errEl.classList.add("hidden");
  answerWrap.classList.remove("hidden");
  answerEl.textContent = "";
  renderSources([]);
  const t0 = performance.now();
  try {
    if (useStream) {
      const res = await fetch("/ask/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
      }
      if (!res.body) throw new Error("No stream body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let sources = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          try {
            const obj = JSON.parse(payload);
            if (obj.token) answerEl.textContent += obj.token;
            if (obj.answer) answerEl.textContent = obj.answer;
            if (obj.done && obj.source_documents) sources = obj.source_documents;
            if (obj.error) throw new Error(obj.error);
          } catch (e) {
            if (e.message && e.message.includes("ERROR")) throw e;
          }
        }
      }
      renderSources(sources);
      latencyEl.textContent = Math.round(performance.now() - t0);
    } else {
      const res = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      answerEl.textContent = data.answer || "";
      renderSources(data.source_documents || []);
      latencyEl.textContent = Math.round(performance.now() - t0);
    }
  } catch (e) {
    errEl.textContent = String(e.message || e);
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  fetchHealth();
  fetchScaleReport();
  const q = document.getElementById("query");
  const c = document.getElementById("char-count");
  q.addEventListener("input", () => { c.textContent = `${q.value.length} / 2000`; });
  document.getElementById("ask-btn").addEventListener("click", () => {
    const val = q.value.trim();
    if (!val) return;
    const useStream = document.getElementById("stream-toggle").checked;
    postAsk(val, useStream);
  });
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      document.getElementById("ask-btn").click();
    }
  });
  setInterval(fetchHealth, 15000);
});
