async function fetchHealth() {
  const started = performance.now();
  try {
    const res = await fetch("/health");
    const trace = res.headers.get("X-Trace-Id") || "-";
    const data = await res.json();
    const latency = Math.round(performance.now() - started);
    document.getElementById("h-status").textContent = data.status;
    document.getElementById("health-status").textContent = data.status;
    document.getElementById("health-status").style.background = data.status === "ok" ? "#dcfce7" : "#fef9c3";
    document.getElementById("h-db").textContent = data.db_ready ? "ready" : "missing";
    document.getElementById("h-count").textContent = data.db_count;
    document.getElementById("h-ollama").textContent = data.ollama_reachable ? "reachable" : "unreachable";
    document.getElementById("h-pipeline").textContent = data.pipeline_initialized ? "initialized" : "lazy";
    document.getElementById("h-hybrid").textContent = data.hybrid_enabled ? "enabled" : "disabled";
    document.getElementById("h-rrf").textContent = data.rrf_k ?? "-";
    document.getElementById("h-bm25").textContent = data.bm25_ready ? "ready" : "missing";
    document.getElementById("h-models").textContent = data.models ? `${data.models.llm} / ${data.models.reranker}` : "-";
    document.getElementById("h-vdb").textContent = data.vector_db || "ChromaDB HNSW";
    document.getElementById("h-pipe").textContent = data.retrieval_pipeline || "-";
    document.getElementById("h-trace").textContent = trace;
    document.getElementById("corpus-count").textContent = data.db_count || 250;
    document.getElementById("h-config").textContent = JSON.stringify({ hybrid: data.hybrid_config, hnsw: data.hnsw_config, models: data.models, pipeline: data.retrieval_pipeline, latency_ms: latency }, null, 2);
  } catch (e) {
    document.getElementById("health-status").textContent = "error";
    document.getElementById("h-config").textContent = String(e);
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
      const text = `${src.id || "doc"} — ${doc.slice(0, 600)}`;
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
  const traceBadge = document.getElementById("trace-badge");
  btn.disabled = true;
  errEl.classList.add("hidden");
  answerWrap.classList.remove("hidden");
  answerEl.textContent = "";
  renderSources([]);
  const t0 = performance.now();
  try {
    if (useStream) {
      const res = await fetch("/ask/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
      const trace = res.headers.get("X-Trace-Id") || "";
      traceBadge.textContent = trace ? `trace ${trace}` : "";
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
      const trace = res.headers.get("X-Trace-Id") || "";
      traceBadge.textContent = trace ? `trace ${trace}` : "";
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
