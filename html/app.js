(() => {
  "use strict";

  const API = "/api/v1";

  const els = {
    thread: document.getElementById("thread"),
    form: document.getElementById("ask-form"),
    question: document.getElementById("question"),
    mode: document.getElementById("mode"),
    submit: document.getElementById("ask-submit"),
    askStatus: document.getElementById("ask-status"),
    confidenceValue: document.getElementById("confidence-value"),
    confidenceMeter: document.getElementById("confidence-meter"),
    confidenceNote: document.getElementById("confidence-note"),
    modeUsed: document.getElementById("mode-used"),
    sources: document.getElementById("sources"),
    sourcesEmpty: document.getElementById("sources-empty"),
    healthChip: document.getElementById("health-chip"),
    healthDetail: document.getElementById("health-detail"),
    desk: document.getElementById("desk"),
    deskOpen: document.getElementById("desk-open"),
    deskClose: document.getElementById("desk-close"),
    deskScrim: document.getElementById("desk-scrim"),
    ingestForm: document.getElementById("ingest-form"),
    docTitle: document.getElementById("doc-title"),
    docText: document.getElementById("doc-text"),
    docUri: document.getElementById("doc-uri"),
    ingestStatus: document.getElementById("ingest-status"),
    ingestSubmit: document.getElementById("ingest-submit"),
    mediaDrop: document.getElementById("media-drop"),
    mediaInput: document.getElementById("media-input"),
    docList: document.getElementById("doc-list"),
    docsEmpty: document.getElementById("docs-empty"),
    docsRefresh: document.getElementById("docs-refresh"),
    commList: document.getElementById("comm-list"),
    commEmpty: document.getElementById("comm-empty"),
    commRebuild: document.getElementById("comm-rebuild"),
    commStatus: document.getElementById("comm-status"),
  };

  let deskOpen = false;
  let lastFocus = null;

  function setStatus(el, message, tone = "") {
    el.textContent = message || "";
    if (tone) el.dataset.tone = tone;
    else delete el.dataset.tone;
  }

  async function api(path, options = {}) {
    const res = await fetch(`${API}${path}`, {
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
      ...options,
    });
    let data = null;
    const text = await res.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { detail: text };
      }
    }
    if (!res.ok) {
      const detail =
        (data && (data.detail || data.message)) ||
        `${res.status} ${res.statusText}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function formatPct(score) {
    if (score == null || Number.isNaN(Number(score))) return "—";
    return `${Math.round(Number(score) * 100)}%`;
  }

  function appendBubble(role, text) {
    const article = document.createElement("article");
    article.className = `bubble bubble--${role}`;
    article.innerHTML = `
      <p class="bubble__label">${role === "user" ? "You" : "GraphRAG"}</p>
      <p class="bubble__body"></p>
    `;
    article.querySelector(".bubble__body").textContent = text;
    els.thread.appendChild(article);
    article.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function renderProof(payload) {
    const confidence = payload.confidence;
    els.confidenceValue.textContent =
      confidence == null ? "—" : formatPct(confidence);
    els.confidenceMeter.style.width =
      confidence == null ? "0%" : `${Math.max(0, Math.min(1, confidence)) * 100}%`;
    els.modeUsed.textContent = `Mode · ${payload.mode_used || "—"}`;
    els.confidenceNote.textContent =
      confidence == null
        ? "No retrieval scores available for this answer."
        : "Aggregate of top retrieval scores. Treat it as guidance, not a warranty.";

    els.sources.innerHTML = "";
    const sources = payload.sources || [];
    els.sourcesEmpty.hidden = sources.length > 0;
    sources.forEach((source, index) => {
      const li = document.createElement("li");
      li.className = "source";
      li.style.animationDelay = `${index * 40}ms`;
      li.innerHTML = `
        <div class="source__top">
          <span class="source__name">${escapeHtml(source.node_name || "Untitled")}</span>
          <span class="source__score">${formatPct(source.score)}</span>
        </div>
        <p class="source__excerpt"></p>
      `;
      li.querySelector(".source__excerpt").textContent = source.excerpt || "";
      els.sources.appendChild(li);
    });
  }

  async function askQuestion(event) {
    event.preventDefault();
    const question = els.question.value.trim();
    if (!question) return;

    appendBubble("user", question);
    els.question.value = "";
    els.submit.disabled = true;
    els.submit.dataset.state = "loading";
    els.submit.textContent = "Working";
    setStatus(els.askStatus, "Retrieving…");

    try {
      const payload = await api("/qa", {
        method: "POST",
        body: JSON.stringify({
          question,
          mode: els.mode.value,
          include_sources: true,
        }),
      });
      appendBubble("assistant", payload.answer || "(empty answer)");
      renderProof(payload);
      setStatus(els.askStatus, "Answered", "ok");
    } catch (err) {
      appendBubble("assistant", `Could not answer: ${err.message}`);
      setStatus(els.askStatus, err.message, "error");
    } finally {
      els.submit.disabled = false;
      delete els.submit.dataset.state;
      els.submit.textContent = "Ask";
    }
  }

  function openDesk() {
    lastFocus = document.activeElement;
    deskOpen = true;
    els.desk.hidden = false;
    requestAnimationFrame(() => els.desk.classList.add("is-open"));
    els.deskClose.focus();
    document.body.style.overflow = "hidden";
    refreshDesk();
  }

  function closeDesk() {
    deskOpen = false;
    els.desk.classList.remove("is-open");
    document.body.style.overflow = "";
    window.setTimeout(() => {
      if (!deskOpen) els.desk.hidden = true;
    }, 220);
    if (lastFocus && typeof lastFocus.focus === "function") lastFocus.focus();
  }

  async function loadHealth() {
    try {
      const health = await api("/health");
      const ok = health.status === "ok" || health.db === true;
      els.healthChip.textContent = ok
        ? `API · ${health.llm_model || "ready"}`
        : "API · degraded";
      els.healthDetail.innerHTML = `
        <span>Status · ${escapeHtml(health.status || "unknown")}</span>
        <span>DB · ${health.db ? "up" : "down"}</span>
        <span>Embed dim · ${escapeHtml(String(health.embedding_dim ?? "—"))}</span>
        <span>LLM · ${escapeHtml(health.llm_model || "—")}</span>
        <span>Embed · ${escapeHtml(health.embedding_model || "—")}</span>
      `;
    } catch (err) {
      els.healthChip.textContent = "API · unreachable";
      els.healthDetail.textContent = err.message;
    }
  }

  async function loadDocuments() {
    try {
      const docs = await api("/documents?limit=50");
      els.docList.innerHTML = "";
      els.docsEmpty.hidden = docs.length > 0;
      docs.forEach((doc) => {
        const li = document.createElement("li");
        li.className = "doc-item";
        const counts = doc.counts || {};
        li.innerHTML = `
          <div class="doc-item__title"></div>
          <div class="doc-item__meta"></div>
          <button type="button" class="btn btn--ghost" data-reindex></button>
        `;
        li.querySelector(".doc-item__title").textContent = doc.title;
        li.querySelector(".doc-item__meta").textContent = [
          doc.status || "unknown",
          counts.chunks != null ? `${counts.chunks} chunks` : null,
          counts.entities != null ? `${counts.entities} entities` : null,
        ]
          .filter(Boolean)
          .join(" · ");
        const btn = li.querySelector("[data-reindex]");
        btn.textContent = "Reindex";
        btn.addEventListener("click", async () => {
          btn.disabled = true;
          btn.dataset.state = "loading";
          btn.textContent = "Queued";
          try {
            await api(`/documents/${doc.id}/reindex`, { method: "POST" });
            setStatus(els.ingestStatus, `Reindex queued for ${doc.title}`, "ok");
            await loadDocuments();
          } catch (err) {
            setStatus(els.ingestStatus, err.message, "error");
            btn.disabled = false;
            delete btn.dataset.state;
            btn.textContent = "Reindex";
          }
        });
        els.docList.appendChild(li);
      });
    } catch (err) {
      els.docsEmpty.hidden = false;
      els.docsEmpty.textContent = err.message;
    }
  }

  async function loadCommunities() {
    try {
      const communities = await api("/communities?limit=50");
      els.commList.innerHTML = "";
      els.commEmpty.hidden = communities.length > 0;
      communities.forEach((c) => {
        const li = document.createElement("li");
        li.className = "doc-item";
        li.innerHTML = `
          <div class="doc-item__title"></div>
          <div class="doc-item__meta"></div>
        `;
        li.querySelector(".doc-item__title").textContent =
          c.label || c.title || `Community ${String(c.id).slice(0, 8)}`;
        li.querySelector(".doc-item__meta").textContent = [
          c.member_count != null ? `${c.member_count} members` : null,
          c.summary ? "has summary" : null,
        ]
          .filter(Boolean)
          .join(" · ");
        els.commList.appendChild(li);
      });
    } catch (err) {
      els.commEmpty.hidden = false;
      els.commEmpty.textContent = err.message;
    }
  }

  function refreshDesk() {
    loadHealth();
    loadDocuments();
    loadCommunities();
  }

  async function ingestDocument(event) {
    event.preventDefault();
    const title = els.docTitle.value.trim();
    const text = els.docText.value.trim();
    const source_uri = els.docUri.value.trim() || null;
    if (!title || !text) return;

    els.ingestSubmit.disabled = true;
    els.ingestSubmit.dataset.state = "loading";
    els.ingestSubmit.textContent = "Indexing";
    setStatus(els.ingestStatus, "Submitting…");

    try {
      const payload = await api("/documents", {
        method: "POST",
        body: JSON.stringify({ title, text, source_uri, props: {} }),
      });
      setStatus(
        els.ingestStatus,
        `Queued · ${payload.document?.status || "pending"}`,
        "ok"
      );
      els.ingestForm.reset();
      await loadDocuments();
    } catch (err) {
      setStatus(els.ingestStatus, err.message, "error");
    } finally {
      els.ingestSubmit.disabled = false;
      delete els.ingestSubmit.dataset.state;
      els.ingestSubmit.textContent = "Index document";
    }
  }

  async function handleFile(file) {
    if (!file) return;
    const name = file.name || "untitled";
    if (!els.docTitle.value.trim()) {
      els.docTitle.value = name.replace(/\.[^.]+$/, "") || name;
    }
    if (!els.docUri.value.trim()) {
      els.docUri.value = name;
    }

    const isText =
      file.type.startsWith("text/") ||
      /\.(txt|md|markdown|csv|json|html|xml|log)$/i.test(name);

    if (isText) {
      const text = await file.text();
      els.docText.value = text;
      setStatus(els.ingestStatus, `Loaded text from ${name}`, "ok");
      return;
    }

    const kind = file.type || "application/octet-stream";
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    const caption = [
      `Media attachment: ${name}`,
      `Type: ${kind}`,
      `Size: ${sizeKb} KB`,
      "",
      "The GraphRAG API indexes text only. Describe what this file contains below,",
      "or replace this block with a transcript / OCR / notes before indexing.",
      "",
      els.docText.value.trim(),
    ]
      .filter((line, i, arr) => !(line === "" && arr[i - 1] === ""))
      .join("\n")
      .trim();
    els.docText.value = caption;
    setStatus(
      els.ingestStatus,
      `Media noted as captioned text · ${name}`,
      "ok"
    );
  }

  function bindDropZone() {
    const zone = els.mediaDrop;
    const input = els.mediaInput;

    const activate = () => input.click();
    zone.addEventListener("click", activate);
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        activate();
      }
    });
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      handleFile(file);
      input.value = "";
    });

    ["dragenter", "dragover"].forEach((type) => {
      zone.addEventListener(type, (e) => {
        e.preventDefault();
        zone.classList.add("is-drag");
      });
    });
    ["dragleave", "drop"].forEach((type) => {
      zone.addEventListener(type, (e) => {
        e.preventDefault();
        zone.classList.remove("is-drag");
      });
    });
    zone.addEventListener("drop", (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      handleFile(file);
    });
  }

  async function rebuildCommunities() {
    els.commRebuild.disabled = true;
    els.commRebuild.dataset.state = "loading";
    els.commRebuild.textContent = "Rebuilding";
    setStatus(els.commStatus, "Rebuilding communities…");
    try {
      const result = await api("/communities/rebuild", { method: "POST" });
      const n = (result.communities || []).length;
      setStatus(els.commStatus, `Rebuilt · ${n} communities`, "ok");
      await loadCommunities();
    } catch (err) {
      setStatus(els.commStatus, err.message, "error");
    } finally {
      els.commRebuild.disabled = false;
      delete els.commRebuild.dataset.state;
      els.commRebuild.textContent = "Rebuild";
    }
  }

  els.form.addEventListener("submit", askQuestion);
  els.deskOpen.addEventListener("click", openDesk);
  els.deskClose.addEventListener("click", closeDesk);
  els.deskScrim.addEventListener("click", closeDesk);
  els.docsRefresh.addEventListener("click", loadDocuments);
  els.commRebuild.addEventListener("click", rebuildCommunities);
  els.ingestForm.addEventListener("submit", ingestDocument);
  bindDropZone();

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && deskOpen) closeDesk();
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && document.activeElement === els.question) {
      els.form.requestSubmit();
    }
  });

  loadHealth();
})();
