// CSV Analyst frontend: talks to the Django REST API under /api/.
// All server and model text is inserted with textContent, never as HTML.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    uploadStatus: $("upload-status"),
    datasetList: $("dataset-list"),
    datasetEmpty: $("dataset-empty"),
    header: $("chat-header"),
    datasetName: $("dataset-name"),
    datasetMeta: $("dataset-meta"),
    columnList: $("column-list"),
    messages: $("messages"),
    welcome: $("welcome"),
    suggestions: $("suggestions"),
    composer: $("composer"),
    question: $("question"),
    send: $("send"),
  };

  const state = { datasets: [], current: null, busy: false };
  const STORAGE_KEY = "csv-analyst:dataset";

  // ---------- helpers ----------
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  async function api(path, options = {}) {
    const response = await fetch(`/api/${path}`, {
      headers: { Accept: "application/json", ...(options.headers || {}) },
      ...options,
    });
    let data = null;
    try { data = await response.json(); } catch { /* empty or non-JSON body */ }
    if (!response.ok && !(data && data.assistant_message)) {
      const detail = (data && (data.detail || data.question || data.file)) || `Request failed (${response.status})`;
      throw new Error(Array.isArray(detail) ? detail.join(" ") : String(detail));
    }
    return data;
  }

  const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  }

  function formatCell(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "number") return numberFormat.format(value);
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  }

  function rememberDataset(id) {
    try { localStorage.setItem(STORAGE_KEY, String(id)); } catch { /* storage unavailable */ }
  }

  function rememberedDataset() {
    try { return Number(localStorage.getItem(STORAGE_KEY)) || null; } catch { return null; }
  }

  function scrollToBottom() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  // ---------- datasets ----------
  function renderDatasetList() {
    els.datasetList.replaceChildren();
    els.datasetEmpty.hidden = state.datasets.length > 0;
    for (const dataset of state.datasets) {
      const item = el("li", "dataset-item");
      if (state.current && state.current.id === dataset.id) item.classList.add("active");

      const select = el("button", "select");
      select.type = "button";
      select.append(
        el("span", "name", dataset.name),
        el("span", "meta", `${dataset.row_count.toLocaleString()} rows · ${dataset.columns.length} columns`),
      );
      select.title = dataset.name;
      select.addEventListener("click", () => selectDataset(dataset.id));

      const remove = el("button", "delete", "✕");
      remove.type = "button";
      remove.title = `Delete ${dataset.name}`;
      remove.setAttribute("aria-label", `Delete ${dataset.name}`);
      remove.addEventListener("click", () => deleteDataset(dataset));

      item.append(select, remove);
      els.datasetList.append(item);
    }
  }

  async function loadDatasets() {
    state.datasets = await api("datasets/");
    renderDatasetList();
    const remembered = rememberedDataset();
    const initial = state.datasets.find((d) => d.id === remembered) || state.datasets[0];
    if (initial) await selectDataset(initial.id);
  }

  async function selectDataset(id) {
    const dataset = state.datasets.find((d) => d.id === id);
    if (!dataset) return;
    state.current = dataset;
    rememberDataset(id);
    renderDatasetList();

    els.header.hidden = false;
    els.datasetName.textContent = dataset.name;
    els.datasetMeta.textContent =
      `${dataset.row_count.toLocaleString()} rows · ${dataset.columns.length} columns · ${formatBytes(dataset.size_bytes)}` +
      (dataset.duplicates ? ` · ${dataset.duplicates} duplicate rows removed` : "");
    els.columnList.replaceChildren(
      ...dataset.columns.map((column) => {
        const li = el("li");
        li.append(el("code", null, column.name), el("span", null, column.kind));
        return li;
      }),
    );

    els.question.disabled = false;
    els.question.placeholder = `Ask about ${dataset.name}…`;
    updateSendState();

    els.messages.replaceChildren();
    let history;
    try {
      history = await api(`datasets/${id}/messages/`);
    } catch (error) {
      if (state.current && state.current.id === id) {
        els.messages.append(renderAssistant({ kind: "error", answer: `Could not load the chat history: ${error.message}` }));
      }
      return;
    }
    if (!state.current || state.current.id !== id) return;
    if (history.length === 0) {
      const intro = el("div", "welcome");
      intro.append(
        el("h1", null, dataset.name),
        el("p", null, "Ask a question below, or start with one of the suggestions."),
      );
      els.messages.append(intro);
    }
    for (const message of history) renderMessage(message);
    renderSuggestions(history.length === 0 ? dataset.suggestions : []);
    scrollToBottom();
    els.question.focus();
  }

  async function deleteDataset(dataset) {
    if (!confirm(`Delete "${dataset.name}" and its chat history?`)) return;
    try {
      await api(`datasets/${dataset.id}/`, { method: "DELETE" });
    } catch (error) {
      setUploadStatus(error.message, true);
      return;
    }
    state.datasets = state.datasets.filter((d) => d.id !== dataset.id);
    if (state.current && state.current.id === dataset.id) {
      state.current = null;
      resetChat();
      if (state.datasets[0]) await selectDataset(state.datasets[0].id);
    }
    renderDatasetList();
  }

  function resetChat() {
    els.header.hidden = true;
    els.messages.replaceChildren(els.welcome);
    els.suggestions.replaceChildren();
    els.question.value = "";
    els.question.disabled = true;
    els.question.placeholder = "Upload a CSV to start…";
    updateSendState();
  }

  // ---------- upload ----------
  function setUploadStatus(text, isError = false) {
    els.uploadStatus.textContent = text;
    els.uploadStatus.classList.toggle("error", isError);
  }

  async function uploadFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setUploadStatus("Please choose a .csv file.", true);
      return;
    }
    const form = new FormData();
    form.append("file", file);
    els.dropzone.classList.add("busy");
    setUploadStatus(`Uploading and profiling ${file.name}…`);
    try {
      const dataset = await api("datasets/", { method: "POST", body: form });
      state.datasets.unshift(dataset);
      setUploadStatus(`Loaded ${dataset.name}.`);
      await selectDataset(dataset.id);
    } catch (error) {
      setUploadStatus(error.message, true);
    } finally {
      els.dropzone.classList.remove("busy");
      els.fileInput.value = "";
    }
  }

  els.fileInput.addEventListener("change", () => uploadFile(els.fileInput.files[0]));
  for (const type of ["dragenter", "dragover"]) {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("dragging");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    els.dropzone.addEventListener(type, () => els.dropzone.classList.remove("dragging"));
  }
  els.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadFile(event.dataTransfer.files[0]);
  });

  // ---------- rendering messages ----------
  function renderTable(table) {
    const wrap = el("div", "table-wrap");
    if (!table || !table.rows.length) {
      wrap.append(el("p", "sub", "No rows."));
      return wrap;
    }
    const numeric = table.columns.map((_, index) =>
      table.rows.every((row) => row[index] === null || typeof row[index] === "number"));
    const tableEl = el("table");
    const head = el("thead");
    const headRow = el("tr");
    table.columns.forEach((column, index) => {
      const th = el("th", numeric[index] ? "num" : null, column);
      th.scope = "col";
      headRow.append(th);
    });
    head.append(headRow);
    const body = el("tbody");
    for (const row of table.rows) {
      const tr = el("tr");
      row.forEach((value, index) => tr.append(el("td", numeric[index] ? "num" : null, formatCell(value))));
      body.append(tr);
    }
    tableEl.append(head, body);
    wrap.append(tableEl);
    return wrap;
  }

  function renderList(title, items, className = "sub") {
    const block = el("div", className);
    block.append(el("strong", null, title));
    const list = el("ul");
    for (const item of items) list.append(el("li", null, item));
    block.append(list);
    return block;
  }

  function renderAssistant(payload, content) {
    const kind = payload.kind || "answer";
    const wrapper = el("div", `msg assistant ${kind === "error" ? "error" : ""}`);
    const bubble = el("div", "bubble");
    wrapper.append(bubble);

    const labels = { unanswerable: "Can't answer from this data", failed: "Query failed", error: "Error", churn: "Churn associations — not causes" };
    if (labels[kind]) bubble.append(el("span", "label", labels[kind]));

    bubble.append(el("p", "answer", payload.answer || content));

    if (payload.suggestions && payload.suggestions.length) {
      const chips = el("div", "inline-suggestions");
      for (const text of payload.suggestions) {
        const chip = el("button", null, text);
        chip.type = "button";
        chip.addEventListener("click", () => ask(text));
        chips.append(chip);
      }
      bubble.append(chips);
    }

    if (payload.profile) bubble.append(el("pre", "profile", payload.profile));

    if (payload.notes && payload.notes.length) {
      const notes = el("ul", "notes");
      for (const note of payload.notes) notes.append(el("li", null, note));
      bubble.append(notes);
    }

    if (payload.chart_url) {
      const link = el("a", "chart");
      link.href = payload.chart_url;
      link.target = "_blank";
      link.rel = "noopener";
      const img = el("img");
      img.src = payload.chart_url;
      img.alt = `Bar chart: ${payload.question || "query result"}`;
      img.loading = "lazy";
      link.append(img);
      bubble.append(link);
    } else if (payload.chart_error) {
      const callout = el("div", "callout warn", payload.chart_error);
      bubble.append(callout);
    }

    for (const table of payload.tables || []) {
      if (table.title) bubble.append(el("p", "table-title", table.title));
      bubble.append(renderTable(table));
    }

    if (payload.warnings && payload.warnings.length) {
      const callout = el("div", "callout warn");
      callout.append(el("strong", null, "Warnings"));
      const list = el("ul");
      for (const warning of payload.warnings) list.append(el("li", null, warning));
      callout.append(list);
      bubble.append(callout);
    }

    const hasDetails = payload.sql || payload.table || (payload.assumptions && payload.assumptions.length) || (payload.repairs && payload.repairs.length);
    if (hasDetails) {
      const details = el("details", "more");
      const rows = payload.row_count ?? (payload.table ? payload.table.rows.length : 0);
      details.append(el("summary", null, payload.table ? `SQL and result (${rows} rows${payload.truncated ? ", truncated" : ""})` : "SQL and details"));
      if (payload.sql) {
        const pre = el("pre", "sql", payload.sql);
        const copy = el("button", "copy", "Copy");
        copy.type = "button";
        copy.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(payload.sql);
            copy.textContent = "Copied";
          } catch {
            copy.textContent = "Copy failed";
          }
          setTimeout(() => { copy.textContent = "Copy"; }, 1500);
        });
        pre.append(copy);
        details.append(pre);
      }
      if (payload.table) details.append(renderTable(payload.table));
      if (payload.assumptions && payload.assumptions.length) details.append(renderList("Assumptions", payload.assumptions));
      if (payload.repairs && payload.repairs.length) details.append(renderList("Query errors the model fixed", payload.repairs));
      bubble.append(details);
    }

    const footer = el("div", "footer");
    if (payload.grounded === true) footer.append(el("span", "badge ok", "Numbers verified against result"));
    if (payload.grounded === false) footer.append(el("span", "badge warn", "Unverified — check the table"));
    if (payload.elapsed_s) footer.append(el("span", null, `${payload.elapsed_s.toFixed(1)}s`));
    if (payload.llm_report) {
      const report = el("span", null, payload.llm_report.split(" | ")[0]);
      report.title = payload.llm_report;
      footer.append(report);
    }
    if (footer.childElementCount) bubble.append(footer);
    return wrapper;
  }

  function renderMessage(message) {
    if (message.role === "user") {
      const wrapper = el("div", "msg user");
      wrapper.append(el("div", "bubble", message.content));
      els.messages.append(wrapper);
    } else {
      els.messages.append(renderAssistant(message.payload || {}, message.content));
    }
  }

  function renderSuggestions(items) {
    els.suggestions.replaceChildren();
    for (const text of items || []) {
      const button = el("button", null, text);
      button.type = "button";
      button.addEventListener("click", () => ask(text));
      els.suggestions.append(button);
    }
  }

  // ---------- asking ----------
  function updateSendState() {
    els.send.disabled = state.busy || !state.current || !els.question.value.trim();
  }

  function autoResize() {
    els.question.style.height = "auto";
    els.question.style.height = `${Math.min(els.question.scrollHeight, 180)}px`;
  }

  async function ask(text) {
    const question = text.trim();
    if (!question || state.busy || !state.current) return;
    const datasetId = state.current.id;
    state.busy = true;
    updateSendState();
    renderSuggestions([]);
    const intro = els.messages.querySelector(".welcome");
    if (intro) intro.remove();

    renderMessage({ role: "user", content: question });
    els.question.value = "";
    autoResize();

    const thinking = el("div", "msg assistant");
    const bubble = el("div", "bubble thinking");
    const dots = el("span", "dots");
    dots.append(el("i"), el("i"), el("i"));
    const status = el("span", null, "Analyzing…");
    bubble.append(dots, status);
    thinking.append(bubble);
    els.messages.append(thinking);
    scrollToBottom();

    const started = Date.now();
    const timer = setInterval(() => {
      const seconds = Math.round((Date.now() - started) / 1000);
      status.textContent = seconds < 8
        ? "Analyzing…"
        : `Analyzing… ${seconds}s (a local model can take a minute)`;
    }, 1000);

    try {
      const data = await api(`datasets/${datasetId}/ask/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      thinking.remove();
      if (state.current && state.current.id === datasetId) renderMessage(data.assistant_message);
    } catch (error) {
      thinking.remove();
      els.messages.append(renderAssistant({ kind: "error", answer: error.message }));
    } finally {
      clearInterval(timer);
      state.busy = false;
      updateSendState();
      scrollToBottom();
      els.question.focus();
    }
  }

  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(els.question.value);
  });
  els.question.addEventListener("input", () => {
    autoResize();
    updateSendState();
  });
  els.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ask(els.question.value);
    }
  });

  loadDatasets().catch((error) => setUploadStatus(`Could not load datasets: ${error.message}`, true));
})();
