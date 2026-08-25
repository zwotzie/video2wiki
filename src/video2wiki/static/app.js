(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const STATUS_LABEL = { queued: "Wartet", running: "Läuft", done: "Fertig", skipped: "Übersprungen", failed: "Fehlgeschlagen" };

  // ---- Formular ----
  const form = $("#submit-form");
  const statusEl = $("#submit-status");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("#submit-btn");
    btn.disabled = true;
    statusEl.className = "status";
    statusEl.textContent = "Reiche Video ein …";

    try {
      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: $("#url").value.trim(),
          category: $("#category").value.trim() || null,
          whisper_model: $("#whisper_model").value,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      statusEl.className = "status ok";
      statusEl.textContent = `Job ${data.job_id} gestartet – verarbeite …`;
      $("#url").value = "";
      $("#category").value = "";
      refreshJobs();
    } catch (err) {
      statusEl.className = "status err";
      statusEl.textContent = `Fehler: ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  });

  // ---- Jobs (Polling alle 2s) ----
  const jobsEl = $("#jobs");
  let jobsRendered = new Set();

  function esc(text) {
    const div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
  }

  function jobHtml(job) {
    const logs = (job.logs || [])
      .map((l) => `<div class="line ${l.level}">${esc(l.message)}</div>`)
      .join("");
    const cat = job.category ? `<span class="job-cat">(${esc(job.category)})</span>` : "";

    let extra = "";
    if (job.status === "done" && job.output_path) {
      extra = ` · <a href="#" class="view-wiki" data-path="${esc(job.output_path)}">Wiki anzeigen</a>`;
    } else if (job.status === "failed" && job.error) {
      extra = ` · ${esc(job.error)}`;
    }

    return `<div class="job" data-id="${esc(job.id)}">
      <div class="job-head">
        <span class="badge ${job.status}">${STATUS_LABEL[job.status] || job.status}</span>
        <span class="job-url">${esc(job.url)}</span>${cat}
        <span class="job-cat">${job.status === "running" || job.status === "queued" ? "…" : ""}</span>
        ${extra}
      </div>
      <div class="job-logs">${logs}</div>
    </div>`;
  }

  async function refreshJobs() {
    try {
      const res = await fetch("/api/jobs");
      const data = await res.json();
      const jobs = data.jobs || [];
      if (!jobs.length) {
        jobsEl.textContent = "Keine Jobs vorhanden.";
        return;
      }
      jobsEl.innerHTML = jobs.map(jobHtml).join("");
    } catch {
      /* Netzwerkfehler ignorieren – nächster Poll */
    }
  }

  // ---- Journal (Polling alle 5s) ----
  const tbody = $("#journal-table tbody");

  async function refreshJournal() {
    try {
      const res = await fetch("/api/journal");
      const data = await res.json();
      const entries = data.entries || [];
      if (!entries.length) {
        tbody.innerHTML = '<tr><td colspan="5">Noch keine Einträge im Journal.</td></tr>';
        return;
      }
      tbody.innerHTML = entries
        .slice(0, 100)
        .map((e) => {
          const path = e.output_path || "";
          const link = path
            ? `<a href="#" class="view-wiki" data-path="${esc(path)}">${esc(e.title || e.url)}</a>`
            : esc(e.title || e.url);
          return `<tr>
            <td>${link}</td>
            <td>${esc(e.channel || "")}</td>
            <td>${esc(e.category || "")}</td>
            <td>${esc(e.processed_at || "")}</td>
            <td>${e.transcription_source ? esc(e.transcription_source) : ""}</td>
          </tr>`;
        })
        .join("");
    } catch {
      /* nächster Poll */
    }
  }

  // ---- Wiki anzeigen (Modal) ----
  const modal = $("#modal");
  const modalTitle = $("#modal-title");
  const modalContent = $("#modal-content");

  document.addEventListener("click", async (e) => {
    const link = e.target.closest(".view-wiki");
    if (!link) return;
    e.preventDefault();
    const path = link.dataset.path;
    if (!path) return;
    modalTitle.textContent = path;
    modalContent.textContent = "Lade …";
    modal.classList.remove("hidden");
    try {
      const res = await fetch(`/api/wiki/${encodeURI(path).replace(/%2F/g, "/")}`);
      const data = await res.json();
      modalContent.textContent = res.ok ? data.content : (data.detail || "Fehler beim Laden.");
    } catch (err) {
      modalContent.textContent = `Fehler: ${err.message}`;
    }
  });

  $("#modal-close").addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });

  // ---- Backfill ----
  $("#backfill-btn").addEventListener("click", async () => {
    const btn = $("#backfill-btn");
    btn.disabled = true;
    try {
      const res = await fetch("/api/backfill", { method: "POST" });
      const data = await res.json();
      btn.textContent = `Backfill: ${data.added} hinzugefügt`;
      refreshJournal();
    } finally {
      btn.disabled = false;
    }
  });

  // ---- Start ----
  refreshJobs();
  refreshJournal();
  setInterval(refreshJobs, 2000);
  setInterval(refreshJournal, 5000);
})();
