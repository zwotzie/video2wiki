"""FastAPI-Webservice für video2wiki.

Bietet ein kleines Web-UI mit:
- Formular zum Einreichen eines YouTube-Videos (wird im Hintergrund verarbeitet)
- Live-Ansicht laufender Jobs (mit Log-Ausgabe)
- Journal-Ansicht (bereits verarbeitete Videos)

Die Verarbeitung wird von einer kleinen Worker-Queue hinter der API
(``JobManager``) übernommen: Alle eingereichten Videos laufen mit begrenzter
Parallelität (Standard: 1 = sequentiell), damit Download, Whisper-Transkription
und LLM-Zusammenfassung auf ressourcenarmen Rechnern nicht zu OOM führen.
"""

import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .logger import get_logger
from .models import VideoSource
from .pipeline import (
    ST_DONE,
    ST_ERROR_AUDIO,
    ST_ERROR_METADATA,
    ST_ERROR_SUMMARY,
    ST_ERROR_TRANSCRIPTION,
    ST_SKIPPED_EXISTING,
    ST_SKIPPED_JOURNAL,
    PipelineOptions,
    existing_categories,
    process_video,
)
from .writer import backfill_journal, load_journal

log = get_logger("webserver")

_PACKAGE_DIR = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))
_STATIC_DIR = _PACKAGE_DIR / "static"

_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


@dataclass
class WebConfig:
    """Konfiguration des Webservice (entspricht den CLI-Optionen von `web`)."""

    output: Path = Path("./wiki")
    work_dir: Path = Path(".video2wiki_work")
    template: Path | None = None
    api_base: str | None = None
    model: str | None = None
    whisper_model: str = "small"
    concurrency: int = 1


class ProcessRequest(BaseModel):
    """Eingabe für das Video-Formular."""

    url: str = Field(..., min_length=1, description="YouTube-URL")
    category: str | None = Field(None, description="Optionale Wiki-Kategorie")
    whisper_model: str = Field("small", description="Whisper-Modellgröße")

    model_config = {
        "json_schema_extra": {"examples": [{"url": "https://youtube.com/watch?v=abc", "category": "energie"}]}
    }


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobManager:
    """Verwaltet Hintergrund-Verarbeitungsjobs in Memory.

    Die Jobs werden in einer FIFO-Warteschlange (``queue.Queue``) gehalten und
    von einer festen Anzahl Worker-Threads (``concurrency``) abgearbeitet.
    Dadurch laufen Download, Whisper-Transkription und LLM-Zusammenfassung nie
    unbegrenzt parallel – wichtig auf ressourcenarmen Rechnern, um OOM zu
    vermeiden. Default ``concurrency=1`` → strikt sequentiell.
    """

    def __init__(self, config: WebConfig, concurrency: int = 1) -> None:
        self._config = config
        self._concurrency = max(1, concurrency)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._workers = [self._spawn_worker(i) for i in range(self._concurrency)]

    def _spawn_worker(self, index: int) -> threading.Thread:
        worker = threading.Thread(
            target=self._worker_loop,
            name=f"video2wiki-worker-{index}",
            daemon=True,
        )
        worker.start()
        return worker

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            except Exception as e:  # noqa: BLE001 – nie soll ein Worker sterben
                log.error("Worker-Fehler bei Job %s: %s", job_id, e)
            finally:
                self._queue.task_done()

    def submit(self, url: str, category: str | None, whisper_model: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "url": url,
            "category": category,
            "whisper_model": whisper_model,
            "status": "queued",
            "logs": [],
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "output_path": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        # In die Warteschlange einreihen, statt einen eigenen Thread zu starten –
        # ein Worker nimmt den Job sofort oder später (falls schon einer läuft).
        self._queue.put(job_id)
        return job_id

    def _run(self, job_id: str) -> None:
        job = self._jobs[job_id]

        def emit(message: str, level: str) -> None:
            with self._lock:
                job["logs"].append({"ts": _now(), "level": level, "message": message})

        with self._lock:
            job["status"] = "running"
            job["started_at"] = _now()

        try:
            options = PipelineOptions(
                output=self._config.output,
                category=job["category"],
                template=self._config.template,
                api_base=self._config.api_base,
                model=self._config.model,
                whisper_model=job["whisper_model"],
                work_dir=self._config.work_dir,
            )
            result = process_video(
                VideoSource(url=job["url"], category=job["category"]),
                options,
                emit_log=emit,
            )
            if result.status == ST_DONE:
                self._finalize(job, "done", output_path=str(result.path))
            elif result.status in (ST_SKIPPED_JOURNAL, ST_SKIPPED_EXISTING):
                self._finalize(job, "skipped")
            else:
                reason = _status_message(result.status)
                with self._lock:
                    job["error"] = reason
                self._finalize(job, "failed")
        except Exception as e:  # unerwartete Fehler abfangen
            emit(str(e), "error")
            with self._lock:
                job["error"] = str(e)
            self._finalize(job, "failed")

    def _finalize(self, job: dict, status: str, output_path: str | None = None) -> None:
        with self._lock:
            job["status"] = status
            job["finished_at"] = _now()
            if output_path is not None:
                job["output_path"] = output_path

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._job_dict(job) if job else None

    def all(self) -> list[dict]:
        with self._lock:
            sorted_jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
            return [self._job_dict(j) for j in sorted_jobs]

    @staticmethod
    def _job_dict(job: dict) -> dict:
        return {
            "id": job["id"],
            "url": job["url"],
            "category": job["category"],
            "status": job["status"],
            "logs": job["logs"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "output_path": job["output_path"],
            "error": job["error"],
        }


def _status_message(status: str) -> str:
    messages = {
        ST_ERROR_METADATA: "Konnte Metadaten nicht abrufen – Video existiert nicht oder ist nicht erreichbar.",
        ST_ERROR_AUDIO: "Audio-Download fehlgeschlagen.",
        ST_ERROR_TRANSCRIPTION: "Transkription ist leer.",
        ST_ERROR_SUMMARY: "Zusammenfassung fehlgeschlagen.",
    }
    return messages.get(status, f"Verarbeitung fehlgeschlagen (Status: {status}).")


def create_app(
    output: Path = Path("./wiki"),
    work_dir: Path = Path(".video2wiki_work"),
    template: Path | None = None,
    api_base: str | None = None,
    model: str | None = None,
    whisper_model: str = "small",
    concurrency: int = 1,
) -> FastAPI:
    """Baut die FastAPI-App für den Webservice auf.

    Args:
        concurrency: Anzahl gleichzeitig laufender Verarbeitungsjobs.
            Default 1 (sequentiell) – verhindert OOM auf ressourcenarmen Rechnern.
    """
    config = WebConfig(
        output=output,
        work_dir=work_dir,
        template=template,
        api_base=api_base,
        model=model,
        whisper_model=whisper_model,
        concurrency=concurrency,
    )
    manager = JobManager(config, concurrency=concurrency)

    app = FastAPI(title="video2wiki", version="0.1.0", description="YouTube → Transkription → Zusammenfassung → Wiki")
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "categories": existing_categories(config.output),
                "whisper_models": _WHISPER_MODELS,
                "default_whisper_model": config.whisper_model,
            },
        )

    @app.post("/api/process")
    def api_process(request: ProcessRequest):
        """Reicht ein Video zur Verarbeitung ein; läuft asynchron im Hintergrund."""
        if not request.url.strip():
            raise HTTPException(status_code=422, detail="URL darf nicht leer sein.")
        job_id = manager.submit(request.url.strip(), request.category, request.whisper_model)
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})

    @app.get("/api/jobs")
    def api_jobs():
        return {"jobs": manager.all()}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job nicht gefunden.")
        return job

    @app.get("/api/journal")
    def api_journal():
        entries = load_journal(config.output)
        entries = sorted(entries, key=lambda e: e.get("processed_at", ""), reverse=True)
        return {"entries": entries}

    @app.post("/api/backfill")
    def api_backfill():
        count = backfill_journal(config.output)
        return {"added": count}

    @app.get("/api/wiki/{path:path}")
    def api_wiki(path: str):
        """Liefert den Inhalt eines Wiki-Markdown (zur Anzeige)."""
        base = config.output.resolve()
        target = (base / path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise HTTPException(status_code=404, detail="Wiki-Eintrag nicht gefunden.")
        return {"path": path, "content": target.read_text(encoding="utf-8")}

    return app


# Kompatibilität mit `from video2wiki.webserver import app`
app = create_app()
