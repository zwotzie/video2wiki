import threading
import time

import pytest
from fastapi.testclient import TestClient

from video2wiki.pipeline import PipelineResult
from video2wiki.webserver import create_app
from video2wiki.writer import save_journal


@pytest.fixture
def app(tmp_path):
    return create_app(output=tmp_path)


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _wait_for_job(client, job_id, timeout=5.0):
    """Pollt, bis der Hintergrund-Job abgeschlossen ist."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} nicht innerhalb von {timeout}s fertig geworden.")


class TestIndex:
    def test_returns_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "video2wiki" in r.text
        assert "Video verarbeiten" in r.text
        assert "Journal" in r.text


class TestProcess:
    def test_submits_job(self, mocker, client, tmp_path):
        fake_path = tmp_path / "out" / "cat" / "x.md"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        mock_process = mocker.patch(
            "video2wiki.webserver.process_video",
            return_value=PipelineResult(status="done", entry=None, path=fake_path),
        )

        r = client.post("/api/process", json={"url": "https://youtube.com/watch?v=abc", "category": "musik"})
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "queued"

        job = _wait_for_job(client, data["job_id"])
        assert job["status"] == "done"
        assert job["output_path"] == str(fake_path)
        # die Pipeline wurde mit der eingereichten URL und Kategorie aufgerufen
        _, kwargs = mock_process.call_args
        assert kwargs["emit_log"] is not None
        source = mock_process.call_args[0][0]
        assert source.url == "https://youtube.com/watch?v=abc"
        assert source.category == "musik"

    def test_rejects_blank_url(self, client):
        r = client.post("/api/process", json={"url": "   "})
        assert r.status_code == 422

    def test_rejects_missing_url(self, client):
        r = client.post("/api/process", json={})
        assert r.status_code == 422


class TestJobs:
    def test_initial_empty(self, client):
        r = client.get("/api/jobs")
        assert r.status_code == 200
        assert r.json() == {"jobs": []}

    def test_job_not_found(self, client):
        r = client.get("/api/jobs/does-not-exist")
        assert r.status_code == 404

    def test_lists_jobs(self, mocker, client):
        mocker.patch(
            "video2wiki.webserver.process_video",
            return_value=PipelineResult(status="done", path=None),
        )
        client.post("/api/process", json={"url": "https://youtube.com/watch?v=abc"})
        r = client.get("/api/jobs")
        jobs = r.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["url"] == "https://youtube.com/watch?v=abc"


class TestJournal:
    def test_empty_journal(self, client):
        r = client.get("/api/journal")
        assert r.status_code == 200
        assert r.json() == {"entries": []}

    def test_returns_entries(self, client, tmp_path):
        save_journal(
            tmp_path,
            [
                {
                    "url": "https://youtube.com/watch?v=aaa",
                    "title": "Erstes",
                    "processed_at": "2024-01-01T00:00:00Z",
                },
                {
                    "url": "https://youtube.com/watch?v=zzz",
                    "title": "Zweites",
                    "processed_at": "2024-02-01T00:00:00Z",
                },
            ],
        )
        r = client.get("/api/journal")
        entries = r.json()["entries"]
        assert [e["title"] for e in entries] == ["Zweites", "Erstes"]  # neueste zuerst


class TestBackfill:
    def test_backfills_journal(self, client, tmp_path):
        (tmp_path / "energie").mkdir(parents=True, exist_ok=True)
        md = tmp_path / "energie" / "test-kanal.md"
        md.write_text(
            "---\n"
            'quelle: "https://youtube.com/watch?v=abc"\n'
            'titel: "Test Video"\n'
            'kanal: "Kanal"\n'
            'erscheinungsdatum: "2024-01-01"\n'
            'sprache: "de"\n'
            'transkription: "subtitles"\n'
            "tags: [a]\n"
            "---\n\nContent"
        )
        r = client.post("/api/backfill")
        assert r.status_code == 200
        assert r.json() == {"added": 1}


class TestWikiView:
    def test_returns_markdown(self, client, tmp_path):
        (tmp_path / "cat").mkdir(parents=True, exist_ok=True)
        f = tmp_path / "cat" / "file.md"
        f.write_text("# Hallo\nInhalt", encoding="utf-8")
        r = client.get("/api/wiki/cat/file.md")
        assert r.status_code == 200
        assert r.json()["content"] == "# Hallo\nInhalt"

    def test_missing_file(self, client):
        r = client.get("/api/wiki/nicht/da.md")
        assert r.status_code == 404

    def test_path_traversal_blocked(self, client):
        r = client.get("/api/wiki/..%2F..%2F..%2Fetc%2Fpasswd")
        assert r.status_code == 404


class TestJobQueue:
    """Verifiziert, dass eingereichte Videos gequeued und mit begrenzter
    Parallelität verarbeitet werden (OOM-Schutz auf schwachen Rechnern)."""

    @staticmethod
    def _mk_state():
        return {
            "lock": threading.Lock(),
            "active": 0,
            "max_active": 0,
            "calls": 0,
            "first_started": threading.Event(),
            "second_started": threading.Event(),
        }

    def _blocking_process(self, state, release):
        """process_video-Mock: zählt parallele Aufrufe und blockt bis release."""

        def fake_process(source, options=None, resolve_category=None, emit_log=None):
            with state["lock"]:
                state["calls"] += 1
                if state["calls"] == 1:
                    state["first_started"].set()
                elif state["calls"] == 2:
                    state["second_started"].set()
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                assert release.wait(timeout=5), "release-Event wurde nicht gesetzt"
            finally:
                with state["lock"]:
                    state["active"] -= 1
            return PipelineResult(status="done", entry=None, path=None)

        return fake_process

    @staticmethod
    def _wait_statuses(client, expected, timeout=5.0):
        """Pollt /api/jobs, bis die sortierte Status-Liste `expected` erreicht ist."""

        def snapshot():
            jobs = client.get("/api/jobs").json()["jobs"]
            return sorted(j["status"] for j in jobs)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if snapshot() == expected:
                return
            time.sleep(0.05)
        pytest.fail(f"Statuses erreichten nie {expected} (zuletzt: {snapshot()}).")

    def test_serializes_jobs_at_concurrency_1(self, mocker, client):
        """Default (concurrency=1): nie mehr als ein Job gleichzeitig aktiv."""
        state = self._mk_state()
        release = threading.Event()
        mocker.patch("video2wiki.webserver.process_video", side_effect=self._blocking_process(state, release))

        r1 = client.post("/api/process", json={"url": "https://youtube.com/watch?v=aaa"}).json()
        r2 = client.post("/api/process", json={"url": "https://youtube.com/watch?v=bbb"}).json()

        # Der erste Job läuft, der zweite wartet eindeutig in der Warteschlange
        assert state["first_started"].wait(timeout=5), "Job 1 sollte starten"
        self._wait_statuses(client, ["queued", "running"])

        # Job 1 freigeben → läuft zu Ende, erst danach beginnt Job 2
        release.set()
        _wait_for_job(client, r1["job_id"])
        assert state["second_started"].wait(timeout=5), "Job 2 sollte erst nach Job 1 starten"
        _wait_for_job(client, r2["job_id"])

        assert state["max_active"] == 1, "Es lief nie mehr als ein Job gleichzeitig."

    def test_concurrency_gt_1_allows_parallelism(self, mocker, tmp_path):
        """Mit concurrency=2 können zwei Jobs gleichzeitig laufen, der dritte wartet."""
        app = create_app(output=tmp_path, concurrency=2)
        state = self._mk_state()
        release = threading.Event()
        mocker.patch("video2wiki.webserver.process_video", side_effect=self._blocking_process(state, release))

        with TestClient(app, raise_server_exceptions=True) as client:
            ids = []
            for u in ("aaa", "bbb", "ccc"):
                ids.append(
                    client.post("/api/process", json={"url": f"https://youtube.com/watch?v={u}"}).json()["job_id"]
                )

            # Beide Worker aktiv, der dritte wartet in der Warteschlange
            self._wait_statuses(client, ["queued", "running", "running"])

            release.set()
            for job_id in ids:
                _wait_for_job(client, job_id)

            assert state["max_active"] == 2

    def test_queued_before_running(self, mocker, client):
        """Der Status eines eingereichten Jobs ist initial 'queued'."""
        state = self._mk_state()
        release = threading.Event()
        mocker.patch("video2wiki.webserver.process_video", side_effect=self._blocking_process(state, release))

        r = client.post("/api/process", json={"url": "https://youtube.com/watch?v=aaa"})
        assert r.status_code == 202
        assert r.json()["status"] == "queued"

        release.set()
        _wait_for_job(client, r.json()["job_id"])
