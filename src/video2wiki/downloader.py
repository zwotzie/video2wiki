import json
import os
import subprocess
import time
from pathlib import Path

from .logger import get_logger

log = get_logger("downloader")

# Verzögerung zwischen yt-dlp-Aufrufen (Sekunden); in Tests auf 0 gesetzt
RATE_LIMIT_DELAY = 1.5


def _ytdlp_extra_args() -> list[str]:
    """Zusätzliche yt-dlp-Argumente gegen YouTube-Blockaden (HTTP 403).

    Seit Mitte 2026 liefert der Default-Player-Client von yt-dlp bei YouTube
    Streaming-URLs, die von YouTube mit 403 blockiert werden. Der `android`-Client
    zusammen mit einer erlaubten JavaScript-Runtime (z. B. node) umgeht das.
    Über die Umgebungsvariablen YTDLP_JS_RUNTIME und YTDLP_PLAYER_CLIENT
    konfigurierbar.
    """
    js_runtime = os.environ.get("YTDLP_JS_RUNTIME", "node")
    player_client = os.environ.get("YTDLP_PLAYER_CLIENT", "android")
    return [
        "--js-runtimes",
        js_runtime,
        "--extractor-args",
        f"youtube:player_client={player_client}",
    ]


def _log_ytdlp_stderr(stderr: str) -> None:
    if not isinstance(stderr, str):
        return
    text = stderr.strip()
    if not text:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("ERROR:"):
            log.error("yt-dlp: %s", line)
        elif line.startswith("WARNING:"):
            log.warning("yt-dlp: %s", line)
        else:
            log.info("yt-dlp: %s", line)


def download_audio(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = get_video_id(url)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--sleep-interval",
                "1",
                *_ytdlp_extra_args(),
                "-x",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "0",
                "-o",
                output_template,
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # Die echte yt-dlp-Fehlermeldung landet sonst nirgends – logge sie.
        _log_ytdlp_stderr(e.stderr)
        raise
    _log_ytdlp_stderr(result.stderr)
    time.sleep(RATE_LIMIT_DELAY)

    expected = output_dir / f"{vid}.mp3"
    if expected.exists():
        return expected

    # Fallback: durchsuche das Verzeichnis nach einer passenden Datei
    for f in output_dir.iterdir():
        if f.stem == vid and f.suffix == ".mp3":
            return f

    raise FileNotFoundError(f"No audio file found for video {vid} in {output_dir}")


def get_video_metadata(url: str) -> dict:
    """Fetch channel, upload_date, tags and available subtitle languages via yt-dlp --dump-json."""
    result = subprocess.run(
        ["yt-dlp", "--sleep-interval", "1", "--dump-json", url],
        capture_output=True,
        text=True,
        check=True,
    )
    _log_ytdlp_stderr(result.stderr)
    time.sleep(RATE_LIMIT_DELAY)
    data = json.loads(result.stdout)
    pub = data.get("upload_date") or ""  # YYYYMMDD
    if len(pub) == 8:
        pub = f"{pub[:4]}-{pub[4:6]}-{pub[6:]}"

    # verfügbare manuelle Untertitel-Sprachen
    subs = sorted(data.get("subtitles") or {})

    return {
        "title": data.get("title") or "",
        "channel": data.get("channel") or data.get("uploader") or "",
        "publish_date": pub,
        "language": data.get("language") or "",
        "tags": data.get("tags") or [],
        "available_subs": subs,
    }


def get_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL."""
    import re

    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else "unknown"


LANG_PRIORITY = ["de", "en"]


def get_subtitles(url: str, output_dir: Path, langs: list[str] | None = None) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = get_video_id(url)

    # Schau zuerst, ob schon Untertitel-Dateien im Cache liegen
    for f in output_dir.iterdir():
        if f.suffix == ".vtt" and f.stem.split(".")[0] == vid:
            return _parse_subtitles(f)

    target = langs or LANG_PRIORITY
    sub_langs = ",".join(target)
    output_template = str(output_dir / "%(id)s")

    result = subprocess.run(
        [
            "yt-dlp",
            "--sleep-interval",
            "1",
            "--write-subs",
            "--sub-langs",
            sub_langs,
            "--skip-download",
            "--sub-format",
            "vtt",
            "-o",
            output_template,
            url,
        ],
        capture_output=True,
        text=True,
    )

    _log_ytdlp_stderr(result.stderr)
    time.sleep(RATE_LIMIT_DELAY)

    if result.returncode != 0:
        return None

    for f in output_dir.iterdir():
        if f.suffix == ".vtt" and f.stem.split(".")[0] == vid:
            return _parse_subtitles(f)

    return None


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS (e.g. 3661.5 → '01:01:01')."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_subtitles(path: Path) -> str:
    """Parse VTT subtitles and return text with timestamp markers.

    Output format:
        [00:01:23 - 00:01:25] Text line
        [00:01:26 - 00:01:30] Next line
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    current_ts = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("WEBVTT") or line.isdigit():
            continue
        if "-->" in line:
            # Zeitstempelzeile: 00:01:23.456 --> 00:01:25.678
            parts = line.split("-->")
            if len(parts) == 2:
                start_raw = parts[0].strip()
                end_raw = parts[1].strip()
                # Millisekunden abschneiden → 00:01:23
                start = start_raw.split(".")[0] if "." in start_raw else start_raw
                end = end_raw.split(".")[0] if "." in end_raw else end_raw
                current_ts = f"[{start} - {end}]"
            continue

        # Textzeile mit Zeitstempel versehen
        if current_ts:
            result.append(f"{current_ts} {line}")
        else:
            result.append(line)

    return "\n".join(result)
