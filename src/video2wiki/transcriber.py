import time
from pathlib import Path

from faster_whisper import WhisperModel

from . import _sslfix  # noqa: F401
from .logger import get_logger

log = get_logger("transcriber")


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS (e.g. 3661.5 → '01:01:01')."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(audio_path: Path, model_size: str = "base") -> str:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file.
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    file_size = audio_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")

    log.info("  Whisper-Modell: %s | Audio: %s (%.1f MB)", model_size, audio_path.name, file_size / 1_024 / 1_024)

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language="de",
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments_list = list(segments)
    lines = [f"[{_fmt_ts(seg.start)} - {_fmt_ts(seg.end)}] {seg.text.strip()}" for seg in segments_list]
    text = "\n".join(lines)
    elapsed = time.time() - t0

    if not text.strip():
        raise RuntimeError(
            f"Transkription ergab leeren Text (Modell: {model_size}, "
            f"Datei: {audio_path.name}, {file_size:,d} Bytes, "
            f"Dauer: {elapsed:.1f}s)"
        )

    log.info("  Transkription abgeschlossen: %s Zeichen in %.1f Sekunden", f"{len(text):,}", elapsed)
    if info is not None:
        log.info("  Sprache: %s (Wahrscheinlichkeit: %.0f%%)", info.language, info.language_probability * 100)

    return text
