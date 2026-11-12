"""Wiederverwendbare Verarbeitungs-Pipeline für ein einzelnes Video.

Die komplette Kette ``Metadaten → Untertitel → Audio/Whisper → LLM → Wiki-Eintrag → Journal``
ist hier gekapselt, damit sowohl die CLI (``cli.py``) als auch der Webservice
(``webserver.py``) dieselbe Logik nutzen. Fortschritt und Fehler werden über einen
optionalen ``emit_log``-Callback an den Aufrufer (z. B. den Webserver-Job) gemeldet.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .downloader import download_audio, get_subtitles, get_video_id, get_video_metadata
from .logger import get_logger
from .models import ProcessedVideo, VideoSource
from .summarizer import ensure_model_available, summarize_with_chunking
from .transcriber import transcribe
from .writer import add_journal_entry, is_in_journal, target_path, write_wiki_entry

log = get_logger("pipeline")

# Status-Konstanten für PipelineResult
ST_DONE = "done"
ST_SKIPPED_JOURNAL = "skipped_journal"
ST_SKIPPED_EXISTING = "skipped_existing"
ST_ERROR_METADATA = "metadata_failed"
ST_ERROR_AUDIO = "audio_download_failed"
ST_ERROR_TRANSCRIPTION = "transcription_failed"
ST_ERROR_SUMMARY = "summary_failed"
ST_ERROR_LLM = "llm_unavailable"

LogCallback = Callable[[str, str], None]  # (message, level)


@dataclass
class PipelineOptions:
    """Konfiguration für einen einzelnen Verarbeitungslauf."""

    output: Path = Path("./wiki")
    category: str | None = None
    subcategory: str | None = None
    template: Path | None = None
    force: bool = False
    api_base: str | None = None
    model: str | None = None
    whisper_model: str = "small"
    work_dir: Path = Path(".video2wiki_work")


@dataclass
class PipelineResult:
    """Ergebnis eines Verarbeitungslaufs."""

    status: str
    entry: ProcessedVideo | None = None
    path: Path | None = None


def process_video(
    source: VideoSource,
    options: PipelineOptions | None = None,
    resolve_category: Callable[[], str] | None = None,
    emit_log: LogCallback | None = None,
) -> PipelineResult:
    """Verarbeitet ein einzelnes Video von der URL bis zum Wiki-Eintrag.

    Args:
        source: Eingabe (URL + optionale Kategorie aus der Quelle).
        options: Pipeline-Konfiguration (Ausgabe, Force, Modell, ...).
        resolve_category: Optionaler Callback, der gefragt wird, wenn keine Kategorie
            erkannt wurde und keine explizite Kategorie gesetzt ist. Nur weitgeben,
            wenn das Aufrufen erwünscht ist (z. B. nur im interaktiven CLI-Modus).
        emit_log: Optionaler Callback ``(message, level)`` für Fortschritt/Fehler.

    Returns:
        PipelineResult mit Status. Bei ``done`` zusätzlich Entry und Zielpfad.
    """
    options = options or PipelineOptions()
    options.work_dir.mkdir(parents=True, exist_ok=True)

    def _emit(level: str, message: str) -> None:
        getattr(log, level)(message)
        if emit_log is not None:
            emit_log(message, level)

    _emit("info", f"Verarbeite: {source.url}")

    # Journal-Check (überspringen, wenn schon verarbeitet – außer mit --force)
    if not options.force and is_in_journal(options.output, source.url):
        _emit("info", "  Übersprungen (bereits im Journal).")
        return PipelineResult(status=ST_SKIPPED_JOURNAL)

    # Erreichbarkeits-Check des LLM, bevor teure Arbeit (Download/Whisper) startet.
    # Ergebnis wird pro (api_base, model) gecacht → ein API-Call pro Lauf.
    try:
        if ensure_model_available(options.api_base, options.model):
            _emit("info", "  LLM-Erreichbarkeit geprüft – Modell antwortet.")
    except Exception as e:
        _emit("error", f"  ✗ Fehler: LLM-Modell nicht erreichbar: {e}")
        return PipelineResult(status=ST_ERROR_LLM)

    try:
        meta = get_video_metadata(source.url)
    except Exception:
        _emit("error", "  ✗ Fehler: Konnte Metadaten nicht abrufen – Video existiert nicht oder ist nicht erreichbar.")
        return PipelineResult(status=ST_ERROR_METADATA)

    _emit("info", f"  Titel: {meta['title']}")
    _emit("info", f"  Kanal: {meta['channel']}")

    cat = options.category or guess_category(meta["title"], meta.get("channel", ""))
    if cat == "uncategorized" and options.category is None and resolve_category is not None:
        cat = resolve_category()
    _emit("info", f"  Kategorie: {cat}")

    # Subkategorie ist immer explizit (kein Raten) und optional
    subcat = options.subcategory or source.subcategory or ""
    if subcat:
        _emit("info", f"  Subkategorie: {subcat}")

    # Prüfe verfügbare Untertitel-Sprachen
    available_subs = meta.get("available_subs") or []
    wanted = [lang for lang in ("de", "en") if lang in available_subs]
    transcription: str | None = None
    transcription_source = "whisper"

    if not wanted:
        _emit("info", f"  Keine manuellen Untertitel ({', '.join(available_subs) if available_subs else 'keine'}).")

    if wanted:
        _emit("info", f"  Untertitel verfügbar ({', '.join(available_subs)}). Versuche {', '.join(wanted)} …")
        transcription = get_subtitles(source.url, options.work_dir, langs=wanted)
        if transcription is not None:
            transcription_source = "subtitles"

    if transcription is None:
        _emit("info", "  Lade Audio herunter …")
        try:
            audio_path = download_audio(source.url, options.work_dir)
            _emit(
                "info",
                f"  Audio-Download abgeschlossen: {audio_path.name} "
                f"({audio_path.stat().st_size / 1_024 / 1_024:.1f} MB)",
            )
        except Exception as e:
            _emit("error", f"  ✗ Fehler: Audio-Download fehlgeschlagen: {e}")
            return PipelineResult(status=ST_ERROR_AUDIO)
        transcription = transcribe(audio_path, model_size=options.whisper_model)

    if not transcription.strip():
        _emit("error", "  ✗ Fehler: Transkription ist leer.")
        return PipelineResult(status=ST_ERROR_TRANSCRIPTION)

    vid = get_video_id(source.url)
    (options.work_dir / f"{vid}.transcription.txt").write_text(transcription, encoding="utf-8")
    _emit("info", f"  Transkription gespeichert: {vid}.txt ({len(transcription):,} Zeichen)")

    _emit("info", "  Analysiere und fasse zusammen...")
    try:
        summary_data = summarize_with_chunking(
            transcription, meta["title"], api_base=options.api_base, model=options.model
        )
    except Exception as e:
        _emit("error", f"  ✗ Fehler: Zusammenfassung fehlgeschlagen: {e}")
        return PipelineResult(status=ST_ERROR_SUMMARY)

    entry = ProcessedVideo(
        url=source.url,
        title=meta["title"],
        titel_deutsch=summary_data.get("titel_deutsch", "") or "",
        channel=meta["channel"],
        publish_date=meta["publish_date"],
        language=meta["language"],
        tags=summary_data.get("stichwörter", []),
        category=cat,
        subcategory=subcat,
        transcription_source=transcription_source,
        transcription=transcription,
        zusammenfassung=summary_data["zusammenfassung"],
        kernpunkte=summary_data["kernpunkte"],
        fazit=summary_data["fazit"],
        markdown_path="",
    )

    path = write_wiki_entry(entry, options.output, template_path=options.template, force=options.force)
    if path is None:
        _emit("info", f"  Übersprungen (existiert bereits): {target_path(entry, options.output)}")
        return PipelineResult(status=ST_SKIPPED_EXISTING, entry=entry)
    entry.markdown_path = str(path)
    add_journal_entry(options.output, entry, str(path.relative_to(options.output)))
    _emit("info", f"  Geschrieben: {path}")

    return PipelineResult(status=ST_DONE, entry=entry, path=path)


_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "gesundheit": [
        "gesundheit",
        "schlaf",
        "krankheit",
        "heilung",
        "parasit",
        "darm",
        "ernährung",
        "training",
        "herz",
        "fasten",
        "insulin",
        "stress",
        "mitochondrien",
        "gehirn",
        "nervensystem",
        "selbstheilung",
        "immunsystem",
        "blutzucker",
        "entzündung",
        "hormon",
        "fettverbrennung",
        "autophagie",
        "probiotika",
        "fermentation",
        "haarausfall",
        "joghurt",
        "reuteri",
        "trockenfasten",
    ],
    "finanzen": [
        "geld",
        "steuern",
        "bitcoin",
        "gold",
        "aktien",
        "invest",
        "vermögen",
        "rente",
        "etf",
        "zentralbank",
        "inflation",
        "bargeld",
        "finanzamt",
        "anlage",
    ],
    "spirituelles": [
        "bewusstsein",
        "meditation",
        "ego",
        "erwachen",
        "spiritualität",
        "achtsamkeit",
        "transformation",
        "seele",
        "shaolin",
        "buddha",
        "buddhismus",
        "hermetismus",
        "matrix",
        "narzisst",
        "energievampir",
    ],
    "geschichte": [
        "geschichte",
        "phantomzeit",
        "mittelalter",
        "archäologie",
        "vergangenheit",
        "mythologie",
        "antike",
    ],
    "energie": [
        "wärmepumpe",
        "photovoltaik",
        "strom",
        "pv",
        "speicher",
        "autarkie",
        "netz",
        "energiewende",
        "bhkw",
        "solar",
    ],
    "ki": [
        "künstliche intelligenz",
        "ki",
        "ai",
        "maschinelles lernen",
        "chatgpt",
        "llm",
        "neuronales netz",
    ],
    "produktivität": [
        "lernen",
        "produktivität",
        "kommunikation",
        "fokus",
        "gewohnheit",
        "schneller lernen",
        "rambling",
    ],
    "musik": [
        "musik",
        "song",
        "album",
        "konzert",
        "instrument",
        "melodie",
        "rhythmus",
    ],
}

_CHANNEL_CATEGORIES: dict[str, str] = {
    "photovoltaik berg": "energie",
    "alles wird strom": "energie",
    "felix & friends": "finanzen",
    "daniel sauer": "finanzen",
    "was bitcoin bringt": "finanzen",
    "jasmin kosubek": "finanzen",
    "seven wands deutsch": "geschichte",
    "ben azadi": "gesundheit",
    "ruth biallowons": "gesundheit",
    "justin sung": "produktivität",
}


def guess_category(title: str, channel: str = "") -> str:
    """Ratet anhand von Titel und Kanal eine Wiki-Kategorie (oder "uncategorized")."""
    title_lower = title.lower()
    channel_lower = channel.lower()
    padded = f" {title_lower} "

    for known, cat in _CHANNEL_CATEGORIES.items():
        if known in channel_lower:
            return cat

    scores: dict[str, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if " " in kw:
                if kw in title_lower:
                    score += 3
            else:
                if f" {kw} " in padded:
                    score += 2
        scores[cat] = score

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "uncategorized"


def existing_categories(output_base: Path) -> list[str]:
    """Liefert die vorhandenen Kategorien (Unterverzeichnisse) im Ausgabeverzeichnis."""
    if not output_base.exists():
        return []
    return sorted(d.name for d in output_base.iterdir() if d.is_dir() and not d.name.startswith("."))


def existing_subcategories(output_base: Path) -> list[str]:
    """Liefert alle vorhandenen Subkategorien (2. Hierarchie-Ebene) im Ausgabeverzeichnis."""
    names: set[str] = set()
    for cat in existing_categories(output_base):
        base = output_base / cat
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                names.add(d.name)
    return sorted(names)
