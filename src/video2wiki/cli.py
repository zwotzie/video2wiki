import sys
from pathlib import Path

import typer

from .downloader import download_audio, get_subtitles, get_video_id, get_video_metadata
from .logger import get_logger
from .models import ProcessedVideo, VideoSource
from .summarizer import summarize_with_chunking
from .transcriber import transcribe
from .writer import (
    add_journal_entry,
    backfill_journal,
    is_in_journal,
    target_path,
    write_wiki_entry,
)

log = get_logger("cli")

app = typer.Typer()


@app.command()
def process(
    url: str | None = typer.Argument(None, help="YouTube-URL"),
    category: str | None = typer.Option(None, "--category", "-c", help="Wiki-Kategorie (Unterverzeichnis)"),
    file: Path | None = typer.Option(
        None, "--file", "-f", help="Datei mit URLs (eine pro Zeile; optionale Kategorie nach Komma)"
    ),
    output: Path = typer.Option(Path("./wiki"), "--output", "-o", help="Ausgabeverzeichnis für das Wiki"),
    template: Path | None = typer.Option(
        None, "--template", "-t", help="Pfad zu einer benutzerdefinierten template.md"
    ),
    force: bool = typer.Option(False, "--force", "-F", help="Vorhandene Einträge überschreiben"),
    api_base: str | None = typer.Option(
        None,
        "--api-base",
        help="OpenAI-kompatible API-Basis-URL (default: $OPENAI_API_BASE / https://api.openai.com/v1)",
    ),
    llm: str | None = typer.Option(None, "--llm", help="LLM-Modellname (default: $OPENAI_MODEL / gpt-4o-mini)"),
    backfill: bool = typer.Option(False, "--backfill", help="Journal aus bestehenden Wiki-Einträgen befüllen"),
    whisper_model: str = typer.Option(
        "small", "--model", "-m", help="Whisper-Modellgröße (tiny, base, small, medium, large-v3)"
    ),
):
    """Lade ein YouTube-Video herunter, transkribiere es, fasse es zusammen und schreibe es als Wiki-Eintrag."""

    from .logger import setup_logging as _setup_logging

    _setup_logging()

    # --backfill: bestehende Einträge ins Journal eintragen
    if backfill:
        count = backfill_journal(output)
        log.info("Journal aktualisiert: %s Einträge hinzugefügt.", count)
        raise typer.Exit(0)

    if file:
        sources = _read_sources(file)
    elif url:
        sources = [VideoSource(url=url, category=category)]
    else:
        log.error("Gib eine URL oder eine --file mit Quellen an.")
        raise typer.Exit(1)

    work_dir = Path(".video2wiki_work")

    for source in sources:
        log.info("Verarbeite: %s", source.url)

        # Journal-Check (überspringen, wenn schon verarbeitet – außer mit --force)
        if not force and is_in_journal(output, source.url):
            log.info("  Übersprungen (bereits im Journal).")
            continue

        try:
            meta = get_video_metadata(source.url)
        except Exception:
            log.error("  ✗ Fehler: Konnte Metadaten nicht abrufen – Video existiert nicht oder ist nicht erreichbar.")
            continue

        log.info("  Titel: %s", meta["title"])
        log.info("  Kanal: %s", meta["channel"])

        cat = source.category or _guess_category(meta["title"], meta.get("channel", ""))
        if cat == "uncategorized" and not source.category and not file:
            cat = _prompt_for_category(output)
        log.info("  Kategorie: %s", cat)

        # Prüfe verfügbare Untertitel-Sprachen
        available_subs = meta.get("available_subs") or []
        wanted = [lang for lang in ("de", "en") if lang in available_subs]
        transcription: str | None = None
        transcription_source = "whisper"

        if not wanted:
            log.info("  Keine manuellen Untertitel (%s).", ", ".join(available_subs) if available_subs else "keine")

        if wanted:
            log.info("  Untertitel verfügbar (%s). Versuche %s …", ", ".join(available_subs), ", ".join(wanted))
            transcription = get_subtitles(source.url, work_dir, langs=wanted)
            if transcription is not None:
                transcription_source = "subtitles"

        if transcription is None:
            log.info("  Lade Audio herunter …")
            try:
                audio_path = download_audio(source.url, work_dir)
                log.info(
                    "  Audio-Download abgeschlossen: %s (%.1f MB)",
                    audio_path.name,
                    audio_path.stat().st_size / 1_024 / 1_024,
                )
            except Exception:
                log.error("  ✗ Fehler: Audio-Download fehlgeschlagen.")
                continue
            transcription = transcribe(audio_path, model_size=whisper_model)

        if not transcription.strip():
            log.error("  ✗ Fehler: Transkription ist leer.")
            continue

        vid = get_video_id(source.url)
        (work_dir / f"{vid}.transcription.txt").write_text(transcription, encoding="utf-8")
        log.info("  Transkription gespeichert: %s.txt (%s Zeichen)", vid, f"{len(transcription):,}")

        log.info("  Analysiere und fasse zusammen...")
        try:
            summary_data = summarize_with_chunking(transcription, meta["title"], api_base=api_base, model=llm)
        except Exception:
            log.error("  ✗ Fehler: Zusammenfassung fehlgeschlagen.")
            continue

        titel_deutsch = summary_data.get("titel_deutsch", "") or ""

        entry = ProcessedVideo(
            url=source.url,
            title=meta["title"],
            titel_deutsch=titel_deutsch,
            channel=meta["channel"],
            publish_date=meta["publish_date"],
            language=meta["language"],
            tags=summary_data.get("stichwörter", []),
            category=cat,
            transcription_source=transcription_source,
            transcription=transcription,
            zusammenfassung=summary_data["zusammenfassung"],
            kernpunkte=summary_data["kernpunkte"],
            fazit=summary_data["fazit"],
            markdown_path="",
        )

        path = write_wiki_entry(entry, output, template_path=template, force=force)
        if path is None:
            log.info("  Übersprungen (existiert bereits): %s", target_path(entry, output))
            continue
        entry.markdown_path = str(path)
        add_journal_entry(output, entry, str(path.relative_to(output)))
        log.info("  Geschrieben: %s", path)


def _read_sources(path: Path) -> list[VideoSource]:
    sources: list[VideoSource] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        url = parts[0]
        category = parts[1] if len(parts) > 1 else None
        sources.append(VideoSource(url=url, category=category))
    return sources


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


def _guess_category(title: str, channel: str = "") -> str:
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


def _get_existing_categories(output_base: Path) -> list[str]:
    if not output_base.exists():
        return []
    return sorted(d.name for d in output_base.iterdir() if d.is_dir() and not d.name.startswith("."))


def _prompt_for_category(output_base: Path) -> str:
    if not sys.stdin.isatty():
        log.warning("Keine passende Kategorie erkannt – nicht-interaktiver Modus, verwende 'uncategorized'.")
        return "uncategorized"

    existing = _get_existing_categories(output_base)
    if not existing:
        existing = ["uncategorized"]

    typer.echo("")
    log.warning("Keine passende Kategorie erkannt.")
    typer.echo("Verfügbare Kategorien:")
    for i, cat in enumerate(existing, 1):
        typer.echo(f"  {i}. {cat}")
    new_idx = len(existing) + 1
    typer.echo(f"  {new_idx}. ⨁ Neue Kategorie anlegen")
    typer.echo("  (Enter ⏎ = uncategorized)")

    choice = typer.prompt("Auswahl", default="").strip()
    if not choice:
        return "uncategorized"

    try:
        idx = int(choice)
        if 1 <= idx <= len(existing):
            return existing[idx - 1]
        if idx == new_idx:
            new_cat = typer.prompt("  Neuer Kategorie-Name").strip()
            return new_cat.lower() if new_cat else "uncategorized"
    except ValueError:
        pass

    return choice.lower()


if __name__ == "__main__":
    app()
