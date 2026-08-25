import sys
from pathlib import Path

import typer

from .logger import get_logger
from .models import VideoSource
from .pipeline import PipelineOptions, existing_categories, process_video
from .pipeline import (  # noqa: F401  (Kompatibilitäts-Alias)
    guess_category as _guess_category,
)
from .writer import backfill_journal

log = get_logger("cli")

app = typer.Typer()

_KNOWN_COMMANDS = {"process", "web"}


def main() -> None:
    """Entry-Point: macht `process` zum Standard-Befehl.

    Damit funktioniert weiterhin `uv run video2wiki "<URL>"` (ohne explizites
    ``process``), auch wenn es inzwischen mehrere Sub-Befehle gibt. Der Wrapper
    fügt den „Default-Command“ in die overgebenen Argumente ein, bevor Typer
    die eigentliche CLI ausführt.
    """
    args = sys.argv[1:]
    if args and args[0] not in _KNOWN_COMMANDS and not args[0].startswith("-"):
        sys.argv[1:] = ["process", *args]
    app()


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

    for source in sources:
        # Kategorie-Interaktion nur im direkten CLI-Modus (nicht bei --file)
        resolve_category = None if file is not None else (lambda: _prompt_for_category(output))

        options = PipelineOptions(
            output=output,
            category=source.category,
            template=template,
            force=force,
            api_base=api_base,
            model=llm,
            whisper_model=whisper_model,
        )
        process_video(source, options, resolve_category=resolve_category)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Host-Adresse"),
    port: int = typer.Option(8000, "--port", "-p", help="Port"),
    output: Path = typer.Option(Path("./wiki"), "--output", "-o", help="Ausgabeverzeichnis für das Wiki"),
    work_dir: Path = typer.Option(
        Path(".video2wiki_work"), "--work-dir", help="Arbeitsverzeichnis für Downloads/Transkriptionen"
    ),
    template: Path | None = typer.Option(
        None, "--template", "-t", help="Pfad zu einer benutzerdefinierten template.md"
    ),
    api_base: str | None = typer.Option(
        None, "--api-base", help="OpenAI-kompatible API-Basis-URL (default: $OPENAI_API_BASE)"
    ),
    llm: str | None = typer.Option(None, "--llm", help="LLM-Modellname (default: $OPENAI_MODEL)"),
    whisper_model: str = typer.Option(
        "small", "--model", "-m", help="Whisper-Modellgröße (tiny, base, small, medium, large-v3)"
    ),
):
    """Starte den Webservice mit Web-UI (Journal + Video-Eingabe)."""

    import uvicorn

    from .logger import setup_logging as _setup_logging
    from .webserver import create_app

    _setup_logging()

    app_obj = create_app(
        output=output,
        work_dir=work_dir,
        template=template,
        api_base=api_base,
        model=llm,
        whisper_model=whisper_model,
    )
    log.info("Starte Webservice auf http://%s:%s (Wiki: %s)", host, port, output)
    uvicorn.run(app_obj, host=host, port=port)


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


def _prompt_for_category(output_base: Path) -> str:
    if not sys.stdin.isatty():
        log.warning("Keine passende Kategorie erkannt – nicht-interaktiver Modus, verwende 'uncategorized'.")
        return "uncategorized"

    existing = existing_categories(output_base)
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
    main()
