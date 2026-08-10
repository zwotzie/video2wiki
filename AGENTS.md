# video2wiki

YouTube-Video → Transkription (Whisper/Untertitel) → LLM-Zusammenfassung → strukturierter Wiki-Eintrag (Markdown).

## Quickstart

```bash
uv sync
uv run video2wiki "https://www.youtube.com/watch?v=VIDEO_ID"
uv run video2wiki "URL" --category musik
uv run video2wiki --file urls.txt
```

## Wichtige Konventionen

- **Sprache**: Deutsch (Prompts, Ausgabe, CLI-Hilfe, Template).
- **Package-Manager**: `uv` (kein pip/poetry). Python ≥3.13 via `.python-version`.
- **Build**: Hatchling (in `pyproject.toml`).
- **Keine monorepo-weiten Tools** – alles in diesem Verzeichnis.

## Stack

| Komponente | Tool | Bemerkung |
|---|---|---|
| CLI | `typer` | `video2wiki`-Command |
| Download | `yt-dlp` | Audio (`-x --audio-format mp3`), Untertitel (`--write-subs` de/en), Metadaten (`--dump-json`) |
| Transkription | `faster-whisper` | Lokal, Default `small`, Sprache `de` |
| LLM | OpenAI-compatible API | Konfigurierbar via `.env` (`OPENAI_API_BASE`, `OPENAI_MODEL`, `OPENAI_SSL_VERIFY`, `OPENAI_STRUCTURED_OUTPUT`) oder CLI-Flags `--api-base`/`--llm` |
| SSL | `_sslfix.py` | Setzt `SSL_CERT_FILE` für interne CAs (nur nötig, wenn `OPENAI_SSL_VERIFY=false`) |

## Projektstruktur

```
video2wiki/
├── pyproject.toml
├── template.md                    # Wiki-Vorlage (string.Template, $platzhalter)
├── .video2wiki_work/              # Cache: *.mp3, *.vtt, *.transcription.txt
├── wiki/                          # Output: <kategorie>/<titel>-<kanal>.md
│   └── .video2wiki_journal.yaml   # Journal: bereits verarbeitete Videos
└── src/video2wiki/
    ├── cli.py               # Typer-CLI, Batch-Verarbeitung
    ├── downloader.py        # yt-dlp-Wrapper
    ├── transcriber.py       # faster-whisper
    ├── summarizer.py        # LLM-Abfrage mit JSON-Struktur
    ├── writer.py            # Template-Rendering + Journal
    ├── models.py            # VideoSource, ProcessedVideo, KernpunktAbschnitt
    └── _sslfix.py
```

## Module im Detail

### `cli.py`
- `process`-Command: URL oder `--file` (eine URL pro Zeile, optional `,Kategorie`).
- Fehlerbehandlung pro Video: `try/except` um Metadaten, Audio-Download, Zusammenfassung → `continue` bei Fehlern.
- Journal-Check vor Verarbeitung: Bereits im Journal → überspringen (außer `--force`).
- `--backfill`: Scannt bestehende Wiki-Einträge und trägt sie ins Journal ein.
- `_guess_category()` gibt immer `"uncategorized"` zurück (Platzhalter).

### `downloader.py`
- `get_video_id(url)` – Regex-Extraktion der YouTube-ID.
- `get_video_metadata(url)` – `yt-dlp --dump-json`, gibt dict mit title/channel/publish_date/language/tags/available_subs.
- `get_subtitles(url, output_dir, langs=["de","en"])` – Prüft Cache (exakter Video-ID-Match via `f.stem.split(".")[0] == vid`), sonst yt-dlp mit `--write-subs --sub-langs <langs>`. Gibt Text oder `None`.
- `download_audio(url, output_dir)` – yt-dlp `-x --audio-format mp3`, sucht zurück **nur** `<vid>.mp3` (kein wildes `*.mp3` mehr). War ein Bug: alte mp3s konnten fälschlich geliefert werden.
- `_parse_subtitles(path)` – Entfernt WEBVTT-Kopf, Zeitstempel, Nummern.

### `transcriber.py`
- `transcribe(audio_path, model_size)` – `WhisperModel("small", device="auto")`, `language="de"`. Gibt joinierte Segmente.

### `summarizer.py`
- `summarize(text, title, api_base, model)` – Ein API-Call, erwartet JSON mit `zusammenfassung` (str), `kernpunkte` (list[{"überschrift": str, "punkte": [str]}]), `fazit` (str).
- `summarize_with_chunking(text, title, api_base, model)` – Automatische Langtext-Verarbeitung:
  - Wenn Transkript ≤ 120.000 Zeichen → direkter `summarize()`-Call.
  - Wenn länger → Split an natürlichen Grenzen (`\n\n` > `\n` > `. ` > Wort), jeder Block einzeln via `summarize()`, dann `_merge_results()` zur finalen Zusammenfassung.
  - Bei API-Fehler (Context exceeded, leere Antwort) → rekursives Halbieren des Blocks (max. 5 Tiefe, min. 2000 Zeichen).
- `_call_llm(system_prompt, user_prompt, api_base, model)` – Gekapselter API-Call mit `response_format`: `json_schema` (wenn `OPENAI_STRUCTURED_OUTPUT=true`) oder `json_object`.
- Prompt ist Deutsch, beschreibt das gewünschte Format exklusive Markdown-Headern (die kommen aus dem Template).
- Defaults via `.env`: `OPENAI_API_BASE`, `OPENAI_MODEL`, `OPENAI_SSL_VERIFY`, `OPENAI_STRUCTURED_OUTPUT`.

### `writer.py`
- `sanitize_filename(name)` – Entfernt Sonderzeichen, lowercase, kürzt auf 80 Zeichen.
- `target_path(video, output_base)` – `<output>/<category>/<sanitized_title>-<sanitized_channel>.md`.
- `write_wiki_entry(video, output_base, template_path, force)` – Überspringt existierende Files außer mit `--force`. Nutzt `string.Template` mit `safe_substitute`.
- `_format_kernpunkte(abschnitte)` – Erzeugt `### Überschrift\n- Punkt\n- Punkt\n\n### ...`.
- `_default_template()` – Fallback, wenn keine `template.md` gesetzt.
- `load_journal(output_base)` / `save_journal()` / `is_in_journal()` / `add_journal_entry()` – YAML-Journal (`wiki/.video2wiki_journal.yaml`) für bereits verarbeitete Videos.
- `backfill_journal(output_base)` – Scannt `wiki/**/*.md`, parsed YAML-Frontmatter und trägt fehlende Einträge ins Journal ein.
- `_parse_frontmatter(content)` – Extrahiert Frontmatter (YAML mit Regex-Fallback bei fehlerhaften Werten).

### `models.py`
- `VideoSource(url, category)` – Input.
- `KernpunktAbschnitt(überschrift, punkte: list[str])` – Strukturierte Kernpunkte.
- `ProcessedVideo(url, title, channel, publish_date, language, tags, category, transcription_source, transcription, zusammenfassung, kernpunkte, fazit, markdown_path)` – Output-Modell.

## Workflow (je Video)

1. Journal-Check: Bereits verarbeitet? → überspringen (außer `--force`).
2. Metadaten holen (`yt-dlp --dump-json`)
3. Untertitel suchen (Cache → yt-dlp)
4. Fallback: Audio runterladen → Whisper-Transkription
5. Rohtranskription als `.video2wiki_work/<id>.transcription.txt` sichern
6. LLM-Zusammenfassung (JSON-strukturiert) – bei langen Transkripten (>120K Zeichen) automatisches Chunking mit Merge
7. Wiki-Eintrag schreiben (Template + YAML-Frontmatter)
8. Journal-Eintrag schreiben

## Output-Format

```markdown
---
quelle: "$url"
titel: "$title"
kanal: "$channel"
erscheinungsdatum: "$publish_date"
sprache: "$language"
transkription: subtitles|whisper
tags: [$tags]
---

## Zusammenfassung

$zusammenfassung

## Kernpunkte

### Überschrift 1
- Punkt 1
- Punkt 2

### Überschrift 2
- Punkt 3
- Punkt 4

# Fazit

$fazit

# Quellen

- [$title]($url)
```

## Bekannte Probleme / Todos

- `_guess_category()` ist ein Dummy – könnte per Kanal-Namen automatisch befüllt werden.
- `.video2wiki_work/` wird nie automatisch geräumt (alte mp3/vtt/txt sammeln sich).
- Template liegt hartcodiert im Projekt-Root; bei `--template` kann man ein eigenes mitgeben.
- `--file` mit Kategorie (`url,kat`) funktioniert, aber Kategorie ohne URL nicht getestet.

## Tests

Keine Test-Infrastruktur vorhanden. Manuelle Tests via `uv run video2wiki`.
