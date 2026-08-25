# video2wiki

Download YouTube-Videos, transkribiere mit Whisper, fasse mit einem LLM (OpenAI-kompatibel) zusammen und schreibe das Ergebnis als strukturiertes Markdown in ein kategorisiertes Wiki.

Nutzung sowohl als **CLI** als auch als **Webservice** mit Web-UI möglich.

## Setup

```bash
uv sync
export OPENAI_API_KEY="sk-..."
```

Die OpenAI-kompatible API wird über Umgebungsvariablen oder eine `.env`-Datei konfiguriert (siehe [Konfiguration](#konfiguration)).

## CLI

```bash
# Einzelnes Video
uv run video2wiki "https://youtube.com/watch?v=..."

# Mit Kategorie
uv run video2wiki "https://youtube.com/watch?v=..." -c tech

# Aus einer Liste (eine URL pro Zeile; optionale Kategorie nach Komma)
uv run video2wiki --file links.txt

# Benutzerdefiniertes Template
uv run video2wiki "https://..." -t mein_template.md

# Vorhandene Einträge überschreiben
uv run video2wiki "https://..." -F

# Whisper-Modell wählen (tiny, base, small, medium, large-v3)
uv run video2wiki "https://..." -m medium

# Ausgabeverzeichnis (Standard: ./wiki)
uv run video2wiki "https://..." -o ./mein_wiki

# Journal aus bestehenden Wiki-Einträgen nachziehen
uv run video2wiki process --backfill

# Webservice starten (siehe unten)
uv run video2wiki web
```

> Die URL kann direkt als erstes Argument genutzt werden – `process` ist der Standard-Befehl
> und darf weggelassen werden (`video2wiki "<URL>"` ≡ `video2wiki process "<URL>"`).

## Webservice

Ein FastAPI-basierter Webservice mit Web-UI, der alle Video-Pipeline-Funktionen im Browser bereitstellt.

**Features:**
- **Video-Eingabe**: Formular zum Einreichen einer YouTube-URL (mit optionaler Kategorie und Whisper-Modell). Die Verarbeitung läuft im Hintergrund und zeigt den Fortschritt als Live-Log.
- **Job-Queue**: Eingereichte Videos werden gequeued und mit begrenzter Parallelität verarbeitet (Standard: 1 = sequentiell), damit Download/Whisper/LLM auf ressourcenarmen Rechnern nicht zu OOM führen.
- **Journal-Ansicht**: Übersicht aller bereits verarbeiteten Videos, inklusive „Wiki anzeigen“ (rendert den Markdown-Eintrag) und „Backfill“-Knopf.

**Starten:**

```bash
uv run video2wiki web
# → http://127.0.0.1:8000
```

Optionen des `web`-Befehls:

```bash
uv run video2wiki web \
  --host 0.0.0.0 \      # Host (Standard: 127.0.0.1)
  --port 8080 \         # Port (Standard: 8000)
  -o ./wiki \           # Ausgabeverzeichnis (Standard: ./wiki)
  --work-dir ./.video2wiki_work \   # Arbeitsverzeichnis (Downloads/Transkriptionen)
  -m small \            # Whisper-Modell (Standard: small)
  --concurrency 1 \     # Gleichzeitige Jobs (Standard: 1 = sequentiell)
  -t template.md \      # Benutzerdefiniertes Template
  --api-base … --llm …  # LLM-Konfiguration (überschreibt .env)
```

### REST-API

| Endpoint | Methode | Beschreibung |
|---|---|---|
| `/` | GET | Web-UI |
| `/api/process` | POST | Video einreichen (`{"url": "...", "category": "musik", "whisper_model": "small"}`) → `job_id` |
| `/api/jobs` | GET | Alle Jobs (Status + Logs) |
| `/api/jobs/{job_id}` | GET | Einzelnen Job abrufen |
| `/api/journal` | GET | Journal als JSON (neueste zuerst) |
| `/api/backfill` | POST | Journal aus bestehenden `.md`-Einträgen nachziehen |
| `/api/wiki/{pfad}` | GET | Markdown-Inhalt eines Wiki-Eintrags |

**Beispiel – Video über die API einreichen und Status pollen:**

```bash
# Einreichen
JOB=$(curl -s -X POST http://127.0.0.1:8000/api/process \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://youtube.com/watch?v=...","category":"energie"}')
# → {"job_id":"a1b2c3d4e5f6","status":"queued"}

# Status abfragen
curl -s http://127.0.0.1:8000/api/jobs/a1b2c3d4e5f6
```

## Konfiguration

Die Konfiguration erfolgt über Umgebungsvariablen bzw. `.env` (siehe `.env.example`):

| Variable | Default | Beschreibung |
|---|---|---|
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | OpenAI-kompatible API-Basis-URL |
| `OPENAI_API_KEY` | – | API-Schlüssel |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM-Modell |
| `OPENAI_SSL_VERIFY` | `true` | SSL-Verifikation (bei internen CAs auf `false`) |
| `OPENAI_STRUCTURED_OUTPUT` | `true` | JSON-Schema-Outputs nutzen (sonst `json_object`) |

### YouTube-Blockaden umgehen

Seit Mitte 2026 blockiert YouTube den Standard-Player-Client von yt-dlp mit HTTP 403.
`video2wiki` umgeht das automatisch mit dem `android`-Client und einer JavaScript-Runtime (z. B. `node`).
Falls nötig, kann beides überschrieben werden:

| Variable | Default | Beschreibung |
|---|---|---|
| `YTDLP_PLAYER_CLIENT` | `android` | YouTube-Player-Client für yt-dlp |
| `YTDLP_JS_RUNTIME` | `node` | JavaScript-Runtime für yt-dlp |

## Ausgabe-Struktur

```
./wiki/<kategorie>/<kanal>-<titel>.md
```

Jeder Eintrag enthält Frontmatter (Quelle, Titel, Kanal, Datum, Tags), Zusammenfassung, Kernpunkte, Fazit und Quellenlink. Verarbeitete Videos werden im Journal (`wiki/.video2wiki_journal.yaml`) nachgeführt.
