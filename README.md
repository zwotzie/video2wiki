# video2wiki

Download YouTube-Videos, transkribiere mit Whisper, fasse mit OpenAI GPT-4o-mini zusammen und schreibe das Ergebnis als strukturiertes Markdown in ein kategorisiertes Wiki.

## Setup

```bash
uv sync
export OPENAI_API_KEY="sk-..."
```

## Usage

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
```

## Ausgabe-Struktur

```
./wiki/<kategorie>/<kanal>-<titel>.md
```

Jeder Eintrag enthält Frontmatter (Quelle, Titel, Kanal, Datum, Tags), Zusammenfassung, Kernpunkte, Fazit und Quellenlink.
