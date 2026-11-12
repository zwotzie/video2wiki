import re
from datetime import UTC, datetime
from pathlib import Path
from string import Template

import yaml

from .models import KernpunktAbschnitt, ProcessedVideo

JOURNAL_FILENAME = ".video2wiki_journal.yaml"

_CANONICAL_KEYS = [
    "url",
    "title",
    "channel",
    "publish_date",
    "language",
    "transcription_source",
    "tags",
    "category",
    "subcategory",
    "processed_at",
    "output_path",
]


def _normalize_entry(entry: dict) -> dict:
    normalized = {}
    for key in _CANONICAL_KEYS:
        if key in entry:
            normalized[key] = entry[key]
    for key in entry:
        if key not in _CANONICAL_KEYS:
            normalized[key] = entry[key]
    op = normalized.get("output_path", "")
    if op and not op.endswith(".md"):
        normalized["output_path"] = op + ".md"
    return normalized


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    name = re.sub(r"[-\s]+", "-", name)
    return name[:80]


def target_path(video: ProcessedVideo, output_base: Path) -> Path:
    safe_channel = sanitize_filename(video.channel) if video.channel else "unknown"
    title = video.titel_deutsch or video.title
    safe_title = sanitize_filename(title)
    filename = f"{safe_title}-{safe_channel}.md"
    base = output_base / video.category
    if video.subcategory:
        base = base / video.subcategory
    return base / filename


def _format_kernpunkte(abschnitte: list[KernpunktAbschnitt]) -> str:
    parts = []
    for a in abschnitte:
        header = f"### {a.überschrift}"
        if a.zeitstempel:
            header += f" [{a.zeitstempel}]"
        parts.append(header)
        for p in a.punkte:
            parts.append(f"- {p}")
        parts.append("")
    return "\n".join(parts).strip()


def write_wiki_entry(
    video: ProcessedVideo,
    output_base: Path,
    template_path: Path | None = None,
    force: bool = False,
) -> Path | None:
    filepath = target_path(video, output_base)

    if filepath.exists() and not force:
        return None

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if template_path and template_path.exists():
        raw = template_path.read_text(encoding="utf-8")
    else:
        raw = _default_template()

    content = Template(raw).safe_substitute(
        url=video.url,
        title=video.title,
        titel_deutsch=video.titel_deutsch or video.title,
        channel=video.channel,
        publish_date=video.publish_date,
        language=video.language,
        transcription_source=video.transcription_source,
        tags=", ".join(video.tags),
        subcategory=video.subcategory or "",
        zusammenfassung=video.zusammenfassung,
        kernpunkte=_format_kernpunkte(video.kernpunkte),
        fazit=video.fazit,
    )
    content = _quote_publish_date_in_frontmatter(content)

    filepath.write_text(content, encoding="utf-8")
    return filepath


def _quote_publish_date_in_frontmatter(content: str) -> str:
    """Ensures `erscheinungsdatum` is a quoted YAML string in frontmatter."""
    match = re.match(r"(?s)^(---\n)(.*?)(\n---\n?)", content)
    if not match:
        return content

    frontmatter = match.group(2)

    def _replace_date_line(m: re.Match[str]) -> str:
        prefix = m.group(1)
        value = m.group(2).strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return m.group(0)
        return f'{prefix}"{value}"'

    updated_frontmatter = re.sub(r"(?m)^(erscheinungsdatum:\s*)(.*?)\s*$", _replace_date_line, frontmatter)
    if updated_frontmatter == frontmatter:
        return content

    return f"{match.group(1)}{updated_frontmatter}{match.group(3)}{content[match.end() :]}"


def journal_path(output_base: Path) -> Path:
    return output_base / JOURNAL_FILENAME


def load_journal(output_base: Path) -> list[dict]:
    path = journal_path(output_base)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def save_journal(output_base: Path, entries: list[dict]) -> None:
    path = journal_path(output_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [_normalize_entry(e) for e in entries]
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def is_in_journal(output_base: Path, url: str) -> bool:
    entries = load_journal(output_base)
    return any(e.get("url") == url for e in entries)


def add_journal_entry(output_base: Path, video: ProcessedVideo, output_path: str) -> None:
    entries = load_journal(output_base)
    # Alten Eintrag für dieselbe URL ersetzen (bei --force)
    entries = [e for e in entries if e.get("url") != video.url]

    entry = _normalize_entry(
        {
            "url": video.url,
            "title": video.title,
            "channel": video.channel,
            "publish_date": video.publish_date,
            "language": video.language,
            "transcription_source": video.transcription_source,
            "tags": video.tags,
            "category": video.category,
            "subcategory": video.subcategory or "",
            "processed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "output_path": output_path,
        }
    )
    entries.append(entry)
    save_journal(output_base, entries)


def backfill_journal(output_base: Path) -> int:
    """Liest alle existierenden Wiki-Einträge und trägt sie ins Journal ein."""
    from datetime import datetime

    entries = load_journal(output_base)
    known_urls = {e.get("url") for e in entries}
    count = 0

    for md_path in output_base.rglob("*.md"):
        content = md_path.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(content)
        if not frontmatter or "quelle" not in frontmatter:
            continue
        url = frontmatter["quelle"]
        if url in known_urls:
            continue

        # Kategorie/Subkategorie aus dem Verzeichnispfad ableiten:
        # wiki/<kat>/<unterkat>/datei.md → beides; wiki/<kat>/datei.md → nur Kategorie
        rel_parent = md_path.parent.relative_to(output_base)
        parts = rel_parent.parts
        category = parts[0] if parts else ""
        path_subcategory = parts[1] if len(parts) > 1 else ""
        subcategory = frontmatter.get("unterkategorie") or path_subcategory
        rel_path = str(md_path.relative_to(output_base))
        mtime = datetime.fromtimestamp(md_path.stat().st_mtime, tz=UTC)

        entry = _normalize_entry(
            {
                "url": url,
                "title": frontmatter.get("titel", ""),
                "channel": frontmatter.get("kanal", ""),
                "publish_date": frontmatter.get("erscheinungsdatum", ""),
                "language": frontmatter.get("sprache", ""),
                "transcription_source": frontmatter.get("transkription", ""),
                "tags": frontmatter.get("tags", []),
                "category": category,
                "subcategory": subcategory,
                "processed_at": mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "output_path": rel_path,
            }
        )
        entries.append(entry)
        known_urls.add(url)
        count += 1

    if count > 0:
        save_journal(output_base, entries)
    return count


def _parse_frontmatter(content: str) -> dict | None:
    """Extrahiert YAML-Frontmatter zwischen --- Markern."""
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    raw = parts[1].strip()

    # Versuche echtes YAML
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        pass

    # Fallback: einfaches Key: Value-Parsing für generische Frontmatter
    result = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Listen-Notation [a, b, c] → echte Liste
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                result[key] = [v.strip().strip("\"'") for v in inner.split(",") if v.strip()]
            else:
                # Anführungszeichen entfernen
                result[key] = val.strip("\"'")
    return result if result else None


def _default_template() -> str:
    return (
        "---\n"
        'quelle: "$url"\n'
        'titel: "$title"\n'
        'kanal: "$channel"\n'
        'erscheinungsdatum: "$publish_date"\n'
        'sprache: "$language"\n'
        'transkription: "$transcription_source"\n'
        "tags: [$tags]\n"
        'unterkategorie: "$subcategory"\n'
        "---\n\n"
        "# $titel_deutsch – $channel\n\n"
        "## Zusammenfassung\n\n"
        "$zusammenfassung\n\n"
        "## Kernpunkte\n\n"
        "$kernpunkte\n\n"
        "# Fazit\n\n"
        "$fazit\n\n"
        "# Quellen\n\n"
        "- [$title]($url)\n"
    )
