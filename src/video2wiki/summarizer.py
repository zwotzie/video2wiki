import json
import os

import httpx
from openai import OpenAI

from .logger import get_logger

log = get_logger("summarizer")


def _build_client(api_base: str) -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "") or "unused"
    ssl_verify = os.environ.get("OPENAI_SSL_VERIFY", "true").lower() in ("true", "1", "yes")

    if ssl_verify and api_base == "https://api.openai.com/v1":
        return OpenAI(api_key=api_key)

    return OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(verify=ssl_verify),
    )


def _resolve_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _structured_output_enabled() -> bool:
    return os.environ.get("OPENAI_STRUCTURED_OUTPUT", "true").lower() in ("true", "1", "yes")


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    api_base: str,
    model: str,
) -> dict:
    client = _build_client(api_base)

    kwargs = {}
    if _structured_output_enabled():
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "video_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "titel_deutsch": {
                            "type": "string",
                            "description": "Deutsche Übersetzung des Video-Titels, kurz und prägnant (max. 80 Zeichen)",
                        },
                        "zusammenfassung": {
                            "type": "string",
                            "description": "Eine prägnante Zusammenfassung des Videoinhalts in 2–3 Absätzen",
                        },
                        "kernpunkte": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "überschrift": {
                                        "type": "string",
                                        "description": "Überschrift des Themenabschnitts",
                                    },
                                    "zeitstempel": {
                                        "type": "string",
                                        "description": "Ungefährer Zeitstempel im Format MM:SS oder HH:MM:SS, "
                                        "z.B. '05:23 – 08:47'. Leer lassen falls unbekannt.",
                                    },
                                    "punkte": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Wichtige Sätze/Aussagen zu diesem Thema",
                                    },
                                },
                                "required": ["überschrift", "zeitstempel", "punkte"],
                                "additionalProperties": False,
                            },
                            "description": "Mehrere Themenabschnitte mit Überschrift, ungefährem Zeitstempel "
                            "und dazugehörigen Kernaussagen",
                        },
                        "fazit": {
                            "type": "string",
                            "description": "Ein abschließendes Fazit mit Einordnung und Bewertung",
                        },
                        "stichwörter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3–8 prägnante Schlagwörter (Tags) zum Video, auf Deutsch",
                            "maxItems": 8,
                        },
                    },
                    "required": ["zusammenfassung", "kernpunkte", "fazit", "stichwörter"],
                    "additionalProperties": False,
                },
            },
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(
        model=model,
        **kwargs,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise RuntimeError("Leere Antwort vom LLM")
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def summarize(
    text: str,
    title: str,
    api_base: str | None = None,
    model: str | None = None,
) -> dict:
    """Einzelnen Transkript-Abschnitt zusammenfassen (ein API-Call)."""
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = model or _resolve_model()

    system_prompt = "Du bist ein hilfreicher Assistent, der Videoinhalte analysiert. Antworte immer auf Deutsch."
    if not _structured_output_enabled():
        system_prompt += (
            " Antworte NUR mit einem JSON-Objekt in folgendem Format, ohne zusätzlichen Text:\n"
            '{"titel_deutsch": "...", "zusammenfassung": "...", '
            '"kernpunkte": [{"überschrift": "...", "zeitstempel": "05:23 – 08:47", '
            '"punkte": ["...", "..."]}], "fazit": "...", "stichwörter": ["...", "..."]}'
        )

    user_prompt = (
        f"Analysiere folgendes Video-Transkript.\n\n"
        f"Titel: {title}\n\n"
        f"Übersetze den Videotitel ins Deutsche und gib ihn als titel_deutsch an.\n\n"
        f"Titel: {title}\n\n"
        f"Das Transkript enthält Zeitmarken im Format [HH:MM:SS - HH:MM:SS]. "
        f"Gib bei jedem Kernpunkt-Abschnitt den ungefähren Zeitstempel an (z.B. '05:23 – 08:47'). "
        f"Wenn keine Zeitmarken vorhanden sind, gib einen leeren String an.\n\n"
        f"Transkript:\n{text}\n"
    )

    return _call_llm(system_prompt, user_prompt, api_base, model)


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Text an natürlichen Grenzen in Stücke teilen (Absätze → Sätze → Wörter)."""
    for sep in ["\n\n", "\n", ". ", "! ", "? "]:
        parts = text.split(sep)
        if len(parts) > 1:
            break
    else:
        parts = text.split(" ")
        sep = " "

    chunks = []
    current = []
    current_len = 0
    sep_len = len(sep)
    for part in parts:
        part_len = len(part) + sep_len
        if current_len + part_len > chunk_size and current:
            chunks.append(sep.join(current))
            current = [part]
            current_len = part_len
        else:
            current.append(part)
            current_len += part_len
    if current:
        joined = sep.join(current)
        if joined:
            chunks.append(joined)
    return chunks


def summarize_with_chunking(
    text: str,
    title: str,
    api_base: str | None = None,
    model: str | None = None,
    chunk_size: int = 120_000,
) -> dict:
    """Automatische Zusammenfassung mit Chunking bei langen Transkripten."""
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = model or _resolve_model()

    if len(text) <= chunk_size:
        return summarize(text, title, api_base, model)

    chunks = _split_into_chunks(text, chunk_size)

    log.info(
        "  Transkript zu lang (%s Zeichen). Teile in %s Blöcke (max. %s Zeichen).",
        f"{len(text):,}",
        len(chunks),
        f"{chunk_size:,}",
    )

    def _process(text: str, depth: int = 0) -> dict:
        try:
            return summarize(text, title, api_base, model)
        except Exception as e:
            if depth >= 5 or len(text) < 2000:
                log.error(
                    "  ✗ Block auch nach %s Teilungen zu groß (%s Zeichen): %s",
                    depth,
                    len(text),
                    e,
                )
                raise
            log.info("  Block zu lang, teile in 2 Unter-Blöcke...")
            mid = len(text) // 2
            nl = text.rfind("\n\n", 0, mid)
            if nl > len(text) // 4:
                mid = nl
            left = text[:mid].strip()
            right = text[mid:].strip()
            left_result = _process(left, depth + 1)
            right_result = _process(right, depth + 1)
            return _merge_results([left_result, right_result], title, api_base, model)

    partials = []
    for i, chunk in enumerate(chunks, 1):
        log.info("  Block %s/%s...", i, len(chunks))
        try:
            result = _process(chunk)
            partials.append(result)
        except Exception as e:
            log.error("  ✗ Block %s/%s fehlgeschlagen: %s", i, len(chunks), e)
            raise

    if len(partials) == 1:
        return partials[0]

    log.info("  Führe %s Teil-Zusammenfassungen zusammen...", len(partials))
    return _merge_results(partials, title, api_base, model)


def _merge_results(
    partials: list[dict],
    title: str,
    api_base: str,
    model: str,
) -> dict:
    """Mehrere Teil-Zusammenfassungen zu einer Gesamt-Zusammenfassung mergen."""
    parts_text = "\n\n---\n\n".join(
        f"Teil {i + 1}:\n"
        f"Titel (deutsch): {p.get('titel_deutsch', '')}\n"
        f"Zusammenfassung: {p['zusammenfassung']}\n"
        f"Kernpunkte: {json.dumps(p['kernpunkte'], ensure_ascii=False)}\n"
        f"Fazit: {p['fazit']}\n"
        f"Stichwörter: {json.dumps(p.get('stichwörter', []), ensure_ascii=False)}"
        for i, p in enumerate(partials)
    )

    system_prompt = "Du bist ein hilfreicher Assistent. Antworte immer auf Deutsch."
    if not _structured_output_enabled():
        system_prompt += (
            " Antworte NUR mit einem JSON-Objekt in folgendem Format, ohne zusätzlichen Text:\n"
            '{"titel_deutsch": "...", "zusammenfassung": "...", '
            '"kernpunkte": [{"überschrift": "...", "zeitstempel": "05:23 – 08:47", '
            '"punkte": ["...", "..."]}], "fazit": "...", "stichwörter": ["...", "..."]}'
        )

    user_prompt = (
        "Du hast mehrere Teil-Zusammenfassungen eines langen Videos erhalten. "
        "Fasse sie zu einer einzigen, kohärenten Gesamt-Zusammenfassung zusammen. "
        "Vermeide Dopplungen und strukturiere die Kernpunkte sinnvoll. "
        "Fasse auch die Stichwörter zusammen und entferne Duplikate.\n\n"
        "Behalte die ungefähren Zeitstempel bei oder passe sie an die Position im Gesamtvideo an. "
        "Gib auch eine deutsche Übersetzung des Titels als titel_deutsch an.\n\n"
        f"Titel: {title}\n\n"
        f"Teil-Zusammenfassungen:\n{parts_text}"
    )

    return _call_llm(system_prompt, user_prompt, api_base, model)
