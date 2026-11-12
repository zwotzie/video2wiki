from pydantic import BaseModel


class VideoSource(BaseModel):
    url: str
    category: str | None = None
    subcategory: str | None = None


class KernpunktAbschnitt(BaseModel):
    überschrift: str
    punkte: list[str]
    zeitstempel: str | None = None


class ProcessedVideo(BaseModel):
    url: str
    title: str
    channel: str
    publish_date: str  # YYYY-MM-DD
    language: str
    tags: list[str]
    category: str
    subcategory: str = ""
    transcription_source: str  # "subtitles" oder "whisper"
    transcription: str
    zusammenfassung: str
    kernpunkte: list[KernpunktAbschnitt]
    fazit: str
    markdown_path: str
    titel_deutsch: str = ""
