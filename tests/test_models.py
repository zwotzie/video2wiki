import pytest
from pydantic import ValidationError

from video2wiki.models import KernpunktAbschnitt, ProcessedVideo, VideoSource


class TestVideoSource:
    def test_minimal(self):
        v = VideoSource(url="https://youtube.com/watch?v=abc123defgh")
        assert v.url == "https://youtube.com/watch?v=abc123defgh"
        assert v.category is None

    def test_with_category(self):
        v = VideoSource(url="https://youtu.be/abc123defgh", category="tech")
        assert v.category == "tech"

    def test_empty_url_allowed_by_pydantic(self):
        """Pydantic allows empty strings for str fields without constraints."""
        v = VideoSource(url="")
        assert v.url == ""


class TestKernpunktAbschnitt:
    def test_minimal(self):
        k = KernpunktAbschnitt(überschrift="Einleitung", punkte=["Punkt 1", "Punkt 2"])
        assert k.überschrift == "Einleitung"
        assert k.punkte == ["Punkt 1", "Punkt 2"]

    def test_empty_points(self):
        k = KernpunktAbschnitt(überschrift="Test", punkte=[])
        assert k.punkte == []

    def test_missing_überschrift_fails(self):
        with pytest.raises(ValidationError):
            KernpunktAbschnitt(punkte=["a"])


class TestProcessedVideo:
    def make_full(self, **overrides):
        data = dict(
            url="https://youtube.com/watch?v=abc123defgh",
            title="Test Video",
            channel="Test Kanal",
            publish_date="2024-01-15",
            language="de",
            tags=["tech", "tutorial"],
            category="tech",
            transcription_source="subtitles",
            transcription="Dies ist ein Test.",
            zusammenfassung="Eine Zusammenfassung.",
            kernpunkte=[KernpunktAbschnitt(überschrift="Punkt 1", punkte=["Detail 1"])],
            fazit="Ein Fazit.",
            markdown_path="tech/test-video-test-kanal.md",
        )
        data.update(overrides)
        return ProcessedVideo(**data)

    def test_full(self):
        p = self.make_full()
        assert p.title == "Test Video"
        assert p.channel == "Test Kanal"
        assert p.transcription_source == "subtitles"
        assert len(p.kernpunkte) == 1
        assert p.kernpunkte[0].überschrift == "Punkt 1"

    def test_empty_tags(self):
        p = self.make_full(tags=[])
        assert p.tags == []

    def test_empty_title_allowed_by_pydantic(self):
        """Pydantic allows empty strings for str fields without constraints."""
        p = self.make_full(title="")
        assert p.title == ""

    def test_invalid_date_format(self):
        """Date is a plain string, so any string is accepted."""
        p = self.make_full(publish_date="kein-datum")
        assert p.publish_date == "kein-datum"

    def test_minimal_required(self):
        """Only url is technically required by model definition."""
        with pytest.raises(ValidationError):
            ProcessedVideo()
