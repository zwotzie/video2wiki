from pathlib import Path

import pytest

from video2wiki.models import KernpunktAbschnitt, ProcessedVideo
from video2wiki.writer import (
    _default_template,
    _format_kernpunkte,
    _parse_frontmatter,
    add_journal_entry,
    backfill_journal,
    is_in_journal,
    load_journal,
    sanitize_filename,
    save_journal,
    target_path,
    write_wiki_entry,
)


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "input,expected",
        [
            ("Hello World", "hello-world"),
            ("Test!!! Special**", "test-special"),
            ("UPPERCASE", "uppercase"),
            ("a" * 100, "a" * 80),
            ("  spaces  ", "spaces"),
            ("special_chars_123", "special_chars_123"),
            ("", ""),
        ],
    )
    def test_sanitizes(self, input, expected):
        assert sanitize_filename(input) == expected


class TestTargetPath:
    def make_video(self, **overrides):
        data = dict(
            url="https://youtube.com/watch?v=abc",
            title="Mein Test Video",
            channel="Mein Kanal",
            publish_date="2024-01-15",
            language="de",
            tags=[],
            category="tech",
            transcription_source="subtitles",
            transcription="Text",
            zusammenfassung="Zusammenfassung",
            kernpunkte=[],
            fazit="Fazit",
            markdown_path="",
        )
        data.update(overrides)
        return ProcessedVideo(**data)

    def test_basic(self):
        v = self.make_video()
        path = target_path(v, Path("/output"))
        assert path == Path("/output/tech/mein-test-video-mein-kanal.md")

    def test_unknown_channel(self):
        v = self.make_video(channel="")
        path = target_path(v, Path("/output"))
        assert "unknown" in str(path)

    def test_special_chars(self):
        v = self.make_video(title="Video: Teil 1!", channel="Kanal/Name")
        path = target_path(v, Path("/output"))
        assert "/" not in path.name

    def test_with_subcategory(self):
        v = self.make_video(category="tech", subcategory="python")
        path = target_path(v, Path("/output"))
        assert path == Path("/output/tech/python/mein-test-video-mein-kanal.md")

    def test_without_subcategory_stays_flat(self):
        v = self.make_video(category="tech")
        path = target_path(v, Path("/output"))
        assert path == Path("/output/tech/mein-test-video-mein-kanal.md")


class TestFormatKernpunkte:
    def test_single_section(self):
        abschnitte = [KernpunktAbschnitt(überschrift="Einleitung", punkte=["Punkt 1", "Punkt 2"])]
        result = _format_kernpunkte(abschnitte)
        assert "### Einleitung" in result
        assert "- Punkt 1" in result
        assert "- Punkt 2" in result

    def test_multiple_sections(self):
        abschnitte = [
            KernpunktAbschnitt(überschrift="A", punkte=["1"]),
            KernpunktAbschnitt(überschrift="B", punkte=["2"]),
        ]
        result = _format_kernpunkte(abschnitte)
        assert result.startswith("### A")
        assert "### B" in result

    def test_empty_points(self):
        abschnitte = [KernpunktAbschnitt(überschrift="Leer", punkte=[])]
        result = _format_kernpunkte(abschnitte)
        assert "### Leer" in result
        assert result.strip().endswith("### Leer")  # No bullet points

    def test_empty_list(self):
        assert _format_kernpunkte([]) == ""


class TestWriteWikiEntry:
    def make_video(self, **overrides):
        data = dict(
            url="https://youtube.com/watch?v=abc",
            title="Test Video",
            channel="Test Kanal",
            publish_date="2024-01-15",
            language="de",
            tags=["tag1", "tag2"],
            category="testcat",
            transcription_source="subtitles",
            transcription="Text",
            zusammenfassung="Eine Zusammenfassung.",
            kernpunkte=[KernpunktAbschnitt(überschrift="Punkt", punkte=["Detail"])],
            fazit="Ein Fazit.",
            markdown_path="",
        )
        data.update(overrides)
        return ProcessedVideo(**data)

    def test_writes_file(self, tmp_path):
        v = self.make_video()
        result = write_wiki_entry(v, tmp_path, force=True)
        assert result is not None
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Test Video" in content
        assert "Test Kanal" in content
        assert "Eine Zusammenfassung" in content
        assert "### Punkt" in content
        assert "- Detail" in content
        assert "Ein Fazit" in content
        assert "tag1, tag2" in content
        assert 'erscheinungsdatum: "2024-01-15"' in content

    def test_skip_existing(self, tmp_path):
        v = self.make_video()
        # Create file first
        write_wiki_entry(v, tmp_path, force=True)
        # Without force, should return None
        result = write_wiki_entry(v, tmp_path, force=False)
        assert result is None

    def test_force_overwrite(self, tmp_path):
        v = self.make_video()
        write_wiki_entry(v, tmp_path, force=True)
        # Modify and force overwrite
        result = write_wiki_entry(v, tmp_path, force=True)
        assert result is not None

    def test_custom_template(self, tmp_path):
        tmpl = tmp_path / "custom_template.md"
        tmpl.write_text("---\ntitle: $title\nerscheinungsdatum: $publish_date\n---\n\n$zusammenfassung")
        v = self.make_video()
        result = write_wiki_entry(v, tmp_path, template_path=tmpl, force=True)
        content = result.read_text(encoding="utf-8")
        assert "Test Video" in content
        assert "Eine Zusammenfassung" in content
        assert 'erscheinungsdatum: "2024-01-15"' in content

    def test_missing_template_fallback(self, tmp_path):
        """When template_path doesn't exist, fall back to default."""
        v = self.make_video()
        result = write_wiki_entry(v, tmp_path, template_path=tmp_path / "nonexistent.md", force=True)
        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "## Zusammenfassung" in content

    def test_creates_category_dir(self, tmp_path):
        v = self.make_video(category="nested/deep")
        result = write_wiki_entry(v, tmp_path, force=True)
        assert result.exists()
        assert "nested/deep" in str(result)


class TestDefaultTemplate:
    def test_contains_placeholders(self):
        tmpl = _default_template()
        assert "$url" in tmpl
        assert "$title" in tmpl
        assert "$channel" in tmpl
        assert "$zusammenfassung" in tmpl
        assert "$kernpunkte" in tmpl
        assert "$fazit" in tmpl
        assert "$subcategory" in tmpl
        assert "## Zusammenfassung" in tmpl
        assert "## Kernpunkte" in tmpl
        assert "# Fazit" in tmpl

    def test_renders_with_safe_substitute(self):
        from string import Template

        result = Template(_default_template()).safe_substitute(
            url="https://x.com",
            title="T",
            channel="C",
            publish_date="2024-01-01",
            language="de",
            transcription_source="whisper",
            tags="a, b",
            subcategory="python",
            zusammenfassung="Z",
            kernpunkte="K",
            fazit="F",
        )
        assert "https://x.com" in result
        assert "Z" in result
        assert "K" in result
        assert "F" in result


class TestJournal:
    def test_load_empty(self, tmp_path):
        assert load_journal(tmp_path) == []

    def test_save_and_load(self, tmp_path):
        entries = [{"url": "https://example.com", "title": "Test"}]
        save_journal(tmp_path, entries)
        loaded = load_journal(tmp_path)
        assert loaded == entries

    def test_is_in_journal(self, tmp_path):
        url = "https://youtube.com/watch?v=abc"
        assert not is_in_journal(tmp_path, url)
        save_journal(tmp_path, [{"url": url}])
        assert is_in_journal(tmp_path, url)

    def test_add_journal_entry(self, tmp_path):
        v = ProcessedVideo(
            url="https://youtube.com/watch?v=abc",
            title="Test",
            channel="Channel",
            publish_date="2024-01-15",
            language="de",
            tags=["tag"],
            category="cat",
            transcription_source="subtitles",
            transcription="T",
            zusammenfassung="Z",
            kernpunkte=[],
            fazit="F",
            markdown_path="",
        )
        add_journal_entry(tmp_path, v, "cat/test-channel.md")
        entries = load_journal(tmp_path)
        assert len(entries) == 1
        assert entries[0]["url"] == "https://youtube.com/watch?v=abc"
        assert entries[0]["subcategory"] == ""

    def test_add_journal_entry_with_subcategory(self, tmp_path):
        v = ProcessedVideo(
            url="https://youtube.com/watch?v=abc",
            title="Test",
            channel="Channel",
            publish_date="2024-01-15",
            language="de",
            tags=["tag"],
            category="energie",
            subcategory="solar",
            transcription_source="subtitles",
            transcription="T",
            zusammenfassung="Z",
            kernpunkte=[],
            fazit="F",
            markdown_path="",
        )
        add_journal_entry(tmp_path, v, "energie/solar/test-channel.md")
        entries = load_journal(tmp_path)
        assert entries[0]["category"] == "energie"
        assert entries[0]["subcategory"] == "solar"

    def test_add_journal_entry_replaces_existing(self, tmp_path):
        url = "https://youtube.com/watch?v=abc"
        v1 = ProcessedVideo(
            url=url,
            title="Old",
            channel="C",
            publish_date="2020-01-01",
            language="de",
            tags=[],
            category="cat",
            transcription_source="subtitles",
            transcription="T",
            zusammenfassung="Z",
            kernpunkte=[],
            fazit="F",
            markdown_path="",
        )
        v2 = ProcessedVideo(
            url=url,
            title="New",
            channel="C",
            publish_date="2024-01-01",
            language="de",
            tags=[],
            category="cat",
            transcription_source="subtitles",
            transcription="T",
            zusammenfassung="Z",
            kernpunkte=[],
            fazit="F",
            markdown_path="",
        )
        add_journal_entry(tmp_path, v1, "old.md")
        add_journal_entry(tmp_path, v2, "new.md")
        entries = load_journal(tmp_path)
        assert len(entries) == 1
        assert entries[0]["title"] == "New"


class TestBackfillJournal:
    def test_no_existing_files(self, tmp_path):
        count = backfill_journal(tmp_path)
        assert count == 0

    def test_backfills_from_md_files(self, tmp_path):
        # Create a wiki entry
        cat_dir = tmp_path / "tech"
        cat_dir.mkdir(parents=True)
        md_file = cat_dir / "test-video-channel.md"
        md_file.write_text(
            "---\n"
            'quelle: "https://youtube.com/watch?v=abc"\n'
            'titel: "Test Video"\n'
            'kanal: "Channel"\n'
            'erscheinungsdatum: "2024-01-15"\n'
            'sprache: "de"\n'
            'transkription: "subtitles"\n'
            "tags: [tag1, tag2]\n"
            "---\n\n"
            "Content"
        )
        count = backfill_journal(tmp_path)
        assert count == 1
        entries = load_journal(tmp_path)
        assert len(entries) == 1
        assert entries[0]["url"] == "https://youtube.com/watch?v=abc"
        assert entries[0]["category"] == "tech"

    def test_backfills_nested_subcategory(self, tmp_path):
        # wiki/tech/python/test-video-channel.md → Kategorie "tech", Subkategorie "python"
        sub_dir = tmp_path / "tech" / "python"
        sub_dir.mkdir(parents=True)
        md_file = sub_dir / "test-video-channel.md"
        md_file.write_text(
            "---\n"
            'quelle: "https://youtube.com/watch?v=abc"\n'
            'titel: "Test Video"\n'
            'kanal: "Channel"\n'
            'erscheinungsdatum: "2024-01-15"\n'
            'sprache: "de"\n'
            'transkription: "subtitles"\n'
            "tags: [tag1, tag2]\n"
            'unterkategorie: "python"\n'
            "---\n\n"
            "Content"
        )
        count = backfill_journal(tmp_path)
        assert count == 1
        entries = load_journal(tmp_path)
        assert entries[0]["category"] == "tech"
        assert entries[0]["subcategory"] == "python"
        assert entries[0]["output_path"] == "tech/python/test-video-channel.md"

    def test_backfills_nested_without_frontmatter_subcategory(self, tmp_path):
        # Ohne unterkategorie im Frontmatter wird sie aus dem Pfad abgeleitet
        sub_dir = tmp_path / "energie" / "solar"
        sub_dir.mkdir(parents=True)
        md_file = sub_dir / "test-video-channel.md"
        md_file.write_text('---\nquelle: "https://youtube.com/watch?v=abc"\ntitel: "T"\n---\n\nC')
        count = backfill_journal(tmp_path)
        assert count == 1
        entries = load_journal(tmp_path)
        assert entries[0]["category"] == "energie"
        assert entries[0]["subcategory"] == "solar"

    def test_skips_known_urls(self, tmp_path):
        cat_dir = tmp_path / "tech"
        cat_dir.mkdir(parents=True)
        md_file = cat_dir / "test-video-channel.md"
        md_file.write_text('---\nquelle: "https://youtube.com/watch?v=abc"\ntitel: "T"\n---\n\nC')
        save_journal(tmp_path, [{"url": "https://youtube.com/watch?v=abc"}])
        count = backfill_journal(tmp_path)
        assert count == 0

    def test_skips_files_without_frontmatter(self, tmp_path):
        cat_dir = tmp_path / "tech"
        cat_dir.mkdir(parents=True)
        no_fm = cat_dir / "no-frontmatter.md"
        no_fm.write_text("Just content without frontmatter")
        count = backfill_journal(tmp_path)
        assert count == 0

    def test_handles_yaml_parse_error_with_fallback(self, tmp_path):
        cat_dir = tmp_path / "tech"
        cat_dir.mkdir(parents=True)
        md_file = cat_dir / "test.md"
        # Invalid YAML: unquoted colon in value
        md_file.write_text('---\nquelle: "https://youtube.com/watch?v=abc"\ntitel: Some: title\n---\n\nContent')
        count = backfill_journal(tmp_path)
        assert count == 1
        entries = load_journal(tmp_path)
        assert entries[0]["url"] == "https://youtube.com/watch?v=abc"


class TestParseFrontmatter:
    def test_valid_yaml(self):
        content = '---\nquelle: "https://x.com"\ntitel: "Test"\n---\n\nBody'
        result = _parse_frontmatter(content)
        assert result["quelle"] == "https://x.com"
        assert result["titel"] == "Test"

    def test_tags_list(self):
        content = "---\ntags: [tag1, tag2]\n---\n\nBody"
        result = _parse_frontmatter(content)
        assert result["tags"] == ["tag1", "tag2"]

    def test_no_frontmatter(self):
        assert _parse_frontmatter("Just content") is None

    def test_incomplete_frontmatter(self):
        content = "---\nkey: value\n"
        result = _parse_frontmatter(content)
        assert result is None

    def test_yaml_error_fallback(self):
        content = "---\nkey: value: with: colons\n---\n\nBody"
        result = _parse_frontmatter(content)
        assert result is not None
        assert "key" in result
