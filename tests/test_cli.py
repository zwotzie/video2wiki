from typer.testing import CliRunner

from video2wiki.cli import _guess_category, _read_sources, app

runner = CliRunner()


class TestReadSources:
    def test_basic(self, tmp_path):
        f = tmp_path / "sources.txt"
        f.write_text("https://youtube.com/watch?v=abc\nhttps://youtu.be/def\n")
        sources = _read_sources(f)
        assert len(sources) == 2
        assert sources[0].url == "https://youtube.com/watch?v=abc"
        assert sources[0].category is None

    def test_with_categories(self, tmp_path):
        f = tmp_path / "sources.txt"
        f.write_text("https://youtube.com/watch?v=abc, tech\nhttps://youtu.be/def, musik\n")
        sources = _read_sources(f)
        assert sources[0].category == "tech"
        assert sources[1].category == "musik"

    def test_skips_empty_and_comments(self, tmp_path):
        f = tmp_path / "sources.txt"
        f.write_text("\n# This is a comment\nhttps://youtube.com/watch?v=abc\n  \n")
        sources = _read_sources(f)
        assert len(sources) == 1

    def test_no_trailing_comma(self, tmp_path):
        f = tmp_path / "sources.txt"
        f.write_text("https://youtube.com/watch?v=abc")
        sources = _read_sources(f)
        assert len(sources) == 1
        assert sources[0].category is None


class TestGuessCategory:
    def test_always_uncategorized(self):
        assert _guess_category("Any Title") == "uncategorized"


class TestCliProcess:
    def test_no_url_no_file(self):
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "Gib eine URL" in result.stderr

    def test_backfill(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.backfill_journal", return_value=3)
        result = runner.invoke(app, ["--backfill", "--output", str(tmp_path)])
        assert result.exit_code == 0
        assert "Journal aktualisiert: 3 Einträge" in result.stderr

    def test_skips_known_video(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=True)
        result = runner.invoke(
            app,
            ["https://youtube.com/watch?v=abc", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Übersprungen (bereits im Journal)" in result.stderr

    def test_metadata_failure(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch("video2wiki.cli.get_video_metadata", side_effect=Exception("fail"))
        result = runner.invoke(
            app,
            ["https://youtube.com/watch?v=abc", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Konnte Metadaten nicht abrufen" in result.stderr

    def test_audio_download_failure(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.cli.get_video_metadata",
            return_value={
                "title": "Test",
                "channel": "Channel",
                "publish_date": "2024-01-01",
                "language": "de",
                "tags": [],
            },
        )
        mocker.patch("video2wiki.cli.get_subtitles", return_value=None)
        mocker.patch("video2wiki.cli.download_audio", side_effect=Exception("no audio"))
        result = runner.invoke(
            app,
            ["https://youtube.com/watch?v=abc", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Audio-Download fehlgeschlagen" in result.stderr

    def test_summary_failure(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.cli.get_video_metadata",
            return_value={
                "title": "Test",
                "channel": "Channel",
                "publish_date": "2024-01-01",
                "language": "de",
                "tags": [],
                "available_subs": ["de"],
            },
        )
        mocker.patch("video2wiki.cli.get_subtitles", return_value="Untertitel Text")
        mocker.patch("video2wiki.cli.get_video_id", return_value="abc123")
        mocker.patch("video2wiki.cli.summarize_with_chunking", side_effect=Exception("LLM fail"))
        result = runner.invoke(
            app,
            ["https://youtube.com/watch?v=abc", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Zusammenfassung fehlgeschlagen" in result.stderr

    def test_full_success_with_subtitles(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.cli.get_video_metadata",
            return_value={
                "title": "Test Video",
                "channel": "Test Kanal",
                "publish_date": "2024-01-15",
                "language": "de",
                "tags": ["tech"],
                "available_subs": ["de"],
            },
        )
        mocker.patch("video2wiki.cli.get_subtitles", return_value="Untertitel")
        mocker.patch("video2wiki.cli.get_video_id", return_value="abc123")
        mocker.patch(
            "video2wiki.cli.summarize_with_chunking",
            return_value={
                "zusammenfassung": "Eine Zusammenfassung.",
                "kernpunkte": [{"überschrift": "Punkt", "punkte": ["Detail"]}],
                "fazit": "Ein Fazit.",
            },
        )
        mocker.patch("video2wiki.cli.write_wiki_entry", return_value=tmp_path / "wiki/test.md")
        mock_add_journal = mocker.patch("video2wiki.cli.add_journal_entry")

        result = runner.invoke(
            app,
            ["https://youtube.com/watch?v=abc", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Geschrieben" in result.stderr
        mock_add_journal.assert_called_once()

    def test_force_flag_passed_through(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.cli.get_video_metadata",
            return_value={
                "title": "Test",
                "channel": "C",
                "publish_date": "2024-01-01",
                "language": "de",
                "tags": [],
                "available_subs": ["en"],
            },
        )
        mocker.patch("video2wiki.cli.get_subtitles", return_value="T")
        mocker.patch("video2wiki.cli.get_video_id", return_value="abc")
        mocker.patch(
            "video2wiki.cli.summarize_with_chunking",
            return_value={
                "zusammenfassung": "Z",
                "kernpunkte": [],
                "fazit": "F",
            },
        )
        # Must return a path inside tmp_path for relative_to() to work
        fake_path = tmp_path / "cat" / "test-kanal.md"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        mock_write = mocker.patch("video2wiki.cli.write_wiki_entry")
        mock_write.return_value = fake_path

        result = runner.invoke(
            app,
            [
                "https://youtube.com/watch?v=abc",
                "--output",
                str(tmp_path),
                "--force",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["force"] is True

    def test_guesses_category_from_source(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.cli.get_video_metadata",
            return_value={
                "title": "Test",
                "channel": "C",
                "publish_date": "2024-01-01",
                "language": "de",
                "tags": [],
            },
        )
        mocker.patch("video2wiki.cli.get_subtitles", return_value="T")
        mocker.patch("video2wiki.cli.get_video_id", return_value="abc")
        mocker.patch(
            "video2wiki.cli.summarize_with_chunking",
            return_value={
                "zusammenfassung": "Z",
                "kernpunkte": [],
                "fazit": "F",
            },
        )
        fake_path = tmp_path / "cat" / "test-kanal.md"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        mock_write = mocker.patch("video2wiki.cli.write_wiki_entry")
        mock_write.return_value = fake_path

        result = runner.invoke(
            app,
            [
                "https://youtube.com/watch?v=abc",
                "--category",
                "musik",
                "--output",
                str(tmp_path),
            ],
        )
        assert "Kategorie: musik" in result.stderr

    def test_file_mode(self, mocker, tmp_path):
        sources_file = tmp_path / "sources.txt"
        sources_file.write_text("https://youtube.com/watch?v=abc, tech\nhttps://youtu.be/def\n")

        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.cli.get_video_metadata",
            return_value={
                "title": "Test",
                "channel": "C",
                "publish_date": "2024-01-01",
                "language": "de",
                "tags": [],
            },
        )
        mocker.patch("video2wiki.cli.get_subtitles", return_value="T")
        mocker.patch("video2wiki.cli.get_video_id", return_value="abc")
        mocker.patch(
            "video2wiki.cli.summarize_with_chunking",
            return_value={
                "zusammenfassung": "Z",
                "kernpunkte": [],
                "fazit": "F",
            },
        )
        fake_path = tmp_path / "cat" / "test-kanal.md"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        mock_write = mocker.patch("video2wiki.cli.write_wiki_entry")
        mock_write.return_value = fake_path

        result = runner.invoke(
            app,
            ["--file", str(sources_file), "--output", str(tmp_path)],
        )
        assert result.exit_code == 0

    def test_whisper_model_passed(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.cli.get_video_metadata",
            return_value={
                "title": "Test",
                "channel": "C",
                "publish_date": "2024-01-01",
                "language": "de",
                "tags": [],
            },
        )
        mocker.patch("video2wiki.cli.get_subtitles", return_value=None)
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x")
        mocker.patch("video2wiki.cli.download_audio", return_value=audio_file)
        mocker.patch("video2wiki.cli.get_video_id", return_value="abc")
        mock_transcribe = mocker.patch("video2wiki.cli.transcribe")
        mock_transcribe.return_value = "Transkribiert"
        mocker.patch(
            "video2wiki.cli.summarize_with_chunking",
            return_value={
                "zusammenfassung": "Z",
                "kernpunkte": [],
                "fazit": "F",
            },
        )
        fake_path = tmp_path / "cat" / "test-kanal.md"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        mock_write = mocker.patch("video2wiki.cli.write_wiki_entry")
        mock_write.return_value = fake_path

        result = runner.invoke(
            app,
            [
                "https://youtube.com/watch?v=abc",
                "--model",
                "medium",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        mock_transcribe.assert_called_once_with(mocker.ANY, model_size="medium")
