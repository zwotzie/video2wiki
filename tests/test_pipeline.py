from video2wiki.models import VideoSource
from video2wiki.pipeline import (
    ST_DONE,
    ST_ERROR_AUDIO,
    ST_ERROR_METADATA,
    ST_ERROR_SUMMARY,
    ST_SKIPPED_EXISTING,
    ST_SKIPPED_JOURNAL,
    PipelineOptions,
    existing_categories,
    process_video,
)


def _options(tmp_path, **overrides):
    data = dict(
        output=tmp_path,
        category=None,
        force=False,
        api_base=None,
        model=None,
        whisper_model="small",
        work_dir=tmp_path / "work",
    )
    data.update(overrides)
    return PipelineOptions(**data)


def _meta(**overrides):
    data = dict(
        title="Test Video",
        channel="Test Kanal",
        publish_date="2024-01-15",
        language="de",
        tags=["tech"],
        available_subs=["de"],
    )
    data.update(overrides)
    return data


def _summary(**overrides):
    data = dict(
        zusammenfassung="Eine Zusammenfassung.",
        kernpunkte=[{"überschrift": "Punkt", "punkte": ["Detail"]}],
        fazit="Ein Fazit.",
    )
    data.update(overrides)
    return data


class TestProcessVideo:
    def test_skips_known_video(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=True)
        result = process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        assert result.status == ST_SKIPPED_JOURNAL

    def test_metadata_failure(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.pipeline.get_video_metadata",
            side_effect=Exception("boom"),
        )
        result = process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        assert result.status == ST_ERROR_METADATA

    def test_audio_download_failure(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta(available_subs=[]))
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value=None)
        mocker.patch("video2wiki.pipeline.download_audio", side_effect=Exception("no audio"))
        result = process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        assert result.status == ST_ERROR_AUDIO

    def test_summary_failure(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta())
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="Untertitel Text")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc123")
        mocker.patch(
            "video2wiki.pipeline.summarize_with_chunking",
            side_effect=Exception("LLM fail"),
        )
        result = process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        assert result.status == ST_ERROR_SUMMARY

    def test_full_success_with_subtitles(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta())
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="Untertitel")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc123")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary())
        fake_path = tmp_path / "cat" / "test-video-test-kanal.md"
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)
        mock_add = mocker.patch("video2wiki.pipeline.add_journal_entry")

        result = process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        assert result.status == ST_DONE
        assert result.path == fake_path
        assert result.entry is not None
        assert result.entry.transcription_source == "subtitles"
        assert result.entry.category == "uncategorized"
        mock_add.assert_called_once()
        # erforderliche Arg-Keys
        args = mock_add.call_args[0]
        assert len(args) == 3

    def test_force_flag_passed_through(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta(available_subs=["en"]))
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="T")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        fake_path = tmp_path / "cat" / "test-kanal.md"
        mock_write = mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)

        process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path, force=True))
        assert mock_write.call_args[1]["force"] is True

    def test_category_from_options(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta())
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="T")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        fake_path = tmp_path / "musik" / "test-kanal.md"
        mock_write = mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)
        mock_add = mocker.patch("video2wiki.pipeline.add_journal_entry")

        process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path, category="musik"))
        entry = mock_add.call_args[0][1]
        assert entry.category == "musik"
        assert mock_write.call_args[0][0].category == "musik"

    def test_guess_category_from_title(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch(
            "video2wiki.pipeline.get_video_metadata",
            return_value=_meta(title="Bitcoin vs. Gold", channel="Finanz-Kanal", available_subs=["en"]),
        )
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="T")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        fake_path = tmp_path / "finanzen" / "test-kanal.md"
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)
        mock_add = mocker.patch("video2wiki.pipeline.add_journal_entry")

        process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        entry = mock_add.call_args[0][1]
        assert entry.category == "finanzen"

    def test_resolve_category_callback(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta())
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="T")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        fake_path = tmp_path / "cat" / "test-kanal.md"
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)
        mock_add = mocker.patch("video2wiki.pipeline.add_journal_entry")

        resolver = mocker.Mock(return_value="geschichte")
        process_video(
            VideoSource(url="https://youtube.com/watch?v=abc"),
            _options(tmp_path),
            resolve_category=resolver,
        )
        resolver.assert_called_once()
        entry = mock_add.call_args[0][1]
        assert entry.category == "geschichte"

    def test_whisper_model_passed(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta(available_subs=[]))
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value=None)
        audio_file = tmp_path / "abc.mp3"
        audio_file.write_bytes(b"x")
        mocker.patch("video2wiki.pipeline.download_audio", return_value=audio_file)
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mock_transcribe = mocker.patch("video2wiki.pipeline.transcribe", return_value="Transkribiert")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        fake_path = tmp_path / "cat" / "test-kanal.md"
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)

        process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path, whisper_model="medium"))
        mock_transcribe.assert_called_once_with(mocker.ANY, model_size="medium")

    def test_whisper_fallback_uses_whisper_source(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta(available_subs=[]))
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value=None)
        audio_file = tmp_path / "abc.mp3"
        audio_file.write_bytes(b"x")
        mocker.patch("video2wiki.pipeline.download_audio", return_value=audio_file)
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mocker.patch("video2wiki.pipeline.transcribe", return_value="Transkribiert")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        fake_path = tmp_path / "cat" / "test-kanal.md"
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)
        mock_add = mocker.patch("video2wiki.pipeline.add_journal_entry")

        process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        entry = mock_add.call_args[0][1]
        assert entry.transcription_source == "whisper"

    def test_skips_existing_file(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta())
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="T")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=None)
        mock_add = mocker.patch("video2wiki.pipeline.add_journal_entry")

        result = process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path))
        assert result.status == ST_SKIPPED_EXISTING
        mock_add.assert_not_called()

    def test_emit_log_callback(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta())
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="T")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary(kernpunkte=[]))
        fake_path = tmp_path / "cat" / "test-kanal.md"
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)

        logs = []
        process_video(
            VideoSource(url="https://youtube.com/watch?v=abc"),
            _options(tmp_path),
            emit_log=lambda msg, level: logs.append((level, msg)),
        )
        assert any("Verarbeite:" in msg for _, msg in logs)
        assert any("Geschrieben:" in msg for _, msg in logs)

    def test_transcription_saved(self, mocker, tmp_path):
        mocker.patch("video2wiki.pipeline.is_in_journal", return_value=False)
        mocker.patch("video2wiki.pipeline.get_video_metadata", return_value=_meta())
        mocker.patch("video2wiki.pipeline.get_subtitles", return_value="Untertitel")
        mocker.patch("video2wiki.pipeline.get_video_id", return_value="abc123")
        mocker.patch("video2wiki.pipeline.summarize_with_chunking", return_value=_summary())
        fake_path = tmp_path / "cat" / "test.md"
        mocker.patch("video2wiki.pipeline.write_wiki_entry", return_value=fake_path)
        mocker.patch("video2wiki.pipeline.add_journal_entry")

        work = tmp_path / "work"
        process_video(VideoSource(url="https://youtube.com/watch?v=abc"), _options(tmp_path, work_dir=work))
        assert (work / "abc123.transcription.txt").read_text(encoding="utf-8") == "Untertitel"


class TestExistingCategories:
    def test_empty(self, tmp_path):
        assert existing_categories(tmp_path) == []

    def test_lists_directories(self, tmp_path):
        (tmp_path / "energie").mkdir()
        (tmp_path / "musik").mkdir()
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "file.txt").write_text("x")
        assert existing_categories(tmp_path) == ["energie", "musik"]
