from pathlib import Path

import pytest

from video2wiki.transcriber import transcribe


def _mock_audio(mocker, path="/fake/audio.mp3", size=1_000_000):
    """Helper: mocket Path.exists() und Path.stat() für einen gültigen Audio-File."""
    mocker.patch.object(Path, "exists", return_value=True)
    mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=size))


class TestTranscribe:
    def test_success(self, mocker):
        _mock_audio(mocker)
        mock_model = mocker.patch("video2wiki.transcriber.WhisperModel")
        mock_instance = mock_model.return_value
        mock_segments = [
            mocker.Mock(text=" Hallo Welt ", start=0.0, end=2.5),
            mocker.Mock(text=" Das ist ein Test.", start=3.0, end=5.0),
        ]
        mock_info = mocker.Mock(language="de", language_probability=0.95)
        mock_instance.transcribe.return_value = (iter(mock_segments), mock_info)

        result = transcribe(Path("/fake/audio.mp3"), model_size="base")

        assert result == "[00:00:00 - 00:00:02] Hallo Welt\n[00:00:03 - 00:00:05] Das ist ein Test."
        mock_model.assert_called_once_with("base", device="auto", compute_type="auto")
        mock_instance.transcribe.assert_called_once_with(
            str(Path("/fake/audio.mp3")),
            language="de",
            vad_filter=True,
            condition_on_previous_text=False,
        )

    def test_empty_segments_raises(self, mocker):
        _mock_audio(mocker)
        mock_model = mocker.patch("video2wiki.transcriber.WhisperModel")
        mock_instance = mock_model.return_value
        mock_instance.transcribe.return_value = (iter([]), None)

        with pytest.raises(RuntimeError, match="leeren Text"):
            transcribe(Path("/fake/empty.mp3"))

    def test_single_segment(self, mocker):
        _mock_audio(mocker)
        mock_model = mocker.patch("video2wiki.transcriber.WhisperModel")
        mock_instance = mock_model.return_value
        mock_segments = [mocker.Mock(text=" Nur ein Segment. ", start=0.0, end=3.0)]
        mock_instance.transcribe.return_value = (iter(mock_segments), None)

        result = transcribe(Path("/fake/single.mp3"))
        assert result == "[00:00:00 - 00:00:03] Nur ein Segment."

    def test_default_model_size(self, mocker):
        _mock_audio(mocker)
        mock_model = mocker.patch("video2wiki.transcriber.WhisperModel")
        mock_instance = mock_model.return_value
        mock_segments = [mocker.Mock(text="x", start=0.0, end=1.0)]
        mock_instance.transcribe.return_value = (iter(mock_segments), None)

        transcribe(Path("/fake/audio.mp3"))
        args, _ = mock_model.call_args
        assert args[0] == "base"

    def test_custom_model_size(self, mocker):
        _mock_audio(mocker)
        mock_model = mocker.patch("video2wiki.transcriber.WhisperModel")
        mock_instance = mock_model.return_value
        mock_segments = [mocker.Mock(text="x", start=0.0, end=1.0)]
        mock_instance.transcribe.return_value = (iter(mock_segments), None)

        transcribe(Path("/fake/audio.mp3"), model_size="large-v3")
        args, _ = mock_model.call_args
        assert args[0] == "large-v3"

    def test_file_not_found(self, mocker):
        mocker.patch.object(Path, "exists", return_value=False)

        with pytest.raises(FileNotFoundError, match="not found"):
            transcribe(Path("/fake/missing.mp3"))

    def test_empty_file(self, mocker):
        _mock_audio(mocker, size=0)

        with pytest.raises(ValueError, match="empty"):
            transcribe(Path("/fake/empty.mp3"))
