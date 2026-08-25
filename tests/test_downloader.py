import json
import subprocess

import pytest

from video2wiki.downloader import (
    _parse_subtitles,
    _ytdlp_extra_args,
    download_audio,
    get_subtitles,
    get_video_id,
    get_video_metadata,
)


class TestGetVideoId:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://youtube.com/watch?v=abc123DEFgh", "abc123DEFgh"),
            ("https://youtu.be/abc123DEFgh", "abc123DEFgh"),
            ("https://www.youtube.com/watch?v=abc123DEFgh&t=30s", "abc123DEFgh"),
            ("https://www.youtube.com/shorts/abc123DEFgh", "abc123DEFgh"),
            ("https://m.youtube.com/watch?v=abc123DEFgh", "abc123DEFgh"),
            ("https://youtube.com/live/abc123DEFgh?si=xyz", "abc123DEFgh"),
        ],
    )
    def test_extracts_id(self, url, expected):
        assert get_video_id(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com",
            "not-a-url",
            "",
        ],
    )
    def test_invalid_returns_unknown(self, url):
        assert get_video_id(url) == "unknown"


class TestGetVideoMetadata:
    def test_success(self, mocker):
        fake_data = {
            "title": "Mein Test Video",
            "channel": "Mein Kanal",
            "uploader": "Alter Name",
            "upload_date": "20240520",
            "language": "de",
            "tags": ["tag1", "tag2"],
        }
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.return_value = mocker.Mock(stdout=json.dumps(fake_data))

        result = get_video_metadata("https://youtube.com/watch?v=abc123DEFgh")

        assert result["title"] == "Mein Test Video"
        assert result["channel"] == "Mein Kanal"
        assert result["publish_date"] == "2024-05-20"
        assert result["language"] == "de"
        assert result["tags"] == ["tag1", "tag2"]

    def test_empty_fields(self, mocker):
        fake_data = {
            "title": None,
            "upload_date": None,
            "language": None,
            "tags": None,
        }
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.return_value = mocker.Mock(stdout=json.dumps(fake_data))

        result = get_video_metadata("https://youtube.com/watch?v=abc123DEFgh")
        assert result["title"] == ""
        assert result["channel"] == ""  # No channel/uploader key
        assert result["publish_date"] == ""
        assert result["language"] == ""
        assert result["tags"] == []

    def test_correct_command(self, mocker):
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.return_value = mocker.Mock(stdout=json.dumps({"title": "T"}))

        get_video_metadata("https://youtube.com/watch?v=abc123DEFgh")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "yt-dlp"
        assert "--dump-json" in cmd
        assert "https://youtube.com/watch?v=abc123DEFgh" in cmd

    def test_subprocess_error(self, mocker):
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, [])

        with pytest.raises(subprocess.CalledProcessError):
            get_video_metadata("https://youtube.com/watch?v=abc123DEFgh")


class TestYtdlpExtraArgs:
    """YouTube-403-Workaround: Android-Player-Client + node-Runtime."""

    def test_defaults(self):
        args = _ytdlp_extra_args()
        assert "--js-runtimes" in args
        assert "node" in args
        assert "--extractor-args" in args
        assert "youtube:player_client=android" in args

    def test_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv("YTDLP_JS_RUNTIME", "deno")
        monkeypatch.setenv("YTDLP_PLAYER_CLIENT", "tv")
        args = _ytdlp_extra_args()
        assert "deno" in args
        assert "youtube:player_client=tv" in args


class TestDownloadAudio:
    def test_download_success(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        expected_file = tmp_path / f"{vid}.mp3"
        expected_file.write_text("fake audio")
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")

        result = download_audio(url, tmp_path)

        assert result == expected_file
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "yt-dlp"
        assert "-x" in cmd
        assert "--audio-format" in cmd
        assert "mp3" in cmd

    def test_includes_yt_403_workaround_args(self, mocker, tmp_path):
        """Die Extra-Argumente gegen YouTube-Blockaden müssen im yt-dlp-Aufruf stecken."""
        vid = "abc123DEFgh"
        expected_file = tmp_path / f"{vid}.mp3"
        expected_file.write_text("fake audio")
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")

        download_audio(f"https://youtube.com/watch?v={vid}", tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--js-runtimes" in cmd
        assert "node" in cmd
        assert "--extractor-args" in cmd
        assert "youtube:player_client=android" in cmd

    def test_file_not_found(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mocker.patch("video2wiki.downloader.subprocess.run")

        with pytest.raises(FileNotFoundError, match="No audio file found"):
            download_audio(url, tmp_path)

    def test_fallback_iterdir(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        # expected path without .mp3 → fallback finds it
        actual = tmp_path / f"{vid}.mp3"
        actual.write_text("fake")
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mocker.patch("video2wiki.downloader.subprocess.run")

        result = download_audio(url, tmp_path)
        assert result == actual

    def test_creates_output_dir(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        nested = tmp_path / "sub" / "dir"
        expected_file = nested / f"{vid}.mp3"
        expected_file.parent.mkdir(parents=True)
        expected_file.write_text("fake")
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mocker.patch("video2wiki.downloader.subprocess.run")

        result = download_audio(url, nested)
        assert result == expected_file

    def test_subprocess_error(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, [])

        with pytest.raises(subprocess.CalledProcessError):
            download_audio(url, tmp_path)


class TestGetSubtitles:
    def test_cache_hit(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        vtt_file = tmp_path / f"{vid}.de.vtt"
        vtt_file.write_text(
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nHallo Welt\n\n2\n00:00:05.000 --> 00:00:08.000\nTest läuft\n"
        )
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")

        result = get_subtitles(url, tmp_path)

        assert result == "[00:00:01 - 00:00:04] Hallo Welt\n[00:00:05 - 00:00:08] Test läuft"
        mock_run.assert_not_called()

    def test_download_success(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0)
        # Create file after "download"
        vtt_file = tmp_path / f"{vid}.de.vtt"

        def _create_file(*args, **kwargs):
            vtt_file.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nNur ein Test\n")
            return mocker.Mock(returncode=0)

        mock_run.side_effect = _create_file

        result = get_subtitles(url, tmp_path)
        assert result == "[00:00:01 - 00:00:04] Nur ein Test"
        mock_run.assert_called_once()

    def test_download_fails_returns_none(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=1)

        result = get_subtitles(url, tmp_path)
        assert result is None

    def test_no_vtt_file_after_download(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0)

        result = get_subtitles(url, tmp_path)
        assert result is None

    def test_subprocess_error_not_captured(self, mocker, tmp_path):
        """subprocess.run is called with capture_output=True, so CalledProcessError
        is not raised automatically. We simulate a non-zero returncode."""
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, [])

        with pytest.raises(subprocess.CalledProcessError):
            get_subtitles(url, tmp_path)

    def test_correct_sub_arguments(self, mocker, tmp_path):
        vid = "abc123DEFgh"
        url = f"https://youtube.com/watch?v={vid}"
        mocker.patch("video2wiki.downloader.get_video_id", return_value=vid)
        mock_run = mocker.patch("video2wiki.downloader.subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0)

        get_subtitles(url, tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--write-subs" in cmd
        assert "--sub-langs" in cmd
        assert "de,en" in cmd
        assert "--skip-download" in cmd
        assert "--sub-format" in cmd
        assert "vtt" in cmd


class TestParseSubtitles:
    def test_simple(self, tmp_path):
        vtt = tmp_path / "test.vtt"
        vtt.write_text(
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nHallo Welt\n\n2\n00:00:05.000 --> 00:00:08.000\nZweite Zeile\n"
        )
        result = _parse_subtitles(vtt)
        assert result == "[00:00:01 - 00:00:04] Hallo Welt\n[00:00:05 - 00:00:08] Zweite Zeile"

    def test_multiline_cues(self, tmp_path):
        vtt = tmp_path / "test.vtt"
        vtt.write_text(
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:05.000\nErste Zeile\n"
            "Zweite Zeile\n\n2\n00:00:06.000 --> 00:00:10.000\nDritte Zeile\n"
        )
        result = _parse_subtitles(vtt)
        assert (
            result == "[00:00:01 - 00:00:05] Erste Zeile\n"
            "[00:00:01 - 00:00:05] Zweite Zeile\n"
            "[00:00:06 - 00:00:10] Dritte Zeile"
        )

    def test_empty_lines_and_timestamps_removed(self, tmp_path):
        vtt = tmp_path / "test.vtt"
        vtt.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n\n2\n00:01:00.000 --> 00:01:02.000\nText\n")
        result = _parse_subtitles(vtt)
        assert result == "[00:01:00 - 00:01:02] Text"

    def test_notes_kind_preserved(self, tmp_path):
        """Header lines after WEBVTT (like 'Kind:') are currently preserved."""
        vtt = tmp_path / "test.vtt"
        vtt.write_text("WEBVTT\nKind: captions\nLanguage: de\n\n1\n00:00:01.000 --> 00:00:03.000\nHallo\n")
        result = _parse_subtitles(vtt)
        assert "Kind: captions" in result
        assert "Hallo" in result

    def test_empty_file(self, tmp_path):
        vtt = tmp_path / "test.vtt"
        vtt.write_text("")
        result = _parse_subtitles(vtt)
        assert result == ""

    def test_only_webvtt_header(self, tmp_path):
        vtt = tmp_path / "test.vtt"
        vtt.write_text("WEBVTT")
        result = _parse_subtitles(vtt)
        assert result == ""
