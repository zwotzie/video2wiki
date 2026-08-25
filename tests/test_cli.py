import sys

from typer.testing import CliRunner

from video2wiki.cli import _guess_category, _read_sources, app, main

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

    def test_keyword_hit(self):
        assert _guess_category("Bitcoin und Gold kaufen") == "finanzen"


class TestCliProcess:
    """CLI-Hülle: Prozess-Fehlerbehandlung und Weiterleitung werden hier nur
    oberflächlich getestet – die eigentliche Pipeline-Logik liegt in
    ``video2wiki.pipeline`` (siehe tests/test_pipeline.py)."""

    def test_no_url_no_file(self):
        result = runner.invoke(app, ["process"])
        assert result.exit_code == 1
        assert "Gib eine URL" in result.stderr

    def test_backfill(self, mocker, tmp_path):
        mocker.patch("video2wiki.cli.backfill_journal", return_value=3)
        result = runner.invoke(app, ["process", "--backfill", "--output", str(tmp_path)])
        assert result.exit_code == 0
        assert "Journal aktualisiert: 3 Einträge" in result.stderr

    def test_unknown_command(self):
        result = runner.invoke(app, ["nonsense"])
        assert result.exit_code != 0


class TestMainDefaultCommand:
    """Der Entry-Point macht ``process`` zum Standard-Befehl."""

    def test_injects_process_for_url(self, mocker):
        mocker.patch.object(sys, "argv", ["video2wiki", "https://youtube.com/watch?v=abc", "--output", "/x"])
        mock_app = mocker.patch("video2wiki.cli.app", return_value=None)
        main()
        mock_app.assert_called_once()
        assert sys.argv[1:3] == ["process", "https://youtube.com/watch?v=abc"]

    def test_keeps_known_command(self, mocker):
        mocker.patch.object(sys, "argv", ["video2wiki", "web", "--host", "0.0.0.0"])
        mock_app = mocker.patch("video2wiki.cli.app", return_value=None)
        main()
        mock_app.assert_called_once()
        assert sys.argv[1:3] == ["web", "--host"]

    def test_keeps_flag_without_url(self, mocker):
        mocker.patch.object(sys, "argv", ["video2wiki", "--help"])
        mock_app = mocker.patch("video2wiki.cli.app", return_value=None)
        main()
        mock_app.assert_called_once()
        assert sys.argv[1:] == ["--help"]
