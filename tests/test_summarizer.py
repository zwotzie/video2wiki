import json
import os

import pytest

from video2wiki.summarizer import (
    _build_client,
    _call_llm,
    _merge_results,
    _resolve_model,
    _split_into_chunks,
    summarize,
    summarize_with_chunking,
)


class TestBuildClient:
    def test_openai_official(self, mocker):
        mocker.patch.dict(os.environ, {"OPENAI_SSL_VERIFY": "true", "OPENAI_API_KEY": ""})
        mock_openai = mocker.patch("video2wiki.summarizer.OpenAI")
        _build_client("https://api.openai.com/v1")
        mock_openai.assert_called_once_with(api_key=mocker.ANY)

    def test_custom_base_url(self, mocker):
        mocker.patch.dict(os.environ, {"OPENAI_SSL_VERIFY": "true", "OPENAI_API_KEY": ""})
        mock_openai = mocker.patch("video2wiki.summarizer.OpenAI")
        mocker.patch("video2wiki.summarizer.httpx.Client")
        _build_client("https://custom.api/v1")
        mock_openai.assert_called_once_with(
            base_url="https://custom.api/v1",
            api_key=mocker.ANY,
            http_client=mocker.ANY,
        )

    def test_api_key_from_env(self, mocker):
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-my-secret", "OPENAI_SSL_VERIFY": "true"})
        mock_openai = mocker.patch("video2wiki.summarizer.OpenAI")
        _build_client("https://api.openai.com/v1")
        mock_openai.assert_called_once_with(api_key="sk-my-secret")

    def test_no_api_key_uses_unused(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        mock_openai = mocker.patch("video2wiki.summarizer.OpenAI")
        mocker.patch("video2wiki.summarizer.httpx.Client")
        _build_client("https://custom.api/v1")
        mock_openai.assert_called_once_with(
            base_url="https://custom.api/v1",
            api_key="unused",
            http_client=mocker.ANY,
        )


class TestResolveModel:
    def test_from_env(self, mocker):
        mocker.patch.dict(os.environ, {"OPENAI_MODEL": "my-model"})
        assert _resolve_model() == "my-model"

    def test_default(self, mocker):
        # kein OPENAI_MODEL gesetzt → Fallback
        mocker.patch.dict(os.environ, {}, clear=True)
        assert _resolve_model() == "gpt-4o-mini"


class TestSplitIntoChunks:
    def test_shorter_than_chunk_size(self):
        text = "Kurzer Text."
        result = _split_into_chunks(text, 1000)
        assert result == ["Kurzer Text."]

    def test_splits_at_double_newline(self):
        text = "Absatz 1.\n\nAbsatz 2.\n\nAbsatz 3."
        result = _split_into_chunks(text, 20)
        assert len(result) >= 2

    def test_splits_at_single_newline_fallback(self):
        text = "Zeile 1\nZeile 2\nZeile 3\n"
        result = _split_into_chunks(text, 10)
        assert len(result) >= 2

    def test_splits_at_sentence_boundary(self):
        text = "Satz eins. Satz zwei. Satz drei. Satz vier."
        result = _split_into_chunks(text, 20)
        assert len(result) >= 2

    def test_all_chunks_under_limit(self):
        text = "A. " * 500
        result = _split_into_chunks(text, 100)
        for chunk in result:
            assert len(chunk) <= 100

    def test_single_word_chunk(self):
        text = " ".join(f"wort{i}" for i in range(100))
        result = _split_into_chunks(text, 50)
        assert len(result) > 1

    def test_empty_text(self):
        assert _split_into_chunks("", 100) == []

    def test_exact_size(self):
        text = "Hallo Welt"
        result = _split_into_chunks(text, 10)
        assert len(result) >= 1


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class TestCallLlm:
    def test_success(self, mocker):
        expected = {"zusammenfassung": "Test", "kernpunkte": [], "fazit": "Ok"}
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = FakeResponse(json.dumps(expected))
        mocker.patch("video2wiki.summarizer._build_client", return_value=mock_client)

        result = _call_llm("System", "User", "https://custom.api/v1", "model-x")
        assert result == expected

    def test_empty_response(self, mocker):
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = FakeResponse(None)
        mocker.patch("video2wiki.summarizer._build_client", return_value=mock_client)

        with pytest.raises(RuntimeError, match="Leere Antwort"):
            _call_llm("System", "User", "https://custom.api/v1", "model-x")

    def test_strips_markdown_code_block(self, mocker):
        raw = '```json\n{"zusammenfassung": "A", "kernpunkte": [], "fazit": "B"}\n```'
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = FakeResponse(raw)
        mocker.patch("video2wiki.summarizer._build_client", return_value=mock_client)

        result = _call_llm("System", "User", "https://custom.api/v1", "model-x")
        assert result["zusammenfassung"] == "A"

    def test_openai_structured_output(self, mocker):
        mocker.patch.dict(os.environ, {"OPENAI_STRUCTURED_OUTPUT": "true"})
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = FakeResponse(
            json.dumps({"zusammenfassung": "T", "kernpunkte": [], "fazit": "F"})
        )
        mocker.patch("video2wiki.summarizer._build_client", return_value=mock_client)

        _call_llm("System", "User", "https://api.openai.com/v1", "gpt-4o-mini")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        fmt = call_kwargs["response_format"]
        assert fmt["type"] == "json_schema"

    def test_non_openai_json_object(self, mocker):
        mocker.patch.dict(os.environ, {"OPENAI_STRUCTURED_OUTPUT": "false"})
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = FakeResponse(
            json.dumps({"zusammenfassung": "T", "kernpunkte": [], "fazit": "F"})
        )
        mocker.patch("video2wiki.summarizer._build_client", return_value=mock_client)

        _call_llm("System", "User", "https://custom.api/v1", "model-x")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        fmt = call_kwargs["response_format"]
        assert fmt["type"] == "json_object"


class TestSummarize:
    def test_basic(self, mocker):
        mock_call = mocker.patch("video2wiki.summarizer._call_llm")
        mock_call.return_value = {
            "zusammenfassung": "Zusammenfassung",
            "kernpunkte": [{"überschrift": "Punkt", "punkte": ["Detail"]}],
            "fazit": "Fazit",
        }
        result = summarize(
            "Transkript text",
            "Titel",
            api_base="https://custom.api/v1",
            model="model-x",
        )
        assert result["zusammenfassung"] == "Zusammenfassung"
        mock_call.assert_called_once()

    def test_uses_env_defaults(self, mocker):
        mock_call = mocker.patch("video2wiki.summarizer._call_llm")
        mock_call.return_value = {
            "zusammenfassung": "",
            "kernpunkte": [],
            "fazit": "",
        }
        mocker.patch.dict(
            os.environ,
            {"OPENAI_API_BASE": "https://custom.api/v1", "OPENAI_MODEL": "custom-model"},
        )
        summarize("Text", "Titel")
        # Should not raise


class TestSummarizeWithChunking:
    def test_short_text_calls_direct(self, mocker):
        mock_summarize = mocker.patch("video2wiki.summarizer.summarize")
        mock_summarize.return_value = {
            "zusammenfassung": "Z",
            "kernpunkte": [],
            "fazit": "F",
        }
        result = summarize_with_chunking("Kurzer Text", "Titel", api_base="https://custom.api/v1", model="model-x")
        assert result["zusammenfassung"] == "Z"
        # summarize() is called with positional args
        mock_summarize.assert_called_once_with("Kurzer Text", "Titel", "https://custom.api/v1", "model-x")

    def test_long_text_chunks_and_merges(self, mocker):
        long_text = "A. " * 50000  # ~150k chars
        mock_summarize = mocker.patch("video2wiki.summarizer.summarize")
        mock_summarize.return_value = {
            "zusammenfassung": "Teil-Zusammenfassung",
            "kernpunkte": [{"überschrift": "Teil", "punkte": ["Punkt"]}],
            "fazit": "Teil-Fazit",
        }
        mock_merge = mocker.patch("video2wiki.summarizer._merge_results")
        mock_merge.return_value = {
            "zusammenfassung": "Finale Zusammenfassung",
            "kernpunkte": [{"überschrift": "Final", "punkte": ["Punkt"]}],
            "fazit": "Finales Fazit",
        }

        result = summarize_with_chunking(long_text, "Titel", api_base="https://custom.api/v1", model="model-x")
        assert result["zusammenfassung"] == "Finale Zusammenfassung"
        assert mock_summarize.call_count >= 2
        mock_merge.assert_called_once()

    def test_long_text_single_chunk_after_split(self, mocker):
        """If splitting produces only one chunk (edge case), return it directly."""
        mock_summarize = mocker.patch("video2wiki.summarizer.summarize")
        mock_summarize.return_value = {
            "zusammenfassung": "Z",
            "kernpunkte": [],
            "fazit": "F",
        }
        # Just one chunk
        mocker.patch(
            "video2wiki.summarizer._split_into_chunks",
            return_value=["Nur ein Block"],
        )

        result = summarize_with_chunking("X" * 200_000, "Titel", api_base="https://custom.api/v1", model="model-x")
        assert result["zusammenfassung"] == "Z"

    def test_chunk_api_error_retries(self, mocker):
        """If a chunk is too long and API fails, it gets halved recursively."""
        long_text = "A. " * 50000
        mock_summarize = mocker.patch("video2wiki.summarizer.summarize")
        call_count = [0]

        def side_effect(text, title, api_base, model):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Context too long")
            return {
                "zusammenfassung": f"Z{call_count[0]}",
                "kernpunkte": [],
                "fazit": f"F{call_count[0]}",
            }

        mock_summarize.side_effect = side_effect
        mock_merge = mocker.patch("video2wiki.summarizer._merge_results")
        mock_merge.return_value = {
            "zusammenfassung": "Merged",
            "kernpunkte": [],
            "fazit": "Merged F",
        }

        result = summarize_with_chunking(long_text, "Titel", api_base="https://custom.api/v1", model="model-x")
        assert result["zusammenfassung"] == "Merged"

    def test_all_chunks_fail_raises(self, mocker):
        mock_summarize = mocker.patch("video2wiki.summarizer.summarize")
        mock_summarize.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError):
            summarize_with_chunking("A. " * 50000, "Titel", api_base="https://custom.api/v1", model="model-x")


class TestMergeResults:
    def test_merges_two_parts(self, mocker):
        mock_call = mocker.patch("video2wiki.summarizer._call_llm")
        mock_call.return_value = {
            "zusammenfassung": "Merged",
            "kernpunkte": [{"überschrift": "M", "punkte": ["P"]}],
            "fazit": "F",
        }
        partials = [
            {"zusammenfassung": "Z1", "kernpunkte": [{"überschrift": "A", "punkte": ["1"]}], "fazit": "F1"},
            {"zusammenfassung": "Z2", "kernpunkte": [{"überschrift": "B", "punkte": ["2"]}], "fazit": "F2"},
        ]
        result = _merge_results(partials, "Titel", "https://custom.api/v1", "model-x")
        assert result["zusammenfassung"] == "Merged"

    def test_includes_both_parts_in_prompt(self, mocker):
        mock_call = mocker.patch("video2wiki.summarizer._call_llm")
        mock_call.return_value = {
            "zusammenfassung": "M",
            "kernpunkte": [],
            "fazit": "F",
        }
        partials = [
            {"zusammenfassung": "Z1", "kernpunkte": [{"überschrift": "A", "punkte": ["1"]}], "fazit": "F1"},
            {"zusammenfassung": "Z2", "kernpunkte": [{"überschrift": "B", "punkte": ["2"]}], "fazit": "F2"},
        ]
        _merge_results(partials, "Titel", "https://custom.api/v1", "model-x")
        prompt = mock_call.call_args[0][1]
        assert "Teil 1" in prompt
        assert "Teil 2" in prompt
        assert "Z1" in prompt
        assert "Z2" in prompt
