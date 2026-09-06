import json
from pathlib import Path

from navila_orca import voice_query
from navila_orca.voice_query import (
    _find_text,
    contains_han,
    main,
    transcribe_http,
)


def test_find_text_accepts_hackomania_transcript_response():
    assert _find_text({"transcript": "Where is my bread?"}) == "Where is my bread?"


def test_find_text_accepts_firered_segment_response():
    assert _find_text({"sentences": [{"text": "Where is my bread?"}]}) == (
        "Where is my bread?"
    )


def test_contains_han_distinguishes_chinese_from_english():
    assert contains_han("我的面包在哪里")
    assert not contains_han("Where is my bread?")


def test_http_backend_accepts_hackomania_response(tmp_path: Path, monkeypatch):
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"RIFF-test")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"transcript": "Where is my bread?"}).encode()

    def fake_urlopen(request, timeout):
        assert request.full_url.endswith("/process/wav")
        assert b'name="file"' in request.data
        assert timeout == 120.0
        return _Response()

    monkeypatch.setattr(voice_query, "urlopen", fake_urlopen)
    transcript = transcribe_http(audio, endpoint="http://speech/process/wav")
    assert transcript == "Where is my bread?"


def test_cli_reports_missing_local_firered_configuration(tmp_path: Path, capsys):
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"RIFF-test")
    assert main(["--audio-file", str(audio), "--backend", "firered"]) == 2
    assert "NAVILA_FIRERED_ROOT" in capsys.readouterr().err


def test_cli_skips_translation_for_english(monkeypatch, tmp_path: Path, capsys):
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"RIFF-test")
    monkeypatch.setattr(voice_query, "transcribe_http", lambda *_args, **_kwargs: "Where is my bread?")

    assert main(
        [
            "--audio-file",
            str(audio),
            "--backend",
            "http",
            "--endpoint",
            "http://speech/process/wav",
            "--translate-to-english",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == "Where is my bread?\n"
    assert "skipped" in captured.err


def test_cli_translates_han_transcript(monkeypatch, tmp_path: Path, capsys):
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"RIFF-test")
    monkeypatch.setattr(voice_query, "transcribe_http", lambda *_args, **_kwargs: "我的面包在哪里？")
    calls = []

    def fake_translate(text, *, project_id):
        calls.append((text, project_id))
        return "Where is my bread?"

    monkeypatch.setattr(voice_query, "translate_to_english", fake_translate)

    assert main(
        [
            "--audio-file",
            str(audio),
            "--backend",
            "http",
            "--endpoint",
            "http://speech/process/wav",
            "--translate-to-english",
            "--google-project",
            "my-project",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == "Where is my bread?\n"
    assert calls == [("我的面包在哪里？", "my-project")]
