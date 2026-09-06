"""Transcribe one voice query for the Memory Guide launcher.

The local backend intentionally mirrors the FireRedASR2-AED path used by the
Hackomania speech service.  FireRed is optional and loaded lazily so the
normal text-only OrcaLab installation does not acquire a second ML runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


class VoiceRecognitionError(RuntimeError):
    """Raised when a voice query cannot be transcribed."""


_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _find_text(payload: Mapping[str, Any]) -> str:
    """Extract text from FireRedASR2S, Hackomania, or generic ASR JSON."""

    for key in ("transcript", "text", "1best", "hyp"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Some ASR APIs return the first segment as the only useful text field.
    sentences = payload.get("sentences")
    if isinstance(sentences, list):
        parts = []
        for sentence in sentences:
            if isinstance(sentence, Mapping) and isinstance(sentence.get("text"), str):
                parts.append(sentence["text"].strip())
        if parts:
            return " ".join(part for part in parts if part)
    return ""


def _require_audio_file(audio_file: Path) -> Path:
    audio_file = audio_file.expanduser().resolve()
    if not audio_file.is_file():
        raise VoiceRecognitionError(f"voice audio file does not exist: {audio_file}")
    if audio_file.stat().st_size == 0:
        raise VoiceRecognitionError(f"voice audio file is empty: {audio_file}")
    return audio_file


def _load_firered_model(
    *,
    firered_root: Path,
    model_dir: Path,
    use_gpu: bool,
):
    """Load the standalone FireRedASR2-AED model used by Hackomania."""

    firered_root = firered_root.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    if not firered_root.is_dir():
        raise VoiceRecognitionError(
            "FireRedASR2S source directory was not found: "
            f"{firered_root}. Set NAVILA_FIRERED_ROOT to the cloned repository."
        )
    if not model_dir.is_dir():
        raise VoiceRecognitionError(
            "FireRedASR2-AED model directory was not found: "
            f"{model_dir}. Set NAVILA_FIRERED_MODEL to its downloaded weights."
        )

    root_text = str(firered_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    try:
        from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
    except ImportError as exc:
        raise VoiceRecognitionError(
            "FireRedASR2S is not importable. Install its requirements in "
            "NAVILA_VOICE_PYTHON and set NAVILA_FIRERED_ROOT."
        ) from exc

    config = FireRedAsr2Config(
        use_gpu=use_gpu,
        use_half=False,
        beam_size=3,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=False,
    )
    try:
        return FireRedAsr2.from_pretrained("aed", str(model_dir), config)
    except Exception as exc:  # FireRed exposes several backend-specific errors.
        raise VoiceRecognitionError(
            f"FireRedASR2-AED could not load from {model_dir}: {exc}"
        ) from exc


def transcribe_firered(
    audio_file: Path,
    *,
    firered_root: Path,
    model_dir: Path,
    use_gpu: bool,
) -> str:
    """Transcribe a 16 kHz mono WAV with FireRedASR2-AED."""

    audio_file = _require_audio_file(audio_file)

    # FireRed and torch can emit useful diagnostics on stdout.  The launcher
    # captures stdout as the query, so keep model diagnostics on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        model = _load_firered_model(
            firered_root=firered_root,
            model_dir=model_dir,
            use_gpu=use_gpu,
        )
        try:
            results = model.transcribe([audio_file.stem], [str(audio_file)])
        except Exception as exc:
            raise VoiceRecognitionError(f"FireRedASR2 transcription failed: {exc}") from exc

    if not results or not isinstance(results[0], Mapping):
        raise VoiceRecognitionError("FireRedASR2 returned no transcription")
    transcript = _find_text(results[0])
    if not transcript:
        raise VoiceRecognitionError(
            "FireRedASR2 returned an empty transcription; please speak a little "
            "closer to the microphone and try again"
        )
    return transcript


def _multipart_audio(audio_file: Path) -> tuple[bytes, str]:
    boundary = f"----navila-voice-{uuid.uuid4().hex}"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{audio_file.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("ascii")
    return header + audio_file.read_bytes() + footer, boundary


def transcribe_http(audio_file: Path, *, endpoint: str, timeout: float = 120.0) -> str:
    """Send audio to a FireRed-compatible HTTP transcription endpoint.

    This accepts both the Hackomania ``/process/wav`` response
    (``{"transcript": "..."}``) and the FireRed demo response
    (``{"text": "..."}``).
    """

    audio_file = _require_audio_file(audio_file)
    endpoint = endpoint.strip()
    if not endpoint:
        raise VoiceRecognitionError(
            "HTTP voice backend requires NAVILA_VOICE_ENDPOINT or --voice-endpoint"
        )
    body, boundary = _multipart_audio(audio_file)
    request = Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise VoiceRecognitionError(
            f"voice transcription endpoint returned HTTP {exc.code}: {detail[:400]}"
        ) from exc
    except URLError as exc:
        raise VoiceRecognitionError(f"could not reach voice transcription endpoint: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceRecognitionError("voice transcription endpoint returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise VoiceRecognitionError("voice transcription endpoint returned a non-object JSON value")
    transcript = _find_text(payload)
    if not transcript:
        raise VoiceRecognitionError("voice transcription endpoint returned an empty transcription")
    return transcript


def contains_han(text: str) -> bool:
    """Return whether a transcript contains Han characters.

    FireRed emits Mandarin and Chinese-dialect speech as Han text.  This lets
    English queries bypass Google Translation entirely, avoiding unnecessary
    network calls and preserving the original English wording.
    """

    return bool(_HAN_RE.search(text))


def translate_to_english(text: str, *, project_id: str | None = None) -> str:
    """Translate one Chinese transcript to English using ADC credentials."""

    try:
        import google.auth
        from google.auth.transport.requests import Request as GoogleAuthRequest
    except ImportError as exc:
        raise VoiceRecognitionError(
            "Google ADC support is not installed; run "
            "'python -m pip install google-auth' in the Python environment "
            "used by the voice launcher"
        ) from exc

    try:
        credentials, adc_project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(GoogleAuthRequest())
        billing_project_id = (
            project_id
            or os.environ.get("NAVILA_GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or adc_project_id
        )
        if not billing_project_id:
            raise VoiceRecognitionError(
                "Google Cloud project ID is unavailable; set "
                "NAVILA_GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT"
            )

        request_body = json.dumps(
            {
                "q": [text],
                "target": "en",
                "format": "text",
            }
        ).encode("utf-8")
        request = Request(
            "https://translation.googleapis.com/language/translate/v2",
            data=request_body,
            headers={
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json; charset=utf-8",
                "x-goog-user-project": billing_project_id,
            },
            method="POST",
        )
        with urlopen(request, timeout=30.0) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise VoiceRecognitionError(
            f"Google Cloud Translation returned HTTP {exc.code}: {detail[:500]}"
        ) from exc
    except URLError as exc:
        raise VoiceRecognitionError(
            f"could not reach Google Cloud Translation: {exc}"
        ) from exc
    except VoiceRecognitionError:
        raise
    except Exception as exc:  # ADC and HTTP clients expose several errors.
        raise VoiceRecognitionError(
            "Google Cloud Translation authentication failed. Check ADC, billing/API enablement, "
            f"and the project ID: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceRecognitionError(
            "Google Cloud Translation returned invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise VoiceRecognitionError("Google Cloud Translation returned an invalid response")

    data = payload.get("data")
    translations = data.get("translations") if isinstance(data, Mapping) else None
    first_translation = (
        translations[0]
        if isinstance(translations, list) and translations
        else None
    )
    translated = (
        first_translation.get("translatedText")
        if isinstance(first_translation, Mapping)
        else None
    )
    if not isinstance(translated, str) or not translated.strip():
        error = payload.get("error")
        detail = error.get("message") if isinstance(error, Mapping) else None
        raise VoiceRecognitionError(
            "Google Cloud Translation returned no translated text"
            + (f": {detail}" if detail else "")
        )
    return html.unescape(translated).strip()


def _default_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe one Memory Guide voice query")
    parser.add_argument("--audio-file", required=True, type=Path)
    parser.add_argument("--backend", choices=("firered", "http"), default="firered")
    parser.add_argument("--firered-root", type=Path, default=_env_path("NAVILA_FIRERED_ROOT"))
    parser.add_argument("--firered-model", type=Path, default=_env_path("NAVILA_FIRERED_MODEL"))
    parser.add_argument("--endpoint", default=os.environ.get("NAVILA_VOICE_ENDPOINT", ""))
    parser.add_argument("--cpu", action="store_true", help="force local FireRed inference on CPU")
    parser.add_argument(
        "--translate-to-english",
        action="store_true",
        help="translate Han-character transcripts to English with Google Cloud Translation",
    )
    parser.add_argument(
        "--google-project",
        default=os.environ.get("NAVILA_GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project ID (otherwise inferred from ADC)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.backend == "http":
            transcript = transcribe_http(args.audio_file, endpoint=args.endpoint)
        else:
            if args.firered_root is None or args.firered_model is None:
                raise VoiceRecognitionError(
                    "local FireRed voice recognition needs NAVILA_FIRERED_ROOT and "
                    "NAVILA_FIRERED_MODEL (or the corresponding command-line flags)"
                )
            transcript = transcribe_firered(
                args.audio_file,
                firered_root=args.firered_root,
                model_dir=args.firered_model,
                use_gpu=not args.cpu and _default_gpu(),
            )
    except VoiceRecognitionError as exc:
        print(f"voice recognition: {exc}", file=sys.stderr)
        return 2

    if args.translate_to_english:
        if contains_han(transcript):
            original_transcript = transcript
            transcript = translate_to_english(
                transcript,
                project_id=args.google_project,
            )
            print(f"voice transcript: {original_transcript}", file=sys.stderr)
            print(f"voice translation: {transcript}", file=sys.stderr)
        else:
            print("voice translation: skipped (transcript is already non-Chinese)", file=sys.stderr)

    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
