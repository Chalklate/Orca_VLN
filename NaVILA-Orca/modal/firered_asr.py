"""Modal-hosted FireRedASR2-AED service for Orca Memory Guide queries.

Deploy:
    modal deploy --stream-logs modal/firered_asr.py

Run a remote smoke test:
    modal run modal/firered_asr.py --audio-path /path/to/query.wav

The service deliberately contains ASR only.  It does not require an OpenAI
secret and does not run the Hackomania LLM/TTS pipeline.
"""

import os
from pathlib import Path
import subprocess
import tempfile

import modal


app = modal.App(name="navila-orca-firered-asr")

FIREREDASR_REPO = "https://github.com/FireRedTeam/FireRedASR2S"
HF_ASR_REPO = "FireRedTeam/FireRedASR2-AED"
MODEL_ROOT = "/root/pretrained_models"
ASR_DIR = f"{MODEL_ROOT}/FireRedASR2-AED"
IMAGE_REVISION = "transformers-2026-09-06-v2"
SERVICE_REVISION = "bytes-upload-2026-09-07-v1"

asr_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",
        "kaldiio==2.18.0",
        "kaldi_native_fbank==1.15",
        "soundfile==0.12.1",
        "transformers==4.51.3",
        "cn2an==0.5.23",
        "sentencepiece==0.1.99",
        "peft>=0.13.2",
        "textgrid",
        "huggingface-hub>=0.23.0",
        "fastapi>=0.110.0",
        "python-multipart",
    )
    .run_commands(
        "python -c \"import transformers; print('transformers', transformers.__version__)\"",
        f"git clone --depth=1 {FIREREDASR_REPO} /root/FireRedASR2S",
    )
    .env(
        {
            "PYTHONPATH": "/root/FireRedASR2S",
            "NAVILA_ASR_IMAGE_REVISION": IMAGE_REVISION,
        }
    )
)

model_volume = modal.Volume.from_name(
    "navila-orca-firered-weights",
    create_if_missing=True,
)


@app.cls(
    image=asr_image,
    gpu="T4",
    cpu=4,
    memory=16384,
    volumes={MODEL_ROOT: model_volume},
    timeout=10 * 60,
    min_containers=0,
    scaledown_window=300,
)
class FireRedASR:
    """Single-model remote ASR service."""

    @modal.enter()
    def load_model(self):
        import torch
        from huggingface_hub import snapshot_download
        from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

        if not torch.cuda.is_available():
            raise RuntimeError("Modal allocated no CUDA GPU; expected an NVIDIA T4")
        self.gpu_name = torch.cuda.get_device_name(0)
        if "T4" not in self.gpu_name.upper():
            raise RuntimeError(
                f"Modal allocated {self.gpu_name!r}; this service requires an NVIDIA T4"
            )
        print(f"Using GPU: {self.gpu_name}")

        if not (Path(ASR_DIR) / "model.pth.tar").exists():
            print(f"Downloading {HF_ASR_REPO} into {ASR_DIR} ...")
            snapshot_download(HF_ASR_REPO, local_dir=ASR_DIR)
            model_volume.commit()

        self.model = FireRedAsr2.from_pretrained(
            "aed",
            ASR_DIR,
            FireRedAsr2Config(
                use_gpu=True,
                use_half=False,
                beam_size=3,
                nbest=1,
                decode_max_len=0,
                softmax_smoothing=1.25,
                aed_length_penalty=0.6,
                eos_penalty=1.0,
                return_timestamp=False,
            ),
        )
        print("FireRedASR2-AED ready.")

    @staticmethod
    def _transcript(result: dict) -> str:
        for key in ("text", "transcript", "1best", "hyp"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _transcribe_wav(self, wav_bytes: bytes) -> dict:
        """Normalize an upload, then transcribe it with FireRed."""

        input_fd, input_path = tempfile.mkstemp(
            prefix="navila-voice-input-", suffix=".audio"
        )
        output_fd, output_path = tempfile.mkstemp(
            prefix="navila-voice-normalized-", suffix=".wav"
        )
        os.close(output_fd)
        try:
            with os.fdopen(input_fd, "wb") as audio_file:
                audio_file.write(wav_bytes)

            # FireRedASR2 is sensitive to PCM value ranges.  In particular,
            # passing float32 WAVs through unchanged can produce confident
            # nonsense, so normalize every upload at the service boundary.
            conversion = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    input_path,
                    "-map",
                    "0:a:0",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    output_path,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if conversion.returncode != 0:
                error = conversion.stderr.decode(errors="replace").strip()
                raise ValueError(f"audio normalization failed: {error}")

            results = self.model.transcribe(
                [Path(output_path).stem], [output_path]
            )
        finally:
            for path in (input_path, output_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

        result = results[0] if results else {}
        text = self._transcript(result)
        if not text:
            raise ValueError("FireRedASR2 returned an empty transcription")
        return {
            "text": text,
            "confidence": result.get("confidence"),
            "duration_s": result.get("dur_s"),
            "backend": "fireredasr2-aed",
            "gpu": self.gpu_name,
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_format": "pcm_s16le",
        }

    @modal.method()
    def transcribe_wav(self, wav_bytes: bytes) -> dict:
        return self._transcribe_wav(wav_bytes)

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, File, HTTPException
        from fastapi.responses import JSONResponse

        web_app = FastAPI(title="NaVILA FireRed ASR", version="1.0")

        @web_app.get("/health")
        async def health():
            return {
                "status": "ok",
                "backend": "fireredasr2-aed",
                "gpu": self.gpu_name,
                "service_revision": SERVICE_REVISION,
            }

        @web_app.post("/transcribe/wav")
        async def transcribe_wav_endpoint(file: bytes = File(...)):
            # Use bytes instead of UploadFile so FastAPI/Pydantic does not
            # need to resolve a locally imported UploadFile annotation.
            wav_bytes = file
            if not wav_bytes:
                raise HTTPException(status_code=400, detail="empty audio upload")
            try:
                result = self._transcribe_wav(wav_bytes)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return JSONResponse(result)

        return web_app


@app.local_entrypoint()
def smoke_test(audio_path: str = ""):
    """Run one remote transcription to validate Modal, GPU, weights, and ASR."""

    if not audio_path:
        raise SystemExit(
            "Pass a 16 kHz mono WAV: modal run modal/firered_asr.py "
            "--audio-path /path/to/query.wav"
        )
    if not os.path.isfile(audio_path):
        raise SystemExit(f"Audio file does not exist: {audio_path}")
    with open(audio_path, "rb") as audio_file:
        result = FireRedASR().transcribe_wav.remote(audio_file.read())
    print(result)
