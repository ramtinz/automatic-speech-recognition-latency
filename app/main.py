from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from mini_asr_lab.audio import load_audio
from mini_asr_lab.backends import build_backend

app = FastAPI(title="Low-Latency ASR Lab")
backend = build_backend("none", "openai/whisper-tiny.en")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = Path(tmp_file.name)

        audio = load_audio(tmp_path)
        text = backend.transcribe(audio.samples, audio.sample_rate)
        return {
            "text": text,
            "sample_rate": audio.sample_rate,
            "duration_s": round(audio.duration_s, 3),
        }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals():
            tmp_path.unlink(missing_ok=True)
