from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ASRBackend(Protocol):
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        ...


@dataclass
class NoOpBackend:
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return ""


class WhisperTransformersBackend:
    def __init__(self, model_id: str = "openai/whisper-tiny.en") -> None:
        from transformers import pipeline

        self.model_id = model_id
        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            device="cpu",
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        result = self.pipeline({"array": samples, "sampling_rate": sample_rate})
        return str(result["text"]).strip()


def build_backend(name: str, model_id: str) -> ASRBackend:
    if name == "none":
        return NoOpBackend()
    if name == "whisper":
        return WhisperTransformersBackend(model_id=model_id)
    raise ValueError(f"Unsupported backend: {name}")

