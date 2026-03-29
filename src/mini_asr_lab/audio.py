from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass
class AudioSample:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate)


def load_audio(path: str | Path, target_sr: int = 16_000) -> AudioSample:
    sample_rate, samples = wavfile.read(str(path))
    samples = _to_float32_mono(samples)
    if sample_rate != target_sr:
        samples = resample_poly(samples, target_sr, sample_rate).astype(np.float32)
        sample_rate = target_sr
    samples = peak_normalize(samples)
    return AudioSample(samples=samples, sample_rate=sample_rate)


def _to_float32_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        scale = np.iinfo(samples.dtype).max
        samples = samples.astype(np.float32) / float(scale)
    else:
        samples = samples.astype(np.float32)
    return samples


def peak_normalize(samples: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak < eps:
        return samples.astype(np.float32)
    return (samples / peak).astype(np.float32)


def frame_audio(samples: np.ndarray, sample_rate: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    if len(samples) < frame_len:
        pad = np.zeros(frame_len - len(samples), dtype=np.float32)
        samples = np.concatenate([samples, pad])
    num_frames = 1 + max(0, (len(samples) - frame_len) // hop_len)
    frames = np.stack(
        [samples[i * hop_len : i * hop_len + frame_len] for i in range(num_frames)],
        axis=0,
    )
    return frames


def energy_vad(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    hop_ms: int = 10,
    threshold_scale: float = 2.0,
) -> np.ndarray:
    frames = frame_audio(samples, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    energies = np.mean(np.square(frames), axis=1)
    baseline = max(np.median(energies), 1e-8)
    threshold = baseline * threshold_scale
    return energies > threshold


def chunk_audio(
    samples: np.ndarray,
    sample_rate: int,
    chunk_ms: int = 1000,
    stride_ms: int = 250,
) -> list[np.ndarray]:
    chunk_len = max(1, int(sample_rate * chunk_ms / 1000))
    stride_len = max(1, int(sample_rate * stride_ms / 1000))
    if len(samples) <= chunk_len:
        return [samples]

    chunks: list[np.ndarray] = []
    for start in range(0, len(samples), stride_len):
        end = start + chunk_len
        chunk = samples[start:end]
        if len(chunk) == 0:
            break
        if len(chunk) < chunk_len:
            pad = np.zeros(chunk_len - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        chunks.append(chunk)
        if end >= len(samples):
            break
    return chunks
