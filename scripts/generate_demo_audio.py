from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile


DEFAULT_TEXT = (
    "This is a low latency speech recognition demo. "
    "We chunk audio, remove silence, and compare batch against streaming style inference."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--output", default="tmp/demo.wav")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aiff_path = output_path.with_suffix(".aiff")

    try:
        subprocess.run(
            ["/usr/bin/say", "-v", args.voice, "-o", str(aiff_path), args.text],
            check=True,
        )
        subprocess.run(
            [
                "/usr/local/bin/ffmpeg",
                "-y",
                "-i",
                str(aiff_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        aiff_path.unlink(missing_ok=True)
        sample_rate, samples = wavfile.read(output_path)
        if getattr(samples, "size", 0) == 0:
            raise ValueError("Generated audio was empty")
    except Exception:
        synthesize_demo_wave(output_path)
    print(output_path)


def synthesize_demo_wave(output_path: Path, sample_rate: int = 16_000) -> None:
    segments = [
        voiced_segment(0.55, 155.0, sample_rate),
        np.zeros(int(sample_rate * 0.18), dtype=np.float32),
        voiced_segment(0.45, 185.0, sample_rate),
        np.zeros(int(sample_rate * 0.12), dtype=np.float32),
        voiced_segment(0.70, 145.0, sample_rate),
        np.zeros(int(sample_rate * 0.20), dtype=np.float32),
        voiced_segment(0.60, 175.0, sample_rate),
    ]
    samples = np.concatenate(segments)
    wavfile.write(output_path, sample_rate, (samples * 32767.0).astype(np.int16))


def voiced_segment(duration_s: float, f0: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate), dtype=np.float32) / float(sample_rate)
    envelope = np.sin(np.pi * np.clip(t / max(duration_s, 1e-6), 0.0, 1.0)) ** 2
    source = (
        0.65 * np.sin(2.0 * np.pi * f0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 3.0 * f0 * t)
    )
    return (0.25 * envelope * source).astype(np.float32)


if __name__ == "__main__":
    main()
