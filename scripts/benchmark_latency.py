from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mini_asr_lab.audio import chunk_audio, energy_vad, load_audio
from mini_asr_lab.backends import build_backend


def timed_call(fn):
    start = time.perf_counter()
    result = fn()
    duration_ms = (time.perf_counter() - start) * 1000.0
    return result, duration_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--backend", choices=["none", "whisper"], default="none")
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--stride-ms", type=int, default=250)
    parser.add_argument("--results", default="results/benchmark.json")
    args = parser.parse_args()

    audio, load_ms = timed_call(lambda: load_audio(args.audio))
    chunks = chunk_audio(audio.samples, audio.sample_rate, args.chunk_ms, args.stride_ms)
    speech_mask, vad_ms = timed_call(lambda: energy_vad(audio.samples, audio.sample_rate))

    backend, backend_init_ms = timed_call(lambda: build_backend(args.backend, args.model))

    _, batch_ms = timed_call(lambda: backend.transcribe(audio.samples, audio.sample_rate))

    chunk_latencies_ms: list[float] = []
    for chunk in chunks:
        _, chunk_ms = timed_call(lambda c=chunk: backend.transcribe(c, audio.sample_rate))
        chunk_latencies_ms.append(chunk_ms)

    active_frames = int(speech_mask.sum())
    total_frames = int(len(speech_mask))
    results = {
        "audio_path": str(Path(args.audio)),
        "backend": args.backend,
        "model": args.model if args.backend != "none" else None,
        "audio_seconds": round(audio.duration_s, 3),
        "load_ms": round(load_ms, 3),
        "vad_ms": round(vad_ms, 3),
        "backend_init_ms": round(backend_init_ms, 3),
        "batch_ms": round(batch_ms, 3),
        "batch_rtf": round(batch_ms / (audio.duration_s * 1000.0), 4),
        "chunk_count": len(chunks),
        "mean_chunk_ms": round(sum(chunk_latencies_ms) / max(len(chunk_latencies_ms), 1), 3),
        "max_chunk_ms": round(max(chunk_latencies_ms, default=0.0), 3),
        "estimated_first_chunk_latency_ms": round(chunk_latencies_ms[0] if chunk_latencies_ms else 0.0, 3),
        "active_vad_frames": active_frames,
        "total_vad_frames": total_frames,
        "active_vad_ratio": round(active_frames / max(total_frames, 1), 4),
    }

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

