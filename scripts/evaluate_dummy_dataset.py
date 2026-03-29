from __future__ import annotations

import argparse
import json
import re
import time
from statistics import mean, median
from pathlib import Path

from datasets import load_dataset

from mini_asr_lab.audio import chunk_audio
from mini_asr_lab.backends import build_backend
from mini_asr_lab.dtw_alignment import estimate_overlap_offset, stitch_chunk_texts_with_dtw


def timed_call(fn):
    start = time.perf_counter()
    result = fn()
    duration_ms = (time.perf_counter() - start) * 1000.0
    return result, duration_ms


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / max(len(ref_words), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--backend", choices=["whisper"], default="whisper")
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--chunk-ms", type=int, default=1200)
    parser.add_argument("--stride-ms", type=int, default=400)
    parser.add_argument("--results", default="results/dummy_dataset_eval.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.config, split=args.split)
    backend = build_backend(args.backend, args.model)
    examples = []
    overlap_ms = max(args.chunk_ms - args.stride_ms, 0)

    for row in dataset.select(range(min(args.limit, len(dataset)))):
        audio = row["audio"]
        samples = audio["array"]
        sample_rate = int(audio["sampling_rate"])
        reference = row.get("text", "")

        batch_text, batch_ms = timed_call(lambda: backend.transcribe(samples, sample_rate))
        chunks = chunk_audio(samples, sample_rate, chunk_ms=args.chunk_ms, stride_ms=args.stride_ms)

        chunk_outputs = []
        chunk_latencies_ms = []
        for chunk in chunks:
            chunk_text, chunk_ms = timed_call(lambda c=chunk: backend.transcribe(c, sample_rate))
            chunk_outputs.append(chunk_text)
            chunk_latencies_ms.append(chunk_ms)
        naive_chunk_text = " ".join(text for text in chunk_outputs if text).strip()

        dtw_offsets = []
        if overlap_ms > 0 and len(chunks) > 1:
            overlap_len = int(sample_rate * overlap_ms / 1000)
            for left, right in zip(chunks[:-1], chunks[1:]):
                dtw_offsets.append(
                    estimate_overlap_offset(
                        left[-overlap_len:],
                        right[:overlap_len],
                        sample_rate=sample_rate,
                    )
                )
        dtw_stitched_text = stitch_chunk_texts_with_dtw(
            chunk_outputs,
            dtw_offsets,
            chunk_ms=args.chunk_ms,
            overlap_ms=overlap_ms,
        )

        examples.append(
            {
                "id": row.get("id"),
                "reference": reference,
                "batch_text": batch_text,
                "naive_chunk_text": naive_chunk_text,
                "dtw_stitched_text": dtw_stitched_text,
                "batch_wer": round(word_error_rate(reference, batch_text), 4),
                "naive_chunk_wer": round(word_error_rate(reference, naive_chunk_text), 4),
                "dtw_chunk_wer": round(word_error_rate(reference, dtw_stitched_text), 4),
                "audio_seconds": round(len(samples) / sample_rate, 3),
                "batch_ms": round(batch_ms, 3),
                "mean_chunk_ms": round(sum(chunk_latencies_ms) / max(len(chunk_latencies_ms), 1), 3),
                "estimated_first_chunk_latency_ms": round(chunk_latencies_ms[0] if chunk_latencies_ms else 0.0, 3),
                "mean_dtw_distance": round(
                    sum(item["distance"] for item in dtw_offsets) / max(len(dtw_offsets), 1),
                    3,
                ),
                "mean_dtw_offset_ms": round(
                    sum(item["offset_ms"] for item in dtw_offsets) / max(len(dtw_offsets), 1),
                    3,
                ),
            }
        )

    summary_metrics = {
        "mean_batch_wer": round(mean(item["batch_wer"] for item in examples), 4),
        "mean_naive_chunk_wer": round(mean(item["naive_chunk_wer"] for item in examples), 4),
        "mean_dtw_chunk_wer": round(mean(item["dtw_chunk_wer"] for item in examples), 4),
        "median_batch_wer": round(median(item["batch_wer"] for item in examples), 4),
        "median_naive_chunk_wer": round(median(item["naive_chunk_wer"] for item in examples), 4),
        "median_dtw_chunk_wer": round(median(item["dtw_chunk_wer"] for item in examples), 4),
        "mean_batch_latency_ms": round(mean(item["batch_ms"] for item in examples), 3),
        "mean_first_chunk_latency_ms": round(mean(item["estimated_first_chunk_latency_ms"] for item in examples), 3),
    }

    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "limit": args.limit,
        "model": args.model,
        "chunk_ms": args.chunk_ms,
        "stride_ms": args.stride_ms,
        "summary_metrics": summary_metrics,
        "examples": examples,
    }

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
