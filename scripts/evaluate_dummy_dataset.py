from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from statistics import mean, median
from pathlib import Path

from datasets import load_dataset
from rapidfuzz.distance import Levenshtein

from mini_asr_lab.audio import chunk_audio, load_audio
from mini_asr_lab.backends import build_backend
from mini_asr_lab.dtw_alignment import estimate_overlap_offset, stitch_chunk_texts
from mini_asr_lab.text import tokenize_text


def timed_call(fn):
    start = time.perf_counter()
    result = fn()
    duration_ms = (time.perf_counter() - start) * 1000.0
    return result, duration_ms


TRANSCRIPT_FIELDS = {
    "batch": "batch_text",
    "naive_chunk": "naive_chunk_text",
    "text_only": "text_only_text",
    "fixed_overlap": "fixed_overlap_text",
    "dtw_chunk": "dtw_stitched_text",
}


def word_errors(reference_words: list[str], hypothesis: str) -> dict[str, int]:
    operations = Levenshtein.editops(reference_words, tokenize_text(hypothesis))
    counts = Counter(operation.tag for operation in operations)
    return {
        "substitutions": counts["replace"],
        "deletions": counts["delete"],
        "insertions": counts["insert"],
    }


def score_transcripts(examples: list[dict]) -> dict:
    totals = {method: Counter() for method in TRANSCRIPT_FIELDS}
    reference_word_count = 0
    for example in examples:
        reference_words = tokenize_text(example["reference"])
        reference_word_count += len(reference_words)
        example["reference_words"] = len(reference_words)
        example["word_errors"] = {}
        for method, field in TRANSCRIPT_FIELDS.items():
            errors = word_errors(reference_words, example[field])
            example["word_errors"][method] = errors
            example[f"{method}_wer"] = round(sum(errors.values()) / max(len(reference_words), 1), 4)
            totals[method].update(errors)

    return {
        "reference_words": reference_word_count,
        "corpus_word_errors": {
            method: {
                **counts,
                "wer": round(sum(counts.values()) / reference_word_count, 4)
                if reference_word_count else None,
            }
            for method, counts in totals.items()
        },
    }


def manifest_rows(path: Path, limit: int):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows[:limit]:
        audio = load_audio(path.parent / row["audio"])
        yield {
            "id": row["id"],
            "text": row["text"],
            "cohort": row.get("cohort", "unspecified"),
            "audio": {"array": audio.samples, "sampling_rate": audio.sample_rate},
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--manifest", type=Path, help="Local JSON list of WAV clips and references")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--backend", choices=["whisper"], default="whisper")
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--chunk-ms", type=int, default=1200)
    parser.add_argument("--stride-ms", type=int, default=400)
    parser.add_argument("--results", default="results/dummy_dataset_eval.json")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.manifest:
        rows = manifest_rows(args.manifest, args.limit)
    else:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
        rows = dataset.select(range(min(args.limit, len(dataset))))
    backend = build_backend(args.backend, args.model)
    examples = []
    overlap_ms = max(args.chunk_ms - args.stride_ms, 0)

    for row in rows:
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
        dtw_stitched_text = stitch_chunk_texts(
            chunk_outputs,
            dtw_offsets,
            chunk_ms=args.chunk_ms,
            overlap_ms=overlap_ms,
        )

        fixed_overlap_text = stitch_chunk_texts(
            chunk_outputs, [], chunk_ms=args.chunk_ms, overlap_ms=overlap_ms
        )
        text_only_text = stitch_chunk_texts(
            chunk_outputs, None, chunk_ms=args.chunk_ms, overlap_ms=overlap_ms
        )

        examples.append(
            {
                "id": row.get("id"),
                "reference": reference,
                "cohort": row.get("cohort", "unspecified"),
                "batch_text": batch_text,
                "naive_chunk_text": naive_chunk_text,
                "dtw_stitched_text": dtw_stitched_text,
                "fixed_overlap_text": fixed_overlap_text,
                "text_only_text": text_only_text,
                "chunk_outputs": chunk_outputs,
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

    if not examples:
        parser.error("No examples selected")

    summary_metrics = {
        **score_transcripts(examples),
        "mean_batch_wer": round(mean(item["batch_wer"] for item in examples), 4),
        "mean_naive_chunk_wer": round(mean(item["naive_chunk_wer"] for item in examples), 4),
        "mean_fixed_overlap_wer": round(mean(item["fixed_overlap_wer"] for item in examples), 4),
        "mean_text_only_wer": round(mean(item["text_only_wer"] for item in examples), 4),
        "mean_dtw_chunk_wer": round(mean(item["dtw_chunk_wer"] for item in examples), 4),
        "median_batch_wer": round(median(item["batch_wer"] for item in examples), 4),
        "median_naive_chunk_wer": round(median(item["naive_chunk_wer"] for item in examples), 4),
        "median_fixed_overlap_wer": round(median(item["fixed_overlap_wer"] for item in examples), 4),
        "median_text_only_wer": round(median(item["text_only_wer"] for item in examples), 4),
        "median_dtw_chunk_wer": round(median(item["dtw_chunk_wer"] for item in examples), 4),
        "mean_batch_latency_ms": round(mean(item["batch_ms"] for item in examples), 3),
        "mean_first_chunk_latency_ms": round(mean(item["estimated_first_chunk_latency_ms"] for item in examples), 3),
    }

    cohort_metrics = {}
    if args.manifest:
        for cohort in sorted({item["cohort"] for item in examples}):
            cohort_examples = [item for item in examples if item["cohort"] == cohort]
            cohort_metrics[cohort] = {"utterances": len(cohort_examples), **score_transcripts(cohort_examples)}

    source = (
        {"manifest": str(args.manifest)}
        if args.manifest
        else {"dataset": args.dataset, "config": args.config, "split": args.split}
    )
    summary = {
        **source,
        "cohort_metrics": cohort_metrics,
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
