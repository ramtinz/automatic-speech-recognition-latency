from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text())


def make_wer_boxplot(examples: list[dict], output_path: Path) -> None:
    batch = [item["batch_wer"] for item in examples]
    naive = [item["naive_chunk_wer"] for item in examples]
    dtw = [item["dtw_chunk_wer"] for item in examples]

    plt.figure(figsize=(7.5, 4.5))
    box = plt.boxplot(
        [batch, naive, dtw],
        tick_labels=["Batch", "Naive chunked", "DTW-guided"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], ["#7aa6c2", "#d98c5f", "#6fb38f"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    plt.ylabel("Word Error Rate")
    plt.title("WER Across Transcription Strategies")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_latency_vs_wer_plot(examples: list[dict], output_path: Path) -> None:
    batch_latency = [item["batch_ms"] for item in examples]
    first_chunk_latency = [item["estimated_first_chunk_latency_ms"] for item in examples]
    batch_wer = [item["batch_wer"] for item in examples]
    naive_wer = [item["naive_chunk_wer"] for item in examples]
    dtw_wer = [item["dtw_chunk_wer"] for item in examples]

    plt.figure(figsize=(7.5, 4.5))
    plt.scatter(batch_latency, batch_wer, label="Batch", s=55, color="#7aa6c2")
    plt.scatter(first_chunk_latency, naive_wer, label="Naive chunked", s=55, color="#d98c5f")
    plt.scatter(first_chunk_latency, dtw_wer, label="DTW-guided", s=55, color="#6fb38f")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Word Error Rate")
    plt.title("Latency-Accuracy Trade-off")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def render_markdown(results: dict, figures_dir: Path) -> str:
    metrics = results["summary_metrics"]
    examples = results["examples"]
    rows = "\n".join(
        [
            f"| {item['id']} | {item['audio_seconds']:.2f} | {item['batch_wer']:.3f} | {item['naive_chunk_wer']:.3f} | {item['dtw_chunk_wer']:.3f} | {item['batch_ms']:.1f} | {item['estimated_first_chunk_latency_ms']:.1f} |"
            for item in examples
        ]
    )
    qualitative = "\n".join(
        [
            (
                f"### Example: {item['id']}\n\n"
                f"Reference: `{item['reference']}`\n\n"
                f"Batch: `{item['batch_text']}`\n\n"
                f"Naive chunked: `{item['naive_chunk_text']}`\n\n"
                f"DTW-guided: `{item['dtw_stitched_text']}`\n"
            )
            for item in examples[:2]
        ]
    )

    return f"""# Short Report: DTW-Guided Chunk Stitching for Low-Latency ASR

## Problem

Low-latency ASR often uses short overlapping chunks so text can appear before the speaker finishes. That helps responsiveness, but naive chunk concatenation often duplicates words, drops context, or fragments phrases at chunk boundaries.

## Data

We evaluated on `{results['dataset']}` (`{results['config']}`, split `{results['split']}`), a small public LibriSpeech-based benchmark chosen for fast and reproducible experimentation. The study used `{results['limit']}` labeled utterances with `{results['model']}`.

## Method

We compared three strategies:

- Batch ASR over the full utterance
- Naive chunked ASR with overlapping windows concatenated directly
- DTW-guided chunk stitching, where Dynamic Time Warping on overlap-region acoustic features estimates the duplicated boundary region before stitching

Chunks used a window length of `{results['chunk_ms']}` ms and a stride of `{results['stride_ms']}` ms.

## Results

Mean batch WER was `{metrics['mean_batch_wer']:.3f}`. Mean naive chunked WER increased to `{metrics['mean_naive_chunk_wer']:.3f}`. The DTW-guided method reduced this to `{metrics['mean_dtw_chunk_wer']:.3f}`.

Mean batch latency was `{metrics['mean_batch_latency_ms']:.1f}` ms, while mean first-chunk latency for the streaming-style methods was `{metrics['mean_first_chunk_latency_ms']:.1f}` ms.

![WER boxplot](figures/wer_boxplot.png)

![Latency vs WER](figures/latency_vs_wer.png)

## Per-Utterance Table

| Example | Audio s | Batch WER | Naive WER | DTW WER | Batch ms | First chunk ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Qualitative Examples

{qualitative}

## Takeaway

On this small public benchmark, naive chunking produced the latency profile we want but badly damaged transcript quality. DTW-guided stitching recovered a substantial part of that loss relative to naive concatenation, making it a stronger starting point for private, low-latency ASR.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/dummy_dataset_eval.json")
    parser.add_argument("--report", default="report/short_report.md")
    args = parser.parse_args()

    results = load_results(args.results)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = report_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    make_wer_boxplot(results["examples"], figures_dir / "wer_boxplot.png")
    make_latency_vs_wer_plot(results["examples"], figures_dir / "latency_vs_wer.png")
    report_path.write_text(render_markdown(results, figures_dir))
    print(report_path)


if __name__ == "__main__":
    main()
