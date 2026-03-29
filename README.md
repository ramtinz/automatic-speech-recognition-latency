# Low-Latency ASR with DTW-Guided Stitching

Short overlapping chunks can make speech recognition feel fast, but they often damage transcript quality by duplicating or fragmenting words at chunk boundaries.

This project tests a simple idea: use Dynamic Time Warping (DTW) to align neighboring audio chunks before stitching their transcripts together.

## What This Repo Contains

- a small low-latency ASR evaluation pipeline
- a DTW-guided chunk stitching baseline
- a reproducible experiment on a public speech dataset
- a short report with figures and qualitative examples

## Experiment

Dataset: `hf-internal-testing/librispeech_asr_dummy`  
Model: `openai/whisper-tiny.en`

Compared methods:

- batch transcription
- naive overlapping chunk transcription
- DTW-guided chunk stitching

Main result on 20 utterances:

- mean batch WER: `0.066`
- mean naive chunked WER: `6.999`
- mean DTW-guided WER: `2.138`
- mean first-chunk latency: `206.9 ms`

DTW-guided stitching does not match full-utterance batch accuracy, but it substantially reduces the failure mode introduced by naive low-latency chunking.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/evaluate_dummy_dataset.py \
  --dataset hf-internal-testing/librispeech_asr_dummy \
  --config clean \
  --split validation \
  --limit 20 \
  --model openai/whisper-tiny.en \
  --results results/experiment_v1.json
PYTHONPATH=src python scripts/make_report.py \
  --results results/experiment_v1.json \
  --report report/short_report.md
```

## Report

- [Short report](report/short_report.md)
- [WER figure](report/figures/wer_boxplot.png)
- [Latency vs WER figure](report/figures/latency_vs_wer.png)
