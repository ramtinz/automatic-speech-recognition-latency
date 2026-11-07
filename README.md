# Low-Latency ASR with DTW-Guided Stitching

Short overlapping chunks can make speech recognition feel fast, but they often damage transcript quality by duplicating or fragmenting words at chunk boundaries.

This project tests a simple idea: use Dynamic Time Warping (DTW) to align neighboring audio chunks before stitching their transcripts together.

## What This Repo Contains

- a small low-latency ASR evaluation pipeline
- a DTW-guided chunk stitching baseline
- a reproducible experiment on a public speech dataset
- a short report with figures and qualitative examples

## Stitching comparison

The pilot used `openai/whisper-tiny.en` on 20 LibriSpeech dummy utterances from one speaker. All chunked methods shared the same transcriptions of 1,200 ms windows with a 400 ms stride.

| Method | Mean per-utterance word error rate |
| --- | ---: |
| Full-utterance transcription | 6.64% |
| Naive chunk concatenation | 699.92% |
| Text-only stitching | 537.10% |
| Original DTW-guided stitching | 213.78% |
| Fixed 800 ms overlap + text matching | 201.41% |

Fixed-overlap stitching had lower error than DTW on six utterances and tied on the other fourteen. The DTW offset averaged 770 ms on every utterance. Some individual Whisper chunks contained long repetitions before stitching, and insertions can push WER above 100%.

The evaluator reports how long the first chunk takes to decode. Live caption latency would also include the time to collect the audio and process and display the output.

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
  --results results/whisper-tiny-stitching.json
```

The results JSON contains the raw chunk transcripts and per-utterance scores for all methods.

## Original pilot report

- [Short report](report/short_report.md)
- [WER figure](report/figures/wer_boxplot.png)
- [Latency vs WER figure](report/figures/latency_vs_wer.png)
