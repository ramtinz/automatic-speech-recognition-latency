# Spanish–English streaming ASR benchmark

How well do transcripts preserve both languages when speakers switch mid-sentence? This extension uses multilingual Whisper in transcription mode with automatic language detection. Tiny and Small should be compared on the same recordings, windows, and stitching rules.

## Prepare clips

Obtain the audio and CHAT transcripts from the [Bangor Miami Corpus](https://talkbank.org/biling/access/Bangor/Miami.html) through TalkBank. Keep audio and references local. Select development and test conversations before tuning, and keep speakers separate across the two sets. Include English-only and Spanish-only controls from the same conversations where available.

Use the transcript time links to export short continuous WAV excerpts. Remove CHAT markup from the reference while keeping the spoken words in their original languages. Exclude regions with untranscribed speech, including lines marked `www`, from scored excerpts. Keep the clip boundaries and inclusion criteria fixed for both models.

Create a JSON manifest alongside the WAV files. These example rows illustrate the format.

```json
[
  {
    "id": "example-01",
    "audio": "clips/example-01.wav",
    "text": "Let's meet mañana por la tarde.",
    "cohort": "mixed"
  },
  {
    "id": "example-02",
    "audio": "clips/example-02.wav",
    "text": "La reunión empieza a las nueve.",
    "cohort": "spanish"
  }
]
```

Paths are relative to the manifest. Use `english`, `spanish`, and `mixed` as cohort labels for the three conditions. Set the limit to the number of clips in the evaluation manifest.

```bash
PYTHONPATH=src python scripts/evaluate_dummy_dataset.py \
  --manifest path/to/test.json --model openai/whisper-tiny \
  --limit 20 --results results/bilingual-tiny.json

PYTHONPATH=src python scripts/evaluate_dummy_dataset.py \
  --manifest path/to/test.json --model openai/whisper-small \
  --limit 20 --results results/bilingual-small.json
```

The evaluator compares batch transcription, naive concatenation, text-only, fixed overlap, and DTW-guided stitching. It reports corpus WER and edit counts for each cohort. The same Unicode-aware tokenization is used for stitching and scoring, so accents and ñ stay intact.

Cohort WER measures whole clips. To study switches directly, annotate switch times and compare errors in windows around those times with matched regions away from switches. Review unintended translation and dropped minority-language words separately. Measuring how long captions take to stabilize after a switch requires real-time replay and a trace of partial and committed captions.

**Dataset citation.** Deuchar, M., & Parafita Couto, M. (2011). *The Bangor Miami Corpus*. TalkBank. [DOI](https://doi.org/10.21415/T5J01D)
