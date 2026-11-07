# What breaks when Whisper transcribes overlapping audio windows?

Live captioning systems have to decide when to show a partial transcript and when to commit it. Short windows let the recognizer start while the speaker is still talking. Overlapping windows can add context and also repeat words at the join. This pilot asks whether an acoustic alignment step improves the join when the audio window boundaries are already known.

## Setup

We used Whisper Tiny English (`openai/whisper-tiny.en`) on the first 20 utterances of the LibriSpeech dummy validation split. All 20 come from one speaker and contain 448 reference words after normalization. We decoded the full utterance as an offline reference and decoded each overlapping window once. All four chunked methods shared those outputs. Window length was 1,200 ms, stride was 400 ms, and nominal overlap was 800 ms.

| Method | How it joins transcripts |
| --- | --- |
| Naive concatenation | Appends every chunk as a diagnostic for the effect of overlapping audio. |
| Text-only | Matches the previous transcript suffix against the new prefix and keeps the prefix when the match is weak. |
| Fixed overlap | Uses the known 800 ms overlap to guide how many words to drop, then refines the join using text similarity. |
| DTW-guided | Uses an offset from the acoustic alignment as the input to the same overlap-guided text matcher. |

## Results

| Method | Substitutions | Deletions | Insertions | Corpus WER ↓ | Median utterance WER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full utterance | 28 | 5 | 1 | 7.59% | 6.28% |
| Naive concatenation | 43 | 0 | 3,151 | 712.95% | 222.65% |
| Text-only | 117 | 10 | 2,323 | 546.88% | 56.09% |
| Fixed overlap | 135 | 22 | 763 | 205.36% | 46.43% |
| DTW-guided | 136 | 22 | 820 | 218.30% | 46.43% |

Corpus WER divides total word edits by total reference words. It can exceed 100% when there are many insertions. The evaluator ignores case and punctuation. Where multiple edit paths have equal cost, the reported breakdown uses the path selected by RapidFuzz.

Fixed overlap had lower utterance WER on six examples and tied on fourteen. The implemented closed-end DTW calculation returned the same mean offset, 770 ms, for every utterance. With equal-length overlap regions and a fixed-endpoint path, the offset comes from the feature frame count. This pilot tests that specific heuristic on audio with known overlap.

### Failure example

Utterance `1272-128104-0008` has an 11-word reference: “As for etchings they are of two kinds British and foreign.” One 1.2-second window alone generated 359 words, repeatedly emitting “there are two things.” This repetition was already present in the raw model output. The final DTW-stitched transcript still had 138 words. The example shows why a production evaluation should track decoder repetitions separately from repeated words at chunk joins.

### What the timing measures

The original run averaged 206.9 ms to decode the first window. A live system using this schedule would first collect 1,200 ms of audio, then decode, stitch, and deliver the text. The benchmark processes files offline and measures compute time for the decode call. Caption delivery, revision, endpointing, and queueing require separate measurements.

## How to test a live system

| Engineering question | Measurement |
| --- | --- |
| How quickly does the user see useful text? | Replay audio in real time. Measure from speech onset to the first visible nonempty partial. Report median and p95. |
| How often does the caption change? | Record each partial and committed transcript. Count words revised after display and measure the lag from spoken words to stable text on time-aligned examples. |
| How soon can a voice interface act on a turn? | Measure from the end of speech to a final event, including endpointing and any delivery delay. Report premature finalizations separately. |
| Can the service keep up? | Track processing time relative to audio duration, queue growth, and p95 finalization delay under a stated concurrency and hardware budget. |

Tune decoding and window settings on development audio, then freeze them for a held-out, multi-speaker evaluation. Include longer recordings with pauses, background noise, and turn changes for the target use case. Compare fixed overlap with an established commit policy such as [WhisperStreaming’s LocalAgreement](https://aclanthology.org/2023.ijcnlp-demo.3/) under the same replay schedule and latency definitions. A larger Whisper checkpoint can then test whether the result carries over. If overlap estimation is a deployment requirement, test it separately by injecting timing shifts with known ground truth.

The pilot supports fixed overlap as the simpler starting point when the window positions are known. Short-window decoder repetition remains the larger accuracy problem to solve.
