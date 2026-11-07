from __future__ import annotations

import numpy as np
from rapidfuzz.distance import Levenshtein

from mini_asr_lab.text import tokenize_text


def overlap_features(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> np.ndarray:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    if len(samples) < frame_len:
        samples = np.pad(samples, (0, frame_len - len(samples)))

    frames = []
    for start in range(0, len(samples) - frame_len + 1, hop_len):
        frame = samples[start : start + frame_len]
        energy = np.log1p(np.mean(np.square(frame)))
        zcr = np.mean(np.abs(np.diff(np.signbit(frame))).astype(np.float32))
        frames.append([energy, zcr])
    return np.asarray(frames, dtype=np.float32)


def estimate_overlap_offset(
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
    sample_rate: int,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> dict[str, float]:
    from dtw import dtw

    left_features = overlap_features(left_overlap, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    right_features = overlap_features(right_overlap, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)

    alignment = dtw(left_features, right_features, keep_internals=False)
    left_end = int(alignment.index1[-1]) if len(alignment.index1) else 0
    right_start = int(alignment.index2[0]) if len(alignment.index2) else 0
    offset_frames = max(0, left_end - right_start)

    return {
        "distance": float(alignment.distance),
        "offset_frames": float(offset_frames),
        "offset_ms": float(offset_frames * hop_ms),
    }


def stitch_chunk_texts(
    chunk_texts: list[str],
    chunk_offsets: list[dict[str, float]] | None,
    chunk_ms: int,
    overlap_ms: int,
    search_radius: int = 4,
) -> str:
    if not chunk_texts:
        return ""

    merged_words = tokenize_text(chunk_texts[0])
    for index, next_text in enumerate(chunk_texts[1:]):
        next_words = tokenize_text(next_text)
        if not next_words:
            continue

        estimated_drop = 0
        if chunk_offsets is None:
            candidate_range = range(min(12, len(merged_words), len(next_words)) + 1)
        else:
            overlap_hint_ms = (
                chunk_offsets[index]["offset_ms"] if index < len(chunk_offsets) else overlap_ms
            )
            estimated_drop = round(len(next_words) * min(overlap_hint_ms, overlap_ms) / max(chunk_ms, 1))
            candidate_range = range(
                max(0, estimated_drop - search_radius),
                min(len(next_words), estimated_drop + search_radius) + 1,
            )

        best_drop = 0
        best_score = -1.0
        for drop_words in candidate_range:
            overlap_length = min(drop_words, len(merged_words), 12)
            if overlap_length == 0:
                score = 0.0
            else:
                left_tail = " ".join(merged_words[-overlap_length:])
                right_head = " ".join(next_words[max(0, drop_words - overlap_length) : drop_words])
                score = Levenshtein.normalized_similarity(left_tail, right_head)
                if chunk_offsets is not None:
                    score -= 0.03 * abs(drop_words - estimated_drop)
            if score > best_score:
                best_score = score
                best_drop = drop_words

        if best_score < 0.35:
            best_drop = min(estimated_drop, len(next_words))

        merged_words.extend(next_words[best_drop:])

    return " ".join(merged_words)


# Preserve the original entry point for existing callers.
stitch_chunk_texts_with_dtw = stitch_chunk_texts
