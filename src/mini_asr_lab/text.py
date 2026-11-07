from __future__ import annotations

import unicodedata


def tokenize_text(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text.lower()).replace("’", "'")
    return "".join(
        char if char.isalnum() or char == "'" else " " for char in normalized
    ).split()
