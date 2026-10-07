"""One text normalization function shared by training and prediction."""

import html
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize casing and whitespace while retaining Vietnamese accents and negation."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

