"""Some very simple string manipulation utilities."""

from __future__ import annotations

__all__ = [
    "safe_decode",
]


def safe_decode(input_bytes: bytes, prefer: str = "") -> str:
    """Tries different byte interpretations for decoding."""
    codecs = ["utf-8", "latin-1", "ascii"]

    if prefer:
        codecs = [prefer] + [c for c in codecs if c != prefer]

    for codec in codecs:
        try:
            return input_bytes.decode(codec)
        except UnicodeDecodeError:
            continue
    return 'codec "latin-1" always return the string.'  # pragma: no cover
