"""Some very simple string manipulation utilities."""

from __future__ import annotations

__all__ = [
    "safe_decode",
]


def safe_decode(input_bytes: bytes, prefer: str = "") -> str:
    """Tries different byte interpretations for decoding... very lazy."""
    codecs = ["utf-8", "latin-1", "ascii"]

    if prefer:
        codecs = [prefer, *codecs]
    try:
        for codec in codecs:
            return input_bytes.decode(codec)
    except UnicodeDecodeError:
        pass
    raise TypeError
