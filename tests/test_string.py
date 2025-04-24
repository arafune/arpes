"""Unit test for string.py."""

from arpes.utilities.string import safe_decode


def test_safe_decode():
    assert safe_decode(b"hello") == "hello"  # utf-8 by default
    assert safe_decode("café".encode()) == "café"  # utf-8 decoding
    assert safe_decode("café".encode("latin-1"), prefer="latin-1") == "café"  # latin-1 decoding

    assert safe_decode(b"hello", prefer="ascii") == "hello"  # ASCII-compatible bytes
    assert safe_decode(b"\x82\xa0\x82\xa2")
    # Test that it raises TypeError when decoding is not possible
    #    with pytest.raises(TypeError):
    #    safe_decode(b"\xff\xfe\xfd")  # Invalid in utf-8, ascii, and latin-1
