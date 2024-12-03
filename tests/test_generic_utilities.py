"""Test for generic utility."""

import pytest

from arpes.utilities import clean_keys, deep_update


def test_cldean_keys() -> None:
    """Test for clean_keys."""
    test_dict = {"Excitation Energy": 4.03, "Count/Cycle": 100}
    cleaned_dict = clean_keys(test_dict)
    assert cleaned_dict == {"excitation_energy": 4.03, "count_cycle": 100}


@pytest.mark.parametrize(
    ("destination", "source", "expected"),
    [
        ({}, {}, {}),
        ({"a": []}, {"a": []}, {"a": []}),
        ({"a": [1.1]}, {"b": [1.2]}, {"a": [1.1], "b": [1.2]}),
        ({"a": {"b": 5}}, {"a": {"c": 7}}, {"a": {"b": 5, "c": 7}}),
    ],
)
def test_deep_update(destination: dict, source: dict, expected: dict) -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        destination: [TODO:description]
        source: [TODO:description]
        expected: [TODO:description]

    Returns:
        [TODO:description]
    """
    assert deep_update(destination, source) == expected
