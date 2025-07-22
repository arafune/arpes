"""Unit test for arpes._typing."""

import numpy as np
import pytest

from arpes._typing.utils import is_dict_kspacecoords


@pytest.fixture
def valid_dict():
    return {
        "eV": np.array([1.0, 2.0, 3.0]),
        "kp": np.array([0.1, 0.2, 0.3]),
        "kx": np.array([0.4, 0.5, 0.6]),
        "ky": np.array([0.7, 0.8, 0.9]),
        "kz": np.array([1.0, 1.1, 1.2]),
    }


@pytest.fixture
def invalid_dict_missing_keys():
    return {
        "eV": np.array([1.0, 2.0, 3.0]),
        "delay": np.array([0.4, 0.5, 0.6]),
    }


@pytest.fixture
def invalid_dict_wrong_values():
    return {
        "eV": np.array([1.0, 2.0, 3.0]),
        "kp": [0.1, 0.2, 0.3],
        "kx": np.array([0.4, 0.5, 0.6]),
        "ky": np.array([0.7, 0.8, 0.9]),
        "kz": np.array([1.0, 1.1, 1.2]),
    }


@pytest.fixture
def empty_dict():
    return {}


@pytest.fixture
def extra_key_dict(valid_dict: dict):
    extra_dict = valid_dict.copy()
    extra_dict["extra"] = np.array([0.0])
    return extra_dict


def test_valid_dict(valid_dict: dict):
    assert is_dict_kspacecoords(valid_dict) is True


def test_missing_keys(invalid_dict_missing_keys: dict):
    assert is_dict_kspacecoords(invalid_dict_missing_keys) is False


def test_wrong_value_type(invalid_dict_wrong_values: dict):
    assert is_dict_kspacecoords(invalid_dict_wrong_values) is False


def test_empty_dict(empty_dict: dict):
    assert is_dict_kspacecoords(empty_dict) is False


def test_extra_keys_allowed(extra_key_dict: dict):
    assert is_dict_kspacecoords(extra_key_dict) is False
