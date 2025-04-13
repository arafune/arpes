"""Unit test for test_band_analyhsis."""

from unittest.mock import MagicMock

import lmfit as lf
import pytest

from arpes.analysis.band_analysis_utils import param_getter, param_stderr_getter


def test_param_getter():
    param_name = "test_param"
    model_result = MagicMock(spec=lf.model.ModelResult)
    model_result.params = {param_name: lf.Parameter(name="a", value=1.23)}

    getter = param_getter(param_name)
    assert getter(model_result) == pytest.approx(1.23)


def test_param_getter_missing():
    param_name = "missing_param"
    model_result = MagicMock(spec=lf.model.ModelResult)
    model_result.params = {}
    getter = param_getter(param_name)
    with pytest.raises(KeyError):
        assert getter(model_result)


def test_param_stderr_getter():
    param_name = "test_param"
    model_result = MagicMock(spec=lf.model.ModelResult)

    model_result.params = {param_name: lf.Parameter(name="a", value=1.23)}
    model_result.params[param_name].stderr = 0.05
    getter = param_stderr_getter(param_name)
    assert getter(model_result) == pytest.approx(0.05)


def test_param_stderr_getter_missing():
    param_name = "missing_param"
    model_result = MagicMock(spec=lf.model.ModelResult)
    model_result.params = {}
    getter = param_stderr_getter(param_name)
    with pytest.raises(KeyError):
        assert getter(model_result)
