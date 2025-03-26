"""Provides utilities used internally by `arpes.analysis.band_analysis`."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import lmfit as lf


class ParamType(NamedTuple):
    """Parameter type."""

    value: float
    stderr: float


def param_getter(param_name: str) -> Callable[..., float]:
    """Constructs a function to extract a parameter value by name.

    Useful to extract data from inside an array of `lmfit.ModelResult` instances.

    Args:
        param_name: Parameter name to retrieve. If you performed a
          composite model fit, make sure to include the prefix.
        safe: Guards against NaN values. This is typically desirable but
          sometimes it is advantageous make sure to include the prefix.
          to have NaNs fail an analysis quickly.

    Returns:
        A function which fetches the fitted value for this named parameter.
    """
    safe_param = ParamType(value=np.nan, stderr=np.nan)

    def getter(x: lf.model.ModelResult) -> float:
        return x.params.get(param_name, safe_param).value

    return getter


def param_stderr_getter(param_name: str) -> Callable[..., float]:
    """Constructs a function to extract a parameter value by name.

    Useful to extract data from inside an array of `lmfit.ModelResult` instances.

    Args:
        param_name: Parameter name to retrieve. If you performed a
          composite model fit, make sure to include the prefix.
        safe: Guards against NaN values. This is typically desirable but
          sometimes it is advantageous make sure to include the prefix.
          to have NaNs fail an analysis quickly.

    Returns:
        A function which fetches the standard error for this named parameter.

    """
    safe_param = ParamType(value=np.nan, stderr=np.nan)

    def getter(x: lf.model.ModelResult) -> float:
        return x.params.get(param_name, safe_param).stderr

    return getter
