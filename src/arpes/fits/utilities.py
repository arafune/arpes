"""Provides broadcasted and process parallel curve fitting for PyARPES.

1. Passing xr.DataArray values to parameter guesses and bounds, which can be interpolated/selected
   to allow changing conditions throughout the curve fitting session.
2. A strategy allowing retries with initial guess taken from the previous fit. This is similar
   to some adaptive curve fitting routines that have been proposed in the literature.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Literal, TypeVar

from arpes.debug import setup_logger
from arpes.utilities.jupyter import get_tqdm

if TYPE_CHECKING:
    import lmfit

__all__ = ("result_to_hints",)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

T = TypeVar("T")

tqdm = get_tqdm()


def result_to_hints(
    model_result: lmfit.model.ModelResult | None,
    defaults: dict[str, dict[Literal["value"], float]] | None = None,
) -> dict[str, dict[Literal["value"], float]] | None:
    """Turns an `lmfit.model.ModelResult` into a dictionary with initial guesses.

    Args:
        model_result: The model result to extract parameters from
        defaults: Returned if `model_result` is None, useful for cell re-evaluation in Jupyter

    Returns:
        A dict containing parameter specifications in key-value rather than `lmfit.Parameter`
        format, as you might pass as `params=` to PyARPES fitting code.
    """
    if model_result is None:
        return defaults
    return {k: {"value": model_result.params[k].value} for k in model_result.params}
