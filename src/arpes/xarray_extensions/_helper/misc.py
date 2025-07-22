"""Helper functions in xarray_extensions."""

import numpy as np
from lmfit.model import ModelResult


def safe_error(model_result_instance: ModelResult | None) -> float:
    r"""Calculates the mean squared error (MSE) from the residuals of a model fit.

    This function safely computes the MSE, returning `np.nan` if the
    `model_result_instance` is `None`, which is useful for handling cases
    where a fit might not have converged or was not performed.

    The mean squared error is calculated as the mean of the squared residuals:
    $MSE = \\frac{1}{N} \\sum_{i=1}^{N} (r_i)^2$, where $r_i$ are the residuals.

    Parameters:
        model_result_instance (ModelResult | None): An instance of a model result
            object (e.g., `lmfit.model.ModelResult`) which is expected to have
            a `residual` attribute that is a NumPy array. If `None`, the function
            returns `np.nan`.

    Returns:
        float: The mean squared error of the residuals. Returns `np.nan` if
            `model_result_instance` is `None`.

    Raises:
        AssertionError: If `model_result_instance` is not `None` but its
            `residual` attribute is not a NumPy array. This ensures that the
            `residual` can be properly processed.

    Examples:
        >>> from lmfit.model import ModelResult # (If lmfit is used)
        >>> # Assuming a ModelResult-like object with residuals
        >>> class MockModelResult:
        ...     def __init__(self, residuals):
        ...         self.residual = np.array(residuals)
        >>>
        >>> result1 = MockModelResult(residuals=[1, -2, 3])
        >>> safe_error(result1)
        4.666666666666667

        >>> result2 = MockModelResult(residuals=[0.5, 0.5])
        >>> safe_error(result2)
        0.25

        >>> safe_error(None)
        nan
    """
    if model_result_instance is None:
        return np.nan
    assert isinstance(model_result_instance.residual, np.ndarray)
    return (model_result_instance.residual**2).mean()
