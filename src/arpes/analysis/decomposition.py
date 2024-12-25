"""Provides array decomposition approaches like principal component analysis for xarray types."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import xarray as xr
from sklearn.decomposition import FactorAnalysis, FastICA

from arpes.constants import TWO_DIMENSION
from arpes.provenance import Provenance, provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    import numpy as np
    import sklearn
    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator

__all__ = (
    "factor_analysis_along",
    "ica_along",
    "nmf_along",
    "pca_along",
)


class PCAParam(TypedDict, total=False):
    n_composition: float | Literal["mle", "auto"] | None
    copy: bool
    whiten: str | bool
    svd_solver: Literal["auto", "full", "arpack", "randomiozed"]
    tol: float
    iterated_power: int | Literal["auto"]
    n_oversamples: int
    power_interation_normalizer: Literal["auto", "QR", "LU", "none"]
    random_state: int | None


class FastICAParam(TypedDict, total=False):
    n_composition: float | None
    algorithm: Literal["Parallel", "deflation"]
    whiten: bool | Literal["unit-variance", "arbitrary-variance"]
    fun: Literal["logosh", "exp", "cube"]
    fun_args: dict[str, float] | None
    max_iter: int
    tol: float
    w_int: NDArray[np.float64]
    whiten_solver: Literal["eigh", "svd"]
    random_state: int | None


class NMFParam(TypedDict, total=False):
    n_composition: int | Literal["auto"] | None
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"] | None
    solver: Literal["cd", "mu"]
    beta_loss: float | Literal["frobenius", "kullback-leibler", "itakura-saito"]
    tol: float
    max_iter: int
    random_state: int | None
    alpha_W: float
    alpha_H: float
    l1_ratio: float
    verbose: int
    shuffle: bool


class FactorAnalysisParam(TypedDict, total=False):
    n_composition: int | None
    tol: float
    copy: bool
    max_iter: int
    noise_variance_init: NDArray[np.float64] | None
    svd_method: Literal["lapack", "randomized"]
    iterated_power: int
    rotation: Literal["varimax", "quartimax"] | None
    random_state: int | None


class DecompositionParam(PCAParam, FastICAParam, NMFParam, FactorAnalysisParam):  # type: ignore[misc]
    pass


class DecompositionParamBase(TypedDict, total=False):
    n_composition: int | None


def decomposition_along(
    data: xr.DataArray,
    axes: list[str],
    *,
    decomposition_cls: type[BaseEstimator],
    correlation: bool = False,
    **kwargs: Unpack[DecompositionParam],
) -> tuple[xr.DataArray, sklearn.base.BaseEstimator]:
    """Change the basis of multidimensional data according to `sklearn` decomposition classes.

    This allows for robust and simple PCA, ICA, factor analysis, and other decompositions of your
    data even when it is very high dimensional.

    Generally speaking, PCA and similar techniques work when data is 2D, i.e. a sequence of 1D
    observations. We can make the same techniques work by unravelling a ND dataset into 1D
    (i.e. np.ndarray.ravel()) and unravelling a KD set of observations into a 1D set of
    observations. This is basically grouping axes. As an example, if you had a 4D dataset which
    consisted of 2D-scanning valence band ARPES, then the dimensions on our dataset would be
    "[x, y, eV, phi]". We can group these into [spatial=(x, y), spectral=(eV, phi)] and perform PCA
    or another analysis of the spectral features over different spatial observations.

    If our data was called `f`, this can be accomplished with:

    ```
    transformed, decomp = decomposition_along(f.stack(spectral=['eV', 'phi']), ['x', 'y'], PCA)
    transformed.dims # -> [X, Y, components]
    ```

    The results of `decomposition_along` can be explored with `arpes.widgets.pca_explorer`,
    regardless of the decomposition class.

    Args:
        data: Input data, can be N-dimensional but should only include one "spectral" axis.
        axes: Several axes to be treated as a single axis labeling the list of observations.
        decomposition_cls: A sklearn.decomposition class (such as PCA or ICA) to be used
          to perform the decomposition.
        correlation: Controls whether StandardScaler() is used as the first stage of the data
          ingestion pipeline for sklearn.
        kwargs: forward to ``decomposition_cls``

    Returns:
        A tuple containing the projected data and the decomposition fit instance.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    if len(axes) > 1:
        flattened_data: xr.DataArray = data.stack(fit_axis=axes)
        stacked = True
    else:
        flattened_data = data.transpose(..., axes[0])
        stacked = False

    if len(flattened_data.dims) != TWO_DIMENSION:
        msg = f"Inappropriate number of dimensions after flattening: [{flattened_data.dims}]"
        raise ValueError(
            msg,
        )

    if correlation:
        pipeline: sklearn.Pipeline = make_pipeline(StandardScaler(), decomposition_cls(**kwargs))
    else:
        pipeline = make_pipeline(decomposition_cls(**kwargs))

    pipeline.fit(flattened_data.values.T)

    decomp = pipeline.steps[-1][1]

    transform = decomp.transform(flattened_data.values.T)

    into = flattened_data.copy(deep=True)
    into_first = into.dims[0]
    into = into.isel({into_first: slice(0, transform.shape[1])})
    into = into.rename({into_first: "components"})

    into.values = transform.T

    if stacked:
        into = into.unstack("fit_axis")

    provenance_context: Provenance = {
        "what": "sklearn decomposition",
        "by": "decomposition_along",
        "axes": axes,
        "correlation": False,
        "decomposition_cls": decomposition_cls.__name__,
    }

    provenance(into, data, provenance_context)

    return into, decomp


@wraps(decomposition_along)
def pca_along(
    *args: *tuple[xr.DataArray, list[str]],
    **kwargs: Unpack[PCAParam],
) -> tuple[xr.DataArray, sklearn.decomposition.PCA]:
    """Specializes `decomposition_along` with `sklearn.decomposition.PCA`."""
    from sklearn.decomposition import PCA

    return decomposition_along(*args, **kwargs, decomposition_cls=PCA)


@wraps(decomposition_along)
def factor_analysis_along(
    *args: *tuple[xr.DataArray, list[str]],
    **kwargs: Unpack[FactorAnalysisParam],
) -> tuple[xr.DataArray, sklearn.decomposition.FactorAnalysis]:
    """Specializes `decomposition_along` with `sklearn.decomposition.FactorAnalysis`."""
    return decomposition_along(*args, **kwargs, decomposition_cls=FactorAnalysis)


@wraps(decomposition_along)
def ica_along(
    *args: *tuple[xr.DataArray, list[str]],
    **kwargs: Unpack[FastICAParam],
) -> tuple[xr.DataArray, sklearn.decomposition.FastICA]:
    """Specializes `decomposition_along` with `sklearn.decomposition.FastICA`."""
    return decomposition_along(*args, **kwargs, decomposition_cls=FastICA)


@wraps(decomposition_along)
def nmf_along(
    *args: *tuple[xr.DataArray, list[str]],
    **kwargs: Unpack[NMFParam],
) -> tuple[xr.DataArray, sklearn.decomposition.NMF]:
    """Specializes `decomposition_along` with `sklearn.decomposition.NMF`."""
    from sklearn.decomposition import NMF

    return decomposition_along(*args, **kwargs, decomposition_cls=NMF)
