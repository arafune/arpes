"""Provides array decomposition approaches like principal component analysis for xarray types."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from arpes.constants import TWO_DIMENSION
from arpes.provenance import PROVENANCE, provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    import sklearn
    import xarray as xr
    from _typeshed import Incomplete

    from arpes._typing import DataType
__all__ = (
    "nmf_along",
    "pca_along",
    "ica_along",
    "factor_analysis_along",
)


def decomposition_along(
    data: DataType,
    axes: list[str],
    decomposition_cls: type[sklearn.decomposition],
    *,
    correlation: bool = False,
    **kwargs: Incomplete,
) -> tuple[xr.DataArray, sklearn.base.BaseEstimator]:
    """Change the basis of multidimensional data according to `sklearn` decomposition classes.

    This allows for robust and simple PCA, ICA, factor analysis, and other decompositions of your
    data even when it is very high dimensional.

    Generally speaking, PCA and similar techniques work when data is 2D, i.e. a sequence of 1D
    observations. We can make the same techniques work by unravelling a ND dataset into 1D
    (i.e. np.ndarray.ravel()) and unravelling a KD set of observations into a 1D set of
    observations. This is basically grouping axes. As an example, if you had a 4D dataset which
    consisted of 2D-scanning valence band ARPES, then the dimensions on our dataset would be
    "[x,y,eV,phi]". We can group these into [spatial=(x, y), spectral=(eV, phi)] and perform PCA or
    another analysis of the spectral features over different spatial observations.

    If our data was called `f`, this can be accomplished with:

    ```
    transformed, decomp = decomposition_analysis(f.stack(spectral=['eV', 'phi']), ['x', 'y'], PCA)
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

    if len(axes) > 1:
        flattened_data: xr.DataArray = normalize_to_spectrum(data).stack(fit_axis=axes)
        stacked = True
    else:
        flattened_data = normalize_to_spectrum(data).S.transpose_to_back(axes[0])
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
    into = into.isel(**dict([[into_first, slice(0, transform.shape[1])]]))
    into = into.rename(dict([[into_first, "components"]]))

    into.values = transform.T

    if stacked:
        into = into.unstack("fit_axis")

    provenance_context: PROVENANCE = {
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
    *args: Incomplete,
    **kwargs: Incomplete,
) -> tuple[xr.DataArray, sklearn.decomposition.PCA]:
    """Specializes `decomposition_along` with `sklearn.decomposition.PCA`."""
    from sklearn.decomposition import PCA

    return decomposition_along(*args, **kwargs, decomposition_cls=PCA)


@wraps(decomposition_along)
def factor_analysis_along(
    *args: Incomplete,
    **kwargs: Incomplete,
) -> tuple[xr.DataArray, sklearn.decomposition.FactorAnalysis]:
    """Specializes `decomposition_along` with `sklearn.decomposition.FactorAnalysis`."""
    from sklearn.decomposition import FactorAnalysis

    return decomposition_along(*args, **kwargs, decomposition_cls=FactorAnalysis)


@wraps(decomposition_along)
def ica_along(
    *args: Incomplete,
    **kwargs: Incomplete,
) -> tuple[xr.DataArray, sklearn.decomposition.FastICA]:
    """Specializes `decomposition_along` with `sklearn.decomposition.FastICA`."""
    from sklearn.decomposition import FastICA

    return decomposition_along(*args, **kwargs, decomposition_cls=FastICA)


@wraps(decomposition_along)
def nmf_along(
    *args: Incomplete,
    **kwargs: Incomplete,
) -> tuple[xr.DataArray, sklearn.decomposition.NMF]:
    """Specializes `decomposition_along` with `sklearn.decomposition.NMF`."""
    from sklearn.decomposition import NMF

    return decomposition_along(*args, **kwargs, decomposition_cls=NMF)
