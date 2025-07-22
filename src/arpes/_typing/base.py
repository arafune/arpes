"""Specialized type annotations for use in PyARPES.

In particular, `DataType` refers to either an xarray.DataArray or xarray.Dataset

`NormalizableDataType` referes to anything that can be tuned into datase,
such as by loading from the cache using an ID.
"""

from __future__ import annotations

import uuid
from typing import (
    Literal,
    TypeAlias,
    TypeVar,
)

import numpy as np
import xarray as xr
from numpy.typing import NDArray

DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)
NormalizableDataType: TypeAlias = DataType | str | uuid.UUID

XrTypes: TypeAlias = xr.DataArray | xr.Dataset

ReduceMethod = Literal["sum", "mean"]

MOMENTUM = Literal["kp", "kx", "ky", "kz"]
EMISSION_ANGLE = Literal["phi", "psi"]
ANGLE = Literal["alpha", "beta", "chi", "theta"] | EMISSION_ANGLE
Orientation = Literal["horizontal", "vertical"]

AxisType = Literal["angle", "k"]

HIGH_SYMMETRY_POINTS = Literal["G", "X", "Y", "M", "K", "S", "A1", "H", "C", "H1"]

SpectrumType = Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]

Plot2DStyle = Literal["line", "scatter"]

AnalysisRegion = Literal["copper_prior", "wide_angular", "narrow_angular"]

SelType = float | str | slice | list[float | str] | NDArray[np.float64]
