"""Utilities related to curve-fitting of ARPES data and xarray format data."""
# pyright: reportUnusedImport=false

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Required, TypedDict

from .fit_models.bands import ParabolicDispersionPhiModel
from .fit_models.decay import ExponentialDecayCModel, TwoExponentialDecayCModel
from .fit_models.dirac import DiracDispersionModel
from .fit_models.fermi_edge import (
    AffineBroadenedFD,
    BandEdgeBGModel,
    BandEdgeBModel,
    FermiDiracModel,
    FermiLorentzianModel,
    GStepBModel,
    GStepBStandardModel,
)
from .fit_models.misc import (
    FermiVelocityRenormalizationModel,
    LogRenormalizationModel,
)
from .fit_models.two_dimensional import EffectiveMassModel, Gaussian2DModel
from .utilities import result_to_hints

if TYPE_CHECKING:
    import lmfit

NAN_POLICY = Literal["raise", "propagate", "omit"]


class ModelArgs(TypedDict, total=False):
    """KWargs for lf.Model."""

    independent_vars: list[str]
    param_names: list[str]
    nan_policy: NAN_POLICY
    prefix: str
    name: str
    form: str


class ParametersArgs(TypedDict, total=False):
    """Class for arguments for Parameters."""

    name: str | lmfit.Parameter
    value: float  # initial value
    vary: bool  # Whether the parameter is varied during the fit
    min: float  # Lower bound for value (default, -np.inf)
    max: float  # Upper bound for value (default np.inf)
    expr: str  # Mathematical expression to constrain the value.
    brute_step: float  # step size for grid points in the brute method.
