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
