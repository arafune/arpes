"""Definitions of band shape.  These models are supporsed to be used for after broadcast_model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
import numpy as np
from lmfit.models import update_param_vals

from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arpes.fits import ModelArgs

__all__ = ("ParabolicDispersionPhiModel",)


class ParabolicDispersionPhiModel(XModelMixin):
    """Model for Parabolic Band model for ARPES data (angle-energy relationshipo)."""

    def parabolic_band_dispersion_phi(
        self,
        x: NDArray[np.float64],
        effective_mass: float,
        phi_offset: float,
        energy_offset: float,
    ) -> NDArray[np.float64]:
        """Return the energy at the emission angle under the free electron band model."""
        return energy_offset * effective_mass / (effective_mass - np.sin((x - phi_offset) ** 2))

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "omit")
        super().__init__(self.parabolic_band_dispersion_phi, **kwargs)

        self.set_param_hint("effective_mass", min=0.1)

    def guess(self, **kwargs: float) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = "Model for parabolic band." + lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC