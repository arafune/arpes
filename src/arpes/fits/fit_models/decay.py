"""Defines models useful for studying excited carriers in Tr-ARPES."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import Model, update_param_vals

from arpes._typing.base import XrTypes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arpes._typing.fits import ModelArgs

__all__ = ["ExponentialDecayCModel"]


class ExponentialDecayCModel(Model):
    """A model for fitting an exponential decay with a constant background."""

    @staticmethod
    def exponential_decay_c(
        x: NDArray[np.floating],
        amp: float,
        tau: float,
        t0: float,
        const_bkg: float,
    ) -> NDArray[np.floating]:
        """Represents an exponential decay after a point (delta) impulse.

        This coarsely models the dynamics after excitation in a
        pump-probe experiment.

        Args:
            x: x-value as independent variable
            amp: amplitude
            tau: time consatnt
            t0: t = 0 point
            const_bkg: constant background

        Returns:
            The decay profile.
        """
        dx = x - t0
        mask = (dx >= 0) * 1
        return const_bkg + amp * mask * np.exp(-(x - t0) / tau)

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.exponential_decay_c, **kwargs)

        # amp is also a parameter, but we have no hint for it
        self.set_param_hint("tau", min=0.0)
        # t0 is also a parameter, but we have no hint for it
        self.set_param_hint("const_bkg")

    def guess(
        self,
        data: NDArray[np.floating] | XrTypes,
        x: NDArray[np.floating] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        if isinstance(data, XrTypes):
            data = np.asarray(data.values)
        if isinstance(x, xr.DataArray):
            x = x.values
        pars = self.make_params()
        pars[f"{self.prefix}tau"].set(value=0.2)  # 200fs
        pars[f"{self.prefix}t0"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.mean())
        pars[f"{self.prefix}amp"].set(value=data.max() - data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        "A model for fitting an exponential decay with a constant background."
        + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
