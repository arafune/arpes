"""Defines models useful for studying excited carriers in Tr-ARPES."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
import numpy as np
from lmfit.models import update_param_vals

from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from arpes.fits import ModelARGS

__all__ = ["ExponentialDecayCModel", "TwoExponentialDecayCModel"]


class ExponentialDecayCModel(XModelMixin):
    """A model for fitting an exponential decay with a constant background."""

    @staticmethod
    def exponential_decay_c(
        x: NDArray[np.float_],
        amp: float,
        tau: float,
        t0: float,
        const_bkg: float,
    ) -> NDArray[np.float_]:
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

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
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
        data: xr.Dataset | NDArray[np.float_],
        **kwargs: float,
    ) -> lf.Parameters:
        """Make heuristic estimates of parameters.

        200fs is a reasonable value for the time constant, in fact its probably a bit large.
        We assume data is probably calibrated so that t0 is at 0 delay.
        """
        pars = self.make_params()
        pars["%stau" % self.prefix].set(value=0.2)  # 200fs
        pars["%st0" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.mean())
        pars["%samp" % self.prefix].set(value=data.max() - data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoExponentialDecayCModel(XModelMixin):
    """A model for fitting an exponential decay with a constant background."""

    @staticmethod
    def twoexponential_decay_c(  # noqa: PLR0913
        x: NDArray[np.float_],
        amp: float,
        t0: float,
        tau1: float,
        tau2: float,
        const_bkg: float,
    ) -> NDArray[np.float_]:
        """Like `exponential_decay_c`, except with two timescales.

        This is meant to model if two different quasiparticle decay channels are allowed,
        represented by `tau1` and `tau2`.
        """
        dx = x - t0
        y = const_bkg + amp * (1 - np.exp(-dx / tau1)) * np.exp(-dx / tau2)
        f = y.copy()
        f[dx < 0] = const_bkg
        f[dx >= 0] = y[dx >= 0]
        return f

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.twoexponential_decay_c, **kwargs)

        # amp is also a parameter, but we have no hint for it
        self.set_param_hint("tau1", min=0.0)
        self.set_param_hint("tau2", min=0.0)
        # t0 is also a parameter, but we have no hint for it
        self.set_param_hint("const_bkg")

    def guess(
        self,
        data: NDArray[np.float_] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here."""
        pars: lf.Parameters = self.make_params()

        pars["%stau1" % self.prefix].set(value=0.2)  # 200fs
        pars["%stau2" % self.prefix].set(value=1)  # 1ps
        pars["%st0" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.mean())
        pars["%samp" % self.prefix].set(value=data.max() - data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
