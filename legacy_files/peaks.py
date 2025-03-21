"""Includes multi-peak model definitions.

arpes/fits/fit_models/peaks.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
from lmfit.models import update_param_vals

from .functional_forms import affine_bkg, gaussian, twolorentzian
from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from numpy.typing import NDArray

    from arpes.fits import ModelArgs

__all__ = ("TwoGaussianModel", "TwoLorModel")


class TwoGaussianModel(XModelMixin):
    """A model for two gaussian functions with a linear background.

    **This is typically not necessary, as you can use the + operator on the Model instances.**
    """

    @staticmethod
    def twogaussian(  # noqa: PLR0913
        x: NDArray[np.float64],
        center: float = 0,
        t_center: float = 0,
        width: float = 1,
        t_width: float = 1,
        amp: float = 1,
        t_amp: float = 1,
        lin_slope: float = 0,
        const_bkg: float = 0,
    ) -> NDArray[np.float64]:
        """Two gaussians and an affine background."""
        return (
            gaussian(x, center, width, amp)
            + gaussian(x, t_center, t_width, t_amp)
            + affine_bkg(x, lin_slope, const_bkg)
        )

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Sets physical constraints for peak width and other parameters."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.twogaussian, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_width", min=0)
        self.set_param_hint("lin_slope", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: xr.DataArray | NDArray[np.float64],
        **kwargs: float,
    ) -> lf.Parameters:
        """Very simple heuristics for peak location."""
        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}t_center"].set(value=0)
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.min())
        pars[f"{self.prefix}width"].set(0.02)  # TODO: we can do better than this
        pars[f"{self.prefix}t_width"].set(0.02)
        pars[f"{self.prefix}amp"].set(value=data.mean() - data.min())
        pars[f"{self.prefix}t_amp"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = "Two gaussian model" + lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorModel(XModelMixin):
    """A model for two gaussian functions with a linear background.

    **This is typically not necessary, as you can use the + operator on the Model instances.**
    """

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Sets physical constraints for peak width and other parameters."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(twolorentzian, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("gamma", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_gamma", min=0)
        self.set_param_hint("lin_slope", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: xr.DataArray | NDArray[np.float64],
        **kwargs: float,
    ) -> lf.Parameters:
        """Very simple heuristics for peak location."""
        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}t_center"].set(value=0)
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.min())
        pars[f"{self.prefix}gamma"].set(0.02)  # TODO: we can do better than this
        pars[f"{self.prefix}t_gamma"].set(0.02)
        pars[f"{self.prefix}amp"].set(value=data.mean() - data.min())
        pars[f"{self.prefix}t_amp"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = "Two lorenzian model" + lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
