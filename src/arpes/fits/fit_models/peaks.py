"""Includes multi-peak model definitions."""

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

    from arpes.fits import ModelARGS

__all__ = ["TwoGaussianModel", "TwoLorModel"]


class TwoGaussianModel(XModelMixin):
    """A model for two gaussian functions with a linear background.

    **This is typically not necessary, as you can use the + operator on the Model instances.**
    """

    @staticmethod
    def twogaussian(  # noqa: PLR0913
        x: NDArray[np.float_],
        center: float = 0,
        t_center: float = 0,
        width: float = 1,
        t_width: float = 1,
        amp: float = 1,
        t_amp: float = 1,
        lin_bkg: float = 0,
        const_bkg: float = 0,
    ) -> NDArray[np.float_]:
        """Two gaussians and an affine background."""
        return (
            gaussian(x, center, width, amp)
            + gaussian(x, t_center, t_width, t_amp)
            + affine_bkg(x, lin_bkg, const_bkg)
        )

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Sets physical constraints for peak width and other parameters."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.twogaussian, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: xr.DataArray | NDArray[np.float_],
        **kwargs: float,
    ) -> lf.Parameters:
        """Very simple heuristics for peak location."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%st_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%swidth" % self.prefix].set(0.02)  # TODO: we can do better than this
        pars["%st_width" % self.prefix].set(0.02)
        pars["%samp" % self.prefix].set(value=data.mean() - data.min())
        pars["%st_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorModel(XModelMixin):
    """A model for two gaussian functions with a linear background.

    **This is typically not necessary, as you can use the + operator on the Model instances.**
    """

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Sets physical constraints for peak width and other parameters."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(twolorentzian, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("gamma", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_gamma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: xr.DataArray | NDArray[np.float_],
        **kwargs: float,
    ) -> lf.Parameters:
        """Very simple heuristics for peak location."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%st_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%sgamma" % self.prefix].set(0.02)  # TODO: we can do better than this
        pars["%st_gamma" % self.prefix].set(0.02)
        pars["%samp" % self.prefix].set(value=data.mean() - data.min())
        pars["%st_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
