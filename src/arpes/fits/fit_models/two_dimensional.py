"""Curve fitting models with two independent variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Unpack

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import update_param_vals

from arpes.constants import HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ

from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes.fits import ModelArgs

__all__ = ("Gaussian2DModel", "EffectiveMassModel")

any_dim_sentinel = None


class Gaussian2DModel(XModelMixin):
    """2D Gaussian fitting."""

    n_dims = 2
    dimension_order: ClassVar[list[str | None]] = [any_dim_sentinel, any_dim_sentinel]

    @staticmethod
    def gaussian_2d_bkg(  # noqa: PLR0913
        x: NDArray[np.float_] | xr.DataArray,
        y: NDArray[np.float_] | xr.DataArray,
        amplitude: float = 1,
        xc: float = 0,
        yc: float = 0,
        sigma_x: float = 0.1,
        sigma_y: float = 0.1,
        const_bkg: float = 0,
        x_bkg: float = 0,
        y_bkg: float = 0,
    ) -> NDArray[np.float_]:
        """Defines a multidimensional axis aligned normal."""
        bkg = np.outer(x * 0 + 1, y_bkg * y) + np.outer(x * x_bkg, y * 0 + 1) + const_bkg
        # make the 2D Gaussian matrix
        gauss = (
            amplitude
            * np.exp(
                -(
                    (x[:, None] - xc) ** 2 / (2 * sigma_x**2)
                    + (y[None, :] - yc) ** 2 / (2 * sigma_y**2)
                ),
            )
            / (2 * np.pi * sigma_x * sigma_y)
        )

        # flatten the 2D Gaussian down to 1D
        return np.ravel(gauss + bkg)

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Sets reasonable constraints on the width and constraints the amplitude to be positive."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.gaussian_2d_bkg, **kwargs)

        self.set_param_hint("sigma_x", min=0.0)
        self.set_param_hint("sigma_y", min=0.0)
        self.set_param_hint("amplitude", min=0.0)


class EffectiveMassModel(XModelMixin):
    """A two dimensional model for a quadratic distribution of Lorentzians.

    This can be used for "global" fitting of the effective mass of a band, providing higher quality
    results than iterative fitting for the lineshapes and then performing a
    quadratic fit to the centers.

    This model also provides a representative example of how to implement (2+)D "global"
    lmfit.Model classes which are xarray coordinate aware.

        `dimension_order`: allows specifying allowed sets of dimension for the global fit (here eV
        must be one axis, while either kp or phi is allowed for the other.
    """

    n_dims = 2
    dimension_orde: ClassVar[list[str | list[str]]] = ["eV", ["kp", "phi"]]

    @staticmethod
    def effective_mass_bkg(  # noqa: PLR0913
        eV: NDArray[np.float_],  # noqa: N803
        kp: NDArray[np.float_],
        m_star: float = 0,
        k_center: float = 0,
        eV_center: float = 0,  # noqa: N803
        gamma: float = 1,
        amplitude: float = 1,
        amplitude_k: float = 0,
        const_bkg: float = 0,
        k_bkg: float = 0,
        eV_bkg: float = 0,  # noqa: N803
    ) -> NDArray[np.float_]:
        """Model implementation function for simultaneous 2D curve fitting of band effective mass.

        Allows for an affine background in each dimension, together with variance in the band
        intensity along the band, as a very simple model of matrix elements. Together with
        prenormalizing your data this should allow reasonable fits of a lot of typical ARPES data.
        """
        bkg = np.outer(eV * 0 + 1, k_bkg * kp) + np.outer(eV_bkg * eV, kp * 0 + 1) + const_bkg

        # check units
        dk = kp - k_center
        offset = HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ * dk**2 / (2 * m_star + 1e-6)
        eVk = np.outer(eV, kp * 0 + 1)
        coherent = (
            (amplitude + amplitude_k * dk)
            * (1 / (2 * np.pi))
            * gamma
            / ((eVk - eV_center + offset) ** 2 + (0.5 * gamma) ** 2)
        )
        return (coherent + bkg).ravel()

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Mostly just set parameter hints to physically realistic values here."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.effective_mass_bkg, **kwargs)

        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("amplitude", min=0.0)

    def guess(
        self,
        data: xr.DataArray,
        eV: xr.DataArray,  # noqa: N803
        kp: xr.DataArray,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Use heuristics to estimate the model parameters."""
        assert isinstance(eV, xr.DataArray)
        momentum = kp.values
        eV_arr = eV.values
        data_arr = data.values
        pars = self.make_params()

        pars["%sm_star" % self.prefix].set(value=1)

        pars["%sk_center" % self.prefix].set(value=np.mean(momentum))
        pars["%seV_center" % self.prefix].set(value=np.mean(eV_arr))

        pars["%samplitude" % self.prefix].set(value=np.mean(np.mean(data_arr, axis=0)))
        pars["%sgamma" % self.prefix].set(value=0.25)

        pars["%samplitude_k" % self.prefix].set(value=0)  # can definitely improve here

        # Crude estimate of the background
        left, right = np.mean(data_arr[:5, :], axis=0), np.mean(data_arr[-5:, :], axis=0)
        top, bottom = np.mean(data_arr[:, :5], axis=0), np.mean(data_arr[:, -5:], axis=0)
        left, right = np.percentile(left, 10), np.percentile(right, 10)
        top, bottom = np.percentile(top, 10), np.percentile(bottom, 10)

        pars["%sconst_bkg" % self.prefix].set(value=np.min(np.array([left, right, top, bottom])))
        pars["%sk_bkg" % self.prefix].set(value=(bottom - top) / (eV_arr[-1] - eV_arr[0]))
        pars["%seV_bkg" % self.prefix].set(value=(right - left) / (momentum[-1] - momentum[0]))

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
