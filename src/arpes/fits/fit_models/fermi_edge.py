"""Definitions of models involving Fermi edges."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.lineshapes import gaussian, lorentzian
from lmfit.models import Model, update_param_vals
from scipy import stats

from arpes._typing import XrTypes

from .functional_forms import (
    affine_broadened_fd,
    band_edge_bkg,
    fermi_dirac,
    gstep_stdev,
    gstepb,
)

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes.fits import ModelArgs

__all__ = (
    "AffineBroadenedFD",
    "BandEdgeBGModel",
    "BandEdgeBModel",
    "FermiDiracModel",
    "FermiLorentzianModel",
    "GStepBModel",
    "GStepBStandardModel",
    "GStepBStdevModel",
)


class AffineBroadenedFD(Model):
    r"""A model for affine density of states convoluted with gaussian.

    The model has three Parameters: `center`, `width`, `const_bkg`, `lin_slope` and `sigma`.
    constraints to report full width at half maximum and maximum peak
    height, respectively.

    .. math::

        f(x; center, width, b, a) = \frac{b + a * x}{1+\exp \left(\frac{x-center}{width}\right)}

    where the parameter `const_bkg` corresponds to :math:`b`, `lin_slope` to
    :math:`a`.

    then, f convoluted by gaussian with the standard deviation `sigma`

    Note:
        * The constant stride about x ("eV" in most case) is assumed, internally,
        * From version 5. offset parameter is removed.  Use ConstantModel in lmfit.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    def __init__(
        self,
        **kwargs: Unpack[ModelArgs],
    ) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(affine_broadened_fd, **kwargs)

        self.set_param_hint("width", min=0.0)
        self.set_param_hint("sigma", min=0.0)

    def guess(
        self,
        data: XrTypes | NDArray[np.float64],
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        if isinstance(data, XrTypes):
            ymin, ymax = data.min().item(), data.max().item()
        ymin, ymax = min(data), max(data)
        if isinstance(x, xr.DataArray):
            x = x.values
        xmin, xmax = np.min(x), np.max(x)
        sigma = 0.1 * (xmax - xmin)
        width = 0.1 * (xmax - xmin)

        pars = self.make_params(
            const_bkg=(ymax - ymin),
            center=(xmax + xmin) / 2.0,
            sigma=sigma,
            width=width,
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        "Affine density of states broadened by Fermi-Dirac " + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiLorentzianModel(Model):
    r"""A model for Lorentzian multiplied by a gstepb background.

    This model ...

    .. math:

        f(x; center, width, erf\_amp, a, b, \gamma, center_l) = L(x; center_l, \gamma)
        \frac{erf\_amp}{2}*\mathrm{erfc}\left(\frac{\sqrt{4ln(2)} (x-center)}{width}\right)

    where the parameter `gamma` corresponds to :math:`\gamma`, `lorcenter` to :math:`center_l`.

    Todo: Reconsidering the NEED & Lorentzian height.
    """

    @staticmethod
    def gstepb_mult_lorentzian(  # noqa: PLR0913
        x: NDArray[np.float64],
        center: float = 0,
        width: float = 1,
        erf_amp: float = 1,
        lin_slope: float = 0,
        const_bkg: float = 0,
        gamma: float = 1,
        lorcenter: float = 0,
    ) -> NDArray[np.float64]:
        """A Lorentzian multiplied by a gstepb background."""
        return gstepb(x, center, width, erf_amp, lin_slope, const_bkg) * lorentzian(
            x,
            gamma,
            lorcenter,
            1,
        )

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.gstepb_mult_lorentzian, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_slope", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)
        self.set_param_hint("gamma", min=0.0)

    def guess(
        self,
        data: XrTypes | NDArray[np.float64],
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        if isinstance(data, XrTypes):
            ymin = data.min().item()
            ymean = data.mean()
        else:
            ymin = min(data)
            ymean = np.mean(data)
        if isinstance(x, xr.DataArray):
            xmin, xmax = x.min().item(), x.max().item()
        else:
            xmin, xmax = np.min(x), np.max(x)
        pars = self.make_params(center=(xmax + xmin) / 2.0, erf_amp=(ymean - ymin))

        pars[f"{self.prefix}lorcenter"].set(value=0)
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.min())
        pars[f"{self.prefix}width"].set(0.02)
        pars[f"{self.prefix}erf_amp"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        "Lorentzian multiplied by a gstepb background model" + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiDiracModel(Model):
    r"""A model for the Fermi Dirac function.

    .. math::

    \frac{scale}{\exp\left(\frac{x-center}{width}  +1\right)}
    """

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(fermi_dirac, **kwargs)

        self.set_param_hint("width", min=0)

    def guess(
        self,
        data: NDArray[np.float64] | XrTypes,
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        if isinstance(x, xr.DataArray):
            x = x.values

        ymax = max(data)
        xmin, xmax = min(x), max(x)
        pars = self.make_params(scale=ymax, center=(xmax - xmin) / 2.0, width=(xmax - xmin) / 10)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = "Fermi-Dirc distribution model" + lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBModel(Model):
    r"""A model for fitting Fermi functions with a linear background.

    .. math::
        f(x; center, width, A, a, b)= b+a (x-center) +
        \frac{A}{2}*\mathrm{erfc}\left(\frac{\sqrt{4ln(2)} (x-center)}{width}\right)

    where the parameter `erp_amp` corresponds `A`, `lin_slope` to `a`, and `const_bkg` to `b`.

    """

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(gstepb, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_slope", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        xmin, xmax = min(x), max(x)
        pars = self.make_params(center=(xmax - xmin) / 2)
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.min())

        pars[f"{self.prefix}width"].set(0.02)
        pars[f"{self.prefix}erf_amp"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        """Fermi functions with a linear background model""" + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class BandEdgeBModel(Model):
    """Fitting model for Lorentzian and background multiplied into the fermi dirac distribution."""

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(band_edge_bkg, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("offset", min=-10)

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope = stats.linregress(x, data)[0]
            pars[f"{self.prefix}lor_center"].set(value=x[np.argmax(data - slope * x)])
        else:
            pars[f"{self.prefix}lor_center"].set(value=-0.2)

        pars[f"{self.prefix}gamma"].set(value=0.2)
        pars[f"{self.prefix}amplitude"].set(value=(data.mean() - data.min()) / 1.5)

        pars[f"{self.prefix}const_bkg"].set(value=data.min())
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}offset"].set(value=data.min())

        pars[f"{self.prefix}center"].set(value=0)

        pars[f"{self.prefix}width"].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBGModel(Model):
    """Fitting model Lorentzian and background multiplied into the fermi dirac distribution."""

    @staticmethod
    def band_edge_bkg_gauss(  # noqa: PLR0913
        x: NDArray[np.float64],
        width: float = 0.05,
        amplitude: float = 1,
        gamma: float = 0.1,
        lor_center: float = 0,
        offset: float = 0,
        lin_slope: float = 0,
        const_bkg: float = 0,
    ) -> NDArray[np.float64]:
        """Fitting model for Lorentzian and background multiplied into Fermi dirac distribution."""
        return np.convolve(
            band_edge_bkg(x, 0, width, amplitude, gamma, lor_center, offset, lin_slope, const_bkg),
            gaussian(np.linspace(-6, 6, 800), 0, 0.01, 1 / np.sqrt(2 * np.pi * 0.01**2)),
            mode="same",
        )

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.band_edge_bkg_gauss, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("offset", min=-10)
        self.set_param_hint("center", vary=False)

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | None = None,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.

        Args:
            data: ARPES data
            x (NDArray[np._float],NONE): as variable "x"
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        pars = self.make_params()

        if x is not None:
            slope = stats.linregress(x, data)[0]
            pars[f"{self.prefix}lor_center"].set(value=x[np.argmax(data - slope * x)])
        else:
            pars[f"{self.prefix}lor_center"].set(value=-0.2)

        pars[f"{self.prefix}gamma"].set(value=0.2)
        pars[f"{self.prefix}amplitude"].set(value=(data.mean() - data.min()) / 1.5)

        pars[f"{self.prefix}const_bkg"].set(value=data.min())
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}offset"].set(value=data.min())

        pars[f"{self.prefix}width"].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)


class GStepBStdevModel(Model):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_stdev(  # noqa: PLR0913
        x: NDArray[np.float64],
        center: float = 0,
        sigma: float = 1,
        erf_amp: float = 1,
        lin_slope: float = 0,
        const_bkg: float = 0,
    ) -> NDArray[np.float64]:
        """Fermi function convolved with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            center: center of the step
            sigma: width of the step
            erf_amp: height of the step
            lin_slope: linear background slope
            const_bkg: constant background
        """
        dx = x - center
        return const_bkg + lin_slope * np.min(dx, 0) + gstep_stdev(x, center, sigma, erf_amp)

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.gstepb_stdev, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_slope", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: XrTypes | NDArray[np.float64],
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        if isinstance(x, xr.DataArray):
            x = x.values

        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.min())

        pars[f"{self.prefix}sigma"].set(0.02)
        pars[f"{self.prefix}erf_amp"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        "Fermi-Dirac distribution function with a linear background model"
        + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStandardModel(Model):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_standard(
        x: NDArray[np.float64],
        center: float = 0,
        sigma: float = 1,
        amplitude: float = 1,
        **kwargs: Incomplete,
    ) -> NDArray[np.float64]:
        """Specializes parameters in gstepb."""
        return gstepb(x, center, width=sigma, erf_amp=amplitude, **kwargs)

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.gstepb_standard, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_slope", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        if isinstance(data, XrTypes):
            ymin = data.min().item()
            ymean = data.mean()
        else:
            ymin = min(data)
            ymean = np.mean(data)
        if isinstance(x, xr.DataArray):
            xmin, xmax = x.min().item(), x.max().item()
        else:
            xmin, xmax = np.min(x), np.max(x)
        pars = self.make_params(center=(xmax + xmin) / 2.0, erf_amp=(ymean - ymin))

        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.min())

        pars[f"{self.prefix}sigma"].set(0.02)
        pars[f"{self.prefix}amplitude"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        """A model for fitting Fermi functions with a linear background."""
        + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
