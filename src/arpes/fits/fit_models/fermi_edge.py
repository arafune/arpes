"""Definitions of models involving Fermi edges."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Unpack

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.lineshapes import gaussian, lorentzian
from lmfit.models import Model, update_param_vals
from scipy import stats

from .functional_forms import (
    affine_broadened_fd,
    band_edge_bkg,
    fermi_dirac,
    fermi_dirac_affine,
    gstep_stdev,
    gstepb,
)
from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import XrTypes
    from arpes.fits import ModelArgs

__all__ = (
    "AffineBroadenedFD",
    "BandEdgeBGModel",
    "BandEdgeBModel",
    "FermiDiracAffGaussModel",
    "FermiDiracModel",
    "FermiLorentzianModel",
    "GStepBModel",
    "GStepBStandardModel",
    "GStepBStdevModel",
    "TwoBandEdgeBModel",
)


class AffineBroadenedFD(Model):
    r"""A model based for affine density of states convoluted with gaussian.

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
        data: XrTypes,
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        ymin, ymax = min(data), max(data)
        if isinstance(x, xr.DataArray):
            x = x.values
        xmin, xmax = min(x), max(x)
        pars = self.make_params(const_bkg=(ymax - ymin), center=(xmax + xmin) / 2.0)
        sigma = 0.1 * (xmax - xmin)
        width = 0.1 * (xmax - xmin)
        pars[f"{self.prefix}sigma"].set(value=sigma)
        pars[f"{self.prefix}width"].set(value=width)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        "Affine density of states broadened by Fermi-Dirac " + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiLorentzianModel(XModelMixin):
    """A Lorentzian multiplied by a gstepb background."""

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
        data: XrTypes,
        x,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Makes heuristic guesses for parameters based on input data.

        This function sets initial guesses for a set of parameters based on simple
        heuristics, such as the minimum and mean of the input data. The function
        is a placeholder for future improvements where better guesses can be made.

        Args:
            data (XrTypes): Input data for making parameter guesses. The data is used
                            to estimate initial values like background levels and amplitude.
            kwargs: Additional keyword arguments to update parameter values.

        Returns:
            lf.Parameters: A set of parameters with initial guesses, potentially updated
                        by the provided `kwargs`.
        """
        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
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
    """A model for the Fermi Dirac function."""

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(fermi_dirac, **kwargs)

        self.set_param_hint("width", min=0)

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Makes heuristic guesses for parameters based on input data.

        This function sets initial guesses for a set of parameters based on simple
        heuristics, such as the minimum and mean of the input data. The function
        is a placeholder for future improvements where better guesses can be made.

        Args:
            data (XrTypes): Input data for making parameter guesses. The data is used
                            to estimate initial values like background levels and amplitude.
            kwargs: Additional keyword arguments to update parameter values.

        Returns:
            lf.Parameters: A set of parameters with initial guesses, potentially updated
                        by the provided `kwargs`.
        """
        if isinstance(x, xr.DataArray):
            x = x.values
        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}width"].set(value=0.05)
        pars[f"{self.prefix}scale"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = "Fermi-Dirc distribution model" + lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

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
        x: None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Makes heuristic guesses for parameters based on the input data.

        This function initializes parameter values with simple heuristic estimates,
        such as using the minimum and mean values of the data. The `x` parameter is
        intentionally ignored, and it should always be `None`.

        Args:
            data (XrTypes): The input data used to make initial guesses for parameters.
                            The data's minimum and mean values are used for background
                            and amplitude estimates.
            x (None): This parameter is ignored and should always be `None`.
            kwargs: Additional keyword arguments used to update the guessed parameters.

        Returns:
            lf.Parameters: A set of parameters initialized with heuristic guesses,
                        which may be updated with the provided `kwargs`.
        """
        pars = self.make_params()
        assert x is None
        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=data.min())

        pars[f"{self.prefix}width"].set(0.02)
        pars[f"{self.prefix}erf_amp"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        """Fermi functions with a linear background model""" + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoBandEdgeBModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution.

    TODO, actually implement two_band_edge_bkg (find original author and their intent).
    """

    @staticmethod
    def two_band_edge_bkg() -> NoReturn:
        """Some missing model referenced in old Igor code retained for visibility here."""
        raise NotImplementedError

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.two_band_edge_bkg, **kwargs)

        self.set_param_hint("amplitude_1", min=0.0)
        self.set_param_hint("gamma_1", min=0.0)
        self.set_param_hint("amplitude_2", min=0.0)
        self.set_param_hint("gamma_2", min=0.0)

        self.set_param_hint("offset", min=-10)

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | None = None,
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


class BandEdgeBModel(XModelMixin):
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
        x: NDArray[np.float64] | None = None,
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


class BandEdgeBGModel(XModelMixin):
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


class FermiDiracAffGaussModel(XModelMixin):
    """Fermi Dirac function with affine background multiplied, then all convolved with Gaussian."""

    @staticmethod
    def fermi_dirac_bkg_gauss(  # noqa: PLR0913
        x: NDArray[np.float64],
        center: float = 0,
        width: float = 0.05,
        lin_slope: float = 0,
        const_bkg: float = 0,
        scale: float = 1,
        sigma: float = 0.01,
    ) -> NDArray[np.float64]:
        """Fermi Dirac function with affine background multiplied, convolved with Gaussian."""
        return np.convolve(
            fermi_dirac_affine(x, center, width, lin_slope, const_bkg, scale),
            gaussian(x, (min(x) + max(x)) / 2, sigma, 1 / np.sqrt(2 * np.pi * sigma**2)),
            mode="same",
        )

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.fermi_dirac_bkg_gauss, **kwargs)

        self.set_param_hint("width", vary=False)
        self.set_param_hint("scale", min=0)
        self.set_param_hint("sigma", min=0, vary=True)
        self.set_param_hint("lin_slope", vary=False)
        self.set_param_hint("const_bkg", vary=False)

    def guess(
        self,
        data: XrTypes,
        x: None = None,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here.

        Args:
            data: [TODO:description]
            x (NONE): In this guess function, x should be None.
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        assert x is None  # "x" is not used but for consistency, it should not be removed.
        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}width"].set(value=0.0009264)
        pars[f"{self.prefix}scale"].set(value=data.mean() - data.min())
        pars[f"{self.prefix}lin_slope"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=0)
        pars[f"{self.prefix}sigma"].set(value=0.023)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        "Fermi Dirac function with affine background multiplied, then all convolved with Gaussian"
        + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStdevModel(XModelMixin):
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
        data: XrTypes,
        x: None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here."""
        assert x is None  # "x" is not used but for consistency, it should not be removed.
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


class GStepBStandardModel(XModelMixin):
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
        x: None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here.

        Args:
            data ([TODO:type]): [TODO:description]
            x (NONE): In this guess function, x should be None
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        assert x is None  # "x" is not used but for consistency, it should not be removed.
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
