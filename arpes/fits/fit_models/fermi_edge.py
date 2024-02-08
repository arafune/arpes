"""Definitions of models involving Fermi edges."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Unpack

import lmfit as lf
import numpy as np
from lmfit.models import update_param_vals
from scipy import stats
from scipy.ndimage import gaussian_filter

from .functional_forms import (
    band_edge_bkg,
    fermi_dirac,
    fermi_dirac_affine,
    gaussian,
    gstep,
    gstep_stdev,
    gstepb,
    lorentzian,
    twolorentzian,
)
from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import DataType, XrTypes
    from arpes.fits import ModelARGS

__all__ = [
    "AffineBroadenedFD",
    "FermiLorentzianModel",
    "FermiDiracModel",
    "GStepBModel",
    "TwoBandEdgeBModel",
    "BandEdgeBModel",
    "BandEdgeBGModel",
    "FermiDiracAffGaussModel",
    "GStepBStdevModel",
    "GStepBStandardModel",
    "TwoLorEdgeModel",
]


class AffineBroadenedFD(XModelMixin):
    """Fitting model for affine density of states.

    (with resolution broadened Fermi-Dirac occupation).
    """

    @staticmethod
    def affine_broadened_fd(  # noqa: PLR0913
        x: NDArray[np.float_],
        fd_center: float = 0,
        fd_width: float = 0.003,
        conv_width: float = 0.02,
        const_bkg: float = 1,
        lin_bkg: float = 0,
        offset: float = 0,
    ) -> NDArray[np.float_]:
        """Fermi function convoled with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            fd_center: center of the step
            fd_width: width of the step
            conv_width: The convolution width
            const_bkg: constant background
            lin_bkg: linear (affine) background slope
            offset: constant background
        """
        dx = x - fd_center
        x_scaling = x[1] - x[0]
        fermi = 1 / (np.exp(dx / fd_width) + 1)
        return (
            gaussian_filter((const_bkg + lin_bkg * dx) * fermi, sigma=conv_width / x_scaling)
            + offset
        )

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.affine_broadened_fd, **kwargs)

        self.set_param_hint("offset", min=0.0)
        self.set_param_hint("fd_width", min=0.0)
        self.set_param_hint("conv_width", min=0.0)

    def guess(self, data: XrTypes, **kwargs: float) -> lf.Parameters:
        """Make some heuristic guesses.

        We use the mean value to estimate the background parameters and physically
        reasonable ones to initialize the edge.
        """
        pars: lf.Parameters = self.make_params()

        pars["%sfd_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.mean().item() * 2)
        pars["%soffset" % self.prefix].set(value=data.min().item())

        pars["%sfd_width" % self.prefix].set(0.005)
        pars["%sconv_width" % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiLorentzianModel(XModelMixin):
    """A Lorentzian multiplied by a gstepb background."""

    @staticmethod
    def gstepb_mult_lorentzian(  # noqa: PLR0913
        x: NDArray[np.float_],
        center: float = 0,
        width: float = 1,
        erf_amp: float = 1,
        lin_bkg: float = 0,
        const_bkg: float = 0,
        gamma: float = 1,
        lorcenter: float = 0,
    ) -> NDArray[np.float_]:
        """A Lorentzian multiplied by a gstepb background."""
        return gstepb(x, center, width, erf_amp, lin_bkg, const_bkg) * lorentzian(
            x,
            gamma,
            lorcenter,
            1,
        )

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.gstepb_mult_lorentzian, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)
        self.set_param_hint("gamma", min=0.0)

    def guess(
        self,
        data: XrTypes,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here.

        Args:
            data ([TODO:type]): [TODO:description]
            x (NONE): in this guess function, x should be None.
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slorcenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())

        pars["%swidth" % self.prefix].set(0.02)
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiDiracModel(XModelMixin):
    """A model for the Fermi Dirac function."""

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(fermi_dirac, **kwargs)

        self.set_param_hint("width", min=0)

    def guess(self, data: DataType, **kwargs: Incomplete) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars[f"{self.prefix}center"].set(value=0)
        pars[f"{self.prefix}width"].set(value=0.05)
        pars[f"{self.prefix}scale"].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(gstepb, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
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
            x (NONE): in this guess function, x should be None.
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        pars = self.make_params()
        assert x is None
        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())

        pars["%swidth" % self.prefix].set(0.02)
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoBandEdgeBModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution.

    TODO, actually implement two_band_edge_bkg (find original author and their intent).
    """

    @staticmethod
    def two_band_edge_bkg() -> NoReturn:
        """Some missing model referenced in old Igor code retained for visibility here."""
        raise NotImplementedError

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
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
        x: NDArray[np.float_] | None = None,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        pars["%scenter" % self.prefix].set(value=0)

        pars["%swidth" % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBModel(XModelMixin):
    """Fitting model for Lorentzian and background multiplied into the fermi dirac distribution."""

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
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
        x: NDArray[np.float_] | None = None,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        pars["%scenter" % self.prefix].set(value=0)

        pars["%swidth" % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBGModel(XModelMixin):
    """Fitting model Lorentzian and background multiplied into the fermi dirac distribution."""

    @staticmethod
    def band_edge_bkg_gauss(  # noqa: PLR0913
        x: NDArray[np.float_],
        width: float = 0.05,
        amplitude: float = 1,
        gamma: float = 0.1,
        lor_center: float = 0,
        offset: float = 0,
        lin_bkg: float = 0,
        const_bkg: float = 0,
    ) -> NDArray[np.float_]:
        """Fitting model for Lorentzian and background multiplied into Fermi dirac distribution."""
        return np.convolve(
            band_edge_bkg(x, 0, width, amplitude, gamma, lor_center, offset, lin_bkg, const_bkg),
            gaussian(np.linspace(-6, 6, 800), 0, 0.01, 1 / np.sqrt(2 * np.pi * 0.01**2)),
            mode="same",
        )

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.band_edge_bkg_gauss, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("offset", min=-10)
        self.set_param_hint("center", vary=False)

        """

        """

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float_] | None = None,
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
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        pars["%swidth" % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)


class FermiDiracAffGaussModel(XModelMixin):
    """Fermi Dirac function with affine background multiplied, then all convolved with Gaussian."""

    @staticmethod
    def fermi_dirac_bkg_gauss(  # noqa: PLR0913
        x: NDArray[np.float_],
        center: float = 0,
        width: float = 0.05,
        lin_bkg: float = 0,
        const_bkg: float = 0,
        scale: float = 1,
        sigma: float = 0.01,
    ) -> NDArray[np.float_]:
        """Fermi Dirac function with affine background multiplied, convolved with Gaussian."""
        return np.convolve(
            fermi_dirac_affine(x, center, width, lin_bkg, const_bkg, scale),
            gaussian(x, (min(x) + max(x)) / 2, sigma, 1 / np.sqrt(2 * np.pi * sigma**2)),
            mode="same",
        )

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.fermi_dirac_bkg_gauss, **kwargs)

        self.set_param_hint("width", vary=False)
        self.set_param_hint("scale", min=0)
        self.set_param_hint("sigma", min=0, vary=True)
        self.set_param_hint("lin_bkg", vary=False)
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
        pars[f"{self.prefix}lin_bkg"].set(value=0)
        pars[f"{self.prefix}const_bkg"].set(value=0)
        pars[f"{self.prefix}sigma"].set(value=0.023)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStdevModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_stdev(  # noqa: PLR0913
        x: NDArray[np.float_],
        center: float = 0,
        sigma: float = 1,
        erf_amp: float = 1,
        lin_bkg: float = 0,
        const_bkg: float = 0,
    ) -> NDArray[np.float_]:
        """Fermi function convolved with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            center: center of the step
            sigma: width of the step
            erf_amp: height of the step
            lin_bkg: linear background slope
            const_bkg: constant background
        """
        dx = x - center
        return const_bkg + lin_bkg * np.min(dx, 0) + gstep_stdev(x, center, sigma, erf_amp)

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.gstepb_stdev, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
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

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())

        pars["%ssigma" % self.prefix].set(0.02)
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStandardModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_standard(
        x: NDArray[np.float_],
        center: float = 0,
        sigma: float = 1,
        amplitude: float = 1,
        **kwargs: Incomplete,
    ) -> NDArray[np.float_]:
        """Specializes parameters in gstepb."""
        return gstepb(x, center, width=sigma, erf_amp=amplitude, **kwargs)

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.gstepb_standard, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
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

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())

        pars["%ssigma" % self.prefix].set(0.02)
        pars["%samplitude" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorEdgeModel(XModelMixin):
    """A model for (two lorentzians with an affine background) multiplied by a gstepb.

    **This is typically not necessary, as you can use the + operator on the Model instances.**
    """

    def twolorentzian_gstep(  # noqa: PLR0913
        self,
        x: NDArray[np.float_],
        gamma: float,
        t_gamma: float,
        center: float,
        t_center: float,
        amp: float,
        t_amp: float,
        lin_bkg: float,
        const_bkg: float,
        g_center: float,
        sigma: float,
        erf_amp: float,
    ) -> NDArray[np.float_]:
        """Two Lorentzians, an affine background, and a gstepb edge."""
        TL = twolorentzian(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg)
        GS = gstep(x, g_center, sigma, erf_amp)
        return TL * GS

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.twolorentzian_gstep, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("gamma", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_gamma", min=0)
        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(
        self,
        data: XrTypes,
        x: None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Placeholder for making better heuristic guesses here."""
        assert x is None
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%st_center" % self.prefix].set(value=0)
        pars["%sg_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())

        pars["%sgamma" % self.prefix].set(0.02)
        pars["%st_gamma" % self.prefix].set(0.02)
        pars["%ssigma" % self.prefix].set(0.02)
        pars["%samp" % self.prefix].set(value=data.mean() - data.min())
        pars["%st_amp" % self.prefix].set(value=data.mean() - data.min())
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
