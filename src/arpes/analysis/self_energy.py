"""Contains self-energy analysis routines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import LinearModel, LorentzianModel
from skimage.measure import LineModelND, ransac

from arpes.constants import HBAR_PER_EV, METERS_PER_SECOND_PER_EV_ANGSTROM

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

__all__ = (
    "estimate_bare_band",
    "fit_for_self_energy",
    "quasiparticle_lifetime",
    "to_self_energy",
)


def get_peak_parameter(
    data: xr.DataArray,
    parameter_name: str,
) -> xr.DataArray:
    """Extracts a parameter from a potentially prefixed peak-like component.

    Works so long as there is only a single peak defined in the model.

    Args:
        data: The input data containing the peak fitting results.
        parameter_name: The name of the parameter which should be extracted.

    Returns:
        The array of parameters corresponding to a peak in a single-peak curve fit.
    """
    first_item = data.values.ravel()[0]
    peak_like = (
        lf.models.LorentzianModel,
        lf.models.VoigtModel,
        lf.models.GaussianModel,
        lf.models.PseudoVoigtModel,
    )

    if isinstance(first_item, lf.model.ModelResult):
        if isinstance(first_item.model, lf.model.CompositeModel):
            peak_like_components = [
                c for c in first_item.model.components if isinstance(c, peak_like)
            ]
            assert len(peak_like_components) == 1

            return data.F.p(f"{peak_like_components[0].prefix}{parameter_name}")
        return data.F.p(parameter_name)

    msg = f"Unsupported dispersion {type(data)}, expected xr.DataArray[lmfit.ModelResult]"
    raise ValueError(
        msg,
    )


def local_fermi_velocity(bare_band: xr.DataArray) -> float:
    """Calculates the band velocity under assumptions of a linear bare band."""
    params = LinearModel().guess(bare_band, bare_band.coords["eV"].values)
    fitted_model = bare_band.S.modelfit("eV", model=LinearModel(), params=params)
    raw_velocity: float = fitted_model.params["slope"].value
    if "eV" in bare_band.dims:
        # the "y" values are in `bare_band` are momenta and the "x" values are energy, therefore
        # the slope is dy/dx = dk/dE
        raw_velocity = 1 / raw_velocity

    return raw_velocity * METERS_PER_SECOND_PER_EV_ANGSTROM


def estimate_bare_band(
    dispersion: xr.DataArray,
    bare_band_specification: str = "",
) -> xr.DataArray:
    """Estimates the bare band from a fitted dispersion.

    This can be done in a few ways:
    #. None: Equivalent to 'baseline_linear' below
    #. `'linear'`: A linear fit to the dispersion is used, and this also provides the fermi_velocity
    #. `'ransac_linear'`: A linear fit with random sample consensus (RANSAC) region will be used and
    this also provides the `fermi_velocity`
    #. `'hough'`: Hough transform based method

    Args:
        dispersion: The array of the fitted peak locations.
        bare_band_specification: What kind of bare band to assume.
            One of "linear", "ransac_linear", and "hough".

    Returns:
        An estimate of the bare band dispersion.
    """
    try:
        centers = get_peak_parameter(dispersion, "center")
    except ValueError:
        centers = dispersion

    mom_options = [d for d in dispersion.dims if d in {"k", "kp", "kx", "ky", "kz"}]
    assert len(mom_options) <= 1
    fit_dimension = "eV" if "eV" in dispersion.dims else mom_options[0]

    if not bare_band_specification:
        bare_band_specification = "ransac_linear"

    params = LinearModel().guess(centers, centers.coords["eV"].values)

    initial_linear_fit = centers.S.modelfit("eV", LinearModel(), params)
    if bare_band_specification == "linear":
        fitted_model = initial_linear_fit
    elif bare_band_specification == "ransac_linear":
        min_samples = len(centers.coords[fit_dimension]) // 10
        residual = initial_linear_fit.residual
        assert residual is not None
        residual_threshold = np.median(np.abs(residual)) * 1
        _, inliers = ransac(
            np.stack([centers.coords[fit_dimension], centers]).T,
            LineModelND,
            max_trials=1000,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
        )
        inlier_data = centers.where(
            xr.DataArray(
                inliers,
                coords={fit_dimension: centers.coords[fit_dimension]},
                dims=[fit_dimension],
            ),
            drop=True,
        )
        params = LinearModel().guess(inlier_data, inlier_data.coords[fit_dimension])
        fitted_model = inlier_data.S.modelfit(fit_dimension, LinearModel(), params=params)
    elif bare_band_specification == "hough":
        msg = "Hough Transform estimate of bare band not yet supported."
        raise NotImplementedError(msg)
    else:
        msg = f"Unrecognized bare band type: {bare_band_specification}"
        raise ValueError(msg)

    ys = fitted_model.eval(x=centers.coords[fit_dimension])
    return xr.DataArray(ys, centers.coords, centers.dims)


def quasiparticle_lifetime(
    self_energy: xr.DataArray,
) -> NDArray[np.float64]:
    """Calculates the quasiparticle mean free path in meters (meters!).

    The bare band is used to calculate the band/Fermi velocity
    and internally the procedure to calculate the quasiparticle lifetime is used.

    Args:
        self_energy: The measured or estimated self-energy.
        bare_band: The bare band defining the band velocity.

    Returns:
        An estimate of the quasiparticle lifetime along the band.
    """
    imaginary_part = np.abs(np.imag(self_energy)) / 2
    return HBAR_PER_EV / imaginary_part


def quasiparticle_mean_free_path(
    self_energy: xr.DataArray,
    bare_band: xr.DataArray,
) -> NDArray[np.float64]:
    lifetime = quasiparticle_lifetime(self_energy)
    return lifetime * local_fermi_velocity(bare_band)


def to_self_energy(
    dispersion: xr.DataArray,
    bare_band: xr.DataArray,
    fermi_velocity: float = 0,
    *,
    k_independent: bool = True,
) -> xr.Dataset:
    r"""Converts MDC fit results into the self energy.

    This largely consists of extracting
    out the linewidth and the difference between the dispersion and the bare band value.

    .. math::

        lorentzian(x, amplitude, center, sigma) =
            (amplitude / pi) * sigma/(sigma^2 + ((x-center))**2)

    Once we have the curve-fitted dispersion we can calculate the self energy if we also
    know the bare-band dispersion. If the bare band is not known, then at least the imaginary
    part of the self energy is still calculable, and a guess as to the real part can be
    calculated under assumptions of the bare band dispersion as being free electron like with
    effective mass m* or being Dirac like (these are equivalent at low enough energy).

    Acceptabe bare band specifications are discussed in detail in `estimate_bare_band` above.

    To future readers of the code, please note that the half-width half-max of a Lorentzian is equal
    to the $\gamma$ parameter, which defines the imaginary part of the self energy.

    Args:
        dispersion (xr.DataArray | xr.Dataset): The array of the fitted peak locations.
            When xr.Dataset is set, ".results" is used.
        bare_band (xr.DataArray): the bare band.
        fermi_velocity (float): The fermi velocity. If not set, use local_fermi_velocity
        k_independent: bool

    Returns:
        The equivalent self energy from the bare band and the measured dispersion.
    """
    if not k_independent:
        msg = (
            "PyARPES does not currently support self energy analysis"
            " except in the k-independent formalism."
        )
        raise NotImplementedError(
            msg,
        )

    if isinstance(dispersion, xr.Dataset):
        dispersion = dispersion.results
    assert isinstance(dispersion, xr.DataArray)
    from_mdcs = "eV" in dispersion.dims  # if eV is in the dimensions, then we fitted MDCs
    estimated_bare_band = estimate_bare_band(dispersion, bare_band_specification="ransac_linear")

    if not fermi_velocity:
        fermi_velocity = local_fermi_velocity(estimated_bare_band)
    assert isinstance(fermi_velocity, float)

    imaginary_part = get_peak_parameter(dispersion, "fwhm") / 2
    centers = get_peak_parameter(dispersion, "center")

    if from_mdcs:
        imaginary_part *= fermi_velocity / METERS_PER_SECOND_PER_EV_ANGSTROM
        real_part = -(
            (centers * fermi_velocity / METERS_PER_SECOND_PER_EV_ANGSTROM)
            - dispersion.coords["eV"].values
        )
    else:
        real_part = centers - bare_band

    self_energy = xr.DataArray(
        real_part + 1.0j * imaginary_part,
        coords=dispersion.coords,
        dims=dispersion.dims,
    )

    return xr.Dataset({"self_energy": self_energy, "bare_band": estimated_bare_band})


def fit_for_self_energy(
    data: xr.DataArray,
    bare_band: xr.DataArray,
    method: Literal["mdc", "edc"] = "mdc",
    **kwargs: Incomplete,
) -> xr.Dataset:
    """Fits for the self energy of a dataset containing a single band.

    Args:
        data: The input ARPES data.
        method: Determine the broadcast dimension in broadcast_model, one of 'mdc' and 'edc'
        bare_band: Optionally, the bare band. If None is provided the bare band will be estimated.
        **kwargs: pass to S.modelfit (typical kwargs is param={})

    Returns:
        The self energy resulting from curve-fitting.
    """
    assert data.S.is_kspace
    model = LorentzianModel() + LinearModel()
    if method == "mdc":
        possible_mometum_dims = ("kp", "kx", "ky", "kz")
        mom_axis = next(str(dim) for dim in data.dims if dim in possible_mometum_dims)
        assert mom_axis is not None
        model = LorentzianModel() + LinearModel()
        fit_results = data.S.modelfit(coords=mom_axis, model=model, **kwargs)
    else:
        fit_results = data.S.modelfit(coords="eV", model=model, **kwargs)
    return to_self_energy(fit_results.modelfilt_results, bare_band=bare_band)
