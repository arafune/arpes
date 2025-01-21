"""Rudimentary band analysis code."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage.filters
import xarray as xr

import arpes.fits
from arpes.analysis.band_analysis_utils import param_getter, param_stderr_getter
from arpes.debug import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from xarray.core.coordinates import DataArrayCoordinates
    from xarray.core.indexes import Indexes

    from arpes.fits import XModelMixin

__all__ = [
    "BackgroundBand",
    "Band",
    "MultifitBand",
    "VoigtBand",
]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class Band:
    """Representation of an ARPES band which supports some calculations after fitting.

    Attribute:
        label (str): The label name of the band.  In most case, prefix of lf.Model.
        _data (XrTypes): Xarray consists of several DataArrays representing the fitting results.
            When _data is xr.Dataset, `data_vars` should be "center", "center_stderr", "amplitude",
            "amplitude_stdrr", "sigma", and "sigma_stderr"
    """

    def __init__(
        self,
        label: str,
        data: xr.Dataset | None = None,
    ) -> None:
        """Set the data but don't perform any calculation eagerly."""
        self.label = label
        self._data: xr.Dataset | None = data

    @property
    def velocity(self) -> xr.DataArray:
        """The band velocity.

        Returns: (xr.DataArray)
            [TODO:description]
        """
        spacing: float = self.coords[self.dims[0]][1].item() - self.coords[self.dims[0]][0].item()
        sigma: float = 0.1 / spacing
        raw_values = self.embed_nan(self.center.values, 50)

        masked: NDArray[np.float64] = np.nan_to_num(np.copy(raw_values), nan=0.0)
        nan_mask: NDArray[np.float64] = np.nan_to_num(np.copy(raw_values) * 0 + 1, nan=0.0)

        nan_mask = scipy.ndimage.gaussian_filter(nan_mask, sigma, mode="mirror")
        masked = scipy.ndimage.gaussian_filter(masked, sigma, mode="mirror")

        return xr.DataArray(
            data=np.gradient(masked / nan_mask, spacing)[50:-50],
            coords=self.coords,
            dims=self.dims,
        )

    @property
    def fermi_velocity(self) -> xr.DataArray:
        """The band velocity evaluated at the Fermi level.

        Implicitly assuming that the fit with broadcast_dim = "eV" was performed.
        """
        return self.velocity.sel(eV=0, method="nearest")

    @property
    def band_width(self) -> None:
        """The width along the band."""
        return

    @property
    def self_energy(self) -> None:
        """Calculates the self energy along the band."""
        return

    @property
    def fit_cls(self) -> type[XModelMixin]:
        """Describes which fit class to use for band fitting, default Lorentzian."""
        return arpes.fits.LorentzianModel

    def get_dataarray(
        self,
        var_name: str,  # Literal["center", "amplitude", "sigma""]
        *,
        clean: bool = True,
    ) -> xr.DataArray | NDArray[np.float64]:
        """Converts the underlying data into an array representation."""
        assert isinstance(self._data, xr.Dataset)
        if not clean:
            return self._data[var_name].values

        output = np.copy(self._data[var_name].values)
        output[self._data[var_name + "_stderr"].values > 0.01] = np.nan  # noqa: PLR2004

        return xr.DataArray(
            data=output,
            coords=self._data[var_name].coords,
            dims=self._data[var_name].dims,
        )

    @property
    def center(self) -> xr.DataArray:
        """Gets the peak location along the band."""
        center_array = self.get_dataarray("center")
        assert isinstance(center_array, xr.DataArray)
        return center_array

    @property
    def center_stderr(self) -> NDArray[np.float64]:
        """Gets the peak location stderr along the band."""
        center_stderr = self.get_dataarray("center_stderr", clean=False)
        assert isinstance(center_stderr, np.ndarray)
        return center_stderr

    @property
    def sigma(self) -> xr.DataArray:
        """Gets the peak width along the band."""
        sigma_array = self.get_dataarray("sigma", clean=True)
        assert isinstance(sigma_array, xr.DataArray)
        return sigma_array

    @property
    def amplitude(self) -> xr.DataArray:
        """Gets the peak amplitude along the band."""
        amplitude_array = self.get_dataarray("amplitude", clean=True)
        assert isinstance(amplitude_array, xr.DataArray)
        return amplitude_array

    @property
    def indexes(self) -> Indexes:
        """Fetches the indices of the originating data (after fit reduction)."""
        assert isinstance(self._data, xr.Dataset)
        return self._data.center.indexes

    @property
    def coords(self) -> DataArrayCoordinates:
        """Fetches the coordinates of the originating data (after fit reduction)."""
        assert isinstance(self._data, xr.Dataset)
        return self._data.center.coords

    @property
    def dims(self) -> tuple[str, ...]:
        """Fetches the dimensions of the originating data (after fit reduction)."""
        assert isinstance(self._data, xr.Dataset)
        return self._data.center.dims

    @staticmethod
    def embed_nan(values: NDArray[np.float64], padding: int) -> NDArray[np.float64]:
        """Return np.ndarray padding before and after the original NDArray with nan.

        Args:
            values: [TODO:description]
            padding: the length of the padding

        Returns: NDArray[np.float64]
            [TODO:description]
        """
        embedded: NDArray[np.float64] = np.full(
            shape=(values.shape[0] + 2 * padding,),
            fill_value=np.nan,
            dtype=np.float64,
        )
        embedded[padding:-padding] = values
        return embedded


class MultifitBand(Band):
    """Convenience class that reimplements reading data out of a composite fit result."""

    def get_dataarray(self, var_name: str, *, clean: bool = True) -> xr.DataArray:
        """Converts the underlying data into an array representation."""
        assert isinstance(self._data, xr.DataArray | xr.Dataset)
        full_var_name = self.label + var_name
        if not clean:
            pass
        if "stderr" in full_var_name:
            return self._data.G.map(param_stderr_getter(full_var_name.split("_stderr")[0]))

        return self._data.G.map(param_getter(full_var_name))


class VoigtBand(Band):
    """Uses a Voigt lineshape."""

    @property
    def fit_cls(self) -> type[XModelMixin]:  # guess_fit is used.
        """Fit using `arpes.fits.VoigtModel`."""
        return arpes.fits.VoigtModel


class BackgroundBand(Band):
    """Uses a Gaussian lineshape."""

    @property
    def fit_cls(self) -> type[XModelMixin]:  # guess_fit is used
        """Fit using `arpes.fits.GaussianModel`."""
        return arpes.fits.GaussianModel
