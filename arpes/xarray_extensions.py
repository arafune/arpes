"""Establishes the PyARPES data model by extending the `xarray` types.

This is another core part of PyARPES. It provides a lot of extensions to
what comes out of the box in xarray. Some of these are useful generics,
generally on the .T extension, others collect and manipulate metadata,
interface with plotting routines, provide functional programming utilities,
etc.

If `f` is an ARPES spectrum, then `f.S` should provide a nice representation of your data
in a Jupyter cell. This is a complement to the text based approach that merely printing `f`
offers. Note, as of PyARPES v3.x.y, the xarray version has been bumped and this representation
is no longer necessary as one is provided upstream.

The main accessors are .S, .G, .X. and .F.

The `.S` accessor:
    The `.S` accessor contains functionality related to spectroscopy. Utilities
    which only make sense in this context should be placed here, while more generic
    tools should be placed elsewhere.

The `.G.` accessor:
    This a general purpose collection of tools which exists to provide conveniences over
    what already exists in the xarray data model. As an example, there are various tools
    for simultaneous iteration of data and coordinates here, as well as for vectorized
    application of functions to data or coordinates.

The `.X` accessor:
    This is an accessor which contains tools related to selecting and subselecting
    data. The two most notable tools here are `.X.first_exceeding` which is very useful
    for initializing curve fits and `.X.max_in_window` which is useful for refining
    these initial parameter choices.

The `.F.` accessor:
    This is an accessor which contains tools related to interpreting curve fitting
    results. In particular there are utilities for vectorized extraction of parameters,
    for plotting several curve fits, or for selecting "worst" or "best" fits according
    to some measure.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import itertools
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Collection, Hashable, Mapping, Sequence
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Unpack

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import ndimage as ndi

import arpes
import arpes.constants
import arpes.utilities.math
from arpes.constants import TWO_DIMENSION

from .analysis import param_getter, param_stderr_getter, rebin
from .models.band import MultifitBand
from .plotting.dispersion import (
    LabeledFermiSurfaceParam,
    fancy_dispersion,
    hv_reference_scan,
    labeled_fermi_surface,
    reference_scan_fermi_surface,
    scan_var_reference_plot,
)
from .plotting.fermi_edge import fermi_edge_reference
from .plotting.movie import plot_movie
from .plotting.parameter import plot_parameter
from .plotting.spatial import reference_scan_spatial
from .plotting.spin import spin_polarized_spectrum
from .plotting.utils import fancy_labels, remove_colorbars
from .utilities import apply_dataarray
from .utilities.conversion.core import slice_along_path
from .utilities.region import DesignatedRegions, normalize_region
from .utilities.xarray import unwrap_xarray_item

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import lmfit
    from _typeshed import Incomplete
    from matplotlib import animation
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.typing import RGBColorType
    from numpy.typing import DTypeLike, NDArray

    from ._typing import (
        ANALYZERINFO,
        ANGLE,
        DAQINFO,
        EXPERIMENTINFO,
        LIGHTSOURCEINFO,
        SAMPLEINFO,
        SCANINFO,
        SPECTROMETER,
        DataType,
        PColorMeshKwargs,
        XrTypes,
    )
    from .provenance import PROVENANCE

    IncompleteMPL: TypeAlias = Incomplete

__all__ = ["ARPESDataArrayAccessor", "ARPESDatasetAccessor", "ARPESFitToolsAccessor"]

EnergyNotation = Literal["Binding", "Kinetic"]

ANGLE_VARS = ("alpha", "beta", "chi", "psi", "phi", "theta")

DEFAULT_RADII = {
    "kp": 0.02,
    "kz": 0.05,
    "phi": 0.02,
    "beta": 0.02,
    "theta": 0.02,
    "eV": 0.05,
    "delay": 0.2,
    "T": 2,
    "temperature": 2,
}

UNSPESIFIED = 0.1

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def _iter_groups(
    grouped: dict[str, Sequence[float] | float],
) -> Generator[tuple[str, float], None, None]:
    """Iterates through a flattened sequence.

    Sequentially yields keys and values from each sequence associated with a key.
    If a key "k" is associated to a value "v0" which is not iterable, it will be emitted as a
    single pair (k, v0).

    Otherwise, one pair is yielded for every item in the associated value.

    ToDo: Not tested
    """
    for k, value_or_list in grouped.items():
        if isinstance(value_or_list, Sequence):
            for list_item in value_or_list:
                yield k, list_item
        else:
            yield k, value_or_list


class ARPESAccessorBase:
    """Base class for the xarray extensions in PyARPES."""

    def along(self, directions: NDArray[np.float_], **kwargs: Incomplete) -> xr.DataArray:
        """TODO: Need description.

        ToDo: Test
        """
        assert isinstance(self._obj, xr.DataArray)
        return slice_along_path(self._obj, directions, **kwargs)

    def find(self, name: str) -> list[str]:
        """Return the property names containing the "name".

        Args:
            name (str): string to find.

        Returns: list[str]
            Property list
        """
        return [n for n in dir(self) if name in n]

    @property
    def sherman_function(self) -> Incomplete:
        """TODO:summary.

        Returns: Incomplete
            [TODO:description]

        Raises: ValueError
            [TODO:description]

        ToDo: Test
        """
        for option in ["sherman", "sherman_function", "SHERMAN"]:
            if option in self._obj.attrs:
                return self._obj.attrs[option]
        msg = "No Sherman function could be found on the data. Is this a spin dataset?"
        raise ValueError(msg)

    @property
    def experimental_conditions(
        self,
    ) -> EXPERIMENTINFO:
        """Return experimental condition: hv, polarization, temperature.

        Use this property in plotting/annotations.py/conditions
        """
        return {
            "hv": self.hv,
            "polarization": self.polarization,
            "temperature": self.temp,
        }

    @property
    def polarization(self) -> float | str | tuple[float, float]:
        """Returns the light polarization information.

        ToDo: Test
        """
        if "epu_pol" in self._obj.attrs:
            # merlin: TODO normalize these
            # check and complete
            try:
                return {
                    0: "p",
                    1: "rc",
                    2: "s",
                }.get(int(self._obj.attrs["epu_pol"]), np.nan)
            except ValueError:
                return self._obj.attrs["epu_pol"]
        if "pol" in self._obj.attrs:
            return self._obj.attrs["pol"]
        return np.nan

    @property
    def is_subtracted(self) -> bool:
        """Infers whether a given data is subtracted.

        Args:
            self ([TODO:type]): [TODO:description]

        Returns (bool):
            Return True if the data is subtracted.

        ToDo: Need test
        """
        assert isinstance(self._obj, xr.DataArray)
        if self._obj.attrs.get("subtracted"):
            return True

        threshold_is_5_percent = 0.05
        return (((self._obj < 0) * 1).mean() > threshold_is_5_percent).item()

    @property
    def is_spatial(self) -> bool:
        """Infers whether a given scan has real-space dimensions (SPEM or u/nARPES).

        Returns:
            True if the data is explicltly a "ucut" or "spem" or contains "X", "Y", or "Z"
            dimensions. False otherwise.
        """
        if self.spectrum_type in {"ucut", "spem"}:
            return True

        return any(d in {"X", "Y", "Z"} for d in self._obj.dims)

    @property
    def is_kspace(self) -> bool:
        """Infers whether the scan is k-space converted or not.

        Because of the way this is defined, it will return
        true for XPS spectra, which I suppose is true but trivially.

        Returns:
            True if the data is k-space converted. False otherwise.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        return not any(d in {"phi", "theta", "beta", "angle"} for d in self._obj.dims)

    @property
    def is_slit_vertical(self) -> bool:
        """Infers whether the scan is taken on an analyzer with vertical slit.

        Caveat emptor: this assumes that the alpha coordinate is not some intermediate value.

        Returns:
            True if the alpha value is consistent with a vertical slit analyzer. False otherwise.
        """
        angle_tolerance = 1.0
        if self.angle_unit.startswith("Deg") or self.angle_unit.startswith("deg"):
            return float(np.abs(self.lookup_offset_coord("alpha") - 90.0)) < angle_tolerance
        return float(np.abs(self.lookup_offset_coord("alpha") - np.pi / 2)) < np.pi / 180

    @property
    def endstation(self) -> str:
        """Alias for the location attribute used to load the data.

        Returns:
            The name of loader/location which was used to load data.
        """
        return str(self._obj.attrs["location"])

    def with_values(self, new_values: NDArray[np.float_]) -> xr.DataArray:
        """Copy with new array values.

        Easy way of creating a DataArray that has the same shape as the calling object but data
        populated from the array `new_values`.

        Notes: This method is applicable only for xr.DataArray.  (Not xr.Dataset)

        Args:
            new_values: The new values which should be used for the data.

        Returns:
            A copy of the data with new values but identical dimensions, coordinates, and attrs.

        ToDo: Test
        """
        assert isinstance(self._obj, xr.DataArray)
        return xr.DataArray(
            new_values.reshape(self._obj.values.shape),
            coords=self._obj.coords,
            dims=self._obj.dims,
            attrs=self._obj.attrs,
        )

    @property
    def logical_offsets(self) -> dict[str, float | xr.DataArray]:
        """Return logical offsets.

        Raises:
            ValueError: [TODO:description]

        Returns:
            [TODO:description]

        ToDo: Test
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "long_x" not in self._obj.coords:
            msg = "Logical offsets can currently only be accessed for hierarchical"
            msg += " motor systems like nanoARPES."
            raise ValueError(
                msg,
            )
        return {
            "x": self._obj.coords["long_x"] - self._obj.coords["physical_long_x"],
            "y": self._obj.coords["long_y"] - self._obj.coords["physical_long_y"],
            "z": self._obj.coords["long_z"] - self._obj.coords["physical_long_z"],
        }

    @property
    def hv(self) -> float | xr.DataArray:
        """Return the photon energy.

        Returns: float | xr.DataArray
            Photon energy in eV unit.  (for hv_map type, xr.DataArray is returned.)
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        try:
            return float(self._obj.coords["hv"])
        except TypeError:
            return self._obj.coords["hv"]

    @property
    def scan_type(self) -> str | None:
        scan_type = self._obj.attrs.get("daq_type")
        if scan_type:
            return scan_type
        return None

    @property
    def spectrum_type(self) -> Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]:
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if self._obj.attrs.get("spectrum_type"):
            return self._obj.attrs["spectrum_type"]
        dim_types = {
            ("eV",): "xps",
            ("eV", "phi"): "cut",
            # this should check whether the other angular axis perpendicular to scan axis?
            ("eV", "phi", "beta"): "map",
            ("eV", "phi", "theta"): "map",
            ("eV", "hv", "phi"): "hv_map",
            # kspace
            ("eV", "kp"): "cut",
            ("eV", "kx", "ky"): "map",
            ("eV", "kp", "kz"): "hv_map",
        }
        dims: tuple = tuple(sorted(self._obj.dims))
        return dim_types.get(dims)

    @property
    def is_differentiated(self) -> bool:
        """Return True if the spectrum is differentiated data.

        Returns: bool

        ToDo: Test
        """
        history = self.short_history()
        return "dn_along_axis" in history or "curvature" in history

    def transpose_to_front(self, dim: str) -> XrTypes:
        """Transpose the dimensions (to front).

        Args:
            dim: dimension to front

        Returns: (XrTypes)
            Transposed ARPES data

        ToDo: Test
        """
        dims = list(self._obj.dims)
        assert dim in dims
        dims.remove(dim)
        return self._obj.transpose(*([dim, *dims]))

    def transpose_to_back(self, dim: str) -> XrTypes:
        """Transpose the dimensions (to back).

        Args:
            dim: dimension to back

        Returns: (XrTypes)
            Transposed ARPES data.

        ToDo: Test
        """
        dims = list(self._obj.dims)
        assert dim in dims
        dims.remove(dim)
        return self._obj.transpose(*([*dims, dim]))

    def select_around_data(
        self,
        points: dict[str, xr.DataArray] | xr.Dataset | dict | tuple[float, ...] | list[float],
        radius: dict[str, float] | float | None = None,  # radius={"phi": 0.005}
        *,
        mode: Literal["sum", "mean"] = "sum",
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """Performs a binned selection around a point or points.

        Can be used to perform a selection along one axis as a function of another, integrating a
        region in the other dimensions.

        Example:
            As an example, suppose we have a dataset with dimensions ('eV', 'kp', 'T',)
            and we also by fitting determined the Fermi momentum as a function of T, kp_F('T'),
            stored in the dataarray kFs. Then we could select momentum integrated EDCs in a small
            window around the fermi momentum for each temperature by using

            >>> edcs = full_data.S.select_around_data({'kp': kFs}, radius={'kp': 0.04}, fast=True)

            The resulting data will be EDCs for each T, in a region of radius 0.04 inverse angstroms
            around the Fermi momentum.

        Args:
            points: The set of points where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                    standard sized selection will be made as a compromise.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"
            kwargs: Can be used to pass radii parameters by keyword with `_r` postfix.

        Returns:
            The binned selection around the desired point or points.
        """
        assert isinstance(
            self._obj,
            xr.DataArray,
        ), "Cannot use select_around on Datasets only DataArrays!"

        assert mode in {"sum", "mean"}, "mode parameter should be either sum or mean."

        if isinstance(points, tuple | list):
            warnings.warn("Dangerous iterable points argument to `select_around`", stacklevel=2)
            points = dict(zip(points, self._obj.dims, strict=True))
        if isinstance(points, xr.Dataset):
            points = {str(k): points[k].item() for k in points.data_vars}
        radius = self._radius(points, radius, **kwargs)
        assert isinstance(radius, dict)

        logger.debug(f"iter(points.values()): {iter(points.values())}")

        along_dims = next(iter(points.values())).dims
        selected_dims = list(points.keys())

        stride = self._obj.G.stride(generic_dim_names=False)

        new_dim_order = [d for d in self._obj.dims if d not in along_dims] + list(along_dims)

        data_for = self._obj.transpose(*new_dim_order)
        new_data = data_for.sum(selected_dims, keep_attrs=True)
        for coord, value in data_for.G.iterate_axis(along_dims):
            nearest_sel_params = {}
            # -- originally, if safe == True, the following liens starting from hear
            for d, v in radius.items():
                if v < stride[d]:
                    nearest_sel_params[d] = points[d].sel(coord)

            radius = {d: v for d, v in radius.items() if d not in nearest_sel_params}
            # -- to heari, but as name said, should be alwayws safe.

            selection_slices = {
                d: slice(
                    points[d].sel(coord) - radius[d],
                    points[d].sel(coord) + radius[d],
                )
                for d in points
                if d in radius
            }
            selected = value.sel(selection_slices)

            if nearest_sel_params:
                selected = selected.sel(nearest_sel_params, method="nearest")

            for d in nearest_sel_params:
                # need to remove the extra dims from coords
                del selected.coords[d]

            if mode == "sum":
                new_data.loc[coord] = selected.sum(list(radius.keys())).values
            elif mode == "mean":
                new_data.loc[coord] = selected.mean(list(radius.keys())).values

        return new_data

    def select_around(
        self,
        points: dict[Hashable, float] | xr.Dataset,
        radius: dict[Hashable, float] | float | None = None,
        *,
        mode: Literal["sum", "mean"] = "sum",
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """Selects and integrates a region around a one dimensional point.

        This method is useful to do a small region integration, especially around
        points on a path of a k-point of interest. See also the companion method
        `select_around_data`.

        If the fast flag is set, we will use the Manhattan norm, i.e. sum over square regions
        rather than ellipsoids, as this is less costly.

        If radii are not set, or provided through kwargs as 'eV_r' or 'phi_r' for instance,
        then we will try to use reasonable default values; buyer beware.

        Args:
            points: The points where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                    standard sized selection will be made as a compromise.
            safe: If true, infills radii with default values. Defaults to `True`.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"
            **kwargs: Can be used to pass radii parameters by keyword with `_r` postfix.

        Returns:
            The binned selection around the desired point or points.
        """
        assert isinstance(
            self._obj,
            xr.DataArray,
        ), "Cannot use select_around on Datasets only DataArrays!"

        assert mode in {"sum", "mean"}, "mode parameter should be either sum or mean."

        if isinstance(points, tuple | list):
            warnings.warn("Dangerous iterable point argument to `select_around`", stacklevel=2)
            points = dict(zip(points, self._obj.dims, strict=True))
        if isinstance(points, xr.Dataset):
            points = {k: points[k].item() for k in points.data_vars}
        logger.debug(f"points: {points}")
        radius = self._radius(points, radius, **kwargs)
        logger.debug(f"radius: {radius}")
        nearest_sel_params = {}

        # -- originally, if safe == True, the following liens starting from hear
        stride = self._obj.G.stride(generic_dim_names=False)
        for d, v in radius.items():
            if v < stride[d]:
                nearest_sel_params[d] = points[d]

        radius = {d: v for d, v in radius.items() if d not in nearest_sel_params}
        # -- to heari, but as name said, should be alwayws safe.

        selection_slices = {
            d: slice(points[d] - radius[d], points[d] + radius[d]) for d in points if d in radius
        }
        selected = self._obj.sel(selection_slices)

        if nearest_sel_params:
            selected = selected.sel(nearest_sel_params, method="nearest")

        for d in nearest_sel_params:
            # need to remove the extra dims from coords
            del selected.coords[d]

        if mode == "sum":
            return selected.sum(list(radius.keys()))
        return selected.mean(list(radius.keys()))

    def _radius(
        self,
        points: dict[Hashable, float],
        radius: float | dict[Hashable, float] | None,
        **kwargs: float,
    ) -> dict[Hashable, float]:
        """Helper function. Generate radius dict.

        When radius is dict form, nothing has been done, essentially.

        Args:
            points (dict[Hashable, float]): Selection point
            radius (dict[Hashable, float] | float | None): radius
            kwargs (float): [TODO:description]

        Returns: dict[Hashable, float]
            radius for selection.
        """
        if isinstance(radius, float):
            radius = {str(d): radius for d in points}
        else:
            collectted_terms = {f"{k}_r" for k in points}.intersection(set(kwargs.keys()))
            if collectted_terms:
                radius = {
                    d: kwargs.get(f"{d}_r", DEFAULT_RADII.get(str(d), UNSPESIFIED)) for d in points
                }
            elif radius is None:
                radius = {d: DEFAULT_RADII.get(str(d), UNSPESIFIED) for d in points}
        assert isinstance(radius, dict)
        return {d: radius.get(str(d), DEFAULT_RADII.get(str(d), UNSPESIFIED)) for d in points}

    def short_history(self, key: str = "by") -> list:
        """Return the short version of history.

        Args:
            key (str): [TODO:description]
        """
        return [h["record"][key] if isinstance(h, dict) else h for h in self.history]

    def _calculate_symmetry_points(
        self,
        symmetry_points: dict[str, list[dict[str, float]]] | dict[str, dict[str, float]],
        epsilon: float = 0.01,
    ) -> tuple[defaultdict[str, list[dict[str, float]]], defaultdict[str, list[dict[str, float]]]]:
        # For each symmetry point, we need to determine if it is projected or not
        # if it is projected, we need to calculate its projected coordinates
        """[TODO:summary].

        Args:
            symmetry_points: [TODO:description]
            epsilon: [TODO:description]
        """
        points = collections.defaultdict(list)
        projected_points = collections.defaultdict(list)

        fixed_coords = {k: v for k, v in self._obj.coords.items() if k not in self._obj.indexes}
        index_coords = self._obj.indexes

        for point, locations in symmetry_points.items():
            location_list = locations if isinstance(locations, list) else [locations]
            for location in location_list:
                # determine whether the location needs to be projected
                projected = False
                skip = False
                for axis_name, value in location.items():
                    if (
                        axis_name in fixed_coords
                        and np.abs(value - fixed_coords[axis_name]) > epsilon
                    ):
                        projected = True
                    if axis_name not in fixed_coords and axis_name not in index_coords:
                        # cannot even hope to do anything here, we don't have enough info
                        skip = True
                if skip:
                    continue
                location.copy()  # <== CHECK ME! Original: new_location = location.copy()
                if projected:
                    # Go and do the projection, for now we will assume we just get it by
                    # replacing the value of the mismatched coordinates.
                    # This does not work if the coordinate system is not orthogonal
                    for axis in location:
                        if axis in fixed_coords:
                            fixed_coords[axis]
                            # <== CHECK ME! Original new_locationn = fixed_coords[axis]
                    projected_points[point].append(location)
                else:
                    points[point].append(location)

        return points, projected_points

    def symmetry_points(
        self,
        *,
        raw: bool = False,
        **kwargs: float,
    ) -> (
        dict[str, dict[str, float]]
        | tuple[defaultdict[str, list[dict[str, float]]], defaultdict[str, list[dict[str, float]]]]
    ):
        """[TODO:summary].

        Args:
            raw (bool): [TODO:description]
            kwargs: pass to _calculate_symmetry_points (epsilon)
        """
        symmetry_points: dict[str, dict[str, float]] = {}
        # An example of "symmetry_points": symmetry_points = {"G": {"phi": 0.405}}
        our_symmetry_points = self._obj.attrs.get("symmetry_points", {})

        symmetry_points.update(our_symmetry_points)

        if raw:
            return symmetry_points

        return self._calculate_symmetry_points(symmetry_points, **kwargs)

    @property
    def iter_own_symmetry_points(self) -> Generator[tuple[str, float], None, None]:
        sym_points, _ = self.symmetry_points()
        return _iter_groups(sym_points)

    @property
    def iter_projected_symmetry_points(self) -> Generator[tuple[str, float], None, None]:
        _, sym_points = self.symmetry_points()
        return _iter_groups(sym_points)

    @property
    def iter_symmetry_points(self) -> Generator[tuple[str, float], None, None]:
        yield from self.iter_own_symmetry_points
        yield from self.iter_projected_symmetry_points

    @property
    def history(self) -> list[PROVENANCE | None]:
        provenance_recorded = self._obj.attrs.get("provenance", None)

        def unlayer(
            prov: PROVENANCE | None | str,
        ) -> tuple[list[PROVENANCE | None], PROVENANCE | str | None]:
            if prov is None:
                return [], None  # tuple[list[Incomplete] | None]
            if isinstance(prov, str):
                return [prov], None
            first_layer: PROVENANCE = copy.copy(prov)

            rest = first_layer.pop("parents_provenance", None)
            if isinstance(rest, list):
                warnings.warn(
                    "Encountered multiple parents in history extraction, "
                    "throwing away all but the first.",
                    stacklevel=2,
                )
                rest = rest[0] if rest else None

            return [first_layer], rest

        def _unwrap_provenance(prov: PROVENANCE | None) -> list[PROVENANCE | None]:
            if prov is None:
                return []

            first, rest = unlayer(
                prov,
            )

            return first + _unwrap_provenance(rest)

        return _unwrap_provenance(provenance_recorded)

    @property
    def spectrometer(self) -> SPECTROMETER:
        ds = self._obj
        if "spectrometer_name" in ds.attrs:
            return arpes.constants.SPECTROMETERS.get(ds.attrs["spectrometer_name"], {})
        if isinstance(ds, xr.Dataset):
            if "up" in ds.data_vars or ds.attrs.get("18  MCP3") == 0:
                return arpes.constants.SPECTROMETERS["SToF"]
        elif isinstance(ds, xr.DataArray) and (ds.name == "up" or ds.attrs.get("18  MCP3") == 0):
            return arpes.constants.SPECTROMETERS["SToF"]
        if "location" in ds.attrs:
            return arpes.constants.SPECTROMETERS.get(ds.attrs["location"], {})
        try:
            return arpes.constants.SPECTROMETERS[ds.attrs["spectrometer_name"]]
        except KeyError:
            return {}

    @property
    def dshape(self) -> dict[Hashable, int]:
        """Return dimension type.

        Examples:
            {"phi": 500, "eV" ,200}
        """
        arr = self._obj
        dim_names = tuple(arr.dims)
        return dict(zip(dim_names, arr.shape, strict=True))

    @property
    def scan_name(self) -> str:
        """[TODO:summary].

        Args:
            self ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        for option in ["scan", "file"]:
            if option in self._obj.attrs:
                return Path(self._obj.attrs[option]).name

        id_code = self._obj.attrs.get("id")

        if id_code is None:
            return "No ID"
        return str(id_code)

    @property
    def label(self) -> str:
        return str(self._obj.attrs.get("description", self.scan_name))

    @contextlib.contextmanager
    def with_rotation_offset(self, offset: float) -> Generator:
        """Temporarily rotates the chi_offset by `offset`.

        Args:
            offset (float): offset value about chi.
        """
        old_chi_offset = self.offsets.get("chi", 0)
        self.apply_offsets({"chi": old_chi_offset + offset})
        yield old_chi_offset + offset
        self.apply_offsets({"chi": old_chi_offset})

    def apply_offsets(self, offsets: dict[ANGLE, float]) -> None:
        assert isinstance(self._obj, xr.Dataset | xr.DataArray)
        for k, v in offsets.items():
            self._obj.attrs[f"{k}_offset"] = v

    @property
    def offsets(self) -> dict[str, float]:
        return {
            str(coord): self.lookup_offset(str(coord))
            for coord in self._obj.coords
            if f"{coord}_offset" in self._obj.attrs
        }

    def lookup_offset_coord(self, name: str) -> xr.DataArray | NDArray[np.float_] | float:
        return self.lookup_coord(name) - self.lookup_offset(name)

    def lookup_coord(self, name: str) -> xr.DataArray | float:
        if name in self._obj.coords:
            return unwrap_xarray_item(self._obj.coords[name])

        msg = f"Could not find coordinate {name}.  Check your endstation module."
        raise ValueError(msg)

    def lookup_offset(self, attr_name: str) -> float:
        symmetry_points = self.symmetry_points(raw=True)
        if "G" in symmetry_points:
            gamma_point = symmetry_points["G"]
            if attr_name in gamma_point:
                return unwrap_xarray_item(gamma_point[attr_name])

        offset_name = attr_name + "_offset"
        if offset_name in self._obj.attrs:
            return unwrap_xarray_item(self._obj.attrs[offset_name])

        return unwrap_xarray_item(self._obj.attrs.get("data_preparation", {}).get(offset_name, 0))

    @property
    def beta_offset(self) -> float:
        return self.lookup_offset("beta")

    @property
    def psi_offset(self) -> float:
        return self.lookup_offset("psi")

    @property
    def theta_offset(self) -> float:
        return self.lookup_offset("theta")

    @property
    def phi_offset(self) -> float:
        return self.lookup_offset("phi")

    @property
    def chi_offset(self) -> float:
        return self.lookup_offset("chi")

    @property
    def work_function(self) -> float:
        """Provides the work function of the sample, if present in metadata.

        Otherwise, uses something approximate.

        Note:
            This "work_function" should *NOT* be used for k-conversion!
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "sample_workfunction" in self._obj.attrs:
            return self._obj.attrs["sample_workfunction"]
        return 4.3

    @property
    def analyzer_work_function(self) -> float:
        """Provides the work function of the analyzer, if present in metadata.

        otherwise, use appropriate

        Note:
            Use this value for k-conversion.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "workfunction" in self._obj.attrs:
            return self._obj.attrs["workfunction"]
        return 4.401

    @property
    def inner_potential(self) -> float:
        """Provides the inner potential, if present in metadata.

        Otherwise, 10 eV is assumed.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "inner_potential" in self._obj.attrs:
            return self._obj.attrs["inner_potential"]
        return 10

    def find_spectrum_energy_edges(self, *, indices: bool = False) -> NDArray[np.float_]:
        """Return energy position corresponding to the spectrum edge.

        Spectrum edge is infection point of the peak.

        Args:
            indices (bool): if True, return the pixel (index) number.

        Returns: NDArray
            Energy position
        """
        assert isinstance(
            self._obj,
            xr.DataArray,
        )  # if self._obj is xr.Dataset, values is  function
        energy_marginal = self._obj.sum([d for d in self._obj.dims if d not in ["eV"]])

        embed_size = 20
        embedded: NDArray[np.float_] = np.ndarray(
            shape=[embed_size, *list(energy_marginal.values.shape)],
        )
        embedded[:] = energy_marginal.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        from skimage import feature

        edges = (
            feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True, low_threshold=0.1) * 1
        )
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.G.stride(generic_dim_names=False)
        return edges * delta["eV"] + self._obj.coords["eV"].values[0]

    def find_spectrum_angular_edges_full(
        self,
        *,
        indices: bool = False,
        energy_division: float = 0.05,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], xr.DataArray]:
        # as a first pass, we need to find the bottom of the spectrum, we will use this
        # to select the active region and then to rebin into course steps in energy from 0
        # down to this region
        # we will then find the appropriate edge for each slice, and do a fit to the edge locations
        energy_edge: NDArray[np.float_] = self.find_spectrum_energy_edges()
        low_edge = np.min(energy_edge) + energy_division
        high_edge = np.max(energy_edge) - energy_division

        if high_edge - low_edge < 3 * energy_division:
            # Doesn't look like the automatic inference of the energy edge was valid
            high_edge = np.max(self._obj.coords["eV"].values)
            low_edge = np.min(self._obj.coords["eV"].values)

        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        energy_cut = self._obj.sel(eV=slice(low_edge, high_edge)).S.sum_other(["eV", angular_dim])

        n_cuts = int(np.ceil((high_edge - low_edge) / energy_division))
        new_shape = {"eV": n_cuts}
        new_shape[angular_dim] = len(energy_cut.coords[angular_dim].values)
        rebinned = rebin(energy_cut, shape=new_shape)

        embed_size = 20
        embedded: NDArray[np.float_] = np.ndarray(
            shape=[embed_size, len(rebinned.coords[angular_dim].values)],
        )
        low_edges = []
        high_edges = []
        for e_cut in rebinned.coords["eV"].values:
            e_slice = rebinned.sel(eV=e_cut)
            values = e_slice.values
            values[values > np.mean(values)] = np.mean(values)
            embedded[:] = values
            embedded = ndi.gaussian_filter(embedded, embed_size / 1.5)

            from skimage import feature

            edges = (
                feature.canny(
                    embedded,
                    sigma=4,
                    use_quantiles=False,
                    low_threshold=0.7,
                    high_threshold=1.5,
                )
                * 1
            )
            edges = np.where(edges[int(embed_size / 2)] == 1)[0]
            low_edges.append(np.min(edges))
            high_edges.append(np.max(edges))

        if indices:
            return np.array(low_edges), np.array(high_edges), rebinned.coords["eV"]

        delta = self._obj.G.stride(generic_dim_names=False)

        return (
            np.array(low_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
            np.array(high_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
            rebinned.coords["eV"],
        )

    def zero_spectrometer_edges(
        self,
        cut_margin: int = 0,
        interp_range: float | None = None,
        low: Sequence[float] | NDArray[np.float_] | None = None,
        high: Sequence[float] | NDArray[np.float_] | None = None,
    ) -> xr.DataArray:
        assert isinstance(self._obj, xr.DataArray)
        if low is not None:
            assert high is not None
            assert len(low) == len(high) == 2  # noqa: PLR2004

            low_edges = low
            high_edges = high

        (
            low_edges,
            high_edges,
            rebinned_eV_coord,
        ) = self.find_spectrum_angular_edges_full(indices=True)

        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        if not cut_margin:
            if "pixel" in self._obj.dims:
                cut_margin = 50
            else:
                cut_margin = int(0.08 / self._obj.G.stride(generic_dim_names=False)[angular_dim])
        elif isinstance(cut_margin, float):
            assert angular_dim == "phi"
            cut_margin = int(
                cut_margin / self._obj.G.stride(generic_dim_names=False)[angular_dim],
            )

        if interp_range is not None:
            low_edge = xr.DataArray(low_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
            high_edge = xr.DataArray(high_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
            low_edge = low_edge.sel(eV=interp_range)
            high_edge = high_edge.sel(eV=interp_range)
        other_dims = list(self._obj.dims)
        other_dims.remove("eV")
        other_dims.remove(angular_dim)
        copied = self._obj.copy(deep=True).transpose(*(["eV", angular_dim, *other_dims]))

        low_edges += cut_margin
        high_edges -= cut_margin

        for i, energy in enumerate(copied.coords["eV"].values):
            index = np.searchsorted(rebinned_eV_coord, energy)
            other = index + 1
            if other >= len(rebinned_eV_coord):
                other = len(rebinned_eV_coord) - 1
                index = len(rebinned_eV_coord) - 2

            low_index = int(np.interp(energy, rebinned_eV_coord, low_edges))
            high_index = int(np.interp(energy, rebinned_eV_coord, high_edges))
            copied.values[i, 0:low_index] = 0
            copied.values[i, high_index:-1] = 0

        return copied

    def sum_other(
        self,
        dim_or_dims: list[str],
        *,
        keep_attrs: bool = False,
    ) -> XrTypes:
        assert isinstance(dim_or_dims, list)

        return self._obj.sum(
            [d for d in self._obj.dims if d not in dim_or_dims],
            keep_attrs=keep_attrs,
        )

    def mean_other(
        self,
        dim_or_dims: list[str] | str,
        *,
        keep_attrs: bool = False,
    ) -> XrTypes:
        if isinstance(dim_or_dims, str):
            dim_or_dims = [dim_or_dims]

        return self._obj.mean(
            [d for d in self._obj.dims if d not in dim_or_dims],
            keep_attrs=keep_attrs,
        )

    def find_spectrum_angular_edges(
        self,
        *,
        indices: bool = False,
    ) -> NDArray[np.float_] | NDArray[np.int_]:
        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        energy_edge = self.find_spectrum_energy_edges()
        energy_slice = slice(np.max(energy_edge) - 0.1, np.max(energy_edge))
        assert isinstance(self._obj, xr.DataArray)
        near_ef = self._obj.sel(eV=energy_slice).sum(
            [d for d in self._obj.dims if d not in [angular_dim]],
        )

        embed_size = 20
        embedded: NDArray[np.float_] = np.ndarray(shape=[embed_size, *list(near_ef.values.shape)])
        embedded[:] = near_ef.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        # try to avoid dependency conflict with numpy v0.16
        from skimage import feature  # pylint: disable=import-error

        edges = (
            feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True, low_threshold=0.2) * 1
        )
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.G.stride(generic_dim_names=False)
        return edges * delta[angular_dim] + self._obj.coords[angular_dim].values[0]

    def wide_angle_selector(self, *, include_margin: bool = True) -> slice:
        edges = self.find_spectrum_angular_edges()
        low_edge, high_edge = np.min(edges), np.max(edges)

        # go and build in a small margin
        if include_margin:
            if "pixels" in self._obj.dims:
                low_edge += 50
                high_edge -= 50
            else:
                low_edge += 0.05
                high_edge -= 0.05

        return slice(low_edge, high_edge)

    def meso_effective_selector(self) -> slice:
        energy_edge = self.find_spectrum_energy_edges()
        return slice(np.max(energy_edge) - 0.3, np.max(energy_edge) - 0.1)

    def region_sel(self, *regions: Incomplete) -> XrTypes:
        def process_region_selector(
            selector: slice | DesignatedRegions,
            dimension_name: str,
        ) -> slice | Callable[..., slice]:
            if isinstance(selector, slice):
                return selector

            options = {
                "eV": (
                    DesignatedRegions.ABOVE_EF,
                    DesignatedRegions.BELOW_EF,
                    DesignatedRegions.EF_NARROW,
                    DesignatedRegions.MESO_EF,
                    DesignatedRegions.MESO_EFFECTIVE_EF,
                    DesignatedRegions.ABOVE_EFFECTIVE_EF,
                    DesignatedRegions.BELOW_EFFECTIVE_EF,
                    DesignatedRegions.EFFECTIVE_EF_NARROW,
                ),
                "phi": (
                    DesignatedRegions.NARROW_ANGLE,
                    DesignatedRegions.WIDE_ANGLE,
                    DesignatedRegions.TRIM_EMPTY,
                ),
            }

            options_for_dim = options.get(dimension_name, list(DesignatedRegions))
            assert selector in options_for_dim

            # now we need to resolve out the region
            resolution_methods = {
                DesignatedRegions.ABOVE_EF: slice(0, None),
                DesignatedRegions.BELOW_EF: slice(None, 0),
                DesignatedRegions.EF_NARROW: slice(-0.1, 0.1),
                DesignatedRegions.MESO_EF: slice(-0.3, -0.1),
                DesignatedRegions.MESO_EFFECTIVE_EF: self.meso_effective_selector,
                # Implement me
                # DesignatedRegions.TRIM_EMPTY: ,
                DesignatedRegions.WIDE_ANGLE: self.wide_angle_selector,
                # DesignatedRegions.NARROW_ANGLE: self.narrow_angle_selector,
            }
            resolution_method = resolution_methods[selector]
            if isinstance(resolution_method, slice):
                return resolution_method
            if callable(resolution_method):
                return resolution_method()

            msg = "Unable to determine resolution method."
            raise NotImplementedError(msg)

        obj = self._obj

        def unpack_dim(dim_name: str) -> str:
            if dim_name == "angular":
                return "pixel" if "pixel" in obj.dims else "phi"

            return dim_name

        for region in regions:
            region = {unpack_dim(k): v for k, v in normalize_region(region).items()}

            # remove missing dimensions from selection for permissiveness
            # and to transparent composing of regions
            region = {k: process_region_selector(v, k) for k, v in region.items() if k in obj.dims}
            obj = obj.sel(**region)

        return obj

    def fat_sel(
        self,
        widths: dict[str, Any] | None = None,
        **kwargs: Incomplete,
    ) -> XrTypes:
        """Allows integrating a selection over a small region.

        The produced dataset will be normalized by dividing by the number
        of slices integrated over.

        This can be used to produce temporary datasets that have reduced
        uncorrelated noise.

        Args:
            widths: Override the widths for the slices. Reasonable defaults are used otherwise.
                    Defaults to None.
            kwargs: slice dict. Has the same function as xarray.DataArray.sel

        Returns:
            The data after selection.
        """
        if widths is None:
            widths = {}
        assert isinstance(widths, dict)
        default_widths = {
            "eV": 0.05,
            "phi": 2,
            "beta": 2,
            "theta": 2,
            "kx": 0.02,
            "ky": 0.02,
            "kp": 0.02,
            "kz": 0.1,
        }

        extra_kwargs = {k: v for k, v in kwargs.items() if k not in self._obj.dims}
        slice_kwargs = {k: v for k, v in kwargs.items() if k not in extra_kwargs}
        slice_widths = {
            k: widths.get(k, extra_kwargs.get(k + "_width", default_widths.get(k)))
            for k in slice_kwargs
        }
        slices = {
            k: slice(v - slice_widths[k] / 2, v + slice_widths[k] / 2)
            for k, v in slice_kwargs.items()
        }

        sliced = self._obj.sel(slices)  # Need check.  "**" should not be required.
        thickness = np.prod([len(sliced.coords[k]) for k in slice_kwargs])
        normalized = sliced.sum(slices.keys(), keep_attrs=True) / thickness
        for k, v in slices.items():
            normalized.coords[k] = (v.start + v.stop) / 2
        normalized.attrs.update(self._obj.attrs.copy())
        return normalized

    @property
    def reference_settings(self) -> dict[str, Any]:
        settings = self.spectrometer_settings or {}

        settings.update(
            {
                "hv": self.hv,
            },
        )

        return settings

    @property
    def beamline_settings(self) -> dict[str, Any]:
        find_keys = {
            "entrance_slit": {
                "entrance_slit",
            },
            "exit_slit": {
                "exit_slit",
            },
            "hv": {
                "hv",
                "photon_energy",
            },
            "grating": {},
        }
        settings = {}
        for key, options in find_keys.items():
            for option in options:
                if option in self._obj.attrs:
                    settings[key] = self._obj.attrs[option]
                    break

        return settings

    @property
    def spectrometer_settings(self) -> dict[str, Any]:
        find_keys = {
            "lens_mode": {
                "lens_mode",
            },
            "pass_energy": {
                "pass_energy",
            },
            "scan_mode": {
                "scan_mode",
            },
            "scan_region": {
                "scan_region",
            },
            "slit": {
                "slit",
                "slit_plate",
            },
        }
        settings = {}
        for key, options in find_keys.items():
            for option in options:
                if option in self._obj.attrs:
                    settings[key] = self._obj.attrs[option]
                    break

        if isinstance(settings.get("slit"), float):
            settings["slit"] = int(round(settings["slit"]))

        return settings

    @property
    def sample_pos(self) -> tuple[float, float, float]:
        x, y, z = None, None, None
        with contextlib.suppress(KeyError):
            x = self._obj.attrs["x"]

        with contextlib.suppress(KeyError):
            y = self._obj.attrs["y"]

        with contextlib.suppress(KeyError):
            z = self._obj.attrs["z"]

        def do_float(w: float | None) -> float:
            return float(w) if w is not None else np.nan

        return (do_float(x), do_float(y), do_float(z))

    @property
    def sample_angles(
        self,
    ) -> tuple[
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
    ]:
        """Returns angle information.

        Returns:
        -------
        tuple[xr.DataArray | float, ...]
            beta, theta, chi, phi, psi, alpha
        """
        return (
            # manipulator
            self.lookup_coord("beta"),
            self.lookup_coord("theta"),
            self.lookup_coord("chi"),
            # analyzer
            self.lookup_coord("phi"),
            self.lookup_coord("psi"),
            self.lookup_coord("alpha"),
        )

    @property
    def full_coords(self) -> dict[str, float | xr.DataArray]:
        """[TODO:summary].

        Args:
            self ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        full_coords: dict[str, float | xr.DataArray] = {}

        full_coords.update(dict(zip(["x", "y", "z"], self.sample_pos, strict=True)))
        full_coords.update(
            dict(
                zip(
                    ["beta", "theta", "chi", "phi", "psi", "alpha"],
                    self.sample_angles,
                    strict=True,
                ),
            ),
        )
        full_coords.update(
            {
                "hv": self.hv,
            },
        )

        full_coords.update(self._obj.coords)
        return full_coords

    @property
    def sample_info(self) -> SAMPLEINFO:
        """Return sample info property.

        Returns (dict):
        """
        sample_info: SAMPLEINFO = {
            "id": self._obj.attrs.get("sample_id"),
            "sample_name": self._obj.attrs.get("sample_name"),
            "source": self._obj.attrs.get("sample_source"),
            "reflectivity": self._obj.attrs.get("sample_reflectivity", np.nan),
        }
        return sample_info

    @property
    def scan_info(self) -> SCANINFO:
        scan_info: SCANINFO = {
            "time": self._obj.attrs.get("time"),
            "date": self._obj.attrs.get("date"),
            "type": self.scan_type,
            "spectrum_type": self.spectrum_type,
            "experimenter": self._obj.attrs.get("experimenter"),
            "sample": self._obj.attrs.get("sample_name"),
        }
        return scan_info

    @property
    def experiment_info(self) -> EXPERIMENTINFO:
        """Return experiment info property."""
        experiment_info: EXPERIMENTINFO = {
            "temperature": self.temp,
            "temperature_cryotip": self._obj.attrs.get("temperature_cryotip", np.nan),
            "pressure": self._obj.attrs.get("pressure", np.nan),
            "polarization": self.probe_polarization,
            "photon_flux": self._obj.attrs.get("photon_flux", np.nan),
            "photocurrent": self._obj.attrs.get("photocurrent", np.nan),
            "probe": self._obj.attrs.get("probe"),
            "probe_detail": self._obj.attrs.get("probe_detail"),
            "analyzer": self._obj.attrs.get("analyzer"),
            "analyzer_detail": self.analyzer_detail,
        }
        return experiment_info

    @property
    def pump_info(self) -> LIGHTSOURCEINFO:
        """Return pump info property."""
        pump_info: LIGHTSOURCEINFO = {
            "pump_wavelength": self._obj.attrs.get("pump_wavelength", np.nan),
            "pump_energy": self._obj.attrs.get("pump_energy", np.nan),
            "pump_fluence": self._obj.attrs.get("pump_fluence", np.nan),
            "pump_pulse_energy": self._obj.attrs.get("pump_pulse_energy", np.nan),
            "pump_spot_size": (
                self._obj.attrs.get("pump_spot_size_x", np.nan),
                self._obj.attrs.get("pump_spot_size_y", np.nan),
            ),
            "pump_profile": self._obj.attrs.get("pump_profile"),
            "pump_linewidth": self._obj.attrs.get("pump_linewidth", np.nan),
            "pump_duration": self._obj.attrs.get("pump_duration", np.nan),
            "pump_polarization": self.pump_polarization,
        }
        return pump_info

    @property
    def probe_info(self) -> LIGHTSOURCEINFO:
        """Return probe info property.

        Returns (LIGHTSOURCEINFO):
        """
        probe_info: LIGHTSOURCEINFO = {
            "probe_wavelength": self._obj.attrs.get("probe_wavelength", np.nan),
            "probe_energy": self.hv,
            "probe_fluence": self._obj.attrs.get("probe_fluence", np.nan),
            "probe_pulse_energy": self._obj.attrs.get("probe_pulse_energy", np.nan),
            "probe_spot_size": (
                self._obj.attrs.get("probe_spot_size_x", np.nan),
                self._obj.attrs.get("probe_spot_size_y", np.nan),
            ),
            "probe_profile": self._obj.attrs.get("probe_profile"),
            "probe_linewidth": self._obj.attrs.get("probe_linewidth", np.nan),
            "probe_duration": self._obj.attrs.get("probe_duration", np.nan),
            "probe_polarization": self.probe_polarization,
        }
        return probe_info

    @property
    def laser_info(self) -> LIGHTSOURCEINFO:
        return {
            **self.probe_info,
            **self.pump_info,
            "repetition_rate": self._obj.attrs.get("repetition_rate", np.nan),
        }

    @property
    def analyzer_info(self) -> ANALYZERINFO:
        """General information about the photoelectron analyzer used."""
        analyzer_info: ANALYZERINFO = {
            "lens_mode": self._obj.attrs.get("lens_mode"),
            "lens_mode_name": self._obj.attrs.get("lens_mode_name"),
            "acquisition_mode": self._obj.attrs.get("acquisition_mode"),
            "pass_energy": self._obj.attrs.get("pass_energy", np.nan),
            "slit_shape": self._obj.attrs.get("slit_shape"),
            "slit_width": self._obj.attrs.get("slit_width", np.nan),
            "slit_number": self._obj.attrs.get("slit_number"),
            "lens_table": self._obj.attrs.get("lens_table"),
            "analyzer_type": self._obj.attrs.get("analyzer_type"),
            "mcp_voltage": self._obj.attrs.get("mcp_voltage", np.nan),
            "work_function": self._obj.attrs.get("workfunction", 4.401),
        }
        return analyzer_info

    @property
    def daq_info(self) -> DAQINFO:
        """General information about the acquisition settings for an ARPES experiment."""
        daq_info: DAQINFO = {
            "daq_type": self._obj.attrs.get("daq_type"),
            "region": self._obj.attrs.get("daq_region"),
            "region_name": self._obj.attrs.get("daq_region_name"),
            "center_energy": self._obj.attrs.get("daq_center_energy"),
            "prebinning": self.prebinning,
            "trapezoidal_correction_strategy": self._obj.attrs.get(
                "trapezoidal_correction_strategy",
            ),
            "dither_settings": self._obj.attrs.get("dither_settings"),
            "sweep_settings": self.sweep_settings,
            "frames_per_slice": self._obj.attrs.get("frames_per_slice", np.nan),
            "frame_duration": self._obj.attrs.get("frame_duration", np.nan),
        }
        return daq_info

    @property
    def beamline_info(self) -> LIGHTSOURCEINFO:
        """Information about the beamline or light source used for a measurement."""
        beamline_info: LIGHTSOURCEINFO = {
            "hv": self.hv,
            "linewidth": self._obj.attrs.get("probe_linewidth", np.nan),
            "photon_polarization": self.probe_polarization,
            "undulator_info": self.undulator_info,
            "repetition_rate": self._obj.attrs.get("repetition_rate", np.nan),
            "beam_current": self._obj.attrs.get("beam_current", np.nan),
            "entrance_slit": self._obj.attrs.get("entrance_slit"),
            "exit_slit": self._obj.attrs.get("exit_slit"),
            "monochromator_info": self.monochromator_info,
        }
        return beamline_info

    @property
    def sweep_settings(self) -> dict[str, xr.DataArray | NDArray[np.float_] | float | None]:
        """For datasets acquired with swept acquisition settings, provides those settings."""
        return {
            "high_energy": self._obj.attrs.get("sweep_high_energy"),
            "low_energy": self._obj.attrs.get("sweep_low_energy"),
            "n_sweeps": self._obj.attrs.get("n_sweeps"),
            "step": self._obj.attrs.get("sweep_step"),
        }

    @property
    def probe_polarization(self) -> tuple[float, float]:
        """Provides the probe polarization of the UV/x-ray source."""
        return (
            self._obj.attrs.get("probe_polarization_theta", np.nan),
            self._obj.attrs.get("probe_polarization_alpha", np.nan),
        )

    @property
    def pump_polarization(self) -> tuple[float, float]:
        """For Tr-ARPES experiments, provides the pump polarization."""
        return (
            self._obj.attrs.get("pump_polarization_theta", np.nan),
            self._obj.attrs.get("pump_polarization_alpha", np.nan),
        )

    @property
    def prebinning(self) -> dict[str, Any]:
        """Information about the prebinning performed during scan acquisition."""
        prebinning = {}
        for d in self._obj.indexes:
            if f"{d}_prebinning" in self._obj.attrs:
                prebinning[d] = self._obj.attrs[f"{d}_prebinning"]

        return prebinning

    @property
    def monochromator_info(self) -> dict[str, float]:
        """Details about the monochromator used on the UV/x-ray source."""
        return {
            "grating_lines_per_mm": self._obj.attrs.get("grating_lines_per_mm", np.nan),
        }

    @property
    def undulator_info(self) -> dict[str, str | float | None]:
        """Details about the undulator for data performed at an undulator source."""
        return {
            "gap": self._obj.attrs.get("undulator_gap"),
            "z": self._obj.attrs.get("undulator_z"),
            "harmonic": self._obj.attrs.get("undulator_harmonic"),
            "polarization": self._obj.attrs.get("undulator_polarization"),
            "type": self._obj.attrs.get("undulator_type"),
        }

    @property
    def analyzer_detail(self) -> dict[str, str | float | None]:
        """Details about the analyzer, its capabilities, and metadata."""
        return {
            "name": self._obj.attrs.get("analyzer_name"),
            "parallel_deflectors": self._obj.attrs.get("parallel_deflectors"),
            "perpendicular_deflectors": self._obj.attrs.get("perpendicular_deflectors"),
            "type": self._obj.attrs.get("analyzer_type"),
            "radius": self._obj.attrs.get("analyzer_radius"),
        }

    @property
    def temp(self) -> float | Literal["RT", "LT"]:
        """The temperature at which an experiment was performed."""
        prefered_attrs = [
            "TA",
            "ta",
            "t_a",
            "T_A",
            "T_1",
            "t_1",
            "t1",
            "T1",
            "temp",
            "temp_sample",
            "temperature",
            "temp_cryotip",
            "temperature_sensor_b",
            "temperature_sensor_a",
            "temperature_cryotip",
        ]
        for attr in prefered_attrs:
            if attr in self._obj.attrs:
                return self._obj.attrs[attr]
        msg = "Could not read temperature off any standard attr"
        warnings.warn(msg, stacklevel=2)
        return np.nan

    def generic_fermi_surface(self, fermi_energy: float) -> XrTypes:
        return self.fat_sel(eV=fermi_energy, method="nearest")

    @property
    def fermi_surface(self) -> XrTypes:
        return self.fat_sel(eV=0, method="nearest")

    def __init__(self, xarray_obj: XrTypes) -> None:
        self._obj = xarray_obj

    @staticmethod
    def dict_to_html(d: Mapping[str, float | str]) -> str:
        return """
        <table>
          <thead>
            <tr>
              <th>Key</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(
            rows="".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items()]),
        )

    def _repr_html_full_coords(
        self,
        coords: dict[str, float | xr.DataArray],
    ) -> str:
        significant_coords = {}
        for k, v in coords.items():
            if v is None:
                continue
            if np.any(np.isnan(v)):
                continue
            significant_coords[k] = v

        def coordinate_dataarray_to_flat_rep(
            value: xr.DataArray | float,
        ) -> str | float:
            if not isinstance(value, xr.DataArray):
                return value
            if len(value.dims) == 0:
                tmp = "<span>{var:.5g}</span>"
                return tmp.format(var=value.values)
            tmp = "<span>{min:.3g}<strong> to </strong>{max:.3g}"
            tmp += "<strong> by </strong>{delta:.3g}</span>"
            return tmp.format(
                min=value.min().item(),
                max=value.max().item(),
                delta=value.values[1] - value.values[0],
            )

        return ARPESAccessorBase.dict_to_html(
            {k: coordinate_dataarray_to_flat_rep(v) for k, v in significant_coords.items()},
        )

    def _repr_html_spectrometer_info(self) -> str:
        skip_keys = {
            "dof",
        }
        ordered_settings = OrderedDict(self.spectrometer_settings)
        ordered_settings.update({k: v for k, v in self.spectrometer.items() if k not in skip_keys})

        return ARPESAccessorBase.dict_to_html(ordered_settings)

    @staticmethod
    def _repr_html_experimental_conditions(conditions: EXPERIMENTINFO) -> str:
        """Return the experimental conditions with html format.

        Args:
            conditions (EXPERIMENTINFO): self.confitions is usually used.

        Returns (str):
            html representation of the experimental conditions.
        """

        def _experimentalinfo_to_dict(conditions: EXPERIMENTINFO) -> dict[str, str]:
            transformed_dict = {}
            for k, v in conditions.items():
                if k == "polarrization":
                    assert isinstance(v, (float | str))
                    transformed_dict[k] = {
                        "p": "Linear Horizontal",
                        "s": "Linear Vertical",
                        "rc": "Right Circular",
                        "lc": "Left Circular",
                        "s-p": "Linear Dichroism",
                        "p-s": "Linear Dichroism",
                        "rc-lc": "Circular Dichroism",
                        "lc-rc": "Circular Dichroism",
                    }.get(str(v), str(v))
                if k == "temp":
                    if isinstance(v, float) and not np.isnan(v):
                        transformed_dict[k] = f"{v} Kelvin"
                    elif isinstance(v, str):
                        transformed_dict[k] = v
                if k == "hv":
                    if isinstance(v, xr.DataArray):
                        min_hv = float(v.min())
                        max_hv = float(v.max())
                        transformed_dict[k] = (
                            f"<strong> from </strong> {min_hv} <strong>  to </strong> {max_hv} eV"
                        )
                    elif isinstance(v, float) and not np.isnan(v):
                        transformed_dict[k] = f"{v} eV"
            return transformed_dict

        transformed_dict = _experimentalinfo_to_dict(conditions)
        return ARPESAccessorBase.dict_to_html(transformed_dict)

    def _repr_html_(self) -> str:
        """Return html representation of ARPES data.

        Returns:
            html representation.
        """
        skip_data_vars = {
            "time",
        }

        if isinstance(self._obj, xr.Dataset):
            to_plot = [str(k) for k in self._obj.data_vars if k not in skip_data_vars]
            to_plot = [str(k) for k in to_plot if 1 <= len(self._obj[k].dims) < 3]  # noqa: PLR2004
            to_plot = to_plot[:5]

            if to_plot:
                _, ax = plt.subplots(
                    1,
                    len(to_plot),
                    figsize=(len(to_plot) * 3, 3),
                )
                if len(to_plot) == 1:
                    ax = [ax]

                for i, plot_var in enumerate(to_plot):
                    self._obj[plot_var].T.plot(ax=ax[i])
                    fancy_labels(ax[i])
                    ax[i].set_title(plot_var.replace("_", " "))

                remove_colorbars()

        elif 1 <= len(self._obj.dims) < 3:  # noqa: PLR2004
            _, ax = plt.subplots(1, 1, figsize=(4, 3))
            self._obj.T.plot(ax=ax)
            fancy_labels(ax)
            ax.set_title("")

            remove_colorbars()
        wrapper_style = 'style="display: flex; flex-direction: row;"'

        if "id" in self._obj.attrs:
            name = "ID: " + str(self._obj.attrs["id"])[:9] + "..."
        else:
            name = "No name"

        warning = ""

        if len(self._obj.attrs) < 10:  # noqa: PLR2004
            warning = ':  <span style="color: red;">Few Attributes, Data Is Summed?</span>'

        return f"""
        <header><strong>{name}{warning}</strong></header>
        <div {wrapper_style}>
        <details open>
            <summary>Experimental Conditions</summary>
            {self._repr_html_experimental_conditions(self.experimental_conditions)}
        </details>
        <details open>
            <summary>Full Coordinates</summary>
            {self._repr_html_full_coords(self.full_coords)}
        </details>
        <details open>
            <summary>Spectrometer</summary>
            {self._repr_html_spectrometer_info()}
        </details>
        </div>
        """

    @property
    def angle_unit(self) -> Literal["Degrees", "Radians"]:
        return self._obj.attrs.get("angle_unit", "Radians")

    @angle_unit.setter
    def angle_unit(self, angle_unit: Literal["Degrees", "Radians"]) -> None:
        """Set "angle unit".

        Angle unit should be "Degrees" or "Radians"

        Args:
            self ([TODO:type]): [TODO:description]
            angle_unit: [TODO:description]
        """
        assert angle_unit in (
            "Degrees",
            "Radians",
        ), "Angle unit should be 'Degrees' or 'Radians'"
        self._obj.attrs["angle_unit"] = angle_unit

    def swap_angle_unit(self) -> None:
        """Swap angle unit (radians <-> degrees).

        Change the value of angle related objects/variables in attrs and coords
        """
        if self.angle_unit == "Radians" or self.angle_unit.startswith("rad"):
            self._radian_to_degree()
        elif self.angle_unit == "Degrees" or self.angle_unit.startswith("deg"):
            self._degree_to_radian()
        else:
            msg = 'The angle_unit must be "Radians" or "Degrees"'
            raise TypeError(msg)

    def _radian_to_degree(self) -> None:
        """A Helper function for swap_angle_unit.

        Degree -> Radian
        """
        self.angle_unit = "Degrees"
        for angle in ANGLE_VARS:
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = np.rad2deg(self._obj.attrs.get(angle, np.nan))
            if angle + "_offset" in self._obj.attrs:
                self._obj.attrs[angle + "_offset"] = np.rad2deg(
                    self._obj.attrs.get(angle + "_offset", np.nan),
                )
            if angle in self._obj.coords:
                self._obj.coords[angle] = np.rad2deg(self._obj.coords[angle])

    def _degree_to_radian(self) -> None:
        """A Helper function for swan_angle_unit.

        Radian -> Degree
        """
        self.angle_unit = "Radians"
        for angle in ANGLE_VARS:
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = np.deg2rad(self._obj.attrs.get(angle, np.nan))
            if angle + "_offset" in self._obj.attrs:
                self._obj.attrs[angle + "_offset"] = np.deg2rad(
                    self._obj.attrs.get(angle + "_offset", np.nan),
                )
            if angle in self._obj.coords:
                self._obj.coords[angle] = np.deg2rad(self._obj.coords[angle])


@xr.register_dataarray_accessor("S")
class ARPESDataArrayAccessor(ARPESAccessorBase):
    """Spectrum related accessor for `xr.DataArray`."""

    def plot(
        self,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> None:
        """Utility delegate to `xr.DataArray.plot` which rasterizes`.

        Args:
            rasterized (bool): if True, rasterized (Not vector) drawing
            args: Pass to xr.DataArray.plot
            kwargs: Pass to xr.DataArray.plot
        """
        if len(self._obj.dims) == TWO_DIMENSION and "rasterized" not in kwargs:
            kwargs["rasterized"] = True
        with plt.rc_context(rc={"text.usetex": False}):
            self._obj.plot(*args, **kwargs)

    def show(self, *, detached: bool = False, **kwargs: Incomplete) -> None:
        """Opens the Qt based image tool."""
        from .plotting.qt_tool import qt_tool

        qt_tool(self._obj, detached=detached, **kwargs)

    def fs_plot(
        self,
        pattern: str = "{}.png",
        **kwargs: Incomplete,
    ) -> Path | None | tuple[Figure, Axes]:
        """Provides a reference plot of the approximate Fermi surface."""
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{self.label}_fs")
            kwargs["out"] = out
        return labeled_fermi_surface(self._obj, **kwargs)

    def fermi_edge_reference_plot(
        self,
        pattern: str = "{}.png",
        **kwargs: str | Normalize | None,
    ) -> Path | None:
        """Provides a reference plot for a Fermi edge reference.

        Args:
            pattern ([TODO:type]): [TODO:description]
            kwargs: pass to plotting.fermi_edge.fermi_edge_reference

        Returns:
            [TODO:description]
        """
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{self.label}_fermi_edge_reference")
            kwargs["out"] = out

        return fermi_edge_reference(self._obj, **kwargs)

    def _referenced_scans_for_spatial_plot(
        self,
        *,
        use_id: bool = True,
        pattern: str = "{}.png",
        out: str | bool = "",
    ) -> Path | tuple[Figure, NDArray[np.object_]]:
        """[TODO:summary].

        Args:
            use_id ([TODO:type]): [TODO:description]
            pattern ([TODO:type]): [TODO:description]
            out (str|bool): if str, Path for output figure. if True,
                the file name is automatically set. If False/"", no output is given.
        """
        label = self._obj.attrs["id"] if use_id else self.label
        if isinstance(out, bool) and out is True:
            out = pattern.format(f"{label}_reference_scan_fs")
        elif isinstance(out, bool) and out is False:
            out = ""

        return reference_scan_spatial(self._obj, out=out)

    def _referenced_scans_for_map_plot(
        self,
        pattern: str = "{}.png",
        *,
        use_id: bool = True,
        **kwargs: Unpack[LabeledFermiSurfaceParam],
    ) -> Path | Axes:
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{label}_reference_scan_fs")
            kwargs["out"] = out

        return reference_scan_fermi_surface(self._obj, **kwargs)

    class HvRefScanParam(LabeledFermiSurfaceParam):
        """Parameter for hf_ref_scan."""

        e_cut: float
        bkg_subtraction: float

    def _referenced_scans_for_hv_map_plot(
        self,
        pattern: str = "{}.png",
        *,
        use_id: bool = True,
        **kwargs: IncompleteMPL,
    ) -> Path | None:
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{label}_hv_reference_scan")
            out = f"{label}_hv_reference_scan.png"
            kwargs["out"] = out

        return hv_reference_scan(self._obj, **kwargs)

    def _simple_spectrum_reference_plot(
        self,
        *,
        use_id: bool = True,
        pattern: str = "{}.png",
        **kwargs: IncompleteMPL,
    ) -> Axes | Path:
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{label}_spectrum_reference")
            kwargs["out"] = out

        return fancy_dispersion(self._obj, **kwargs)

    def cut_nan_coords(self) -> XrTypes:
        """Selects data where coordinates are not `nan`.

        Returns (xr.DataArray):
            The subset of the data where coordinates are not `nan`.
        """
        slices = {}
        for cname, cvalue in self._obj.coords.items():
            try:
                end_ind = np.where(np.isnan(cvalue.values))[0][0]
                end_ind = None if end_ind == -1 else end_ind
                slices[cname] = slice(None, end_ind)
            except IndexError:
                pass
        return self._obj.isel(slices)

    def reference_plot(
        self,
        **kwargs: Unpack[LabeledFermiSurfaceParam] | Unpack[PColorMeshKwargs],
    ) -> Axes | Path:
        """Generates a reference plot for this piece of data according to its spectrum type.

        Args:
            kwargs: pass to referenced_scans_for_**

        Raises:
            NotImplementedError: If there is no standard approach for plotting this data.

        Returns:
            The axes which were used for plotting.
        """
        if self.spectrum_type == "map":
            return self._referenced_scans_for_map_plot(**kwargs)
        if self.spectrum_type == "hv_map":
            return self._referenced_scans_for_hv_map_plot(**kwargs)
        if self.spectrum_type == "cut":
            return self._simple_spectrum_reference_plot(**kwargs)
        if self.spectrum_type in {"ucut", "spem"}:
            return self._referenced_scans_for_spatial_plot(**kwargs)
        raise NotImplementedError

    @property
    def energy_notation(self) -> EnergyNotation:
        """Returns the energy notation ("Binding" energy or "Kinetic" energy).

        Note: The "Kinetic" energy refers to the Fermi level.  (not Vacuum level)
        """
        if "energy_notation" in self._obj.attrs:
            if self._obj.attrs["energy_notation"] in (
                "Kinetic",
                "kinetic",
                "kinetic energy",
            ):
                self._obj.attrs["energy_notation"] = "Kinetic"
                return "Kinetic"
            return "Binding"
        self._obj.attrs["energy_notation"] = self._obj.attrs.get("energy_notation", "Binding")
        return "Binding"

    def switch_energy_notation(self, nonlinear_order: int = 1) -> None:
        """Switch the energy notation between binding and kinetic.

        Args:
            nonlinear_order (int): order of the nonliniarity, default to 1
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if self._obj.coords["hv"].ndim == 0:
            if self.energy_notation == "Binding":
                self._obj.coords["eV"] = (
                    self._obj.coords["eV"] + nonlinear_order * self._obj.coords["hv"]
                )
                self._obj.attrs["energy_notation"] = "Kinetic"
            elif self.energy_notation == "Kinetic":
                self._obj.coords["eV"] = (
                    self._obj.coords["eV"] - nonlinear_order * self._obj.coords["hv"]
                )
                self._obj.attrs["energy_notation"] = "Binding"
        else:
            msg = "Not impremented yet."
            raise RuntimeError(msg)

    def corrected_angle_by(
        self,
        angle_for_correction: Literal[
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        ],
    ) -> xr.DataArray:
        """Return xr.DataArray corrected angle by "angle_for_correction".

        if "angle_for_correction" is like "'angle'_offset", the 'angle' corrected by the
        'angle'_offset value. if "angle_for_correction" is "beta" or "theta", the angle "phi" or
        "psi" is shifted.

        Args:
            angle_for_correction(str): should be one of "alpha_offset", "beta_offset",
                                       "chi_offset", "phi_offset", "psi_offset", "theta_offset",
                                       "beta", "theta"

        Returns:
            xr.DataArray
        """
        assert angle_for_correction in (
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        )
        assert angle_for_correction in self._obj.attrs
        arr: xr.DataArray = self._obj.copy(deep=True)
        arr.S.correct_angle_by(angle_for_correction)
        return arr

    def correct_angle_by(
        self,
        angle_for_correction: Literal[
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        ],
    ) -> None:
        """Angle correction in place.

        if "angle_for_correction" is like "'angle'_offset", the 'angle' corrected by the
        'angle'_offset value. if "angle_for_correction" is "beta" or "theta", the angle "phi" or
        "psi" is shifted.

        Args:
            angle_for_correction (str): should be one of "alpha_offset", "beta_offset",
                                        "chi_offset", "phi_offset", "psi_offset", "theta_offset",
                                        "beta", "theta"
        """
        assert angle_for_correction in (
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        )
        assert angle_for_correction in self._obj.attrs
        if "_offset" in angle_for_correction:
            angle = angle_for_correction.split("_")[0]
            if angle in self._obj.coords:
                self._obj.coords[angle] = (
                    self._obj.coords[angle] - self._obj.attrs[angle_for_correction]
                )
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = (
                    self._obj.attrs[angle] - self._obj.attrs[angle_for_correction]
                )
            self._obj.attrs[angle_for_correction] = 0
            return
        if angle_for_correction == "beta" and self._obj.S.is_slit_vertical:
            self._obj.coords["phi"] = (
                self._obj.coords["phi"] + self._obj.attrs[angle_for_correction]
            )
            self._obj.coords[angle_for_correction] = 0
            self._obj.attrs[angle_for_correction] = 0
            return
        if (
            angle_for_correction == "beta"
            and not self._obj.S.is_slit_vertical
            and "psi" in self._obj.dims
        ):
            self._obj.coords["psi"] = (
                self._obj.coords["psi"] + self._obj.attrs[angle_for_correction]
            )
            self._obj.coords[angle_for_correction] = 0
            self._obj.attrs[angle_for_correction] = 0
            return
        if (
            angle_for_correction == "theta"
            and self._obj.S.is_slit_vertical
            and "psi" in self._obj.dims
        ):
            self._obj.coords["psi"] = (
                self._obj.coords["psi"] + self._obj.attrs[angle_for_correction]
            )
            self._obj.coords[angle_for_correction] = 0
            self._obj.attrs[angle_for_correction] = 0
            return
        if angle_for_correction == "theta" and not self._obj.S.is_slit_vertical:
            self._obj.coords["phi"] = (
                self._obj.coords["phi"] + self._obj.attrs[angle_for_correction]
            )
            self._obj.coords[angle_for_correction] = 0
            self._obj.attrs[angle_for_correction] = 0
            return
        msg = f"No change about the angle by {angle_for_correction}"
        warnings.warn(msg, stacklevel=2)
        return


NORMALIZED_DIM_NAMES = ["x", "y", "z", "w"]


@xr.register_dataset_accessor("G")
@xr.register_dataarray_accessor("G")
class GenericAccessorTools:
    _obj: XrTypes

    def round_coordinates(
        self,
        coords: dict[str, list[float] | NDArray[np.float_]],
        *,
        as_indices: bool = False,
    ) -> dict:
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        data = self._obj
        rounded = {
            k: v.item() for k, v in data.sel(coords, method="nearest").coords.items() if k in coords
        }

        if as_indices:
            rounded = {k: data.coords[k].index(v) for k, v in rounded.items()}

        return rounded

    def argmax_coords(self) -> dict[Hashable, float]:
        """Return dict representing the position for maximum value."""
        assert isinstance(self._obj, xr.DataArray)
        data: xr.DataArray = self._obj
        raveled_idx: int = data.argmax(None).item()
        flat_indices = np.unravel_index(raveled_idx, data.values.shape)
        return {d: data.coords[d][flat_indices[i]].item() for i, d in enumerate(data.dims)}

    def apply_over(
        self,
        fn: Callable,
        *,
        copy: bool = True,
        **selections: Incomplete,
    ) -> XrTypes:
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        data = self._obj

        if copy:
            data = data.copy(deep=True)

        try:
            transformed = fn(data.sel(**selections))
        except TypeError:
            transformed = fn(data.sel(**selections).values)

        if isinstance(transformed, xr.DataArray):
            transformed = transformed.values

        data.loc[selections] = transformed
        return data

    def to_unit_range(self, percentile: float | None = None) -> XrTypes:
        assert isinstance(self._obj, xr.DataArray)  # to work with np.percentile
        if percentile is None:
            norm = self._obj - self._obj.min()
            return norm / norm.max()

        percentile = min(percentile, 100 - percentile)
        low, high = np.percentile(self._obj, (percentile, 100 - percentile))
        norm = self._obj - low
        return norm / (high - low)

    def drop_nan(self) -> xr.DataArray:
        assert isinstance(self._obj, xr.DataArray)  # ._obj.values
        assert len(self._obj.dims) == 1

        mask = np.logical_not(np.isnan(self._obj.values))
        return self._obj.isel({self._obj.dims[0]: mask})

    def shift_coords(
        self,
        dims: tuple[str, ...],
        shift: NDArray[np.float_] | float,
    ) -> XrTypes:
        if self._obj is None:
            msg = "Cannot access 'G'"
            raise RuntimeError(msg)
        if not isinstance(shift, np.ndarray):
            shift = np.ones((len(dims),)) * shift

        def transform(data: NDArray[np.float_]) -> NDArray[np.float_]:
            new_shift: NDArray[np.float_] = shift
            for _ in range(len(dims)):
                new_shift = np.expand_dims(new_shift, axis=0)

            return data + new_shift

        return self.transform_coords(dims, transform)

    def scale_coords(
        self,
        dims: tuple[str, ...],
        scale: float | NDArray[np.float_],
    ) -> XrTypes:
        if not isinstance(scale, np.ndarray):
            n_dims = len(dims)
            scale = np.identity(n_dims) * scale
        elif len(scale.shape) == 1:
            scale = np.diag(scale)

        return self.transform_coords(dims, scale)

    def transform_coords(
        self,
        dims: Collection[str],
        transform: NDArray[np.float_] | Callable,
    ) -> XrTypes:
        """Transforms the given coordinate values according to an arbitrary function.

        The transformation should either be a function from a len(dims) x size of raveled coordinate
        array to len(dims) x size of raveled_coordinate array or a linear transformation as a matrix
        which is multiplied into such an array.

        Params:
            dims: A list or tuple of dimensions that should be transformed
            transform: The transformation to apply, can either be a function, or a matrix

        Returns:
            An identical valued array over new coordinates.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        as_array = np.stack([self._obj.data_vars[d].values for d in dims], axis=-1)

        if isinstance(transform, np.ndarray):
            transformed = np.dot(as_array, transform)
        else:
            transformed = transform(as_array)

        copied = self._obj.copy(deep=True)

        for d, arr in zip(dims, np.split(transformed, transformed.shape[-1], axis=-1), strict=True):
            copied.data_vars[d].values = np.squeeze(arr, axis=-1)

        return copied

    def filter_vars(self, f: Callable[[Hashable, xr.DataArray], bool]) -> xr.Dataset:
        assert isinstance(self._obj, xr.Dataset)  # ._obj.data_vars
        return xr.Dataset(
            data_vars={k: v for k, v in self._obj.data_vars.items() if f(k, v)},
            attrs=self._obj.attrs,
        )

    def coordinatize(self, as_coordinate_name: str | None = None) -> XrTypes:
        """Copies data into a coordinate's data, with an optional renaming.

        If you think of an array as a function c => f(c) from coordinates to values at
        those coordinates, this function replaces f by the identity to give c => c

        Remarkably, `coordinatize` is a word.

        For the most part, this is only useful when converting coordinate values into
        k-space "forward".

        Args:
            as_coordinate_name: A new coordinate name for the only dimension. Defaults to None.

        Returns:
            An array which consists of the mapping c => c.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        assert len(self._obj.dims) == 1

        dim = self._obj.dims[0]
        if as_coordinate_name is None:
            as_coordinate_name = str(dim)

        o = self._obj.rename(dict([[dim, as_coordinate_name]])).copy(deep=True)
        o.coords[as_coordinate_name] = o.values

        return o

    def ravel(self) -> Mapping[Hashable, xr.DataArray | NDArray[np.float_]]:
        """Converts to a flat representation where the coordinate values are also present.

        Extremely valuable for plotting a dataset with coordinates, X, Y and values Z(X,Y)
        on a scatter plot in 3D.

        By default the data is listed under the key 'data'.

        Returns:
            A dictionary mapping between coordinate names and their coordinate arrays.
            Additionally, there is a key "data" which maps to the `.values` attribute of the array.
        """
        assert isinstance(self._obj, xr.DataArray)

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        raveled_coordinates = dict(
            zip(
                dims,
                [cs.ravel() for cs in np.meshgrid(*coords_as_list)],
                strict=True,
            ),
        )
        assert "data" not in raveled_coordinates
        raveled_coordinates["data"] = self._obj.values.ravel()

        return raveled_coordinates

    def meshgrid(
        self,
        *,
        as_dataset: bool = False,
    ) -> dict[Hashable, NDArray[np.float_]] | xr.Dataset:
        assert isinstance(self._obj, xr.DataArray)  # ._obj.values is used.

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        meshed_coordinates = dict(zip(dims, list(np.meshgrid(*coords_as_list)), strict=True))
        assert "data" not in meshed_coordinates
        meshed_coordinates["data"] = self._obj.values

        if as_dataset:
            # this could use a bit of cleaning up
            faked = ["x", "y", "z", "w"]
            return xr.Dataset(
                {
                    k: (faked[: len(v.shape)], v)
                    for k, v in meshed_coordinates.items()
                    if k != "data"
                },
            )

        return meshed_coordinates

    def to_arrays(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Converts a (1D) `xr.DataArray` into two plain ``ndarray``s of their coordinate and data.

        Useful for rapidly converting into a format than can be `plt.scatter`ed
        or similar.

        Example:
            We can use this to quickly scatter a 1D dataset where one axis is the coordinate value.

            >>> plt.scatter(*data.G.as_arrays(), marker='s')  # doctest: +SKIP

        Returns:
            A tuple of the coordinate array (first index) and the data array (second index)
        """
        assert isinstance(self._obj, xr.DataArray)
        assert len(self._obj.dims) == 1

        return (self._obj.coords[self._obj.dims[0]].values, self._obj.values)

    def clean_outliers(self, clip: float = 0.5) -> xr.DataArray:
        assert isinstance(self._obj, xr.DataArray)
        low, high = np.percentile(self._obj.values, [clip, 100 - clip])
        copied = self._obj.copy(deep=True)
        copied.values[copied.values < low] = low
        copied.values[copied.values > high] = high
        return copied

    def as_movie(
        self,
        time_dim: str = "delay",
        pattern: str = "{}.png",
        out: str | bool = "",
        **kwargs: Unpack[PColorMeshKwargs],
    ) -> Path | animation.FuncAnimation:
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)

        if isinstance(out, bool) and out is True:
            out = pattern.format(f"{self._obj.S.label}_animation")
        assert isinstance(out, str)
        return plot_movie(self._obj, time_dim, out=out, **kwargs)

    def filter_coord(
        self,
        coordinate_name: str,
        sieve: Callable[[Any, XrTypes], bool],
    ) -> XrTypes:
        """Filters a dataset along a coordinate.

        Sieve should be a function which accepts a coordinate value and the slice
        of the data along that dimension.

        Internally, the predicate function `sieve` is applied to the coordinate and slice to
        generate a mask. The mask is used to select from the data after iteration.

        An improvement here would support filtering over several coordinates.

        Args:
            coordinate_name: The coordinate which should be filtered.
            sieve: A predicate to be applied to the coordinate and data at that coordinate.

        Returns:
            A subset of the data composed of the slices which make the `sieve` predicate `True`.
        """
        mask = np.array(
            [
                i
                for i, c in enumerate(self._obj.coords[coordinate_name])
                if sieve(c, self._obj.isel({coordinate_name: i}))
            ],
        )
        return self._obj.isel({coordinate_name: mask})

    def iterate_axis(
        self,
        axis_name_or_axes: list[str] | str,
    ) -> Generator[tuple[dict[str, float], XrTypes], str, None]:
        """[TODO:summary].

        Args:
            axis_name_or_axes: [TODO:description]

        Returns:
            [TODO:description]
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if isinstance(axis_name_or_axes, str):
            axis_name_or_axes = [axis_name_or_axes]

        coord_iterators: list[NDArray[np.float_]] = [
            self._obj.coords[d].values for d in axis_name_or_axes
        ]
        for indices in itertools.product(*[range(len(c)) for c in coord_iterators]):
            cut_coords = [cs[index] for cs, index in zip(coord_iterators, indices, strict=True)]
            coords_dict = dict(zip(axis_name_or_axes, cut_coords, strict=True))
            yield coords_dict, self._obj.sel(coords_dict, method="nearest")

    def map_axes(
        self,
        axes: list[str] | str,
        fn: Callable[[XrTypes, dict[str, float]], DataType],
        dtype: DTypeLike = None,
    ) -> xr.DataArray:
        """[TODO:summary].

        Args:
            axes ([TODO:type]): [TODO:description]
            fn: [TODO:description]
            dtype: [TODO:description]

        Raises:
            TypeError: [TODO:description]
        """
        if isinstance(self._obj, xr.Dataset):
            msg = "map_axes can only work on xr.DataArrays for now because of how the type"
            msg += " inference works"
            raise TypeError(
                msg,
            )
        assert isinstance(self._obj, xr.DataArray)
        obj = self._obj.copy(deep=True)

        if dtype is not None:
            obj.values = np.ndarray(shape=obj.values.shape, dtype=dtype)

        type_assigned = False
        for coord, value in self.iterate_axis(axes):
            new_value = fn(value, coord)

            if dtype is None:
                if not type_assigned:
                    obj.values = np.ndarray(shape=obj.values.shape, dtype=new_value.data.dtype)
                    type_assigned = True

                obj.loc[coord] = new_value.values
            else:
                obj.loc[coord] = new_value

        return obj

    def transform(
        self,
        axes: str | list[str],
        transform_fn: Callable,
        dtype: DTypeLike = None,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """Applies a vectorized operation across a subset of array axes.

        Transform has similar semantics to matrix multiplication, the dimensions of the
        output can grow or shrink depending on whether the transformation is size preserving,
        grows the data, shinks the data, or leaves in place.

        Examples:
            As an example, let us suppose we have a function which takes the mean and
            variance of the data:

                [dimension], coordinate_value -> [{'mean', 'variance'}]

            And a dataset with dimensions [X, Y]. Then calling transform
            maps to a dataset with the same dimension X but where Y has been replaced by
            the length 2 label {'mean', 'variance'}. The full dimensions in this case are
            ['X', {'mean', 'variance'}].

            >>> data.G.transform('X', f).dims  # doctest: +SKIP
            ["X", "mean", "variance"]

        Please note that the transformed axes always remain in the data because they
        are iterated over and cannot therefore be modified.

        The transform function `transform_fn` must accept the coordinate of the
        marginal at the currently iterated point.

        Args:
            axes: Dimension/axis or set of dimensions to iterate over
            transform_fn: Transformation function that takes a DataArray into a new DataArray
            dtype: An optional type hint for the transformed data. Defaults to None.
            args: args to pass into transform_fn
            kwargs: kwargs to pass into transform_fn

        Raises:
            TypeError: When the underlying object is an `xr.Dataset` instead of an `xr.DataArray`.
            This is due to a constraint related to type inference with a single passed dtype.


        Returns:
            The data consisting of applying `transform_fn` across the specified axes.

        """
        if isinstance(self._obj, xr.Dataset):
            msg = "transform can only work on xr.DataArrays for"
            msg += " now because of how the type inference works"
            raise TypeError(msg)

        assert isinstance(self._obj, xr.DataArray)
        dest = None
        for coord, value in self.iterate_axis(axes):
            new_value = transform_fn(value, coord, *args, **kwargs)

            if dest is None:
                new_value = transform_fn(value, coord, *args, **kwargs)

                original_dims = [d for d in self._obj.dims if d not in value.dims]
                original_shape = [self._obj.shape[self._obj.dims.index(d)] for d in original_dims]
                original_coords = {k: v for k, v in self._obj.coords.items() if k not in value.dims}

                full_shape = original_shape + list(new_value.shape)

                new_coords = original_coords
                new_coords.update(
                    {k: v for k, v in new_value.coords.items() if k not in original_coords},
                )
                new_dims = original_dims + list(new_value.dims)
                dest = xr.DataArray(
                    np.zeros(full_shape, dtype=dtype or new_value.data.dtype),
                    coords=new_coords,
                    dims=new_dims,
                )

            dest.loc[coord] = new_value
        assert isinstance(dest, xr.DataArray)
        return dest

    def map(self, fn: Callable, **kwargs: Incomplete) -> xr.DataArray:
        """[TODO:summary].

        Args:
            fn: [TODO:description]
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        assert isinstance(self._obj, xr.DataArray)
        return apply_dataarray(self._obj, np.vectorize(fn, **kwargs))

    def enumerate_iter_coords(
        self,
    ) -> Generator[tuple[tuple[int, ...], dict[Hashable, float]], None, None]:
        """[TODO:summary].

        Returns:
            Generator of the following data
            ((0, 0), {'phi': -0.2178031280148764, 'eV': 9.0})
            which shows the relationship between pixel position and physical (like "eV" and "phi").
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        coords_list = [self._obj.coords[d].values for d in self._obj.dims]
        for indices in itertools.product(*[range(len(c)) for c in coords_list]):
            cut_coords = [cs[index] for cs, index in zip(coords_list, indices, strict=True)]
            yield indices, dict(zip(self._obj.dims, cut_coords, strict=True))

    def iter_coords(
        self,
        dim_names: tuple[str | Hashable, ...] = (),
    ) -> Generator[dict[Hashable, float], None, None]:
        """[TODO:summary].

        Args:
            dim_names: [TODO:description]

        Returns:
            Generator of the physical position like ("eV" and "phi")
            {'phi': -0.2178031280148764, 'eV': 9.002}
        """
        if not dim_names:
            dim_names = tuple(self._obj.dims)
        for ts in itertools.product(*[self._obj.coords[d].values for d in dim_names]):
            yield dict(zip(dim_names, ts, strict=True))

    def range(
        self,
        *,
        generic_dim_names: bool = True,
    ) -> dict[Hashable, tuple[float, float]]:
        """Return the maximum/minimum value in each dimension.

        Args:
            generic_dim_names (bool): if True, use Generic dimension name, such as 'x', is used.

        Returns: (dict[str, tuple[float, float]])
            The range of each dimension.
        """
        indexed_coords = [self._obj.coords[d] for d in self._obj.dims]
        indexed_ranges = [(np.min(coord.values), np.max(coord.values)) for coord in indexed_coords]

        dim_names: list[str] | tuple[Hashable, ...] = tuple(self._obj.dims)
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        return dict(zip(dim_names, indexed_ranges, strict=True))

    def stride(
        self,
        *args: str | list[str] | tuple[str, ...],
        generic_dim_names: bool = True,
    ) -> dict[str, float] | list[float] | float:
        """Return the stride in each dimension.

        Note that the stride defined in this method is just a difference between first two values.
        In most case, this treatment does not cause a problem.  However, when the data has been
        concatenated, this assumption may not be not valid.

        Args:
            args: The dimension to return.  ["eV", "phi"] or "eV", "phi"
            generic_dim_names (bool): if True, use Generic dimension name, such as 'x', is used.

        Returns:
            The stride of each dimension
        """
        indexed_coords: list[xr.DataArray] = [self._obj.coords[d] for d in self._obj.dims]
        indexed_strides: list[float] = [
            coord.values[1] - coord.values[0] for coord in indexed_coords
        ]

        dim_names: list[str] | tuple[Hashable, ...] = tuple(self._obj.dims)
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        result: dict[Hashable, float] = dict(zip(dim_names, indexed_strides, strict=True))
        if args:
            if len(args) == 1:
                if not isinstance(args[0], str):  # suppose args is list / tuple
                    result = [result[selected_names] for selected_names in args[0]]
                else:
                    result = result[args[0]]
            else:
                # if passed several names as arguments
                result = [result[selected_names] for selected_names in args]

        return result

    def shift_by(  # noqa: PLR0913
        self,
        other: xr.DataArray | NDArray[np.float_],
        shift_axis: str = "",
        by_axis: str = "",
        *,
        zero_nans: bool = True,
        shift_coords: bool = False,
    ) -> xr.DataArray:
        """Data shift along the axis.

        For now we only support shifting by a one dimensional array

        Args:
            other (xr.DataArray | NDArray): [TODO:description]
                 we only support shifting by a one dimensional array
            shift_axis (str): [TODO:description]
            by_axis (str): The dimension name of `other`.  When `other` is xr.DataArray, this value
                 is ignored.
            zero_nans (bool): if True, fill 0 for np.nan
            shift_coords (bool): [TODO:description]

        Returns (xr.DataArray):
            Shifted xr.DataArray
        """
        if not shift_axis:
            msg = "shift_by must take shift_axis argument."
            raise TypeError(msg)
        assert isinstance(self._obj, xr.DataArray)
        data = self._obj.copy(deep=True)

        if isinstance(other, xr.DataArray):
            assert len(other.dims) == 1
            by_axis = str(other.dims[0])
            assert len(other.coords[by_axis]) == len(data.coords[by_axis])
            if shift_coords:
                mean_shift = np.mean(other.values)
                other -= mean_shift
            shift_amount = -other.values / data.G.stride(generic_dim_names=False)[shift_axis]
        else:
            assert isinstance(other, np.ndarray)
            assert other.ndim == 1
            assert other.shape[0] == len(data.coords[by_axis])
            if not by_axis:
                msg = "When np.ndarray is used for shift_by by_axis is required."
                raise TypeError(msg)
            assert other.shape[0] == len(data.coords[by_axis])
            if shift_coords:
                mean_shift = np.mean(other)
                other -= mean_shift
            shift_amount = -other / data.G.stride(generic_dim_names=False)[shift_axis]

        shifted_data: NDArray[np.float_] = arpes.utilities.math.shift_by(
            data.values,
            shift_amount,
            axis=data.dims.index(shift_axis),
            by_axis=data.dims.index(by_axis),
            order=1,
        )

        if zero_nans:
            shifted_data[np.isnan(shifted_data)] = 0

        built_data = xr.DataArray(
            shifted_data,
            data.coords,
            data.dims,
            attrs=data.attrs.copy(),
        )

        if shift_coords:
            built_data = built_data.assign_coords(
                {shift_axis: data.coords[shift_axis] + mean_shift},
            )

        return built_data

    def __init__(self, xarray_obj: XrTypes) -> None:
        self._obj = xarray_obj


@xr.register_dataarray_accessor("X")
class SelectionToolAccessor:
    _obj: xr.DataArray

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def max_in_window(
        self,
        around: xr.DataArray,
        window: float,
        n_iters: int = 1,
    ) -> xr.DataArray:
        # TODO: refactor into a transform and finish the transform refactor to allow
        # simultaneous iteration
        assert isinstance(self._obj, xr.DataArray)
        destination = around.copy(deep=True) * 0

        # should only be one!
        (other_dim,) = list(set(self._obj.dims).difference(around.dims))

        for coord, value in around.G.iterate_axis(around.dims):
            value_item = value.item()
            marg = self._obj.sel(coord)

            if isinstance(value_item, float):
                marg = marg.sel({other_dim: slice(value_item - window, value_item + window)})
            else:
                marg = marg.isel({other_dim: slice(value_item - window, value_item + window)})

            destination.loc[coord] = marg.coords[other_dim][marg.argmax().item()]

        if n_iters > 1:
            return self.max_in_window(destination, window, n_iters - 1)

        return destination

    def first_exceeding(  # noqa: PLR0913
        self,
        dim: str,
        value: float,
        *,
        relative: bool = False,
        reverse: bool = False,
        as_index: bool = False,
    ) -> xr.DataArray:
        data = self._obj

        if relative:
            data = data / data.max(dim)

        cond = data > value
        reindex = data.coords[dim]

        if reverse:
            reindex = np.flip(reindex)
            cond = np.flip(cond, data.dims.index(dim)).values

        indices = cond.argmax(axis=data.dims.index(dim))
        if as_index:
            new_values = indices
            if reverse:
                new_values = -new_values + len(reindex) - 1
        else:
            new_values = reindex[indices]

        with contextlib.suppress(AttributeError):
            new_values = new_values.values

        return data.isel({dim: 0}).S.rith_values(new_values)

    def last_exceeding(self, dim: str, value: float, *, relative: bool = False) -> xr.DataArray:
        return self.first_exceeding(dim, value, relative=relative, reverse=False)


@xr.register_dataset_accessor("F")
class ARPESDatasetFitToolAccessor:
    _obj: xr.Dataset

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def eval(self, *args: Incomplete, **kwargs: Incomplete) -> xr.DataArray:
        return self._obj.results.G.map(lambda x: x.eval(*args, **kwargs))

    def show(self, *, detached: bool = False) -> None:
        from .plotting.fit_tool import fit_tool

        fit_tool(self._obj, detached=detached)

    @property
    def broadcast_dimensions(self) -> list[str]:
        """Returns the dimensions which were used in the fitting process.

        This is a sibling property to `fit_dimensions`.

        Returns:
            The list of the dimensions which were used in any individual fit.
            For example, a broadcast of MDCs across energy on a dataset with dimensions
            `["eV", "kp"]` would produce `["kp"]`.
        """
        assert isinstance(self._obj, xr.Dataset)
        return list(self._obj.results.dims)

    @property
    def fit_dimensions(self) -> list[str]:
        """Returns the dimensions which were broadcasted across, as opposed to fit across.

        This is a sibling property to `broadcast_dimensions`.

        Returns:
            The list of the dimensions which were **not** used in any individual fit.
            For example, a broadcast of MDCs across energy on a dataset with dimensions
            `["eV", "kp"]` would produce `["eV"]`.
        """
        assert isinstance(self._obj, xr.Dataset)
        return list(set(self._obj.data.dims).difference(self._obj.results.dims))

    def best_fits(self) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.best_fits`.

        Orders the fits into a raveled array by the MSE error.
        """
        assert isinstance(self._obj, xr.Dataset)
        return self._obj.results.F.best_fits()

    def worst_fits(self) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.worst_fits`.

        Orders the fits into a raveled array by the MSE error.
        """
        assert isinstance(self._obj, xr.Dataset)
        return self._obj.results.F.worst_fits()

    def mean_square_error(self) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.mean_square_error`.

        Calculates the mean square error of the fit across the fit
        axes for all model result instances in the collection.
        """
        assert isinstance(self._obj, xr.Dataset)
        return self._obj.results.F.mean_square_error()

    def p(self, param_name: str) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.p`.

        Collects the value of a parameter from curve fitting.

        Across an array of fits, walks parameters to collect the value
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the value found by the fitting routine.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        assert isinstance(self._obj, xr.Dataset)
        return self._obj.results.F.p(param_name)

    def s(self, param_name: str) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.s`.

        Collects the standard deviation of a parameter from fitting.

        Across an array of fits, walks parameters to collect the standard error
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the floating point value for the fits.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        assert isinstance(self._obj, xr.Dataset)
        return self._obj.results.F.s(param_name)

    def plot_param(self, param_name: str, **kwargs: tuple[int, int] | RGBColorType) -> None:
        """Alias for `ARPESFitToolsAccessor.plot_param`.

        Creates a scatter plot of a parameter from a multidimensional curve fit.

        Args:
            param_name: The name of the parameter which should be plotted
            kwargs: Passed to plotting routines to provide user control
                figsize =, color =
        """
        assert isinstance(self._obj, xr.Dataset)
        return self._obj.results.F.plot_param(param_name, **kwargs)


@xr.register_dataarray_accessor("F")
class ARPESFitToolsAccessor:
    """Utilities related to examining curve fits."""

    _obj: xr.DataArray

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Initialization hook for xarray.

        This should never need to be called directly.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        self._obj = xarray_obj

    def plot_param(self, param_name: str, **kwargs: tuple[int, int] | RGBColorType) -> None:
        """Creates a scatter plot of a parameter from a multidimensional curve fit.

        Args:
            param_name: The name of the parameter which should be plotted
            kwargs: Passed to plotting routines to provide user control
                figsize=, color=, markersize=,
        """
        plot_parameter(self._obj, param_name, **kwargs)

    def param_as_dataset(self, param_name: str) -> xr.Dataset:
        """Maps from `lmfit.ModelResult` to a Dict parameter summary.

        Args:
            param_name: The parameter which should be summarized.

        Returns:
            A dataset consisting of two arrays: "value" and "error"
            which are the fit value and standard error on the parameter
            requested.
        """
        return xr.Dataset(
            {
                "value": self.p(param_name),
                "error": self.s(param_name),
            },
        )

    def show(self, *, detached: bool = False) -> None:
        """Opens a Qt based interactive fit inspection tool."""
        from .plotting.fit_tool import fit_tool

        fit_tool(self._obj, detached=detached)

    def best_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error."""
        return self.order_stacked_fits(ascending=True)

    def worst_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error."""
        return self.order_stacked_fits(ascending=False)

    def mean_square_error(self) -> xr.DataArray:
        """Calculates the mean square error of the fit across fit axes.

        Producing a scalar metric of the error for all model result instances in
        the collection.
        """
        assert isinstance(self._obj, xr.DataArray)

        def safe_error(model_result_instance: lmfit.model.ModelResult | None) -> float:
            if model_result_instance is None:
                return np.nan
            assert isinstance(model_result_instance.residual, np.ndarray)
            return (model_result_instance.residual**2).mean()

        return self._obj.G.map(safe_error)

    def order_stacked_fits(self, *, ascending: bool = False) -> xr.DataArray:
        """Produces an ordered collection of `lmfit.ModelResult` instances.

        For multidimensional broadcasts, the broadcasted dimensions will be
        stacked for ordering to produce a 1D array of the results.

        Args:
            ascending: Whether the results should be ordered according to ascending
              mean squared error (best fits first) or descending error (worst fits first).

        Returns:
            An xr.DataArray instance with stacked axes whose values are the ordered models.
        """
        assert isinstance(self._obj, xr.DataArray)
        stacked = self._obj.stack({"by_error": self._obj.dims})

        error = stacked.F.mean_square_error()

        if not ascending:
            error = -error

        indices = np.argsort(error.values)
        return stacked[indices]

    def p(self, param_name: str) -> xr.DataArray:
        """Collects the value of a parameter from curve fitting.

        Across an array of fits, walks parameters to collect the value
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the value found by the fitting routine.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        assert isinstance(self._obj, xr.DataArray)
        return self._obj.G.map(param_getter(param_name), otypes=[float])

    def s(self, param_name: str) -> xr.DataArray:
        """Collects the standard deviation of a parameter from fitting.

        Across an array of fits, walks parameters to collect the standard error
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the floating point value for the fits.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        assert isinstance(self._obj, xr.DataArray)
        return self._obj.G.map(param_stderr_getter(param_name), otypes=[float])

    @property
    def bands(self) -> dict[str, MultifitBand]:
        """Collects bands after a multiband fit.

        Returns:
            The collected bands.
        """
        band_names = self.band_names

        return {label: MultifitBand(label=label, data=self._obj) for label in band_names}

    @property
    def band_names(self) -> set[str]:
        """Collects the names of the bands from a multiband fit.

        Heuristically, a band is defined as a dispersive peak so we look for
        prefixes corresponding to parameter names which contain `"center"`.

        Returns:
            The collected prefix names for the bands.

            For instance, if the param name `"a_center"`, the return value
            would contain `"a_"`.
        """
        collected_band_names: set[str] = set()
        assert isinstance(self._obj, xr.DataArray)
        for item in self._obj.values.ravel():
            if item is None:
                continue
            band_names = [k[:-6] for k in item.params if "center" in k]
            collected_band_names = collected_band_names.union(set(band_names))
        return collected_band_names

    @property
    def parameter_names(self) -> set[str]:
        """Collects the parameter names for a multidimensional fit.

        Assumes that the model used is the same for all ``lmfit.ModelResult``s
        so that we can merely extract the parameter names from a single non-null
        result.

        Returns:
            A set of all the parameter names used in a curve fit.

        Todo:
            Test
        """
        collected_parameter_names: set[str] = set()
        assert isinstance(self._obj, xr.DataArray)
        for item in self._obj.values.ravel():
            if item is None:
                continue

            param_names = list(item.params.keys())
            collected_parameter_names = collected_parameter_names.union(set(param_names))

        return collected_parameter_names


@xr.register_dataset_accessor("S")
class ARPESDatasetAccessor(ARPESAccessorBase):
    """Spectrum related accessor for `xr.Dataset`."""

    def __getattr__(self, item: str) -> dict:
        """Forward attribute access to the spectrum, if necessary.

        Args:
            item: Attribute name

        Returns:
            The attribute after lookup on the default spectrum
        """
        return getattr(self._obj.S.spectrum.S, item)

    def polarization_plot(self, **kwargs: IncompleteMPL) -> Axes:
        """Creates a spin polarization plot.

        Returns:
            The axes which were plotted onto for customization.
        """
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = f"{self.label}_spin_polarization.png"
            kwargs["out"] = out
        return spin_polarized_spectrum(self._obj, **kwargs)

    @property
    def is_spatial(self) -> bool:
        """Predicate indicating whether the dataset is a spatial scanning dataset.

        Returns:
            True if the dataset has dimensions indicating it is a spatial scan.
            False otherwise
        """
        assert isinstance(self.spectrum, xr.DataArray | xr.Dataset)

        return self.spectrum.S.is_spatial

    @property
    def spectrum(self) -> xr.DataArray:
        """Isolates a single spectrum from a dataset.

        This is a convenience method which is typically used in startup for
        tools and analysis routines which need to operate on a single
        piece of data. As an example, the image browser `qt_tool` needs
        an `xr.DataArray` to operate but will accept an `xr.Dataset`
        which it will attempt to resolve to a single spectrum.

        In practice, we filter data variables by whether they contain "spectrum"
        in the name before selecting the one with the largest pixel volume.
        This is a heuristic which tries to guarantee we select ARPES data
        above XPS data, if they were collected together.

        Returns:
            A spectrum found in the dataset, if one can be isolated.

            In the case that several candidates are found, a single spectrum
            is selected among the candidates.

            Attributes from the parent dataset are assigned onto the selected
            array as a convenience.

        ToDo: Need test
        """
        if "spectrum" in self._obj.data_vars:
            spectrum = self._obj.spectrum
        elif "raw" in self._obj.data_vars:
            spectrum = self._obj.raw
        elif "__xarray_dataarray_variable__" in self._obj.data_vars:
            spectrum = self._obj.__xarray_dataarray_variable__
        else:
            candidates = self.spectra
            if candidates:
                spectrum = candidates[0]
                best_volume = np.prod(spectrum.shape)
                for c in candidates[1:]:
                    volume = np.prod(c.shape)
                    if volume > best_volume:
                        spectrum = c
                        best_volume = volume
            else:
                msg = "No spectrum found"
                raise RuntimeError(msg)
        return spectrum

    @property
    def spectra(self) -> list[xr.DataArray]:
        """Collects the variables which are likely spectra.

        Returns:
            The subset of the data_vars which have dimensions indicating
            that they are spectra.
        """
        return [dv for dv in self._obj.data_vars.values() if "eV" in dv.dims]

    @property
    def spectrum_type(self) -> Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]:
        """Gives a heuristic estimate of what kind of data is contained by the spectrum.

        Returns:
            The kind of data, coarsely
        """
        return self.spectrum.S.spectrum_type

    @property
    def degrees_of_freedom(self) -> set[str]:
        """The collection of all degrees of freedom.

        Equivalently, dimensions on a piece of data.

        Returns:
            All degrees of freedom as a set.
        """
        collection_set = set()
        for dim in self.spectrum.dims:
            collection_set.add(str(dim))
        return collection_set

    @property
    def spectrum_degrees_of_freedom(self) -> set[str]:
        """Collects the spectrometer degrees of freedom.

        Spectrometer degrees of freedom are any which would be collected by an ARToF
        and their momentum equivalents.

        Returns:
            The collection of spectrum degrees of freedom.
        """
        return self.degrees_of_freedom.intersection({"eV", "phi", "pixel", "kx", "kp", "ky"})

    @property
    def scan_degrees_of_freedom(self) -> set[str]:
        """Collects the scan degrees of freedom.

        Scan degrees of freedom are all of the degrees of freedom which are not recorded
        by the spectrometer but are "scanned over". This includes spatial axes,
        temperature, etc.

        Returns:
            The collection of scan degrees of freedom represented in the array.
        """
        return self.degrees_of_freedom.difference(self.spectrum_degrees_of_freedom)

    def reference_plot(self, **kwargs: IncompleteMPL) -> None:
        """Creates reference plots for a dataset.

        A bit of a misnomer because this actually makes many plots. For full datasets,
        the relevant components are:

        #. Temperature as function of scan DOF
        #. Photocurrent as a function of scan DOF
        #. Photocurrent normalized + unnormalized figures, in particular
            #. The reference plots for the photocurrent normalized spectrum
            #. The normalized total cycle intensity over scan DoF, i.e. cycle vs scan DOF
              integrated over E, phi
            #. For delay scans:
                #. Fermi location as a function of scan DoF, integrated over phi
                #. Subtraction scans
        #. For spatial scans:
            #. energy/angle integrated spatial maps with subsequent measurements indicated
            #. energy/angle integrated FS spatial maps with subsequent measurements indicated

        Args:
            kwargs: Passed to plotting routines to provide user control
        """
        self._obj.sum(*list(self.scan_degrees_of_freedom))
        kwargs.get("out")
        # <== CHECK ME  the above two lines were:

        # make figures for temperature, photocurrent, delay
        make_figures_for = ["T", "IG_nA", "current", "photocurrent"]
        name_normalization = {
            "T": "T",
            "IG_nA": "photocurrent",
            "current": "photocurrent",
        }

        for figure_item in make_figures_for:
            if figure_item not in self._obj.data_vars:
                continue
            name = name_normalization.get(figure_item, figure_item)
            data_var = self._obj[figure_item]
            out = f"{self.label}_{name}_spec_integrated_reference.png"
            scan_var_reference_plot(data_var, title=f"Reference {name}", out=out)

        # may also want to make reference figures summing over cycle, or summing over beta

        # make photocurrent normalized figures
        normalized = self._obj / self._obj.IG_nA
        normalized.S.make_spectrum_reference_plots(prefix="norm_PC_", out=True)

        self.make_spectrum_reference_plots(out=True)

    def make_spectrum_reference_plots(
        self,
        prefix: str = "",
        **kwargs: Incomplete,
    ) -> None:
        """Creates photocurrent normalized + unnormalized figures.

        Creates:
        #. The reference plots for the photocurrent normalized spectrum
        #. The normalized total cycle intensity over scan DoF,
        #  i.e. cycle vs scan DOF integrated over E, phi
        #. For delay scans:

            #. Fermi location as a function of scan DoF, integrated over phi
            #. Subtraction scans

        Args:
            prefix: A prefix inserted into filenames to make them unique.
            kwargs: Passed to plotting routines to provide user control over plotting
                    behavior
        """
        self.spectrum.S.reference_plot(pattern=prefix + "{}.png", **kwargs)

        if self.is_spatial:
            pass
            # <== CHECK ME: original is  referenced = self.referenced_scans
        if "cycle" in self._obj.coords:
            integrated_over_scan = self._obj.sum(*list(self.spectrum_degrees_of_freedom))
            integrated_over_scan.S.spectrum.S.reference_plot(
                pattern=prefix + "sum_spec_DoF_{}.png",
                **kwargs,
            )

        if "delay" in self._obj.coords:
            dims = self.spectrum_degrees_of_freedom
            dims.remove("eV")
            angle_integrated = self._obj.sum(*list(dims))

            # subtraction scan
            self.spectrum.S.subtraction_reference_plots(pattern=prefix + "{}.png", **kwargs)
            angle_integrated.S.fermi_edge_reference_plots(pattern=prefix + "{}.png", **kwargs)

    @property
    def energy_notation(self) -> EnergyNotation:
        """Return the energy notation ("Binding" energy or "Kinetic" Energy).

        .. Note:: The "Kinetic" energy refers to the Fermi level.  (not Vacuum level)
        """
        if "energy_notation" in self._obj.attrs:
            if self.spectrum.attrs["energy_notation"] in (
                "Kinetic",
                "kinetic",
                "kinetic energy",
            ):
                self.spectrum.attrs["energy_notation"] = "Kinetic"
                return "Kinetic"
            return "Binding"
        self.spectrum.attrs["energy_notation"] = self.spectrum.attrs.get(
            "energy_notation",
            "Binding",
        )
        return "Binding"

    def switch_energy_notation(self, nonlinear_order: int = 1) -> None:
        """Switch the energy notation between binding and kinetic.

        Args:
            nonlinear_order (int): order of the nonliniarity, default to 1
        """
        if self._obj.coords["hv"].ndim == 0:
            if self.energy_notation == "Binding":
                self._obj.coords["eV"] = self._obj.coords["eV"] + nonlinear_order * self.hv
                self._obj.attrs["energy_notation"] = "Kinetic"
                for spectrum in self._obj.data_vars.values():
                    spectrum.attrs["energy_notation"] = "Kinetic"
            elif self.energy_notation == "Kinetic":
                self._obj.coords["eV"] = self._obj.coords["eV"] - nonlinear_order * self.hv
                self._obj.attrs["energy_notation"] = "Binding"
                for spectrum in self._obj.data_vars.values():
                    spectrum.attrs["energy_notation"] = "Binding"
        else:
            msg = "Not impremented yet."
            raise RuntimeError(msg)

    @property
    def angle_unit(self) -> Literal["Degrees", "Radians"]:
        """Returns angle unit (Radians/Degrees)."""
        return self._obj.attrs.get("angle_unit", "Radians")

    @angle_unit.setter
    def angle_unit(self, angle_unit: Literal["Degrees", "Radians"]) -> None:
        """Setter of angle_unit (Dataset)."""
        assert angle_unit in (
            "Degrees",
            "Radians",
        ), "Angle unit should be 'Degrees' or 'Radians'"
        self._obj.attrs["angle_unit"] = angle_unit

        for spectrum in self._obj.data_vars.values():
            if "eV" in spectrum.dims:
                spectrum.attrs["angle_unit"] = angle_unit

    def swap_angle_unit(self) -> None:
        """Swap angle unit (radians <-> degeres).

        Change the value of angle related objects/variables in attrs and coords.
        """
        msg = 'The angle_unit must be "Radians" or "Degrees"'
        if self.angle_unit == "Radians" or self.angle_unit.startswith("rad"):
            self._degree_to_radian()
        elif self.angle_unit.startswith("deg") or self.angle_unit == "Degrees":
            self._radian_to_degree()
        else:
            raise TypeError(msg)

    def _degree_to_radian(self) -> None:
        """A Helper function for swap_angle_unit.

        Degrees -> Radians
        """
        self.angle_unit = "Degrees"
        for angle in ANGLE_VARS:
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = np.rad2deg(self._obj.attrs[angle])
                for spectrum in self._obj.data_vars.values():
                    if angle in spectrum.attrs:
                        spectrum.attrs[angle] = np.rad2deg(spectrum.attrs.get(angle, np.nan))
            if angle + "_offset" in self._obj.attrs:
                self._obj.attrs[angle + "_offset"] = np.rad2deg(
                    self._obj.attrs.get(angle + "_offset", 0),
                )
                for spectrum in self._obj.data_vars.values():
                    spectrum.attrs[angle + "_offset"] = np.rad2deg(
                        spectrum.attrs.get(angle + "_offset", 0),
                    )
            if angle in self._obj.coords:
                self._obj.coords[angle] = np.rad2deg(self._obj.coords[angle])
                for spectrum in self._obj.data_vars.values():
                    if angle in spectrum.coords:
                        spectrum.coords[angle] = np.rad2deg(spectrum.coords[angle])

    def _radian_to_degree(self) -> None:
        """A Helper function for swan_angle_unit.

        Radians -> Degrees
        """
        self.angle_unit = "Radians"
        for angle in ANGLE_VARS:
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = np.deg2rad(self._obj.attrs[angle])
                for spectrum in self._obj.data_vars.values():
                    if angle in spectrum.attrs:
                        spectrum.attrs[angle] = np.deg2rad(spectrum.attrs.get(angle, np.nan))
            if angle + "_offset" in self._obj.attrs:
                self._obj.attrs[angle + "_offset"] = np.deg2rad(
                    self._obj.attrs.get(angle + "_offset", 0),
                )
                for spectrum in self._obj.data_vars.values():
                    spectrum.attrs[angle + "_offset"] = np.deg2rad(
                        spectrum.attrs.get(angle + "_offset", np.nan),
                    )
            if angle in self._obj.coords:
                self._obj.coords[angle] = np.deg2rad(self._obj.coords[angle])
                for spectrum in self._obj.data_vars.values():
                    if angle in spectrum.coords:
                        spectrum.coords[angle] = np.deg2rad(spectrum.coords[angle])

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialization hook for xarray.

        This should never need to be called directly.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        super().__init__(xarray_obj)
