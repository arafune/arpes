from __future__ import annotations  # noqa: D100

from logging import DEBUG, INFO
from typing import (
    TYPE_CHECKING,
    Self,
    Unpack,
)

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from arpes.constants import TWO_DIMENSION
from arpes.correction import coords
from arpes.correction.angle_unit import switch_angle_unit, switched_angle_unit
from arpes.debug import setup_logger
from arpes.plotting.dispersion import (
    fancy_dispersion,
    hv_reference_scan,
    labeled_fermi_surface,
    reference_scan_fermi_surface,
    scan_var_reference_plot,
)
from arpes.plotting.fermi_edge import fermi_edge_reference
from arpes.plotting.spatial import reference_scan_spatial
from arpes.plotting.ui import ProfileApp

from .base import ARPESAccessorBase, ARPESDataArrayAccessorBase

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from _typeshed import Incomplete
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from panel.layout import Panel

    from arpes._typing.attrs_property import CoordsOffset, SpectrumType
    from arpes._typing.plotting import (
        HvRefScanParam,
        LabeledFermiSurfaceParam,
        MPLPlotKwargs,
        PColorMeshKwargs,
        ProfileViewParam,
    )

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@xr.register_dataarray_accessor("S")
class ARPESDataArrayAccessor(ARPESDataArrayAccessorBase):
    """Spectrum related accessor for `xr.DataArray`."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Initialize."""
        self._obj: xr.DataArray = xarray_obj
        assert isinstance(self._obj, xr.DataArray)

    def switched_angle_unit(self) -> xr.DataArray:
        """Return the identical data but the angle unit is converted.

        Change the value of angle related objects/variables in attrs and coords

        Returns:
            xr.DataArray:The DataArray in which angle units are converted.
        """
        return switched_angle_unit(self._obj)

    def switch_angle_unit(self) -> None:
        """Switch angle unit (radians <-> degrees) in place.

        Change the value of angle related objects/variables in attrs and coords
        """
        return switch_angle_unit(self._obj)

    def corrected_coords(
        self,
        correction_types: CoordsOffset | Sequence[CoordsOffset],
    ) -> xr.DataArray:
        """Apply the specified coordinate corrections to the DataArray.

        Args:
            correction_types (CoordsOffset | Sequence[CoordsOffset]): The types of corrections to
                apply.

        Returns:
            xr.DataArray: The corrected DataArray.
        """
        return coords.corrected_coords(self._obj, correction_types)

    def correct_coords(
        self,
        correction_types: CoordsOffset | Sequence[CoordsOffset],
    ) -> None:
        """Correct the coordinates of the DataArray in place.

        Args:
            correction_types (CoordsOffset | Sequence[CoordsOffset, ...]): The types of corrections
                to apply.
        """
        array = coords.corrected_coords(self._obj, correction_types)
        self._obj.attrs = array.attrs
        self._obj.coords.update(array.coords)

    # --- Mehhods about plotting
    # --- TODO : [RA] Consider refactoring/removing
    def plot(
        self: Self,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> None:
        """Utility delegate to `xr.DataArray.plot` which rasterizes`.

        Args:
            rasterized (bool): if True, rasterized (Not vector) drawing
            args: Pass to xr.DataArray.plot
            kwargs: Pass to xr.DataArray.plot
        """
        if len(self._obj.dims) == TWO_DIMENSION:
            kwargs.setdefault("rasterized", True)
        with plt.rc_context(rc={"text.usetex": False}):
            self._obj.plot(*args, **kwargs)

    def show(self, **kwargs: Unpack[ProfileViewParam]) -> Panel:
        """Show holoviews based plot."""
        return ProfileApp(self._obj, **kwargs).panel()

    def fs_plot(
        self: Self,
        pattern: str = "{}.png",
        **kwargs: Unpack[LabeledFermiSurfaceParam],
    ) -> Path | tuple[Figure | None, Axes]:
        """Provides a reference plot of the approximate Fermi surface."""
        assert isinstance(self._obj, xr.DataArray)
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{self.label}_fs")
            kwargs["out"] = out
        return labeled_fermi_surface(self._obj, **kwargs)

    def fermi_edge_reference_plot(
        self: Self,
        pattern: str = "{}.png",
        out: str | Path = "",
        **kwargs: Unpack[MPLPlotKwargs],
    ) -> Path | Axes:
        """Provides a reference plot for a Fermi edge reference.

        This function generates a reference plot for a Fermi edge, which can be useful for analyzing
        energy spectra. It calls the `fermi_edge_reference` function and passes any additional
        keyword arguments to it for plotting customization. The output file name can be specified
        using the `out` argument, with a default name pattern.

        Args:
            pattern (str): A string pattern for the output file name. The pattern can include
                placeholders that will be replaced by the label or other variables.
                Default is "{}.png".
            out (str | Path): The path for saving the output figure. If set to `None` or `False`,
                no figure will be saved. If a boolean `True` is passed, it will use the `pattern`
                to generate the filename.
            kwargs: Additional arguments passed to the `fermi_edge_reference` function for
                customizing the plot.

        Returns:
            Path | Axes: The path to the saved figure (if `out` is provided), or the Axes object of
            the plot.Provides a reference plot for a Fermi edge reference.

        """
        assert isinstance(self._obj, xr.DataArray)
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{self.label}_fermi_edge_reference")
        return fermi_edge_reference(self._obj, out=out, **kwargs)

    def _referenced_scans_for_spatial_plot(
        self: Self,
        *,
        use_id: bool = True,
        pattern: str = "{}.png",
        out: str | Path = "",
    ) -> Path | tuple[Figure, NDArray[np.object_]]:
        """Helper function for generating a spatial plot of referenced scans.

        This function assists in generating a spatial plot for referenced scans, either by using a
        unique identifier or a predefined label. The output file name can be automatically generated
        or specified by the user. The function calls `reference_scan_spatial` for generating the
        plot and optionally saves the output figure.

        Args:
            use_id (bool): If `True`, uses the "id" attribute from the object's metadata as the
                label. If `False`, uses the predefined label. Default is `True`.
            pattern (str): A string pattern for the output file name. The placeholder `{}` will be
                replaced by the label or identifier. Default is `"{}.png"`.
            out (str | bool): The path to save the output figure. If `True`, the file name is
                generated using the `pattern`. If `False` or an empty string (`""`), no output is
                saved.

        Returns:
            Path | tuple[Figure, NDArray[np.object_]]:
                - If `out` is provided, returns the path to the saved figure.
                - Otherwise, returns the Figure and an array of the spatial data.

        """
        label = self._obj.attrs["id"] if use_id else self.label
        if isinstance(out, bool) and out is True:
            out = pattern.format(f"{label}_reference_scan_fs")
        elif isinstance(out, bool) and out is False:
            out = ""

        return reference_scan_spatial(self._obj, out=out)

    def _referenced_scans_for_map_plot(
        self: Self,
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

    def _referenced_scans_for_hv_map_plot(
        self: Self,
        pattern: str = "{}.png",
        *,
        use_id: bool = True,
        **kwargs: Unpack[HvRefScanParam],
    ) -> Path | Axes:
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{label}_hv_reference_scan")
            out = f"{label}_hv_reference_scan.png"
            kwargs["out"] = out

        return hv_reference_scan(self._obj, **kwargs)

    def _simple_spectrum_reference_plot(
        self: Self,
        *,
        use_id: bool = True,
        pattern: str = "{}.png",
        out: str | Path = "",
        **kwargs: Unpack[PColorMeshKwargs],
    ) -> Axes | Path:
        assert isinstance(self._obj, xr.DataArray)
        label = self._obj.attrs["id"] if use_id else self.label
        if isinstance(out, bool):
            out = pattern.format(f"{label}_spectrum_reference")

        return fancy_dispersion(self._obj, out=out, **kwargs)

    def reference_plot(
        self,
        **kwargs: Incomplete,
    ) -> Axes | Path | tuple[Figure, NDArray[np.object_]]:
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
        piece of data.
        Historically, the handling of Dataset and Dataarray was a mess in previous pyarpes.
        Most of the current pyarpes methods/function are sufficient to treat DataArray as the main
        object. (The few exceptions are S.modelfit, whose return value is a Dataset, which is
        reasonable.) For backward compatibility, the return of load_data is still a Dataset,
        so in many cases, using this property for a DataArray will provide a more robust analysing
        environment.

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

        Todo: Need test
        """
        if "spectrum" in self._obj.data_vars:
            return self._obj.spectrum
        if "raw" in self._obj.data_vars:
            return self._obj.raw
        if "__xarray_dataarray_variable__" in self._obj.data_vars:
            return self._obj.__xarray_dataarray_variable__
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
    def spectrum_type(self) -> SpectrumType:
        """Gives a heuristic estimate of what kind of data is contained by the spectrum.

        Returns:
            The kind of data, coarsely
        """
        return self.spectrum.S.spectrum_type

    def reference_plot(self: Self, **kwargs: Incomplete) -> None:
        """Creates reference plots for a dataset.

        A bit of a misnomer because this actually makes many plots. For full datasets,
        the relevant components are:

        #. Temperature as function of scan DOF
        #. Photocurrent as a function of scan DOF
        #. Photocurrent normalized + unnormalized figures, in particular

            #. The reference plots for the photocurrent normalized spectrum
            #. The normalized total cycle intensity over scan DoF, i.e. cycle vs scan DOF integrated
                over E, phi

            #. For delay scans

                #. Fermi location as a function of scan DoF, integrated over phi
                #. Subtraction scans

        #. For spatial scans

            #. energy/angle integrated spatial maps with subsequent measurements indicated
            #. energy/angle integrated FS spatial maps with subsequent measurements indicated

        Args:
            kwargs: Passed to plotting routines to provide user control
        """
        spectrum_degrees_of_freedom = set(self.spectrum.dims).intersection(
            {"eV", "phi", "pixel", "kx", "kp", "ky"},
        )
        scan_degrees_of_freedom = set(self.spectrum.dims).difference(spectrum_degrees_of_freedom)
        self._obj.sum(scan_degrees_of_freedom)
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
            data_var: xr.DataArray = self._obj[figure_item]
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
           i.e. cycle vs scan DOF integrated over E, phi
        #. For delay scans

            #. Fermi location as a function of scan DoF, integrated over phi
            #. Subtraction scans

        Args:
            prefix: A prefix inserted into filenames to make them unique.
            kwargs: Passed to plotting routines to provide user control over plotting
                    behavior
        """
        self.spectrum.S.reference_plot(pattern=prefix + "{}.png", **kwargs)
        spectrum_degrees_of_freedom = set(self.spectrum.dims).intersection(
            {"eV", "phi", "pixel", "kx", "kp", "ky"},
        )
        if self.is_spatial:
            pass
            # <== CHECK ME: original is  referenced = self.referenced_scans
        if "cycle" in self._obj.coords:
            integrated_over_scan = self._obj.sum(spectrum_degrees_of_freedom)
            integrated_over_scan.S.spectrum.S.reference_plot(
                pattern=prefix + "sum_spec_DoF_{}.png",
                **kwargs,
            )

        if "delay" in self._obj.coords:
            dims = spectrum_degrees_of_freedom
            dims.remove("eV")
            angle_integrated = self._obj.sum(dims)

            # subtraction scan
            self.spectrum.S.subtraction_reference_plots(pattern=prefix + "{}.png", **kwargs)
            angle_integrated.S.fermi_edge_reference_plots(pattern=prefix + "{}.png", **kwargs)

    def switch_energy_notation(self, nonlinear_order: int = 1) -> None:
        """Switch the energy notation between binding and kinetic.

        Args:
            nonlinear_order (int): order of the nonliniarity, default to 1
        """
        super().switch_energy_notation(nonlinear_order=nonlinear_order)
        for data in self._obj.data_vars.values():
            if data.S.energy_notation == "Binding":
                data.attrs["energy_notation"] = "Final"
            else:
                data.attrs["energy_notation"] = "Binding"

    def switched_angle_unit(self) -> xr.Dataset:
        """Return the identical data but the angle unit is converted.

        Change the value of angle related objects/variables in attrs and coords

        Returns:
            xr.Dataset: The Dataset in which angle units are converted.
        """
        data = switched_angle_unit(self._obj)
        for spectral_name, spectral_array in data.data_vars.items():
            data[spectral_name] = switched_angle_unit(spectral_array)
        return data

    def switch_angle_unit(self) -> None:
        """Switch angle unit in place.

        Change the value of angle related objects/variables in attrs and coords
        """
        for data in self._obj.data_vars.values():
            switch_angle_unit(data)
        switch_angle_unit(self._obj)

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialization hook for xarray.

        Args:
            xarray_obj: The parent object which this is an accessor for

        Note:
            This class should not be called directly.
        """
        self._obj: xr.Dataset
        super().__init__(xarray_obj)
