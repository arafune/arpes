"""Public plotting API with lazy imports to avoid circular dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "DifferentiateApp",
    "ProfileApp",
    "SmoothingApp",
    "TailorApp",
    "annotate_cuts",
    "annotate_experimental_conditions",
    "annotate_point",
    "concat_along_phi_ui",
    "cut_dispersion_plot",
    "dark_background",
    "fancy_dispersion",
    "fancy_labels",
    "fermi_edge_reference",
    "fermi_surface_slices",
    "fit_inspection",
    "flat_stack_plot",
    "h_gradient_fill",
    "hv_reference_scan",
    "labeled_fermi_surface",
    "magnify_circular_regions_plot",
    "make_overview",
    "make_reference_plots",
    "offset_scatter_plot",
    "plot_core_levels",
    "plot_dispersion",
    "plot_dos",
    "plot_movie",
    "plot_parameter",
    "plot_spatial_reference",
    "plot_with_bands",
    "profile_view",
    "reference_scan_fermi_surface",
    "reference_scan_spatial",
    "remove_colorbars",
    "savefig",
    "scan_var_reference_plot",
    "simple_ax_grid",
    "spin_colored_spectrum",
    "spin_difference_spectrum",
    "spin_polarized_spectrum",
    "stack_dispersion_plot",
    "v_gradient_fill",
    "waterfall_dispersion",
]

# TYPE_CHECKING 用に明示的インポート
if TYPE_CHECKING:
    from .annotations import annotate_cuts, annotate_experimental_conditions, annotate_point
    from .bands import plot_with_bands
    from .basic import make_overview, make_reference_plots
    from .dark_bg import dark_background
    from .decoration import h_gradient_fill, v_gradient_fill
    from .dispersion import (
        cut_dispersion_plot,
        fancy_dispersion,
        hv_reference_scan,
        labeled_fermi_surface,
        plot_dispersion,
        reference_scan_fermi_surface,
        scan_var_reference_plot,
    )
    from .dos import plot_core_levels, plot_dos
    from .fermi_edge import fermi_edge_reference
    from .fermi_surface import fermi_surface_slices, magnify_circular_regions_plot
    from .movie import plot_movie
    from .parameter import plot_parameter
    from .spatial import plot_spatial_reference, reference_scan_spatial
    from .spin import spin_colored_spectrum, spin_difference_spectrum, spin_polarized_spectrum
    from .stack_plot import (
        flat_stack_plot,
        offset_scatter_plot,
        stack_dispersion_plot,
        waterfall_dispersion,
    )
    from .ui import (
        DifferentiateApp,
        ProfileApp,
        SmoothingApp,
        TailorApp,
        concat_along_phi_ui,
        fit_inspection,
        profile_view,
    )
    from .utils import fancy_labels, remove_colorbars, savefig, simple_ax_grid

# モジュール名マップ
_module_map = {
    "annotate_cuts": "arpes.plotting.annotations",
    "annotate_experimental_conditions": "arpes.plotting.annotations",
    "annotate_point": "arpes.plotting.annotations",
    "plot_with_bands": "arpes.plotting.bands",
    "make_overview": "arpes.plotting.basic",
    "make_reference_plots": "arpes.plotting.basic",
    "dark_background": "arpes.plotting.dark_bg",
    "h_gradient_fill": "arpes.plotting.decoration",
    "v_gradient_fill": "arpes.plotting.decoration",
    "cut_dispersion_plot": "arpes.plotting.dispersion",
    "fancy_dispersion": "arpes.plotting.dispersion",
    "hv_reference_scan": "arpes.plotting.dispersion",
    "labeled_fermi_surface": "arpes.plotting.dispersion",
    "plot_dispersion": "arpes.plotting.dispersion",
    "reference_scan_fermi_surface": "arpes.plotting.dispersion",
    "scan_var_reference_plot": "arpes.plotting.dispersion",
    "plot_core_levels": "arpes.plotting.dos",
    "plot_dos": "arpes.plotting.dos",
    "fermi_edge_reference": "arpes.plotting.fermi_edge",
    "fermi_surface_slices": "arpes.plotting.fermi_surface",
    "magnify_circular_regions_plot": "arpes.plotting.fermi_surface",
    "plot_movie": "arpes.plotting.movie",
    "plot_parameter": "arpes.plotting.parameter",
    "plot_spatial_reference": "arpes.plotting.spatial",
    "reference_scan_spatial": "arpes.plotting.spatial",
    "spin_colored_spectrum": "arpes.plotting.spin",
    "spin_difference_spectrum": "arpes.plotting.spin",
    "spin_polarized_spectrum": "arpes.plotting.spin",
    "flat_stack_plot": "arpes.plotting.stack_plot",
    "offset_scatter_plot": "arpes.plotting.stack_plot",
    "stack_dispersion_plot": "arpes.plotting.stack_plot",
    "waterfall_dispersion": "arpes.plotting.stack_plot",
    "DifferentiateApp": "arpes.plotting.ui",
    "ProfileApp": "arpes.plotting.ui",
    "SmoothingApp": "arpes.plotting.ui",
    "TailorApp": "arpes.plotting.ui",
    "concat_along_phi_ui": "arpes.plotting.ui",
    "fit_inspection": "arpes.plotting.ui",
    "profile_view": "arpes.plotting.ui",
    "fancy_labels": "arpes.plotting.utils",
    "remove_colorbars": "arpes.plotting.utils",
    "savefig": "arpes.plotting.utils",
    "simple_ax_grid": "arpes.plotting.utils",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy import for public API.

    Note: The return type is annotated as `Any` because the attributes
    can be functions, classes, or other objects, and their types vary
    dynamically at runtime. A static type cannot be determined, so `Any`
    is necessary to satisfy type checkers.
    """
    if name not in _module_map:
        msg = f"module {__name__} has no attribute {name}"
        raise AttributeError(msg)
    mod = __import__(_module_map[name], fromlist=[name])
    return getattr(mod, name)
