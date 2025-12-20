"""Contains common ARPES analysis routines with lazy import to avoid circular dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "align",
    "apply_mask",
    "apply_mask_to_coords",
    "approximate_core_levels",
    "boxcar_filter_arr",
    "build_crosscorrelation",
    "condense",
    "curvature1d",
    "curvature2d",
    "curves_along_pocket",
    "d1_along_axis",
    "d2_along_axis",
    "deconvolve_ice",
    "deconvolve_rl",
    "determine_broadened_fermi_distribution",
    "dn_along_axis",
    "edcs_along_pocket",
    "estimate_bare_band",
    "factor_analysis_along",
    "fit_bands",
    "fit_fermi_edge",
    "fit_for_effective_mass",
    "fit_for_self_energy",
    "gaussian_filter_arr",
    "ica_along",
    "kfermi_from_mdcs",
    "make_psf1d",
    "minimum_gradient",
    "nmf_along",
    "normalize_by_fermi_dirac",
    "normalize_by_fermi_distribution",
    "pca_along",
    "pocket_parameters",
    "polys_to_mask",
    "quasiparticle_lifetime",
    "radial_edcs_along_pocket",
    "raw_poly_to_mask",
    "rebin",
    "relative_change",
    "savgol_filter_multi",
    "savitzky_golay_filter",
    "symmetrize",
    "symmetrize_axis",
    "to_self_energy",
    "unpack_bands_from_fit",
]

if TYPE_CHECKING:
    from .align import align
    from .band_analysis import fit_bands, fit_for_effective_mass, unpack_bands_from_fit
    from .decomposition import factor_analysis_along, ica_along, nmf_along, pca_along
    from .deconvolution import deconvolve_ice, deconvolve_rl, make_psf1d
    from .derivative import (
        curvature1d,
        curvature2d,
        d1_along_axis,
        d2_along_axis,
        dn_along_axis,
        minimum_gradient,
    )
    from .filters import (
        boxcar_filter_arr,
        gaussian_filter_arr,
        savgol_filter_multi,
        savitzky_golay_filter,
    )
    from .gap import determine_broadened_fermi_distribution, normalize_by_fermi_dirac, symmetrize
    from .general import (
        condense,
        fit_fermi_edge,
        normalize_by_fermi_distribution,
        rebin,
        symmetrize_axis,
    )
    from .kfermi import kfermi_from_mdcs
    from .mask import apply_mask, apply_mask_to_coords, polys_to_mask, raw_poly_to_mask
    from .pocket import (
        curves_along_pocket,
        edcs_along_pocket,
        pocket_parameters,
        radial_edcs_along_pocket,
    )
    from .self_energy import (
        estimate_bare_band,
        fit_for_self_energy,
        quasiparticle_lifetime,
        to_self_energy,
    )
    from .tarpes import build_crosscorrelation, relative_change
    from .xps import approximate_core_levels


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy import functions to avoid circular imports while keeping public API.

    Note: The return type is annotated as `Any` because the attributes
    can be functions, classes, or other objects, and their types vary
    dynamically at runtime. A static type cannot be determined, so `Any`
    is necessary to satisfy type checkers.
    """
    if name not in __all__:
        msg = f"module {__name__} has no attribute {name}"
        raise AttributeError(msg)

    _module_map = {
        "align": "arpes.analysis.align",
        "fit_bands": "arpes.analysis.band_analysis",
        "fit_for_effective_mass": "arpes.analysis.band_analysis",
        "unpack_bands_from_fit": "arpes.analysis.band_analysis",
        "factor_analysis_along": "arpes.analysis.decomposition",
        "ica_along": "arpes.analysis.decomposition",
        "nmf_along": "arpes.analysis.decomposition",
        "pca_along": "arpes.analysis.decomposition",
        "deconvolve_ice": "arpes.analysis.deconvolution",
        "deconvolve_rl": "arpes.analysis.deconvolution",
        "make_psf1d": "arpes.analysis.deconvolution",
        "curvature1d": "arpes.analysis.derivative",
        "curvature2d": "arpes.analysis.derivative",
        "d1_along_axis": "arpes.analysis.derivative",
        "d2_along_axis": "arpes.analysis.derivative",
        "dn_along_axis": "arpes.analysis.derivative",
        "minimum_gradient": "arpes.analysis.derivative",
        "boxcar_filter_arr": "arpes.analysis.filters",
        "gaussian_filter_arr": "arpes.analysis.filters",
        "savgol_filter_multi": "arpes.analysis.filters",
        "savitzky_golay_filter": "arpes.analysis.filters",
        "determine_broadened_fermi_distribution": "arpes.analysis.gap",
        "normalize_by_fermi_dirac": "arpes.analysis.gap",
        "symmetrize": "arpes.analysis.gap",
        "condense": "arpes.analysis.general",
        "fit_fermi_edge": "arpes.analysis.general",
        "normalize_by_fermi_distribution": "arpes.analysis.general",
        "rebin": "arpes.analysis.general",
        "symmetrize_axis": "arpes.analysis.general",
        "kfermi_from_mdcs": "arpes.analysis.kfermi",
        "apply_mask": "arpes.analysis.mask",
        "apply_mask_to_coords": "arpes.analysis.mask",
        "polys_to_mask": "arpes.analysis.mask",
        "raw_poly_to_mask": "arpes.analysis.mask",
        "curves_along_pocket": "arpes.analysis.pocket",
        "edcs_along_pocket": "arpes.analysis.pocket",
        "pocket_parameters": "arpes.analysis.pocket",
        "radial_edcs_along_pocket": "arpes.analysis.pocket",
        "estimate_bare_band": "arpes.analysis.self_energy",
        "fit_for_self_energy": "arpes.analysis.self_energy",
        "quasiparticle_lifetime": "arpes.analysis.self_energy",
        "to_self_energy": "arpes.analysis.self_energy",
        "build_crosscorrelation": "arpes.analysis.tarpes",
        "relative_change": "arpes.analysis.tarpes",
        "approximate_core_levels": "arpes.analysis.xps",
    }

    module_name = _module_map[name]
    mod = __import__(module_name, fromlist=[name])
    return getattr(mod, name)
