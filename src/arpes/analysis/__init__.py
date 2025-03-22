"""Contains common ARPES analysis routines."""

# pyright: reportUnusedImport=false
from __future__ import annotations

from .align import align
from .band_analysis import fit_bands, fit_for_effective_mass, unpack_bands_from_fit
from .band_analysis_utils import param_getter, param_stderr_getter
from .decomposition import (
    factor_analysis_along,
    ica_along,
    nmf_along,
    pca_along,
)
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
    boxcar_filter,
    boxcar_filter_arr,
    gaussian_filter,
    gaussian_filter_arr,
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
from .tarpes import build_crosscorrelation, normalized_relative_change, relative_change
from .xps import approximate_core_levels
