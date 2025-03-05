"""Provides standard corrections for datasets.

Largely, this covers:
1. Fermi edge corrections
2. Background estimation and subtraction

It also contains utilities related to identifying a piece of data
earlier in a dataset which can be used to furnish equivalent references.

"""

# pyright: reportUnusedImport=false
from __future__ import annotations

from .background import remove_incoherent_background
from .coords import (
    adjust_coords_to_limit,
    corrected_coords,
    extend_coords,
    is_equally_spaced,
    shift_by,
)
from .fermi_edge import (
    apply_direct_fermi_edge_correction,
    apply_photon_energy_fermi_edge_correction,
    apply_quadratic_fermi_edge_correction,
    build_direct_fermi_edge_correction,
    build_photon_energy_fermi_edge_correction,
    build_quadratic_fermi_edge_correction,
    find_e_fermi_linear_dos,
)
from .intensity_map import shift
