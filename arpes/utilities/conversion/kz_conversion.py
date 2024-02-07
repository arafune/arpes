"""Coordinate conversion classes for photon energy scans."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numba
import numpy as np

from arpes.constants import HV_CONVERSION, K_INV_ANGSTROM

from .base import K_SPACE_BORDER, MOMENTUM_BREAKPOINTS, CoordinateConverter
from .bounds_calculations import calculate_kp_kz_bounds
from .calibration import DetectorCalibration

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import MOMENTUM

__all__ = ["ConvertKpKzV0", "ConvertKxKyKz", "ConvertKpKz"]


@numba.njit(parallel=True, cache=True)
def _kspace_to_hv(
    kp: NDArray[np.float_],
    kz: NDArray[np.float_],
    hv: NDArray[np.float_],
    energy_shift: NDArray[np.float_],
    *,
    is_constant_shift: bool,
) -> None:
    """Efficiently perform the inverse coordinate transform to photon energy."""
    shift_ratio = 0 if is_constant_shift else 1

    for i in numba.prange(len(kp)):
        hv[i] = HV_CONVERSION * (kp[i] ** 2 + kz[i] ** 2) + energy_shift[i * shift_ratio]


@numba.njit(parallel=True, cache=True)
def _kp_to_polar(
    kinetic_energy: NDArray[np.float_],
    kp: NDArray[np.float_],
    phi: NDArray[np.float_],
    inner_potential: float,
    angle_offset: float,
) -> None:
    """Efficiently performs the inverse coordinate transform phi(hv, kp)."""
    for i in numba.prange(len(kp)):
        phi[i] = (
            np.arcsin(kp[i] / (K_INV_ANGSTROM * np.sqrt(kinetic_energy[i] + inner_potential)))
            + angle_offset
        )


class ConvertKpKzV0(CoordinateConverter):
    """Implements inner potential broadcasted hv Fermi surfaces."""

    # TODO(<RA>): implement
    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """TODO, implement this."""
        super().__init__(*args, **kwargs)
        raise NotImplementedError


class ConvertKxKyKz(CoordinateConverter):
    """Implements 4D data volume conversion."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """TODO, implement this."""
        super().__init__(*args, **kwargs)
        raise NotImplementedError


class ConvertKpKz(CoordinateConverter):
    """Implements single angle photon energy scans."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Cache the photon energy coordinate we calculate backwards from kz."""
        super().__init__(*args, **kwargs)
        self.hv: NDArray[np.float_] | None = None

    def get_coordinates(
        self,
        resolution: Incomplete | None = None,
        bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    ) -> dict[str, NDArray[np.float_]]:
        """Calculates appropriate coordinate bounds."""
        if resolution is None:
            resolution = {}
        if bounds is None:
            bounds = {}
        coordinates = super().get_coordinates(resolution=resolution, bounds=bounds)
        ((kp_low, kp_high), (kz_low, kz_high)) = calculate_kp_kz_bounds(self.arr)
        if "kp" in bounds:
            kp_low, kp_high = bounds["kp"]
        if "kz" in bounds:
            kz_low, kz_high = bounds["kz"]
        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["phi"])
        inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][-1]
        # go a bit finer here because it would otherwise be very coarse
        inferred_kz_res = (kz_high - kz_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["hv"])
        inferred_kz_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kz_res][-1]

        coordinates["kp"] = np.arange(
            kp_low - K_SPACE_BORDER,
            kp_high + K_SPACE_BORDER,
            resolution.get("kp", inferred_kp_res),
        )
        coordinates["kz"] = np.arange(
            kz_low - K_SPACE_BORDER,
            kz_high + K_SPACE_BORDER,
            resolution.get("kz", inferred_kz_res),
        )
        base_coords = {
            str(k): v for k, v in self.arr.coords.items() if k not in ["eV", "phi", "hv"]
        }  # should v.values ?
        coordinates.update(base_coords)
        return coordinates

    def kspace_to_hv(
        self,
        binding_energy: NDArray[np.float_],
        kp: NDArray[np.float_],
        kz: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Converts from momentum back to the raw photon energy."""
        if self.hv is None:
            inner_v = self.arr.S.inner_potential
            wf = self.arr.S.analyzer_work_function

            is_constant_shift = True
            if not isinstance(binding_energy, np.ndarray):
                is_constant_shift = True
                binding_energy = np.array([binding_energy])

            self.hv = np.zeros_like(kp)
            _kspace_to_hv(
                kp,
                kz,
                self.hv,
                -inner_v - binding_energy + wf,
                is_constant_shift=is_constant_shift,
            )  # <== **FIX ME**

        return self.hv

    def kspace_to_phi(
        self,
        binding_energy: NDArray[np.float_],
        kp: NDArray[np.float_],
        kz: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Converts from momentum back to the hemisphere angle axis.

        Args:
            binding_energy(NDArray[np.float_]): [TODO:description]
            kp (NDArray[np.float_]): [TODO:description]
            kz (NDArray[np.float_]): [TODO:description]

        Returns:
            [TODO:description]
        """
        if self.phi is not None:
            return self.phi
        if self.hv is None:
            self.kspace_to_hv(binding_energy, kp, kz)
        assert self.hv is not None
        if self.arr.S.energy_notation == "Binding":
            kinetic_energy = binding_energy + self.hv - self.arr.S.analyzer_work_function
        elif self.arr.S.energy_notation == "Kinetic":
            kinetic_energy = binding_energy - self.arr.S.analyzer_work_function
        else:
            warnings.warn(
                "Energy notation is not specified. Assume the Binding energy notation",
                stacklevel=2,
            )
            kinetic_energy = binding_energy + self.hv - self.arr.S.analyzer_work_function
        self.phi = np.zeros_like(self.hv)
        _kp_to_polar(
            kinetic_energy,
            kp,
            self.phi,
            self.arr.S.inner_potential,
            self.arr.S.phi_offset,
        )
        if isinstance(self.calibration, DetectorCalibration):
            self.phi = self.calibration.correct_detector_angle(eV=binding_energy, phi=self.phi)
        return self.phi

    def conversion_for(self, dim: str) -> Callable[[NDArray[np.float_]], NDArray[np.float_]]:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""
        return {
            "eV": self.kspace_to_BE,
            "hv": self.kspace_to_hv,
            "phi": self.kspace_to_phi,
        }.get(dim, self.identity_transform)
