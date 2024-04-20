"""Calculates momentum bounds from the angular coordinates.

Mostly these are used as common helper routines to the coordinate conversion code,
which is responsible for actually outputting the desired bounds.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from arpes.constants import K_INV_ANGSTROM

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

__all__ = (
    "calculate_kp_kz_bounds",
    "calculate_kx_ky_bounds",
    "calculate_kp_bounds",
    "full_angles_to_k",
)


def full_angles_to_k(  # noqa: PLR0913
    kinetic_energy: NDArray[np.float_] | float,
    phi: float | NDArray[np.float_],
    psi: float | NDArray[np.float_],
    alpha: float | NDArray[np.float_],
    beta: float | NDArray[np.float_],
    theta: float | NDArray[np.float_],
    chi: float | NDArray[np.float_],
    inner_potential: float,
) -> tuple[float, float, float] | tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    """Converts from the full set of standard PyARPES angles to momentum.

    More details on angle to momentum conversion can be found at
    `the momentum conversion notes <https://arpes.readthedocs.io/momentum-conversion>`.

    Args:
        kinetic_energy (float | xr.DataArray): kinetic energy
        phi (float): angle along analyzer
        psi (float): analyzer deflector angle
        alpha (float): analyzer rotation angle
        beta (float): scan angle perpendicular to theta
        theta (float): goniometer azimuthal angle
        chi (float): sample azimuthal angle
        inner_potential (float): material inner potential in eV

    Returns:
        [(float, float, float)]: [(kx, ky, kz)]
    """
    chi = -chi

    # use the full direct momentum conversion
    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
    sin_beta, cos_beta = np.sin(beta), np.cos(beta)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_chi, cos_chi = np.sin(chi), np.cos(chi)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)

    vx = cos_alpha * cos_psi * sin_phi - sin_alpha * sin_psi
    vy = sin_alpha * cos_psi * sin_phi + cos_alpha * sin_psi
    vz = cos_phi * cos_psi

    # perform theta rotation
    vrtheta_x = cos_theta * vx - sin_theta * vz
    vrtheta_y = vy
    vrtheta_z = sin_theta * vx + cos_theta * vz

    # perform beta rotation
    vrbeta_x = vrtheta_x
    vrbeta_y = cos_beta * vrtheta_y - sin_beta * vrtheta_z
    vrbeta_z = sin_beta * vrtheta_y + cos_beta * vrtheta_z

    # Perform chi rotation
    vrchi_x = cos_chi * vrbeta_x - sin_chi * vrbeta_y
    vrchi_y = sin_chi * vrbeta_x + cos_chi * vrbeta_y
    vrchi_z = vrbeta_z

    v_par_sq = vrchi_x**2 + vrchi_y**2

    """
    velocity -> momentum in each of parallel and perpendicular directions
    in the perpendicular case, we need the value of the cos^2(zeta) for the polar declination
    angle zeta in the sample (emission) frame. The total in plane velocity v_parallel is
    proportional to sin(zeta), so by the trig identity:

    1 = cos^2(zeta) + sin^2(zeta)

    we may substitute cos^2(zeta) for 1 - sin^2(zeta)
    which is 1 - (vrchi_x **2 + vrchi_y ** 2) above.
    """
    k_par = K_INV_ANGSTROM * np.sqrt(kinetic_energy)
    k_perp = K_INV_ANGSTROM * np.sqrt(kinetic_energy * (1 - v_par_sq) + inner_potential)

    return k_par * vrchi_x, k_par * vrchi_y, k_perp * vrchi_z


def euler_to_kx(
    kinetic_energy: NDArray[np.float_],
    phi: NDArray[np.float_] | float,
    beta: NDArray[np.float_] | float,
    theta: float = 0,
    *,
    slit_is_vertical: bool = False,
) -> NDArray[np.float_]:
    """Calculates kx from the phi/beta Euler angles given the experimental geometry."""
    if slit_is_vertical:
        return K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(beta) * np.cos(phi)
    return K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(phi + theta)


def euler_to_ky(
    kinetic_energy: NDArray[np.float_],
    phi: NDArray[np.float_] | float,
    beta: NDArray[np.float_] | float,
    theta: float = 0,
    *,
    slit_is_vertical: bool = False,
) -> NDArray[np.float_]:
    """Calculates ky from the phi/beta Euler angles given the experimental geometry."""
    if slit_is_vertical:
        return (
            K_INV_ANGSTROM
            * np.sqrt(kinetic_energy)
            * (np.cos(theta) * np.sin(phi) + np.cos(beta) * np.cos(phi) * np.sin(theta))
        )
    return K_INV_ANGSTROM * np.sqrt(kinetic_energy) * (np.cos(phi + theta) * np.sin(beta),)


def euler_to_kz(  # noqa: PLR0913
    kinetic_energy: NDArray[np.float_],
    phi: NDArray[np.float_] | float,
    beta: NDArray[np.float_] | float,
    theta: float = 0,
    inner_potential: float = 10,
    *,
    slit_is_vertical: bool = False,
) -> NDArray[np.float_]:
    """Calculates kz from the phi/beta Euler angles given the experimental geometry."""
    if slit_is_vertical:
        beta_term = -np.sin(theta) * np.sin(phi) + np.cos(theta) * np.cos(beta) * np.cos(phi)
    else:
        beta_term = np.cos(phi + theta) * np.cos(beta)
    return K_INV_ANGSTROM * np.sqrt(kinetic_energy * beta_term**2 + inner_potential)


def spherical_to_kx(
    kinetic_energy: float,
    theta: float,
    phi: float,
) -> float:
    """Calculates kx from the sample spherical (emission, not measurement) coordinates."""
    return K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(theta) * np.cos(phi)


def spherical_to_ky(
    kinetic_energy: float,
    theta: float,
    phi: float,
) -> float:
    """Calculates ky from the sample spherical (emission, not measurement) coordinates."""
    return K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(theta) * np.sin(phi)


def spherical_to_kz(
    kinetic_energy: float,
    theta: float,
    inner_potential: float,
) -> float:
    r"""Calculates the out of plane momentum from sample spherical (not measurement) coordinates.

    K_INV_ANGSTROM encodes that k_z = \frac{\sqrt{2 * m * E_kin * \cos^2\theta + V_0}}{\hbar}

    Args:
        kinetic_energy: kinetic energy (E_kin)
        theta: angle theta
        phi: angle phi
        inner_potential(float): inner potential (V_0)

    Returns:
        The out of plane momentum, kz.
    """
    return K_INV_ANGSTROM * np.sqrt(kinetic_energy * np.cos(theta) ** 2 + inner_potential)


def calculate_kp_kz_bounds(arr: xr.DataArray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculates kp and kz bounds for angle-hv Fermi surfaces."""
    phi_offset = arr.S.phi_offset
    phi_min = arr.coords["phi"].min().item() - phi_offset
    phi_max = arr.coords["phi"].max().item() - phi_offset
    binding_energy_min, binding_energy_max = (
        arr.coords["eV"].min().item(),
        arr.coords["eV"].max().item(),
    )
    hv_min, hv_max = arr.coords["hv"].min().item(), arr.coords["hv"].max().item()
    wf = arr.S.analyzer_work_function  # <= **FIX ME!!**
    kx_min = min(
        spherical_to_kx(hv_max - binding_energy_max - wf, phi_min, 0.0),
        spherical_to_kx(hv_min - binding_energy_max - wf, phi_min, 0.0),
    )
    kx_max = max(
        spherical_to_kx(hv_max - binding_energy_max - wf, phi_max, 0.0),
        spherical_to_kx(hv_min - binding_energy_max - wf, phi_max, 0.0),
    )
    angle_max = max(abs(phi_min), abs(phi_max))
    assert isinstance(angle_max, float)
    inner_V = arr.S.inner_potential
    kz_min = spherical_to_kz(hv_min + binding_energy_min - wf, angle_max, inner_V)
    kz_max = spherical_to_kz(hv_max + binding_energy_max - wf, 0.0, inner_V)
    return (
        (round(kx_min, 2), round(kx_max, 2)),  # kp
        (round(kz_min, 2), round(kz_max, 2)),  # kz
    )


def calculate_kp_bounds(arr: xr.DataArray) -> tuple[float, float]:
    """Calculates kp bounds for a single ARPES cut.

    Args:
        arr (xr.DataArray): ARPES 'cut'-type (the number of the anglar axis is 1 ("phi")) data

    Returns (tuple[float, float]):
        Minimum and maximum value of K region from the ARPES data
    """
    phi_coords = arr.coords["phi"].values - arr.S.phi_offset
    beta = float(arr.coords["beta"]) - arr.S.beta_offset

    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    phi_mid = (phi_high + phi_low) / 2

    sampled_phi_values = np.array([phi_low, phi_mid, phi_high])

    if arr.S.energy_notation == "Binding":
        max_kinetic_energy = max(
            arr.coords["eV"].max().item(),
            arr.S.hv - arr.S.analyzer_work_function,
        )
    elif arr.S.energy_notation == "Kinetic":
        max_kinetic_energy = arr.coords["eV"].max().item()
    else:
        warnings.warn(
            "Energyi notation is not specified. Assume the Binding energy notatation",
            stacklevel=2,
        )
        max_kinetic_energy = max(
            arr.coords["eV"].max().item(),
            arr.S.hv - arr.S.analyzer_work_function,
        )
    kps = K_INV_ANGSTROM * np.sqrt(max_kinetic_energy) * np.sin(sampled_phi_values) * np.cos(beta)
    return round(np.min(kps), 2), round(np.max(kps), 2)


def calculate_kx_ky_bounds(
    arr: xr.DataArray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculates the kx and ky range for a dataset with a fixed photon energy.

    This is used to infer the gridding that should be used for a k-space conversion.
    Based on Jonathan Denlinger's old codes

    Args:
        arr: Dataset that includes a key indicating the photon energy of
          the scan

    Returns:
        ((kx_low, kx_high,), (ky_low, ky_high,))
    """
    phi_coords, beta_coords = (
        arr.coords["phi"] - arr.S.phi_offset,
        arr.coords["beta"] - arr.S.beta_offset,
    )
    # Sample hopefully representatively along the edges
    phi_low: float
    phi_high: float
    phi_low, phi_high = phi_coords.min().item(), phi_coords.max().item()
    phi_mid: float = (phi_high + phi_low) / 2
    sampled_phi_values: NDArray[np.float_] = np.array(
        [
            phi_high,
            phi_high,
            phi_mid,
            phi_low,
            phi_low,
            phi_low,
            phi_mid,
            phi_high,
            phi_high,
        ],
    )
    beta_low, beta_high = beta_coords.min().item(), beta_coords.max().item()
    beta_mid = (beta_high + beta_low) / 2
    sampled_beta_values = np.array(
        [
            beta_mid,
            beta_high,
            beta_high,
            beta_high,
            beta_mid,
            beta_low,
            beta_low,
            beta_low,
            beta_mid,
        ],
    )
    if arr.S.energy_notation == "Binding":
        kinetic_energy = max(
            arr.coords["eV"].max().item(),
            arr.S.hv - arr.S.analyzer_work_function,
        )
    elif arr.S.energy_notation == "Kinetic":
        kinetic_energy = arr.coords["eV"].max().item()
    else:
        warnings.warn(
            "Energy notation is not specified. Assume the Binding energy notation",
            stacklevel=2,
        )
        kinetic_energy = max(
            arr.coords["eV"].max().item(),
            arr.S.hv - arr.S.analyzer_work_function,
        )
    # note that the type of the kinetic_energy is float in below.
    kxs: NDArray[np.float_] = K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(sampled_phi_values)
    kys: NDArray[np.float_] = (
        K_INV_ANGSTROM
        * np.sqrt(kinetic_energy)
        * np.cos(sampled_phi_values)
        * np.sin(sampled_beta_values)
    )
    return (
        (round(np.min(kxs), 2).astype(float), round(np.max(kxs), 2).astype(float)),
        (round(np.min(kys), 2).astype(float), round(np.max(kys), 2).astype(float)),
    )
