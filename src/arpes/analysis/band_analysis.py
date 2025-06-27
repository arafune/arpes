"""Provides some band analysis tools."""

from __future__ import annotations

import contextlib
import functools
import operator
from itertools import pairwise, permutations, product
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import LinearModel, LorentzianModel, QuadraticModel
from scipy.spatial import distance

from arpes import models
from arpes.constants import HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ, TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.models.band import Band
from arpes.provenance import update_provenance
from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward
from arpes.utilities.jupyter import get_tqdm

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

    from _typeshed import Incomplete
    from lmfit import Parameter
    from lmfit.model import ModelResult
    from numpy.typing import NDArray

    from arpes._typing import XrTypes
    from arpes.fits import ParametersArgs

__all__ = (
    "fit_bands",
    "fit_for_effective_mass",
    "unpack_bands_from_fit",
)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

tqdm = get_tqdm()


class BandDescription(TypedDict, total=False):
    """TypedDict Object for band_description."""

    band: Required[Band]
    name: str
    params: dict[Hashable, ParametersArgs]


def fit_for_effective_mass(
    data: xr.DataArray,
    fit_kwargs: dict | None = None,
) -> float:
    """Fits for the effective mass in a piece of data.

    Performs an effective mass fit by first fitting for Lorentzian lineshapes and then fitting
    a quadratic model to the result. This is an alternative to global effective mass fitting.

    In the case that data is provided in anglespace, the Lorentzian fits are performed in anglespace
    before being converted to momentum where the effective mass is extracted.

    We should probably include uncertainties here.

    Args:
        data (xr.DataArray): ARPES data
        fit_kwargs: Passthrough for arguments to `broadcast_model`, used internally to
          obtain the Lorentzian peak locations

    Returns:
        The effective mass in units of the bare mass.
    """
    fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
    mom_dim = next(
        dim for dim in ["kp", "kx", "ky", "kz", "phi", "beta", "theta"] if dim in data.dims
    )
    model = LorentzianModel() + LinearModel()
    fit_results = data.S.modelfit(coords="eV", model=model, **fit_kwargs)
    if mom_dim in {"phi", "beta", "theta"}:
        forward = convert_coordinates_to_kspace_forward(data)
        assert isinstance(forward, xr.Dataset)
        final_mom = next(dim for dim in ["kx", "ky", "kp", "kz"] if dim in forward)
        eVs = fit_results.results.F.p("a_center").values
        kps = [
            forward[final_mom].sel({mom_dim: ang}, eV=eV, method="nearest")
            for eV, ang in zip(eVs, data.coords[mom_dim].values, strict=True)
        ]
        quad_fit = QuadraticModel().fit(eVs, x=np.array(kps))

        return HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ / (2 * quad_fit.params["a"].value)
    quad_fit = QuadraticModel().fit(fit_results.modelfit_results.F.p("a_center"))
    return HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ / (2 * quad_fit.params["a"].value)


def unpack_bands_from_fit(
    band_results: xr.DataArray,
    weights: tuple[float, float, float] = (2, 0, 10),
) -> list[Band]:
    """Deconvolve the band identities of a series of overlapping bands.

    Sometimes through the fitting process, or across a place in the band structure where there is a
    nodal point, the identities of the bands across sequential fits can get mixed up.

    We can try to restore this identity by using the cosine similarity of fits, where the fit is
    represented as a vector by:

        v_band =  (sigma, amplitude, center) * weights
        weights = (5, 1/5, 10)

    For any point in the band structure, we find the closest place where we have fixed the band
    identities. Let the bands be indexed by i so that the bands are b_i and b_i_0 at the point of
    interest and at the reference respectively.

    Then, we calculate the matrix:
        s_ij = sim(b_i, b_j_0)

    The band identities are subsequently chosen so that the trace of this matrix is maximized among
    possible ways of labelling the bands b_i.

    The value of the weights parameter is chosen only to scale the dimensions so that they are
    closer to the same magnitude.

    Args:
        band_results (xr.DataArray): Results of spectrum fit.
            The value must be the array of lmfit.model.ModelResuls, which is the return of
            broadcast_model().results, in most case.
        weights (tuple[float, float, float]): weight values for sigma, amplitude, center

    Returns:
        Unpacked bands.
    """
    band_results = band_results if isinstance(band_results, xr.DataArray) else band_results.results
    band_names: list[str] = list(band_results.F.band_names)
    identified_band_results = _identified_band_results(
        band_results=band_results,
        weights=weights,
    )

    bands: list[Band] = []
    for i in range(len(band_names)):
        label = identified_band_results[0][i]

        def dataarray_for_value(
            param_name: Literal["center", "amplitude", "sigma", "gamma"],
            i: int = i,
            *,
            is_value: bool,
        ) -> xr.DataArray | None:
            """Return a DataArray representing the fit results for a specific parameter.

            This function retrieves the values (or standard errors) of a specified fit parameter
            (such as "center", "amplitude", "sigma", or "gamma") for each band in the
            `identified_band_results`. The result is returned as an `xr.DataArray`. If the parameter
            is not available in the fitting results for a given band, `None` is returned.

            Args:
                param_name (Literal["center", "amplitude", "sigma", "gamma"]): The name of the fit
                    parameter whose values are being retrieved (e.g., "center", "amplitude", etc.).
                i (int): Index for band names in the `identified_band_results` list. It is used to
                    identify the correct band in the results.
                is_value (bool): If `True`, the function returns the fit parameter's value; if
                    `False`, it returns the standard error (stderr) of the fit.

            Returns:
                xr.DataArray | None: An `xr.DataArray` containing the fit parameter values
                    (or stderr). Returns `None` if the corresponding parameter is not found for the
                    given index.
            """
            values: NDArray[np.float64] = np.zeros_like(
                band_results.values,
                dtype=float,
            )
            with np.nditer(values, flags=["multi_index"], op_flags=[["writeonly"]]) as it:
                while not it.finished:
                    prefix = identified_band_results[it.multi_index][i]
                    try:
                        param = band_results.values[it.multi_index].params[prefix + param_name]
                        it[0] = param.value if is_value else param.stderr
                    except KeyError:
                        return None
                    finally:
                        it.iternext()
            return band_results.G.with_values(values, keep_attrs=False)

        band_data = xr.Dataset({})
        center = dataarray_for_value(param_name="center", is_value=True)
        if center is not None:
            band_data.update(
                {
                    "center": center,
                    "center_stderr": dataarray_for_value(param_name="center", is_value=False),
                },
            )

        amplitude = dataarray_for_value(param_name="amplitude", is_value=True)
        if amplitude is not None:
            band_data.update(
                {
                    "amplitude": amplitude,
                    "amplitude_stderr": dataarray_for_value(param_name="amplitude", is_value=False),
                },
            )
        sigma = dataarray_for_value(param_name="sigma", is_value=True)
        if sigma is not None:
            band_data.update(
                {
                    "sigma": sigma,
                    "sigma_stderr": dataarray_for_value(param_name="sigma", is_value=False),
                },
            )
        gamma = dataarray_for_value(param_name="gamma", is_value=True)
        if gamma is not None:
            band_data.update(
                {
                    "gamma": gamma,
                    "gamma_stderr": dataarray_for_value(param_name="gamma", is_value=False),
                },
            )

        bands.append(Band(label, data=band_data))

    return bands


def _identified_band_results(
    band_results: xr.DataArray,
    weights: tuple[float, float, float] = (2, 0, 10),
) -> NDArray[np.object_]:
    """Helper function to generate identified band.

    Args:
        band_results (xr.DataArray): Results of spectrum fit.
            The value must be the array of lmfit.model.ModelResuls, which is the return of
            broadcast_model().results, in most case.
        weights (tuple[float, float, float]): weight values for sigma, amplitude, center

    Returns: NDArray[np.object_]
        identified_band_results. The each item of the list stores the order of the band name (suffix
        used in lmfit procedure.)
    """
    band_results = band_results if isinstance(band_results, xr.DataArray) else band_results.results
    prefixes: list[str] = [
        component.prefix for component in band_results.values[0].model.components
    ]
    identified_band_results: list[list[str]] = []
    identified_by_coordinate: dict[tuple[float, ...], tuple[list[str], ModelResult]] = {}
    for coordinate in band_results.G.iter_coords():
        fit_result: ModelResult = band_results.loc[coordinate].values.item()
        frozen_coord: tuple[float, ...] = tuple(coordinate[d] for d in band_results.dims)
        logger.debug(f"frozen_coord: {frozen_coord}")
        closest_identified: tuple[list[str], ModelResult] | None = None
        dist = np.inf
        for coord, identified_band in identified_by_coordinate.items():
            current_dist = np.dot(coord, frozen_coord)
            if current_dist < dist:
                closest_identified = identified_band
                dist = current_dist
        if closest_identified is None:
            closest_identified = (
                [c.prefix for c in fit_result.model.components],
                fit_result,
            )
            logger.debug(f"closest_identified: {closest_identified}")
            identified_by_coordinate[frozen_coord] = closest_identified
        closest_prefixes, closest_fit = closest_identified
        mat_shape: tuple[int, int] = (len(prefixes), len(prefixes))
        dist_mat: NDArray[np.float64] = np.zeros(shape=mat_shape)
        for i, prefix_i in enumerate(prefixes):
            for j, prefix_j in enumerate(closest_prefixes):
                dist_mat[i, j] = distance.euclidean(
                    _modelresult_to_array(
                        model_fit=fit_result,
                        prefix=prefix_i,
                        weights=weights,
                    ),
                    _modelresult_to_array(
                        model_fit=closest_fit,
                        prefix=prefix_j,
                        weights=weights,
                    ),
                )

        best_arrangement: tuple[int, ...] = min(
            permutations(range(len(prefixes))),
            key=lambda p: sum(dist_mat[i, p_i] for i, p_i in enumerate(p)),
        )
        ordered_prefixes: list[str] = [closest_prefixes[p_i] for p_i in best_arrangement]
        identified_by_coordinate[frozen_coord] = ordered_prefixes, fit_result
        identified_band_results.append(ordered_prefixes)

    return np.asarray(identified_band_results, dtype=np.object_)


def _modelresult_to_array(
    model_fit: ModelResult,
    prefix: str = "",
    weights: tuple[float, float, float] = (2, 0, 10),
) -> NDArray[np.float64]:
    """Convert ModelResult to a weighted NDArray of fit parameter values.

    This function extracts the values and standard errors for the parameters
    "sigma", "gamma", "amplitude", and "center" from the `model_fit` object,
    applies weights for each parameter (sigma, amplitude, center), and
    returns the result as a NumPy array.

    If any parameter is missing from `model_fit`, a default value and
    standard error are assigned. The weights are applied to the parameters
    during the conversion process.

    Args:
        model_fit (ModelResult): The model fitting result containing the parameters.
        prefix (str): Prefix to be added to parameter names for identification.
        weights (tuple[float, float, float]): Weights for the parameters in the order
            (sigma, amplitude, center). Default is (2, 0, 10).

    Returns:
        NDArray[np.float64]: A NumPy array containing the weighted parameter values.
    """
    parameter_names: set[str] = set(model_fit.params.keys())
    if prefix + "sigma" in parameter_names:
        param_width: Parameter = model_fit.params[prefix + "sigma"]
    else:
        param_width = lf.Parameter(name=prefix + "sigma", value=1)
        param_width.stderr = 1
        weights = (0.0, weights[1], weights[2])
    if prefix + "gamma" in parameter_names:
        param_width = model_fit.params[prefix + "gamma"]
    if prefix + "amplitude" in parameter_names:
        param_amplitude = model_fit.params[prefix + "amplitude"]
    else:
        param_amplitude = lf.Parameter(name=prefix + "amplitude", value=1)
        param_amplitude.stderr = 1
        weights = (weights[0], 0.0, weights[2])

    stderr: NDArray[np.float64] = np.array(
        [
            param_width.stderr,
            param_amplitude.stderr,
            model_fit.params[prefix + "center"].stderr,
        ],
    )
    return (
        np.array(
            [
                param_width.value,
                param_amplitude.value,
                model_fit.params[prefix + "center"].value,
            ],
        )
        * weights
        / (1 + stderr)
    )


@update_provenance("Fit bands from pattern")
def fit_patterned_bands(  # noqa: PLR0913
    arr: xr.DataArray,
    band_set: dict[Incomplete, Incomplete],
    fit_direction: str = "",
    stray: float | None = None,
    *,
    background: bool | type[Band] = True,
    interactive: bool = True,
    dataset: bool = True,
) -> XrTypes:
    """Fits bands and determines dispersion in a region of a spectrum.

    The dimensions of the dataset are partitioned into three types:

    1. Fit directions: Coordinates along the 1D (or later 2D) marginals, e.g., energy (E).
    2. Broadcast directions: Directions used to interpolate against the patterned, e.g., k.
    3. Free directions: Broadcasted directions not used to extract the initial parameter values.

    For example, in a spectrum at delta_t=0, if using MDCs, `k_p` could be the fit direction,
    `E` the broadcast direction, and `delay` a free direction.

    Args:
        arr (xr.DataArray): The data array containing the spectrum to fit.
        band_set (dict[Incomplete, Incomplete]): A dictionary defining the bands and points along
            the spectrum.
        fit_direction (str): The direction to fit the data (e.g., "energy").
        stray (float, optional): A parameter used for adjusting fits. Defaults to None.
        background (bool | type[Band]): If True, includes background fitting, otherwise specifies
            the background band class.
        interactive (bool): If True, show an interactive progress bar.
        dataset (bool): If True, return the results as an `xr.Dataset`. If False, return just the
            `band_results`.

    Returns:
        XrTypes: Either an `xr.DataArray` or an `xr.Dataset` depending on the `dataset` argument.
        The returned object contains fitting results, residuals, and normalized residuals.
    """
    free_directions = [dim for dim in arr.dims if str(dim) != fit_direction]

    def resolve_partial_bands_from_description(  # noqa: PLR0913
        coord_dict: dict[str, Incomplete],
        marginal: xr.DataArray | None = None,
        name: str = "",
        band: Band | None = None,
        dims: list[str] | tuple[str, ...] | None = None,
        params: ParametersArgs | None = None,
        points: Incomplete = None,
    ) -> list[BandDescription]:
        # You don't need to supply a marginal, but it is useful because it allows estimation of the
        # initial value for the amplitude from the approximate peak location
        params = params or {}
        dims = dims or ()
        assert band is not None
        coord_name = next(d for d in dims if d in coord_dict)
        partial_band_locations = list(
            _interpolate_intersecting_fragments(
                coord=coord_dict[coord_name],
                coord_index=arr.dims.index(coord_name),
                points=points or [],
            ),
        )
        return [
            {
                "band": band,
                "name": f"{name}_{i}",
                "params": _correct_params_with_stray_marginal(
                    params=params,
                    center=band_center,
                    center_stray=params.get("stray", stray),
                    marginal=marginal,
                ),
            }
            for i, (_, band_center) in enumerate(partial_band_locations)
        ]

    template = arr.sum(fit_direction)
    band_results = template.G.with_values(
        np.ndarray(shape=template.values.shape, dtype=object),
    )

    total_slices = np.prod([len(arr.coords[d]) for d in free_directions])
    for coord_dict in tqdm(
        arr.G.iter_coords(free_directions),
        interactive=interactive,
        desc="fitting",  # Prefix for the progressbar.
        total=total_slices,  # The number of expected iterations. If unspecified,
    ):
        marginal = arr.sel(coord_dict)
        partial_bands = [
            resolve_partial_bands_from_description(
                coord_dict=coord_dict,
                marginal=marginal,
                **band_set_values,
            )
            for band_set_values in band_set.values()
        ]

        partial_bands = [p for p in partial_bands if len(p)]

        if background is not None and partial_bands:
            partial_bands = [*partial_bands, [{"band": background, "name": "", "params": {}}]]

        internal_models = [_instantiate_band(b) for bs in partial_bands for b in bs]

        if not internal_models:
            band_results.loc[coord_dict] = None
            continue

        composite_model = functools.reduce(operator.add, internal_models)
        new_params = composite_model.make_params()
        fit_result = composite_model.fit(
            marginal.values,
            new_params,
            x=marginal.coords[next(iter(marginal.indexes))].values,
        )

        # populate models, sample code
        band_results.loc[coord_dict] = fit_result

    if not dataset:
        band_results.attrs["original_data"] = arr
        return band_results

    residual = arr.G.with_values(np.zeros(arr.shape))

    for coords in band_results.G.iter_coords():
        fit_item = band_results.sel(coords).item()
        if fit_item is None:
            continue

        with contextlib.suppress(Exception):
            residual.loc[coords] = fit_item.residual

    return xr.Dataset(
        data_vars={
            "data": arr,
            "residual": residual,
            "results": band_results,
            "norm_residual": residual / arr,
        },
        coords=residual.coords,
    )


def _is_between(x: float, y0: float, y1: float) -> bool:
    y0, y1 = np.min([y0, y1]), np.max([y0, y1])
    return y0 <= x <= y1


def _instantiate_band(partial_band: dict[str, Any]) -> lf.Model:
    phony_band = partial_band["band"](partial_band["name"])
    built = phony_band.fit_cls(prefix=partial_band["name"], missing="drop")
    for constraint_coord, params in partial_band["params"].items():
        if constraint_coord == "stray":
            continue
        built.set_param_hint(constraint_coord, **params)
    return built


def fit_bands(
    arr: xr.DataArray,
    band_descriptions: list[BandDescription],
    direction: Literal["edc", "mdc", "EDC", "MDC"] = "mdc",
) -> tuple[xr.DataArray | None, None, ModelResult | None]:
    """Fits bands and determines dispersion in some region of a spectrum.

    Args:
        arr(xr.DataArray): ARPES data for fit.
        band_descriptions: List of the description of the bands to fit in the region
        direction: fit direction (along the enegy or momentum),
            default is "mdc" (Momentum Distribution Curve).

    Returns:
        Fitted bands.

    Todo:
        Deep refactoring. The current version may not work.
    """
    assert direction in {"edc", "mdc", "EDC", "MDC"}

    directions, broadcast_direction = list(arr.dims), "eV"

    if direction in {"mdc", "MDC"}:
        possible_directions = set(directions).intersection({"kp", "kx", "ky", "phi"})
        broadcast_direction = str(next(iter(possible_directions)))

    directions.remove(broadcast_direction)

    residual, _ = next(_iterate_marginals(arr, directions))
    residual = residual - np.min(residual.values)

    # Let the first band be given by fitting the raw data to this band
    # Find subsequent peaks by fitting models to the residuals
    raw_bands = [band_description.get("band") for band_description in band_descriptions]
    initial_fits = None
    all_fit_parameters = {}

    for band_description in band_descriptions:
        band_inst: Band = band_description.get("band")
        params = band_description.get("params", {})
        fit_model = band_inst.fit_cls(prefix=band_inst.label)
        initial_fit = fit_model.guess_fit(residual, params=params)
        if initial_fits is None:
            initial_fits = initial_fit.params
        else:
            initial_fits.update(initial_fit.params)

        residual = residual - initial_fit.best_fit
        if isinstance(band_inst, models.band.BackgroundBand):
            # This is an approximation to simulate a constant background band underneath the data
            # Because backgrounds are added to our model only after the initial sequence of fits.
            # This is by no means the most appropriate way to do this, just one that works
            # alright for now
            pass

    template = arr.sum(broadcast_direction)
    band_results = template.G.with_values(np.zeros_like(template.values))
    for marginal, coordinate in _iterate_marginals(arr, directions):
        # Use the closest parameters that have been successfully fit, or use the initial
        # parameters, this should be good enough because the order of the iterator will
        # be stable
        closest_model_params = initial_fits  # fix me
        dist = np.inf
        frozen_coordinate = tuple(coordinate[str(k)] for k in template.dims)
        for c, v in all_fit_parameters.items():
            delta = np.array(c) - frozen_coordinate
            current_distance = delta.dot(delta)
            if current_distance < dist and direction in {"mdc", "MDC"}:  # TODO: remove me
                closest_model_params = v

        # TODO: mix in any params to the model params

        # populate models
        internal_models = [band.fit_cls(prefix=band.label) for band in raw_bands]
        composite_model = functools.reduce(operator.add, internal_models)
        new_params = composite_model.make_params(
            **{k: v.value for k, v in closest_model_params.items()},
        )
        fit_result = composite_model.fit(
            marginal.values,
            new_params,
            x=marginal.coords[next(iter(marginal.indexes))].values,
        )

        # insert fit into the results, insert the parameters into the cache so that we have
        # fitting parameters for the next sequence
        band_results.loc[coordinate] = fit_result
        all_fit_parameters[frozen_coordinate] = fit_result.params

    # Unpack the band results
    unpacked_bands = None
    residual = None

    return band_results, unpacked_bands, residual  # Memo bunt_result is xr.DataArray


def _interpolate_intersecting_fragments(
    coord: Incomplete,
    coord_index: int,
    points: Incomplete,
) -> Iterator[Incomplete]:
    """Finds all consecutive pairs of points in `points`.

    Args:
        coord ([TODO:type]): [TODO:description]
        coord_index ([TODO:type]): [TODO:description]
        points ([TODO:type]): [TODO:description]
    """
    assert len(points[0]) == TWO_DIMENSION

    for point_low, point_high in pairwise(points):
        coord_other_index = 1 - coord_index

        check_coord_low, check_coord_high = point_low[coord_index], point_high[coord_index]
        if _is_between(coord, check_coord_low, check_coord_high):
            # this is unnecessarily complicated
            if check_coord_low < check_coord_high:
                yield (
                    coord,
                    (coord - check_coord_low)
                    / (check_coord_high - check_coord_low)
                    * (point_high[coord_other_index] - point_low[coord_other_index])
                    + point_low[coord_other_index],
                )
            else:
                yield (
                    coord,
                    (coord - check_coord_high)
                    / (check_coord_low - check_coord_high)
                    * (point_low[coord_other_index] - point_high[coord_other_index])
                    + point_high[coord_other_index],
                )


def _iterate_marginals(
    arr: xr.DataArray,
    iterate_directions: list[Hashable] | None = None,
) -> Iterator[tuple[xr.DataArray, dict[Hashable, float]]]:
    iterate_directions = (
        iterate_directions
        if iterate_directions is not None
        else [str(dim) for dim in arr.dims if dim != "eV"]
    )
    selectors = product(*[arr.coords[d] for d in iterate_directions])
    for ss in selectors:
        coords = dict(zip(iterate_directions, [float(s) for s in ss], strict=True))
        yield arr.sel(coords), coords


def _correct_params_with_stray_marginal(
    params: dict[Hashable, ParametersArgs],
    center: float,
    center_stray: float | None = None,
    marginal: xr.DataArray | None = None,
) -> dict[Hashable, ParametersArgs]:
    params["center"] = params.get("center", {})
    params.update({"center": {"value": center}})
    if center_stray is not None:
        params["center"]["min"] = center - center_stray
        params["center"]["max"] = center + center_stray
        params["sigma"] = params.get("sigma", {})
        params["sigma"]["value"] = center_stray
        if marginal is not None:
            near_center = marginal.sel(
                indexers={
                    marginal.dims[0]: slice(
                        center - 1.2 * center_stray,
                        center + 1.2 * center_stray,
                    ),
                },
            )
            low, high = np.percentile(
                near_center.values,
                (20, 80),
            )
            params["amplitude"] = params.get("amplitude", {})
            params["amplitude"]["value"] = high - low
    return params
