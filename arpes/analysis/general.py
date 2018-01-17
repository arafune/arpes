import numpy as np
import xarray as xr
import typing
from collections import defaultdict
import arpes.models.band
import arpes.utilities
import arpes.utilities.math
import itertools
from arpes.provenance import update_provenance
from .filters import gaussian_filter_arr

__all__ = ('normalize_by_fermi_distribution', 'symmetrize_axis', 'condense', 'rebin',)


@update_provenance('Normalized by the 1/Fermi Dirac Distribution at sample temp')
def normalize_by_fermi_distribution(data, max_gain=None, rigid_shift=0, instrumental_broadening=None):
    """
    Normalizes a scan by 1/the fermi dirac distribution. You can control the maximum gain with ``clamp``, and whether
    the Fermi edge needs to be shifted (this is for those desperate situations where you want something that
    "just works") via ``rigid_shift``.

    :param data: Input
    :param clamp: Maximum value for the gain. By default the value used is the mean of the spectrum.
    :param rigid_shift: How much to shift the spectrum chemical potential.
    Pass the nominal value for the chemical potential in the scan. I.e. if the chemical potential is at BE=0.1, pass
    rigid_shift=0.1.
    :param instrumental_broadening: Instrumental broadening to use for convolving the distribution
    :return: Normalized DataArray
    """
    distrib = arpes.utilities.math.fermi_distribution(data.coords['eV'].values - rigid_shift, data.S.temp)

    # don't boost by more than 90th percentile of input, by default
    if max_gain is None:
        max_gain = np.mean(data.values)

    distrib[distrib < 1/max_gain] = 1/max_gain
    distrib_arr = xr.DataArray(
        distrib,
        {'eV': data.coords['eV'].values},
        ['eV']
    )

    if instrumental_broadening is not None:
        distrib_arr = gaussian_filter_arr(distrib_arr, sigma={'eV': instrumental_broadening})

    return data / distrib_arr


#@update_provenance('Symmetrize axis')
def symmetrize_axis(data, axis_name, flip_axes=None):
    selector = {}
    selector[axis_name] = slice(None, None, -1)
    rev = data.sel(**selector).copy()
    rev.coords[axis_name].values = -rev.coords[axis_name].values

    if flip_axes is None:
        flip_axes = []

    for axis in flip_axes:
        selector = {}
        selector[axis] = slice(None, None, -1)
        rev = rev.sel(**selector)
        rev.coords[axis].values = -rev.coords[axis].values

    return rev.combine_first(data)


@update_provenance('Condensed array')
def condense(data: xr.DataArray):
    """
    Clips the data so that only regions where there is substantial weight are included. In
    practice this usually means selecting along the ``eV`` axis, although other selections
    might be made.

    :param data: xarray.DataArray
    :return:
    """
    if 'eV' in data.dims:
        data = data.sel(eV=slice(None, 0.05))

    return data


@update_provenance('Rebinned array')
def rebin(data: xr.DataArray, shape: dict=None, reduction: typing.Union[int, dict]=None, interpolate=False):
    """
    Rebins the data onto a different (smaller) shape. By default the behavior is to
    split the data into chunks that are integrated over. An interpolation option is also
    available.

    Exactly one of ``shape`` and ``reduction`` should be supplied.

    Dimensions corresponding to missing entries in ``shape`` or ``reduction`` will not
    be changed.

    :param data:
    :param interpolate: Use interpolation instead of integration
    :param shape: Target shape
    :param reduction: Factor to reduce each dimension by
    :return:
    """

    if interpolate:
        raise NotImplementedError('The interpolation option has not been implemented')

    assert(shape is None or reduction is None)

    if isinstance(reduction, int):
        reduction = {d: reduction for d in data.dims}

    # we standardize by computing reduction from shape is shape was supplied.
    if shape is not None:
        reduction = {k: len(data.coords[k]) // v for k, v in shape.items()}

    # since we are not interpolating, we need to clip each dimension so that the reduction
    # factor evenly divides the real shape of the input data.
    slices = defaultdict(lambda: slice(None))

    for dim, reduction_factor in reduction.items():
        remainder = len(data.coords[dim]) % reduction_factor
        if remainder != 0:
            slices[dim] = slice(None, -remainder)

    trimmed_data = data.data[[slices[d] for d in data.dims]]
    trimmed_coords = {d: coord[slices[d]] for d, coord in data.coords.items()}

    temp_shape = [[trimmed_data.shape[i] // reduction.get(d, 1), reduction.get(d, 1)]
                  for i, d in enumerate(data.dims)]
    temp_shape = itertools.chain(*temp_shape)
    reduced_data = trimmed_data.reshape(*temp_shape)

    for i in range(len(data.dims)):
        reduced_data = reduced_data.mean(i + 1)

    reduced_coords = {d: coord[::reduction.get(d, 1)] for d, coord in trimmed_coords.items()}

    return xr.DataArray(
        reduced_data,
        reduced_coords,
        data.dims,
        attrs=data.attrs
    )