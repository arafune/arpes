"""Class for base of ARPES xarray extensions."""

from __future__ import annotations

import itertools
import warnings
from logging import DEBUG, INFO
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import xarray as xr
from more_itertools import always_reversible

from arpes.correction import coords
from arpes.debug import setup_logger
from arpes.utilities import selections

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
        Mapping,
        Sequence,
    )

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing.base import (
        ReduceMethod,
        SelType,
        XrTypes,
    )

from .property import ARPESProperty

NORMALIZED_DIM_NAMES = ["x", "y", "z", "w"]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class ARPESAccessorBase(ARPESProperty):
    """Base class for the "S" accessor of xarray extensions in PyARPES.

    This class provides a foundational set of utility methods for interacting
    with and manipulating ARPES data stored in `xarray.Dataset` or `xarray.DataArray`
    objects. It includes functionalities for searching properties, performing
    reductions across dimensions, specialized selections, and interfacing with
    fitting tools.
    """

    def __init__(self, xarray_obj: XrTypes) -> None:
        """Initializes the ARPESAccessorBase with an xarray object.

        Args:
            xarray_obj (XrTypes): The xarray.Dataset or xarray.DataArray instance
                to which this accessor is attached.
        """
        self._obj = xarray_obj

    def find(self, name: str) -> list[str]:
        """Returns a list of property names within the accessor that contain the specified string.

        This method is useful for discovering available functionalities or attributes
        when their exact names are not known, or for quickly listing related methods.

        Args:
            name (str): The substring to search for within the names of the
                accessor's attributes and methods.

        Returns:
            list[str]: A list of strings, where each string is the name of a
            property or method in the accessor that contains the `name` substring.

        Examples:
            >>> # Assuming 'ds' is an xarray.Dataset and 'ds.S' is an instance of this accessor
            >>> class MockAccessor(ARPESAccessorBase):
            ...     def method_a(self): pass
            ...     def another_method(self): pass
            ...     def prop_val(self): return 1
            ...
            >>> mock_obj = xr.Dataset()
            >>> accessor = MockAccessor(mock_obj)
            >>> accessor.find("method")
            ['method_a', 'another_method']
            >>> accessor.find("prop")
            ['prop_val']
        """
        return [n for n in dir(self) if name in n]

    def sum_other(
        self,
        dim_or_dims: list[str],
        *,
        keep_attrs: bool = False,
    ) -> XrTypes:
        """Calculates the sum over all dimensions *except* those specified.

        This is a convenience method for `xarray.Dataset.sum()` or `xarray.DataArray.sum()`
        that inverts the selection of dimensions. Instead of specifying dimensions
        to sum *along*, you specify dimensions to *keep*.

        Args:
            dim_or_dims (list[str]): A list of dimension names to keep. The sum
                operation will be performed over all other dimensions not in this list.
            keep_attrs (bool, optional): If True, attributes (`.attrs`) will be
                preserved on the returned object. Defaults to False.

        Returns:
            XrTypes: A new xarray object (Dataset or DataArray) with the data
            summed along the specified "other" dimensions. Its dimensions will
            only include those listed in `dim_or_dims`.

        Raises:
            AssertionError: If `dim_or_dims` is not a list.

        Examples:
            >>> data = xr.DataArray(np.ones((2, 3, 4)), dims=['x', 'y', 'z'])
            >>> accessor = ARPESAccessorBase(data)
            >>> accessor.sum_other(dim_or_dims=['x']) # Sums over 'y' and 'z'
            <xarray.DataArray (x: 2)>
            array([12., 12.])
            Dimensions without coordinates: y, z
            Coordinates:
              * x        (x) int64 0 1
            >>> accessor.sum_other(dim_or_dims=['y', 'z']) # Sums over 'x'
            <xarray.DataArray (y: 3, z: 4)>
            array([[2., 2., 2., 2.],
                   [2., 2., 2., 2.],
                   [2., 2., 2., 2.]])
            Dimensions without coordinates: x
            Coordinates:
              * y        (y) int64 0 1 2
              * z        (z) int64 0 1 2 3
        """
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
        """Calculates the mean over all dimensions *except* those specified.

        This is a convenience method for `xarray.Dataset.mean()` or `xarray.DataArray.mean()`
        that inverts the selection of dimensions. Instead of specifying dimensions
        to average *along*, you specify dimensions to *keep*.

        Args:
            dim_or_dims (list[str] | str): A list of dimension names to keep, or a single
                dimension name string. The mean operation will be performed over all other
                dimensions not in this list/string.
            keep_attrs (bool, optional): If True, attributes (`.attrs`) will be
                preserved on the returned object. Defaults to False.

        Returns:
            XrTypes: A new xarray object (Dataset or DataArray) with the data
            averaged along the specified "other" dimensions. Its dimensions will
            only include those listed in `dim_or_dims`.

        Raises:
            AssertionError: If `dim_or_dims` is not a list (note: the type hint allows `str`
                but the assertion explicitly checks for `list`). This discrepancy should
                be resolved for consistency. For now, the docstring reflects the assertion.

        Examples:
            >>> data = xr.DataArray(np.arange(12).reshape(2, 2, 3), dims=['x', 'y', 'z'])
            >>> accessor = ARPESAccessorBase(data)
            >>> accessor.mean_other(dim_or_dims=['x']) # Averages over 'y' and 'z'
            <xarray.DataArray (x: 2)>
            array([2.5, 8.5])
            Coordinates:
              * x        (x) int64 0 1
            >>> accessor.mean_other(dim_or_dims=['y', 'z']) # Averages over 'x'
            <xarray.DataArray (y: 2, z: 3)>
            array([[2.5, 3.5, 4.5],
                   [5.5, 6.5, 7.5]])
            Coordinates:
              * y        (y) int64 0 1
              * z        (z) int64 1 2 3
        """
        assert isinstance(dim_or_dims, list)

        return self._obj.mean(
            [d for d in self._obj.dims if d not in dim_or_dims],
            keep_attrs=keep_attrs,
        )

    def fat_sel(
        self,
        widths: dict[Hashable, float] | None = None,
        method: ReduceMethod = "mean",
        **kwargs: float,
    ) -> XrTypes:
        """Performs a 'fat' selection, integrating data over small regions specified by widths.

        This method allows for integrating a selection over a small coordinate region
        (defined by `widths` or keyword arguments), effectively reducing noise
        by averaging or summing over nearby slices. The resulting dataset will
        be normalized by the number of slices integrated over if `method="mean"`.

        Args:
            widths (dict[Hashable, float] | None, optional): A dictionary
                specifying the width of the integration window for each dimension.
                Keys are dimension names (Hashable), and values are float widths.
                Overrides any widths specified in `kwargs`. Defaults to None,
                in which case `selections.fat_sel` might use default widths.
            method (ReduceMethod, optional): The reduction method to apply within
                the selection window. Can be "mean" (default) or "sum".
            **kwargs (float): Keyword arguments that can define specific selection
                points (e.g., `eV=1.5`) or widths (e.g., `eV_width=0.1`).
                **Note**: Using `*_width` in kwargs for specifying widths is
                deprecated. Prefer the `widths` dictionary argument.

        Returns:
            XrTypes: The xarray.DataArray or xarray.Dataset after the 'fat'
            selection and reduction have been applied. The dimensions for which
            a width was specified will effectively be reduced or removed.

        Note:
            The `widths` argument is the preferred way to specify integration
            widths. Using `*_width` through `kwargs` is deprecated and may
            be removed in future versions.
        """
        return selections.fat_sel(data=self._obj, widths=widths, method=method, **kwargs)

    def modelfit(self, *args: Incomplete, **kwargs: Incomplete) -> xr.Dataset:
        """Performs curve fitting using `lmfit` via the `xarray-lmfit` extension.

        This method acts as a direct wrapper around `xarray.Dataset.xlm.modelfit`
        (or `xarray.DataArray.xlm.modelfit`). It allows applying complex
        fitting models defined with `lmfit` directly to xarray objects,
        leveraging xarray's labeled dimensions and broadcasting capabilities.

        For detailed usage and available parameters, refer to the
        `xarray-lmfit` documentation.

        Args:
            *args (Incomplete): Positional arguments to be passed directly to
                `xlm.modelfit`. These typically include the `lmfit.Model`
                instance and initial parameters.
            **kwargs (Incomplete): Keyword arguments to be passed directly to
                `xlm.modelfit`. These can include `dim` for specifying fitting
                dimensions, `weights`, `nan_policy`, etc.

        Returns:
            xr.Dataset: An xarray Dataset containing the fitting results.
                This typically includes best-fit parameters, uncertainties,
                the best-fit curve, residuals, and other diagnostic information
                from `lmfit`.

        Raises:
            AttributeError: If the `xlm` accessor (from `xarray-lmfit`) is
                not registered on the xarray object, which means `xarray-lmfit`
                might not be installed or imported correctly.
        """
        return self._obj.xlm.modelfit(*args, **kwargs)


class ARPESDataArrayAccessorBase(ARPESAccessorBase):
    """Base class for accessors specifically designed for `xarray.DataArray` objects in PyARPES.

    This class extends `ARPESAccessorBase` and provides methods tailored for
    single-variable ARPES data, such as inferring background subtraction
    status and performing advanced selections around specific data points.
    """

    _obj: xr.DataArray

    @property
    def is_subtracted(self) -> bool:
        """Infers whether a given data is subtracted.

        Returns (bool):
            Return True if the data is subtracted.
        """
        assert isinstance(self._obj, xr.DataArray)
        if self._obj.attrs.get("subtracted"):
            return True

        threshold_is_5_percent = 0.05
        if (((self._obj < 0) * 1).mean() > threshold_is_5_percent).item():
            self._obj.attrs["subtracted"] = True
            return True
        return False

    def select_around_data(
        self,
        points: Mapping[Hashable, xr.DataArray],
        radius: dict[Hashable, float] | float | None = None,  # radius={"phi": 0.005}
        *,
        mode: ReduceMethod = "sum",
    ) -> xr.DataArray:
        """Performs a binned selection around a point or points.

        Can be used to perform a selection along one axis as a function of another, integrating a
        region in the other dimensions.

        Example:
            As an example, suppose we have a dataset with dimensions ('eV', 'kp', 'T',)
            and we also by fitting determined the Fermi momentum as a function of T, kp_F('T'),
            stored in the dataset kFs. Then we could select momentum integrated EDCs in a small
            window around the fermi momentum for each temperature by using

            >>> edcs = full_data.S.select_around_data(points={'kp': kFs}, radius={'kp': 0.04})

            The resulting data will be EDCs for each T, in a region of radius 0.04 inverse angstroms
            around the Fermi momentum.

        Args:
            points: The set of points where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                    standard sized selection will be made as a compromise.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"

        Returns:
            The binned selection around the desired point or points.
        """
        return selections.select_around_data(
            data=self._obj,
            points=points,
            radius=radius,
            mode=mode,
        )

    def select_around(
        self,
        point: dict[Hashable, float],
        radius: dict[Hashable, float] | float | None,
        *,
        mode: ReduceMethod = "sum",
    ) -> xr.DataArray:
        """Selects and integrates a region around a one dimensional point.

        This method is useful to do a small region integration, especially around
        point on a path of a k-point of interest. See also the companion method
        `select_around_data`.

        Args:
            point: The point where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                    standard sized selection will be made as a compromise.
            safe: If true, infills radii with default values. Defaults to `True`.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"

        Returns:
            The binned selection around the desired point.
        """
        return selections.select_around(data=self._obj, point=point, radius=radius, mode=mode)


class GenericAccessorBase:
    """Class for general-purpose utility methods for xarray.Dataset and xarray.DataArray objects.

    This class offers functionalities such as coordinate manipulation,
    applying functions to data regions, and iterating over coordinates,
    which can be broadly useful across different types of scientific data.
    """

    _obj: XrTypes

    def round_coordinates(
        self,
        coords_to_round: dict[str, list[float] | NDArray[np.float64]],
        *,
        as_indices: bool = False,
    ) -> dict:
        """Rounds specified coordinates to their nearest existing values in the dataset.

        This method takes a dictionary of target coordinate values and finds the
        closest actual coordinate values present in the xarray object along those
        dimensions using `method="nearest"`. It can optionally return these
        rounded coordinates as their integer indices.

        Args:
            coords_to_round (dict[str, list[float] | NDArray[np.float64]]):
                A dictionary where keys are dimension names (strings) and values
                are the target coordinate points (floats or arrays of floats)
                to be rounded to the nearest existing coordinate in the dataset.
            as_indices (bool, optional): If True, the rounded coordinates are
                returned as their integer indices within the respective dimensions.
                If False (default), the actual float coordinate values are returned.

        Returns:
            dict[str, float | int]: A dictionary mapping dimension names to their
            rounded coordinate values (float) or their corresponding integer indices (int),
            depending on the `as_indices` parameter. Only dimensions specified in
            `coords_to_round` will be included in the output.

        Raises:
            AssertionError: If the internal `_obj` is not an `xarray.DataArray`
                or `xarray.Dataset`.

        Examples:
            >>> data = xr.DataArray(np.arange(10), dims=['x'], coords={'x': np.linspace(0, 9, 10)})
            >>> accessor = GenericAccessorBase(data)
            >>> accessor.round_coordinates({'x': [3.1]})
            {'x': 3.0}
            >>> accessor.round_coordinates({'x': [3.9]}, as_indices=True)
            {'x': 4}
            >>> data_md = xr.DataArray(np.random.rand(5,5), dims=['a', 'b'],
            ...                      coords={'a': np.arange(5), 'b': np.arange(5,10)})
            >>> accessor_md = GenericAccessorBase(data_md)
            >>> accessor_md.round_coordinates({'a': [2.2], 'b': [6.8]})
            {'a': 2.0, 'b': 7.0}
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        data = self._obj
        rounded = {
            k: v.item()
            for k, v in data.sel(coords_to_round, method="nearest").coords.items()
            if k in coords_to_round
        }

        if as_indices:
            rounded = {k: data.coords[k].index(v) for k, v in rounded.items()}

        return rounded

    def apply_over(
        self,
        fn: Callable[[xr.DataArray | xr.Dataset], XrTypes | NDArray[np.float64]],
        *,
        copy: bool = True,
        selections: Mapping[str, SelType] | None = None,
        **selections_kwargs: SelType,
    ) -> XrTypes:
        """Applies a function to a data region and updates the dataset with the result.

        Args:
            fn (Callable): The function to apply.
            copy (bool, optional): If True, operates on a deep copy of the data.
                If False, modifies the data in-place. Defaults to True.
            selections (Incomplete): Keyword arguments specifying the region of the data to select.
            **selections_kwargs: Selection keys and values as keyword arguments.

        Returns:
            XrTypes: The dataset after the function has been applied.

        Todo:
            - Add tests.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)

        data = self._obj.copy(deep=True) if copy else self._obj

        if selections is None:
            combined_selections: Mapping[str, SelType] = selections_kwargs
        else:
            combined_selections = {**(selections or {}), **selections_kwargs}

        selected: xr.DataArray | xr.Dataset = data.sel(**(combined_selections))  # type: ignore[arg-type]
        transformed = fn(selected)

        if isinstance(transformed, xr.DataArray | xr.Dataset):
            transformed = transformed.values

        data.loc[combined_selections] = transformed
        return data

    def coordinatize(self, as_coordinate_name: str | None = None) -> XrTypes:  # pragma: no cover
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

        Todo:
            Test
        """
        warnings.warn(
            "This method will be deprecated. Don't use it.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        assert len(self._obj.dims) == 1

        dim = self._obj.dims[0]
        if as_coordinate_name is None:
            as_coordinate_name = str(dim)

        o = self._obj.rename({dim: as_coordinate_name})
        o.coords[as_coordinate_name] = o.values

        return o

    def enumerate_iter_coords(
        self,
        dim_names: Sequence[Hashable] = (),
    ) -> Iterator[tuple[tuple[int, ...], dict[Hashable, float]]]:
        """Return an iterator for pixel and physical coordinates.

        Aargs:
            dir_names (Sequence[Hashable]): Dimension names for iterateion.

        Yields:
            Iteratoring the data like:
            ((0, 0), {'phi': -0.2178031280148764, 'eV': 9.0})
            which shows the relationship between pixel position and physical (like "eV" and "phi").
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        dim_names = dim_names if dim_names else tuple(self._obj.dims)
        dim_names = [dim_names] if isinstance(dim_names, str) else dim_names
        coords_list = [self._obj.coords[d].values for d in dim_names]
        for indices in itertools.product(*[range(len(c)) for c in coords_list]):
            cut_coords = [cs[index] for cs, index in zip(coords_list, indices, strict=True)]
            yield indices, dict(zip(self._obj.dims, cut_coords, strict=True))

    def iter_coords(
        self,
        dim_names: Sequence[Hashable] = (),
        *,
        reverse: bool = False,
    ) -> Iterator[dict[Hashable, float]]:
        """Iterator for coordinates along the axis.

        Args:
            dim_names (Sequence[Hashable]): Dimensions for iteration.
            reverse: return the "reversivle" iterator.

        Yields:
            Iterator of the physical position like ("eV" and "phi")
            {'phi': -0.2178031280148764, 'eV': 9.002}
        """
        dim_names = dim_names if dim_names else tuple(self._obj.dims)
        dim_names = [dim_names] if isinstance(dim_names, str) else dim_names
        the_iterator: Iterator = itertools.product(*[self._obj.coords[d].values for d in dim_names])
        the_iterator = always_reversible(the_iterator) if reverse else the_iterator
        for ts in the_iterator:
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
        indexed_ranges = [(coord.min().item(), coord.max().item()) for coord in indexed_coords]

        dim_names: list[str] | tuple[Hashable, ...] = tuple(self._obj.dims)
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        return dict(zip(dim_names, indexed_ranges, strict=True))

    def stride(
        self,
        *args: str | Sequence[str],
        generic_dim_names: bool = True,
    ) -> dict[Hashable, float] | list[float] | float:
        """Return the stride in each dimension.

        Args:
            args: The dimension to return.  ["eV", "phi"] or "eV", "phi"
            generic_dim_names (bool): if True, use Generic dimension name, such as 'x', is used.

        Returns:
            The stride of each dimension
        """
        indexed_strides: list[float] = [
            coords.is_equally_spaced(
                self._obj.coords[dim],
                dim,
            )
            for dim in self._obj.dims
        ]

        dim_names: list[str] | tuple[Hashable, ...] = tuple(self._obj.dims)
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        result: dict[Hashable, float] = dict(zip(dim_names, indexed_strides, strict=True))
        if args:
            if isinstance(args[0], str):
                return (
                    result[args[0]]
                    if len(args) == 1
                    else [result[str(selected_names)] for selected_names in args]
                )
            return [result[selected_names] for selected_names in args[0]]
        return result

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

        Todo:
            Test
        """
        mask = np.array(
            [
                i
                for i, c in enumerate(self._obj.coords[coordinate_name])
                if sieve(c, self._obj.isel({coordinate_name: i}))
            ],
        )
        return self._obj.isel({coordinate_name: mask})
