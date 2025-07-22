from __future__ import annotations  # noqa: D100

from logging import DEBUG, INFO
from typing import (
    TYPE_CHECKING,
    Any,
    Unpack,
)

import numpy as np
import xarray as xr

from arpes.correction import coords, intensity_map
from arpes.debug import setup_logger
from arpes.plotting.movie import plot_movie
from arpes.utilities import apply_dataarray

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Hashable,
        Mapping,
    )
    from pathlib import Path

    from _typeshed import Incomplete
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation
    from matplotlib.figure import Figure
    from numpy.typing import DTypeLike, NDArray

    from arpes._typing.base import DataType, XrTypes
    from arpes._typing.plotting import PColorMeshKwargs
from .base import GenericAccessorBase

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@xr.register_dataset_accessor("G")
class GenericDatasetAccessor(GenericAccessorBase):
    """A collection of generic accessors for xarray.Dataset objects.

    This accessor provides utility methods for filtering data variables,
    and transforming meshgrid coordinates within an xarray Dataset.

    Usage:
        Register this accessor using `@xr.register_dataset_accessor("G")`.
        Then, you can access its methods via the `.G` attribute on any xarray.Dataset:
        `ds.G.filter_vars(...)` or `ds.G.shift_meshgrid(...)`.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialization hook for xarray.Dataset.

        This constructor is automatically called when the accessor is accessed
        on an xarray.Dataset object (e.g., `ds.G`). It initializes the accessor
        with a reference to the parent Dataset.

        Args:
            xarray_obj: The parent xarray.Dataset object to which this accessor is attached.

        Note:
            This class should not be instantiated directly by users. It's intended
            to be accessed via the `.G` property of an `xarray.Dataset` object.
        """
        self._obj = xarray_obj
        assert isinstance(self._obj, xr.Dataset)

    def filter_vars(
        self,
        f: Callable[[Hashable, xr.DataArray], bool],
    ) -> xr.Dataset:
        """Filters data variables based on the specified condition and returns a new dataset.

        This method iterates through the data variables of the Dataset and applies
        a user-defined filtering function. Only variables for which the function
        returns `True` will be included in the new Dataset. The original Dataset
        remains unchanged.

        Args:
            f (Callable[[Hashable, xr.DataArray], bool]): A predicate function
                that takes two arguments: the name (hashable key) of a data variable
                and its corresponding `xr.DataArray`. It must return `True` to
                include the variable in the filtered dataset, or `False` to exclude it.

        Returns:
            xr.Dataset: A new xarray.Dataset containing only the data variables
                that satisfied the filtering condition. The attributes of the
                original dataset are preserved.

        Examples:
            >>> import xarray as xr
            >>> ds = xr.Dataset(
            ...     {"temp": (("x", "y"), [[1, 2], [3, 4]]),
            ...      "pressure": (("x", "y"), [[5, 6], [7, 8]])}
            ... )
            >>> # Filter variables whose names start with 't'
            >>> filtered_ds = ds.G.filter_vars(lambda k, v: k.startswith("t"))
            >>> print(filtered_ds)
            <xarray.Dataset>
            Dimensions:  (x: 2, y: 2)
            Data variables:
                temp     (x, y) int64 1 2 3 4
            >>> # Filter variables with more than 2 elements
            >>> filtered_ds = ds.G.filter_vars(lambda k, v: v.size > 2)
            >>> print(filtered_ds)
            <xarray.Dataset>
            Dimensions:  (x: 2, y: 2)
            Data variables:
                temp     (x, y) int64 1 2 3 4
                pressure (x, y) int64 5 6 7 8

        See Also:
            :py:meth:`xarray.Dataset.drop_vars`: To explicitly remove variables by name.
        """
        assert isinstance(self._obj, xr.Dataset)  # ._obj.data_vars
        return xr.Dataset(
            data_vars={k: v for k, v in self._obj.data_vars.items() if f(k, v)},
            attrs=self._obj.attrs,
        )

    def shift_meshgrid(
        self,
        dims: tuple[str, ...],
        shift: NDArray[np.float64] | float,
    ) -> xr.Dataset:
        """Shifts the meshgrid coordinates for specified dimensions.

        This method applies an additive shift to the coordinates of the specified
        dimensions. It's particularly useful when your coordinates represent
        a meshgrid (e.g., from `np.meshgrid` or `xr.Dataset.meshgrid`). The
        transformation is applied to the underlying coordinate arrays, resulting
        in a new Dataset with shifted coordinates.

        Args:
            dims (tuple[str, ...]): A tuple of strings specifying the names of
                the dimensions whose coordinates will be shifted. These dimensions
                should typically form a meshgrid.
            shift (NDArray[np.float64] | float): The amount(s) by which to shift
                the coordinates.
                - If a `float`, the same scalar shift is applied uniformly to all dimensions.
                - If an `NDArray[np.float64]`, it must be a 1D array with a
                  length equal to `len(dims)`. Each element in the array
                  corresponds to the shift applied to the coordinate of the
                  respective dimension in `dims`.

        Returns:
            xr.Dataset: A new `xarray.Dataset` with the coordinates of the
                specified dimensions shifted by the given amount(s). The original
                Dataset remains unchanged.

        Raises:
            AssertionError: If an invalid shift amount is provided (e.g., an array
                with an incorrect shape).

        Examples:
            >>> import xarray as xr
            >>> x = np.arange(2)
            >>> y = np.arange(3)
            >>> XX, YY = np.meshgrid(x, y)
            >>> ds = xr.Dataset(
            ...     coords={"x_coord": (("y", "x"), XX), "y_coord": (("y", "x"), YY)},
            ...     data_vars={"data": (("y", "x"), np.random.rand(3, 2))}
            ... )
            >>> # Shift both x_coord and y_coord by 1.0
            >>> shifted_ds = ds.G.shift_meshgrid(dims=("x_coord", "y_coord"), shift=1.0)
            >>> print(shifted_ds["x_coord"].values)
            [[1. 2.]
             [1. 2.]
             [1. 2.]]
            >>> # Shift x_coord by 0.5 and y_coord by -0.5
            >>> shifted_ds_individual = ds.G.shift_meshgrid(
            ...     dims=("x_coord", "y_coord"), shift=np.array([0.5, -0.5])
            ... )
            >>> print(shifted_ds_individual["y_coord"].values)
            [[-0.5 -0.5]
             [ 0.5  0.5]
             [ 1.5  1.5]]

        See Also:
            `.GenericDatasetAccessor.scale_meshgrid`: For applying multiplicative scaling.
            `.GenericDatasetAccessor.transform_meshgrid`: For arbitrary transformations using a
                function or matrix.
        """
        shift_array = np.ones((len(dims),)) * shift if isinstance(shift, float) else shift

        def transform(data: NDArray[np.float64]) -> NDArray[np.float64]:
            assert isinstance(shift_array, np.ndarray)
            new_shift: NDArray[np.float64] = shift_array
            for _ in range(len(dims)):
                new_shift = np.expand_dims(new_shift, axis=0)

            return data + new_shift

        return self.transform_meshgrid(dims, transform)

    def scale_meshgrid(
        self,
        dims: tuple[str, ...],
        scale: float | NDArray[np.float64],
    ) -> xr.Dataset:
        """Scales the meshgrid coordinates for specified dimensions.

        This method applies a multiplicative scaling to the coordinates of the
        specified dimensions. Similar to `shift_meshgrid`, this is designed
        for datasets where coordinates represent a meshgrid.

        Args:
            dims (tuple[str, ...]): A tuple of strings specifying the names of the dimensions whose
                coordinates will be scaled.
            scale (float | NDArray[np.float64]): The amount(s) by which to scale the coordinates.

                - If a `float`, the same scalar scaling factor is applied uniformly to all
                  specified dimensions.
                - If an `NDArray[np.float64]`, it can be a 1D array or a 2D matrix.
                  If 1D, its length must equal `len(dims)`. Each element represents the scaling
                  factor for the corresponding dimension. This is converted internally into a
                  diagonal scaling matrix. If 2D, it must be a square matrix of shape
                  `(len(dims), len(dims))`. This matrix represents a linear transformation
                  (e.g., rotation, shear, non-uniform scaling) to be applied to the stacked
                  coordinate vectors.

        Returns:
            xr.Dataset: A new `xarray.Dataset` with the coordinates of the
                specified dimensions scaled by the given amount(s). The original
                Dataset remains unchanged.

        Raises:
            AssertionError: If an invalid scale amount is provided (e.g., an array
                with an incorrect shape).

        Examples:
            >>> import xarray as xr
            >>> x = np.arange(2)
            >>> y = np.arange(3)
            >>> XX, YY = np.meshgrid(x, y)
            >>> ds = xr.Dataset(
            ...     coords={"x_coord": (("y", "x"), XX), "y_coord": (("y", "x"), YY)},
            ...     data_vars={"data": (("y", "x"), np.random.rand(3, 2))}
            ... )
            >>> # Scale both x_coord and y_coord by 2.0
            >>> scaled_ds = ds.G.scale_meshgrid(dims=("x_coord", "y_coord"), scale=2.0)
            >>> print(scaled_ds["x_coord"].values)
            [[0. 2.]
             [0. 2.]
             [0. 2.]]
            >>> # Scale x_coord by 0.5 and y_coord by 1.5 (using 1D array)
            >>> scaled_ds_individual = ds.G.scale_meshgrid(
            ...     dims=("x_coord", "y_coord"), scale=np.array([0.5, 1.5])
            ... )
            >>> print(scaled_ds_individual["x_coord"].values)
            [[0.  0.5]
             [0.  0.5]
             [0.  0.5]]
            >>> print(scaled_ds_individual["y_coord"].values)
            [[0.  0. ]
             [1.5 1.5]
             [3.  3. ]]

        See Also:
            `~.GenericDatasetAccessor.shift_meshgrid`: For applying additive shifts.
            `~.GenericDatasetAccessor.transform_meshgrid`: For arbitrary transformations using a
                function or matrix.
        """
        if not isinstance(scale, np.ndarray):
            n_dims = len(dims)
            scale = np.identity(n_dims) * scale
        elif len(scale.shape) == 1:
            scale = np.diag(scale)

        return self.transform_meshgrid(dims, scale)

    def transform_meshgrid(
        self,
        dims: Collection[str],
        transform: NDArray[np.float64] | Callable,
    ) -> xr.Dataset:
        r"""Transforms the given meshgrid coordinates by an arbitrary function or matrix.

        This is the core method for applying complex transformations to meshgrid
        coordinates. It takes a collection of dimension names whose coordinates
        form a meshgrid, stacks their values into a single NumPy array, applies
        a user-defined transformation (either a function or a linear
        transformation matrix), and then updates the original coordinates
        in a new Dataset.

        The transformation operates on a reshaped view of the coordinate data.
        Specifically, for N dimensions, the coordinate arrays (e.g., `X`, `Y`, `Z`)
        are stacked along a new last axis. If the original coordinates are
        (M1, M2, ..., MN) for dims `d1, d2, ..., dN`, then the stacked array
        will have shape `(M1, M2, ..., MN, N)`. Each "row" in the last axis
        represents a coordinate vector `[coord_d1, coord_d2, ..., coord_dN]`
        at a specific grid point.

        Args:
            dims (Collection[str]): A list or tuple of strings representing the names of the
                dimensions whose coordinates should be transformed. These dimensions are assumed to
                form a meshgrid. The order of dimensions in this collection matters, as it defines
                the order of columns in the stacked coordinate array passed to `transform`.
            transform (NDArray[np.float64] | Callable[[NDArray[np.float64]], NDArray[np.float64]]):
                The transformation to apply to the stacked meshgrid coordinates.
                This can be one of two types:

                - `NDArray[np.float64]`: A 2D NumPy array representing a **linear transformation
                  matrix**. This matrix will be right-multiplied onto the stacked coordinate array.
                  Its shape must be `(len(dims), len(dims))`. This is suitable for operations like
                  rotation, scaling (including non-uniform), and shearing.
                - `Callable[[NDArray[np.float64]], NDArray[np.float64]]`: A
                  function that takes a single NumPy array as input and returns
                  a NumPy array. The input array will have the shape
                  `(..., len(dims))`, where `...` represents the original
                  meshgrid dimensions. The function must return an array of
                  the *same shape* as the input, containing the transformed
                  coordinate values. This allows for arbitrary, non-linear
                  transformations (e.g., spherical to Cartesian conversion,
                  custom distortions).

        Returns:
            xr.Dataset: A new `xarray.Dataset` object with the specified
                meshgrid coordinates updated after applying the transformation.
                The original Dataset is not modified.

        Raises:
            AssertionError: If the input `transform` matrix has an incorrect
                shape.
            ValueError: If the `transform` callable does not return an array
                of the expected shape.

        Examples:
            >>> import xarray as xr
            >>> x = np.arange(2)
            >>> y = np.arange(3)
            >>> XX, YY = np.meshgrid(x, y)
            >>> ds = xr.Dataset(
            ...     coords={"x_coord": (("y", "x"), XX), "y_coord": (("y", "x"), YY)},
            ...     data_vars={"data": (("y", "x"), np.random.rand(3, 2))}
            ... )

            >>> # Example 1: Linear transformation (rotation by 90 degrees)
            >>> # Rotate [x, y] to [-y, x]
            >>> rotation_matrix = np.array([[0, 1], [-1, 0]])
            >>> rotated_ds = ds.G.transform_meshgrid(dims=("x_coord", "y_coord"),
                transform=rotation_matrix)
            >>> print("Rotated x_coord:\\n", rotated_ds["x_coord"].values)
            Rotated x_coord:
             [[ 0.  0.]
             [-1. -1.]
             [-2. -2.]]
            >>> print("Rotated y_coord:\\n", rotated_ds["y_coord"].values)
            Rotated y_coord:
             [[0. 1.]
             [0. 1.]
             [0. 1.]]

            >>> # Example 2: Non-linear transformation (squaring each coordinate)
            >>> def square_coords(coords_array: NDArray[np.float64]) -> NDArray[np.float64]:
            ...     return coords_array**2
            >>> squared_ds = ds.G.transform_meshgrid(dims=("x_coord", "y_coord"),
                transform=square_coords)
            >>> print("Squared x_coord:\\n", squared_ds["x_coord"].values)
            Squared x_coord:
             [[0. 1.]
             [0. 1.]
             [0. 1.]]
            >>> print("Squared y_coord:\\n", squared_ds["y_coord"].values)
            Squared y_coord:
             [[0. 0.]
             [1. 1.]
             [4. 4.]]

        See Also:
            `~.GenericDatasetAccessor.shift_meshgrid`: A specialized linear transformation for
                additive shifts.
            `~.GenericDatasetAccessor.scale_meshgrid`: A specialized linear transformation for
                multiplicative scaling.
            `numpy.meshgrid`: For understanding how meshgrid coordinates are typically structured.
        """
        assert isinstance(self._obj, xr.Dataset)
        as_ndarray = np.stack([self._obj.data_vars[d].values for d in dims], axis=-1)

        if isinstance(transform, np.ndarray):
            transformed = np.dot(as_ndarray, transform)
        else:
            transformed = transform(as_ndarray)

        copied = self._obj.copy(deep=True)

        for d, arr in zip(dims, np.split(transformed, transformed.shape[-1], axis=-1), strict=True):
            copied.data_vars[d].values = np.squeeze(arr, axis=-1)

        return copied


@xr.register_dataarray_accessor("G")
class GenericDataArrayAccessor(GenericAccessorBase):
    """A collection of generic accessors for xarray.DataArray objects.

    This accessor provides utility methods for common operations on `xarray.DataArray`,
    including finding maximum value coordinates, reshaping, creating meshgrids,
    handling outliers, generating animations, and applying custom transformations.

    Usage:
        Register this accessor using `@xr.register_dataarray_accessor("G")`.
        Then, you can access its methods via the `.G` attribute on any xarray.DataArray:
        `da.G.argmax_coords()` or `da.G.ravel()`.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Initialization hook for xarray.DataArray.

        This constructor is automatically called when the accessor is accessed
        on an xarray.DataArray object (e.g., `da.G`). It initializes the accessor
        with a reference to the parent DataArray.

        Args:
            xarray_obj: The parent xarray.DataArray object to which this accessor is attached.

        Note:
            This class should not be instantiated directly by users. It's intended
            to be accessed via the `.G` property of an `xarray.DataArray` object.
        """
        self._obj: xr.DataArray = xarray_obj
        assert isinstance(self._obj, xr.DataArray)

    def argmax_coords(self) -> dict[Hashable, float]:
        """Return dict representing the position for maximum value."""
        data: xr.DataArray = self._obj
        raveled = data.argmax(None)
        assert isinstance(raveled, xr.DataArray)
        idx = raveled.item()
        flat_indices = np.unravel_index(idx, data.values.shape)
        return {d: data.coords[d][flat_indices[i]].item() for i, d in enumerate(data.dims)}

    def ravel(self) -> Mapping[Hashable, xr.DataArray | NDArray[np.float64]]:
        """Converts to a flat representation where the coordinate values are also present.

        Extremely valuable for plotting a dataset with coordinates, X, Y and values Z(X,Y)
        on a scatter plot in 3D.

        By default the data is listed under the key 'data'.

        Returns:
            A dictionary mapping between coordinate names and their coordinate arrays.
            Additionally, there is a key "data" which maps to the `.values` attribute of the array.
        """
        assert isinstance(self._obj, xr.DataArray)

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        raveled_coordinates = dict(
            zip(
                dims,
                [cs.ravel() for cs in np.meshgrid(*coords_as_list)],
                strict=True,
            ),
        )
        assert "data" not in raveled_coordinates
        raveled_coordinates["data"] = self._obj.values.ravel()

        return raveled_coordinates

    def meshgrid(
        self,
        *,
        as_dataset: bool = False,
    ) -> dict[Hashable, NDArray[np.float64]] | xr.Dataset:
        r"""Creates a meshgrid from the DataArray's dimensions and includes its values.

        optionally returning it as an xarray.Dataset.

        This method is useful for reconstructing meshgrid-like coordinate arrays
        from a DataArray's dimensions, similar to `numpy.meshgrid`. It also
        includes the original DataArray's values, preserving their multi-dimensional shape.

        Args:
            as_dataset (bool, optional): If `True`, the result is returned as an
                `xarray.Dataset` where coordinate arrays and the data array are
                represented as `DataArray` objects with explicit dimensions.
                If `False` (default), the result is a dictionary of NumPy arrays.

        Returns:
            dict[Hashable, NDArray[np.float64]] | xr.Dataset:
                - If `as_dataset` is `False`: A dictionary where keys are dimension names
                  and `"data"`, and values are multi-dimensional NumPy arrays
                  representing the meshgrid coordinates and the original data values.
                - If `as_dataset` is `True`: An `xarray.Dataset` containing `DataArray`
                  objects for each meshgrid coordinate and the original data, with
                  dimensions explicitly named (e.g., "x", "y", "z" for the meshgrid,
                  and the original dims for "data").

        Examples:
            >>> import xarray as xr
            >>> import numpy as np
            >>> data = xr.DataArray(
            ...     [[1, 2], [3, 4]],
            ...     coords={"x": [10, 20], "y": [100, 200]},
            ...     dims=("x", "y")
            ... )
            >>> # As a dictionary of NumPy arrays
            >>> meshed_dict = data.G.meshgrid()
            >>> print("x_meshgrid:\\n", meshed_dict["x"])
            x_meshgrid:
             [[10 20]
             [10 20]]
            >>> print("y_meshgrid:\\n", meshed_dict["y"])
            y_meshgrid:
             [[100 100]
             [200 200]]
            >>> print("data:\\n", meshed_dict["data"])
            data:
             [[1 2]
             [3 4]]

            >>> # As an xarray.Dataset
            >>> meshed_ds = data.G.meshgrid(as_dataset=True)
            >>> print(meshed_ds)
            <xarray.Dataset>
            Dimensions:  (x: 2, y: 2)
            Coordinates:
                x_coord  (x, y) int64 10 20 10 20
                y_coord  (x, y) int64 100 100 200 200
            Data variables:
                data     (x, y) int64 1 2 3 4
            >>> print(meshed_ds["x_coord"]) # Note the internal renaming
            <xarray.DataArray 'x_coord' (x: 2, y: 2)>
            array([[10, 20],
                   [10, 20]])
            Coordinates:
              * x        (x) int64 10 20
              * y        (y) int64 100 200

        See Also:
            `numpy.meshgrid`: The core NumPy function for creating coordinate grids.
            `~.GenericDataArrayAccessor.ravel`: For flattening the data and coordinates.
        """
        assert isinstance(self._obj, xr.DataArray)  # ._obj.values is used.

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        meshed_coordinates = dict(zip(dims, list(np.meshgrid(*coords_as_list)), strict=True))
        assert "data" not in meshed_coordinates
        meshed_coordinates["data"] = self._obj.values

        if as_dataset:
            # this could use a bit of cleaning up
            faked = ["x", "y", "z", "w"]
            return xr.Dataset(
                {
                    k: (faked[: len(v.shape)], v)
                    for k, v in meshed_coordinates.items()
                    if k != "data"
                },
            )

        return meshed_coordinates

    def clean_outliers(self, clip: float = 0.5) -> xr.DataArray:
        """Clip outliers in the DataArray by limiting values to a specified percentile range.

        This method modifies the values of an `xarray.DataArray` to ensure that they fall within a
        specified range defined by percentiles. Any value below the lower percentile is set to the
        lower limit, and any value above the upper percentile is set to the upper limit.

        Args:
            clip (float, optional): The percentile range to use for clipping. The lower and upper
                bounds are determined by the `clip` value and its complement:

                - Lower bound: `clip` percentile.
                - Upper bound: `(100 - clip)` percentile.

                For example, if `clip=0.5`, the lower 0.5% and upper 99.5% of the data will be
                    clipped. Default is 0.5.

        Returns:
        xr.DataArray: A new DataArray with outliers clipped to the specified range.

        Raises:
            AssertionError: If the underlying object is not an `xarray.DataArray`.

        Todo:
            - Add unit tests to ensure the method behaves as expected.
        """
        assert isinstance(self._obj, xr.DataArray)
        low, high = np.percentile(self._obj.values, [clip, 100 - clip])
        copied = self._obj.copy(deep=True)
        copied.values[copied.values < low] = low
        copied.values[copied.values > high] = high
        return copied

    def as_movie(
        self,
        time_dim: str = "delay",
        *,
        out: str | None = None,
        **kwargs: Unpack[PColorMeshKwargs],
    ) -> Path | HTML | Figure | FuncAnimation:
        """Create an animation or save images showing the DataArray's evolution over time.

            This method creates a time-based visualization of an `xarray.DataArray`, either as an
            animation or as a sequence of images saved to disk. The `time_dim` parameter specifies
            the dimension used for the temporal progression.

        Args:
            time_dim (str, optional): The name of the dimension representing time or progression
                in the DataArray. Defaults to "delay".
            out (str , optional): Determines the output format.  If a string is provided, it is used
                as the base name for the output file or directory. otherwise, the animation is
                returned without saving.
            kwargs (optional): Additional keyword arguments passed to the `plot_movie` function.
                These can customize the appearance of the generated images or animation.

        Returns:
            Path | animation.FuncAnimation:
                - If `out` is specified (as a string or `True`), returns a `Path` to the saved file.
                - If `out` is `False` or an empty string, returns a
                  `matplotlib.animation.FuncAnimation` object.

        Raises:
            AssertionError: If the underlying object is not an `xarray.DataArray`.
            AssertionError: If `out` is not a valid string when required.

        Example:

        .. code-block:: python

            import xarray as xr

            # Create a sample DataArray with a time dimension
            data = xr.DataArray(
                [[[i + j for j in range(10)] for i in range(10)] for _ in range(5)],
                dims=("delay", "x", "y"),
                coords={"delay": range(5), "x": range(10), "y": range(10)},
                )
            # Generate an animation
            animation = data.G.as_movie(time_dim="delay")
        """
        assert isinstance(self._obj, xr.DataArray)

        return plot_movie(self._obj, time_dim, out=out, **kwargs)

    def map_axes(
        self,
        axes: list[str] | str,
        fn: Callable[[XrTypes, dict[Hashable, float]], DataType],
        dtype: DTypeLike = None,
    ) -> xr.DataArray:
        """Apply a function along specified axes of the DataArray, creating a new DataArray.

        This method iterates over the coordinates of the specified axes, applies the provided
        function to each coordinate, and assigns the result to the corresponding position
        in the output DataArray. Optionally, the data type of the output array can be specified.

        Args:
            axes (list[str] | str): The axis or axes along which to iterate and apply the function.
            fn (Callable[[XrTypes, dict[str, float]], DataType]): A function that takes the selected
                data and its coordinates as input and returns the transformed data.
            dtype (DTypeLike, optional): The desired data type for the output DataArray. If not
                specified, the type is inferred from the function's output.

        Returns:
            xr.DataArray: A new DataArray with the function applied along the specified axes.

        Raises:
            TypeError: If the input arguments or operations result in a type mismatch.

        Example:

        .. code-block python

            import xarray as xr
            import numpy as np
            # Create a sample DataArray
            data = xr.DataArray(
                np.random.rand(5, 5),
                dims=["x", "y"],
                coords={"x": range(5), "y": range(5)},
                )
            # Define a function to scale data
            def scale_fn(data, coord):
                scale_factor = coord["x"] + 1
                return data * scale_factor
            result = data.map_axes(axes="x", fn=scale_fn)
            print(result)

        Todo:
            - Add tests to validate the behavior with complex axes configurations.
            - Optimize performance for high-dimensional DataArrays.

        """
        obj = self._obj.copy(deep=True)

        if dtype is not None:
            obj.values = np.ndarray(shape=obj.values.shape, dtype=dtype)

        type_assigned = False
        for coord in self.iter_coords(axes):
            value = self._obj.sel(coord, method="nearest")
            new_value = fn(value, coord)

            if dtype is None:
                if not type_assigned:
                    obj.values = np.ndarray(shape=obj.values.shape, dtype=new_value.data.dtype)
                    type_assigned = True

                obj.loc[coord] = new_value.values
            else:
                obj.loc[coord] = new_value

        return obj

    def transform(
        self,
        axes: str | list[str],
        transform_fn: Callable,
        dtype: DTypeLike = None,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """Applies a vectorized operation across a subset of array axes.

        Transform has similar semantics to matrix multiplication, the dimensions of the
        output can grow or shrink depending on whether the transformation is size preserving,
        grows the data, shrinks the data, or leaves in place.

        Examples:
            As an example, let us suppose we have a function which takes the mean and
            variance of the data:

                [dimension], coordinate_value -> [{'mean', 'variance'}]

            And a dataset with dimensions [X, Y]. Then calling transform
            maps to a dataset with the same dimension X but where Y has been replaced by
            the length 2 label {'mean', 'variance'}. The full dimensions in this case are
            ['X', {'mean', 'variance'}].

            >>> data.G.transform('X', f).dims  # doctest: +SKIP
            ["X", "mean", "variance"]

        Please note that the transformed axes always remain in the data because they
        are iterated over and cannot therefore be modified.

        The transform function `transform_fn` must accept the coordinate of the
        marginal at the currently iterated point.

        Args:
            axes: Dimension/axis or set of dimensions to iterate over
            transform_fn: Transformation function that takes a DataArray into a new DataArray
            dtype: An optional type hint for the transformed data. Defaults to None.
            args: args to pass into transform_fn
            kwargs: kwargs to pass into transform_fn

        Raises:
            TypeError: When the underlying object is an `xr.Dataset` instead of an `xr.DataArray`.
                This is due to a constraint related to type inference with a single passed dtype.


        Returns:
            The data consisting of applying `transform_fn` across the specified axes.

        Todo:
            Test
        """
        dest = None
        for coord in self._obj.G.iter_coords(axes):
            value = self._obj.sel(coord, method="nearest")
            new_value = transform_fn(value, coord, *args, **kwargs)

            if dest is None:
                new_value = transform_fn(value, coord, *args, **kwargs)

                original_dims = [d for d in self._obj.dims if d not in value.dims]
                original_shape = [self._obj.shape[self._obj.dims.index(d)] for d in original_dims]
                original_coords = {k: v for k, v in self._obj.coords.items() if k not in value.dims}

                full_shape = original_shape + list(new_value.shape)

                new_coords = original_coords
                new_coords.update(
                    {k: v for k, v in new_value.coords.items() if k not in original_coords},
                )
                new_dims = original_dims + list(new_value.dims)
                dest = xr.DataArray(
                    data=np.zeros(full_shape, dtype=dtype or new_value.data.dtype),
                    coords=new_coords,
                    dims=new_dims,
                )

            dest.loc[coord] = new_value
        assert isinstance(dest, xr.DataArray)
        return dest

    def map(
        self,
        fn: Callable[[NDArray[np.float64], Any], NDArray[np.float64]],
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """Applies the specified function to the values of an xarray and returns a new DataArray.

        Args:
            fn (Callable): The function to apply to the xarray values.
            kwargs: Additional arguments to pass to the function.

        Returns:
            xr.DataArray: A new DataArray with the function applied to the values.
        """
        return apply_dataarray(self._obj, np.vectorize(fn, **kwargs))

    def shift_by(
        self,
        other: xr.DataArray | NDArray[np.float64],
        shift_axis: str = "",
        by_axis: str = "",
        *,
        extend_coords: bool = False,
        shift_coords: bool = False,
    ) -> xr.DataArray:
        """Shifts the data along the specified axis.

        Currently, only supports shifting by a one-dimensional array.

        Args:
            other (xr.DataArray | NDArray): Data to shift by. Only supports one-dimensional array.
            shift_axis (str): The axis to shift along, which is 1D.
            by_axis (str): The dimension name of `other`. Ignored when `other` is an xr.DataArray.
            extend_coords (bool): If True, the coords expands.  Default is False.
            shift_coords (bool): Whether to shift the coordinates as well.
                The arg will be removed, because it is not unique way to shift from the "other".
                Currently it uses mean value of "other".

        Returns:
            xr.DataArray: The shifted xr.DataArray.

        Todo:
            - Add tests.Data shift along the axis.

        Note:
            zero_nans is removed.  Use DataArray.fillna(0), if needed.
        """
        return intensity_map.shift(
            self._obj,
            other=other,
            shift_axis=shift_axis,
            by_axis=by_axis,
            extend_coords=extend_coords,
            shift_coords=shift_coords,
        )

    def shift_coords_by(
        self,
        shift_values: dict[str, float],
    ) -> xr.DataArray:
        """Shifts the coordinates by the specified values.

        Args:
            shift_values (dict[str, float]): A dictionary where keys are coordinate names and values
            are the amounts to shift.

        Returns:
            xr.DataArray: The DataArray with shifted coordinates.
        """
        data_shifted = self._obj.copy(deep=True)
        for coord, shift in shift_values.items():
            data_shifted = coords.shift_by(data_shifted, coord, shift)
        return data_shifted

    def with_values(
        self,
        new_values: NDArray[np.float64],
        *,
        keep_attrs: bool = True,
    ) -> xr.DataArray:
        """Copy with new array values.

        Easy way of creating a DataArray that has the same shape as the calling object but data
        populated from the array `new_values`.

        Notes: This method is applicable only for xr.DataArray.  (Not xr.Dataset)

        Args:
            new_values: The new values which should be used for the data.
            keep_attrs (bool): If True, attributes are also copied.

        Returns:
            A copy of the data with new values but identical dimensions, coordinates, and attrs.

        ToDo: Test
        """
        assert isinstance(self._obj, xr.DataArray)
        if keep_attrs:
            return xr.DataArray(
                data=new_values.reshape(self._obj.values.shape),
                coords=self._obj.coords,
                dims=self._obj.dims,
                attrs=self._obj.attrs,
            )
        return xr.DataArray(
            data=new_values.reshape(self._obj.values.shape),
            coords=self._obj.coords,
            dims=self._obj.dims,
        )
