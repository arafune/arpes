"""Base class and shared logic for ARPES endstation plugins.

This module defines `EndstationBase`, the superclass for all ARPES endstation
plugins. Plugins inheriting from this class are responsible for implementing
custom loading and normalization logic specific to different experimental setups
(e.g., synchrotrons, hemispherical analyzers, MAESTRO FITS data).

The base class provides:

- Interface definitions (`load_single_frame`, `concatenate_frames`, etc.)
- Coordinate consistency checks and merging logic
- File search heuristics based on regular expressions and workspace settings
- Class-level configuration via attributes like `ALIASES`, `RENAME_KEYS`, etc.

Plugin authors should subclass `EndstationBase` and override/extend its methods
as needed for their specific data formats and conventions.

For more information, see the plugin documentation:
https://arpes.readthedocs.io/writing-plugins
"""

from __future__ import annotations

import re
import warnings
from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.configuration.interface import get_workspace_path
from arpes.debug import setup_logger
from arpes.helper import rename_keys
from arpes.utilities.xarray import rename_dataarray_attrs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from _typeshed import Incomplete

    from arpes._typing import DataType, ScanDesc, Spectrometer

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class EndstationBase:
    """Implements the core features of ARPES data loading.

    A thorough documentation
    is available at `the plugin documentation <https://arpes.readthedocs.io/writing-plugins>`_.

    To summarize, a plugin has a few core jobs:

    1. Load data, including collating any data that is in a multi-file format
       This is accomplished with `.load`, which delegates loading `frames` (single files)
       to `load_single_frame`. Frame collation is then performed by `concatenate_frames`.
    2. Loading and attaching metadata.
    3. Normalizing metadata to standardized names. These are documented at the
       `data model documentation <https://arpes.readthedocs.io/spectra>`_.
    4. Ensuring all angles and necessary coordinates are attached to the data.
       Data should permit immediate conversion to angle space after being loaded.

    Plugins are in one-to-many correspondence with the values of the "location" column in
    analysis spreadsheets. This binding is provided by PRINCIPAL_NAME and ALIASES.

    The simplest way to normalize metadata is by renaming keys, but sometimes additional
    work is required. RENAME_KEYS is provided to make this simpler, and is implemented in
    scan post-processessing.
    """

    ALIASES: ClassVar[list[str]] = []
    PRINCIPAL_NAME = ""
    ATTR_TRANSFORMS: ClassVar[dict[str, Callable[..., dict[str, float | list[str] | str]]]] = {}
    MERGE_ATTRS: ClassVar[Spectrometer] = {}

    _SEARCH_DIRECTORIES: tuple[str, ...] = (
        "",
        "hdf5",
        "fits",
        "../Data",
        "../Data/hdf5",
        "../Data/fits",
    )
    _SEARCH_PATTERNS: tuple[str, ...] = (
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
    )
    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".h5",
        ".nc",
        ".fits",
        ".pxt",
        ".nxs",
        ".itx",
        ".txt",
    }
    _USE_REGEX = True

    # adjust as needed
    ENSURE_COORDS_EXIST: ClassVar[set[str]] = {
        "x",
        "y",
        "z",
        "theta",
        "beta",
        "chi",
        "hv",
        "alpha",
        "psi",
    }
    CONCAT_COORDS: ClassVar[list[str]] = [
        "hv",
        "chi",
        "psi",
        "timed_power",
        "tilt",
        "beta",
        "theta",
    ]

    # phi because this happens sometimes at BL4 with core level scans
    SUMMABLE_NULL_DIMS: ClassVar[list[str]] = ["phi", "cycle"]

    RENAME_KEYS: ClassVar[dict[str, str]] = {}

    def __init__(self) -> None:
        """Initialize."""

    @classmethod
    def is_file_accepted(
        cls: type[EndstationBase],
        file: str | Path,
    ) -> bool:
        """Determines whether this loader can load this file."""
        if Path(file).exists() and Path(file).is_file():
            p = Path(file)

            if p.suffix not in cls._TOLERATED_EXTENSIONS:
                return False

            for pattern in cls._SEARCH_PATTERNS:
                regex = re.compile(pattern.format(r"[0-9]+"))
                if regex.match(p.stem):
                    return True

            return False
        try:
            _ = cls.find_first_file(int(file))  # type: ignore[arg-type]
        except ValueError:
            return False
        return True

    @classmethod
    def files_for_search(cls: type[EndstationBase], directory: str | Path) -> list[Path]:
        """Filters files in a directory for candidate scans.

        Here, this just means collecting the ones with extensions acceptable to the loader.
        """
        return [f for f in Path(directory).iterdir() if Path(f).suffix in cls._TOLERATED_EXTENSIONS]

    @classmethod
    def find_first_file(
        cls: type[EndstationBase],
        file_number: int,
    ) -> Path:  # pragma no cover
        """Attempts to find file associated to the scan given the user provided path or scan number.

        This is mostly done by regex matching over available options.
        Endstations which do not require further control here can just provide class attributes:

        * `._SEARCH_DIRECTORIES`: Defining which paths should be checked for scans
        * `._SEARCH_PATTERNS`: Defining acceptable filenames
        * `._USE_REGEX`: Controlling literal or regex filename checking
        * `._TOLERATED_EXTENSIONS`: Controlling whether files should be rejected based on their
          extension.
        """
        warnings.warn(
            "Thisi is the EndstationBase's `first_find_file`. "
            "While it would be the result of best effort for serching the `first file` in"
            "the directrory, but the resultant file may not agree with what you really expected."
            "Considering the explicit file name specification, or writing your own code to"
            "return the file name from the arbitrary number in your own endstation plugin class.",
            stacklevel=2,
        )

        workspace_path = get_workspace_path()
        base_dir: Path = workspace_path
        dir_options: list[Path] = [base_dir / option for option in cls._SEARCH_DIRECTORIES]

        logger.debug(f"dir_options: {dir_options}")
        # another plugin related option here is we can restrict the number of regexes by allowing
        # plugins to install regexes for particular endstations, if this is needed in the future it
        # might be a good way of preventing clashes where there is ambiguity in file naming scheme
        # across endstations

        patterns = [re.compile(m.format(file_number)) for m in cls._SEARCH_PATTERNS]

        if not cls._USE_REGEX:
            msg = "This endstation class does not allow to use the find_first_file."
            raise RuntimeError(msg)

        for directory in dir_options:
            try:
                files: list[Path] = cls.files_for_search(directory)
            except FileNotFoundError:
                continue
            for pattern in patterns:
                for f in files:
                    m = pattern.match(f.stem)
                    if m is not None and m.string == f.stem:
                        return directory / f

        msg = f"Could not find file associated to {file_number}"
        raise ValueError(msg)

    def concatenate_frames(
        self,
        frames: list[xr.Dataset],
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Performs concatenation of frames in multi-frame scans.

        The way this happens is that we look for an axis on which the frames are changing uniformly
        among a set of candidates (`.CONCAT_COORDS`). Then we delegate to xarray to perform the
        concatenation and clean up the merged coordinate.
        """
        if scan_desc:
            logger.debug("scan_desc is not supported at this level")
        if not frames:
            msg = "Could not read any frames."
            raise ValueError(msg)

        if len(frames) == 1:
            return frames[0]

        # determine which axis to stitch them together along, and then do this
        scan_coord = None
        max_different_values = -np.inf
        for possible_scan_coord in self.CONCAT_COORDS:
            coordinates = [f.attrs.get(possible_scan_coord, None) for f in frames]
            n_different_values = len(set(coordinates))
            if n_different_values > max_different_values and None not in coordinates:
                max_different_values = n_different_values
                scan_coord = possible_scan_coord

        assert isinstance(scan_coord, str)

        for f in frames:
            f.coords[scan_coord] = f.attrs[scan_coord]

        frames.sort(key=lambda x: x.coords[scan_coord].min().item())
        return xr.concat(frames, scan_coord)

    def resolve_frame_locations(self, scan_desc: ScanDesc | None = None) -> list[Path]:
        """Determine all files and frames associated to this piece of data.

        This always needs to be overridden in subclasses to handle data appropriately.
        """
        if scan_desc:
            msg = "You need to define resolve_frame_locations or subclass SingleFileEndstation."
        msg = "You need to define resolve_frame_locations or subclass SingleFileEndstation."
        raise NotImplementedError(msg)

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Hook for loading a single frame of data.

        This always needs to be overridden in subclasses to handle data appropriately.
        """
        if frame_path:
            msg = "You need to define load_single_frame."
            raise NotImplementedError(msg)
        if scan_desc:
            msg = "You need to define load_single_frame."
            raise NotImplementedError(msg)
        if kwargs:
            msg = "You need to define load_single_frame."
            raise NotImplementedError(msg)
        return xr.Dataset()

    def postprocess(self, frame: xr.Dataset) -> xr.Dataset:
        """Performs frame level normalization of scan data.

        Here, we currently:
        1. Remove dimensions if they only have a single point, i.e. if the scan has shape [1,N] it
          gets squeezed to have size [N]
        2. Rename attributes
        """
        frame = xr.Dataset(
            {k: rename_dataarray_attrs(v, self.RENAME_KEYS) for k, v in frame.data_vars.items()},
            attrs=rename_keys(frame.attrs, self.RENAME_KEYS),
        )

        sum_dims = [
            dim
            for dim in frame.dims
            if len(frame.coords[dim]) == 1 and dim in self.SUMMABLE_NULL_DIMS
        ]

        if sum_dims:
            frame = frame.sum(sum_dims, keep_attrs=True)

        return frame

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Perform final normalization of scan data.

        This defines the common codepaths for attaching extra information to scans at load time.
        Currently this means we:

        1. Attach a normalized "type" or "kind" of the spectrum indicating what sort of scan it is
        2. Ensure standard coordinates are represented
        3. Apply attribute renaming and attribute transformations defined by class attrs
        4. Ensure the scan endianness matches the system for performance reasons down the line
        """
        # attach the 'spectrum_type'
        # TODO: move this logic into xarray extensions and customize here
        # only as necessary
        scan_desc = scan_desc or {}
        coord_names: tuple[str, ...] = tuple(sorted([str(c) for c in data.dims if c != "cycle"]))
        spectrum_type = _spectrum_type(coord_names)

        modified_data = [
            self._modify_a_data(a_data, spectrum_type)
            for a_data in [data, *[dv for dv in data.data_vars.values() if "eV" in dv.dims]]
        ]
        for a_data in [
            _ensure_coords(a_data, self.ENSURE_COORDS_EXIST) for a_data in modified_data
        ]:
            if "chi" in a_data.coords and "chi_offset" not in a_data.attrs:
                a_data.attrs["chi_offset"] = a_data.coords["chi"].item()

        # go and change endianness and datatypes to something reasonable
        # this is done for performance reasons in momentum space conversion, primarily
        return data.assign(
            {
                name: v.byteswap().newbyteorder()
                if isinstance(v, np.ndarray) and not v.dtype.isnative
                else v
                for name, v in data.data_vars.items()
            },
        )

    def load_from_path(self, path: str | Path) -> xr.Dataset:
        """Convenience wrapper around `.load` which references an explicit path."""
        path = str(path)
        return self.load(
            scan_desc={
                "file": path,
                "location": self.PRINCIPAL_NAME,
            },
        )

    def load(self, scan_desc: ScanDesc | None = None, **kwargs: Incomplete) -> xr.Dataset:
        """Loads a scan from a single file or a sequence of files.

        This method provides the standard procedure for loading data from one or more files:
        1. Resolves file locations (`.resolve_frame_locations`).
        2. Loads each file sequentially (`.load_single_frame`).
        3. Applies any cleaning or processing to each frame (`.postprocess`).
        4. Concatenates the loaded frames into a single dataset (`.concatenate_frames`).
        5. Applies any final postprocessing to the concatenated dataset (`.postprocess_final`).

        This loading workflow can be customized by overriding the respective methods for specific
        beamlines or data sources. It provides a flexible way to integrate with different types of
        data formats and handling strategies.

        Args:
            scan_desc (ScanDesc): The description of the scan, which may contain information such as
                file paths or other metadata.
            kwargs: Additional keyword arguments that will be passed to the `.load_single_frame`
                method for loading each frame.

        Returns:
            xr.Dataset: The concatenated and processed dataset containing the scan data.

        Raises:
            RuntimeError: If no files are found or if there is an error in loading the scan data.
        """
        scan_desc = scan_desc or {}
        logger.debug("Resolving frame locations")
        resolved_frame_locations = self.resolve_frame_locations(scan_desc)
        logger.debug(f"resolved_frame_locations: {resolved_frame_locations}")
        if not resolved_frame_locations:
            msg = "File not found"
            raise RuntimeError(msg)
        logger.debug(f"Found frames: {resolved_frame_locations}")
        frames = [
            self.load_single_frame(fpath, scan_desc, **kwargs) for fpath in resolved_frame_locations
        ]
        frames = [self.postprocess(f) for f in frames]
        concatted = self.concatenate_frames(frames, scan_desc)
        concatted = self.postprocess_final(concatted, scan_desc)

        if "id" in scan_desc:
            concatted.attrs["id"] = scan_desc["id"]

        return concatted

    def _modify_a_data(self, a_data: DataType, spectrum_type: str | None) -> DataType:
        """Helper function to modify the Dataset and DataArray that are contained in the Dataset.

        This method modifies the attributes and coordinates of a given data object
            (either an xarray Dataset or DataArray). It ensures that the "phi" coordinate is
            set to 0 if it doesn't exist, updates the "spectrum_type" attribute, and applies any
            transformations defined in `ATTR_TRANSFORMS`. Additionally, it ensures that default
            attributes from `MERGE_ATTRS` are added to the dataset if they don't already exist.

        Args:
            a_data (DataType): The data object (either an xarray Dataset or DataArray) to modify.
            spectrum_type (str | None): The spectrum type to set as an attribute for the data
                object.

        Returns:
            DataType: The modified data object with updated attributes and coordinates.
        """
        if "phi" not in a_data.coords:
            a_data.coords["phi"] = 0
        a_data.attrs["spectrum_type"] = spectrum_type
        for k, key_fn in self.ATTR_TRANSFORMS.items():
            if k in a_data.attrs:
                transformed = key_fn(a_data.attrs[k])
                if isinstance(transformed, dict):
                    a_data.attrs.update(transformed)
                else:
                    a_data.attrs[k] = transformed
        for k, v in self.MERGE_ATTRS.items():
            a_data.attrs.setdefault(k, v)
        return a_data


class SynchrotronEndstation(EndstationBase):
    """Base class code for ARPES setups at synchrotrons.

    Synchrotron endstations have somewhat complicated light source metadata.
    This stub exists to attach commonalities, such as a resolution table which
    can be interpolated into to retrieve the x-ray linewidth at the
    experimental settings. Additionally, subclassing this is used in resolution
    calculations to signal that such a resolution lookup is required.
    """

    RESOLUTION_TABLE = None


class HemisphericalEndstation(EndstationBase):
    """Base class code for ARPES setups using hemispheres.

    An endstation definition for a hemispherical analyzer should include
    everything needed to determine energy + k resolution, angle conversion,
    and ideally correction databases for dead pixels + detector nonlinearity
    information
    """

    ANALYZER_INFORMATION = None
    SLIT_ORIENTATION = None
    PIXELS_PER_DEG = None


def _spectrum_type(
    coord_names: Sequence[str],
) -> str | None:
    if any(d in coord_names for d in ("x", "y", "z")):
        coord_names = tuple(c for c in coord_names if c not in {"x", "y", "z"})
        spectrum_types = {
            ("eV",): "spem",
            ("eV", "phi"): "ucut",
        }
        return spectrum_types.get(coord_names)
    spectrum_types = {
        ("eV",): "xps",
        ("eV", "phi", "theta"): "map",
        ("eV", "phi", "psi"): "map",
        ("beta", "eV", "phi"): "map",
        ("eV", "hv", "phi"): "hv_map",
        ("eV", "phi"): "cut",
    }
    return spectrum_types.get(tuple(coord_names))


def _ensure_coords(spectrum: DataType, coords_exist: set[str]) -> DataType:
    for coord in coords_exist:
        if coord not in spectrum.coords:
            if coord in spectrum.attrs:
                spectrum.coords[coord] = spectrum.attrs[coord]
            else:
                warnings_msg = f"Could not assign coordinate {coord} from attributes,"
                warnings_msg += "assigning np.nan instead."
                warnings.warn(
                    warnings_msg,
                    stacklevel=2,
                )
                spectrum.coords[coord] = np.nan
    return spectrum
