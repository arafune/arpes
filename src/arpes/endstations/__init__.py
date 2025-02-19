"""Plugin facility to read and normalize information from different sources to a common format."""

from __future__ import annotations

import contextlib
import copy
import re
import warnings
from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict

import h5py
import numpy as np
import xarray as xr
from astropy.io import fits

from arpes import DATA_PATH
from arpes.config import CONFIG, load_plugins
from arpes.debug import setup_logger
from arpes.load_pxt import find_ses_files_associated, read_single_pxt
from arpes.provenance import Provenance, provenance_from_file
from arpes.utilities.dict import rename_dataarray_attrs

from .fits_utils import find_clean_coords
from .igor_utils import shim_wave_note

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    from _typeshed import Incomplete

    from arpes._typing import DataType, Spectrometer

__all__ = [
    "EndstationBase",
    "FITSEndstation",
    "HemisphericalEndstation",
    "ScanDesc",
    "SingleFileEndstation",
    "SynchrotronEndstation",
    "add_endstation",
    "endstation_from_alias",
    "endstation_name_from_alias",
    "load_scan",
    "resolve_endstation",
]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


_ENDSTATION_ALIASES: dict[str, type[EndstationBase]] = {}


class ScanDesc(TypedDict, total=False):
    """TypedDict based class for scan_desc."""

    file: str | Path
    location: str
    path: str | Path
    note: dict[str, str | float]  # used as attrs basically.
    id: int | str


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
    ) -> Path:
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
        workspace = CONFIG["WORKSPACE"]
        assert "path" in workspace
        workspace_path = Path(workspace["path"]) / "data" if workspace else Path()
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
        from arpes.utilities import rename_keys

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


class SingleFileEndstation(EndstationBase):
    """Abstract endstation which loads data from a single file.

    This just specializes the routine used to determine the location of files on disk.

    Unlike general endstations, if your data comes in a single file you can trust that the
    file given to you in the spreadsheet or direct load calls is all there is.
    """

    def resolve_frame_locations(self, scan_desc: ScanDesc | None = None) -> list[Path]:
        """Single file endstations just use the referenced file from the scan description."""
        if scan_desc is None:
            msg = "Must pass dictionary as file scan_desc to all endstation loading code."
            raise ValueError(
                msg,
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert original_data_loc
        if not Path(original_data_loc).exists():
            if DATA_PATH is not None:
                original_data_loc = Path(DATA_PATH) / original_data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)
        return [Path(original_data_loc)]


class SESEndstation(EndstationBase):
    """Provides collation and loading for Scienta's SESWrapper and endstations using it.

    These files have special frame names, at least at the beamlines Conrad has encountered.
    """

    def resolve_frame_locations(self, scan_desc: ScanDesc | None = None) -> list[Path]:
        if scan_desc is None:
            msg = "Must pass dictionary as file scan_desc to all endstation loading code."
            raise ValueError(
                msg,
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert original_data_loc
        if not Path(original_data_loc).exists():
            if DATA_PATH is not None:
                original_data_loc = Path(DATA_PATH) / original_data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)

        return find_ses_files_associated(Path(original_data_loc))

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: bool,
    ) -> xr.Dataset:
        """Load the single frame from the specified file.

        This method loads a single frame of data from a file.
            If the file is in NetCDF (".nc") format, it loads the data using the `load_SES_nc`
            method, passing along the `scan_desc` dictionary and any additional keyword arguments.
            If the file is in PXT format, it reads the data, negates the energy values, and returns
            the data as an `xarray.Dataset` with the `spectrum` key.

        Args:
            frame_path (str | Path): The path to the file containing the single frame of data.
            scan_desc (ScanDesc | None): A description of the scan, which is passed to the
            `load_SES_nc` function if the file is in NetCDF format. Defaults to `None`.
            kwargs (bool): Additional keyword arguments passed to `load_SES_nc`. The only accepted
            argument is "robust_dimension_labels".

        Returns:
            xr.Dataset: The dataset containing the loaded spectrum data.
            Load the single frame from the file.
        """
        ext = Path(frame_path).suffix
        if scan_desc is None:
            scan_desc = {}
        if "nc" in ext:
            # was converted to hdf5/NetCDF format with Conrad's Igor scripts
            scan_desc["path"] = Path(frame_path)
            return self.load_SES_nc(scan_desc=scan_desc, **kwargs)

        # it's given by SES PXT files

        pxt_data = read_single_pxt(frame_path).assign_coords(
            {"eV": -read_single_pxt(frame_path).eV.values},
        )  # negate energy
        return xr.Dataset({"spectrum": pxt_data}, attrs=pxt_data.attrs)

    def postprocess(self, frame: xr.Dataset) -> xr.Dataset:
        import arpes.xarray_extensions  # pylint: disable=unused-import, redefined-outer-name

        frame = super().postprocess(frame)
        return frame.assign_attrs(frame.S.spectrum.attrs)

    def load_SES_nc(
        self,
        scan_desc: ScanDesc | None = None,
        *,
        robust_dimension_labels: bool = False,
    ) -> xr.Dataset:
        """Imports an hdf5 dataset exported from Igor that was originally generated in SES format.

        In order to understand the structure of these files have a look at Conrad's saveSESDataset
        in Igor Pro.

        Args:
            scan_desc: Dictionary with extra information to attach to the xr.Dataset, must contain
              the location of the file
            robust_dimension_labels: safety control, used to load despite possibly malformed
              dimension names
            kwargs: kwargs, unused currently

        Returns:
            Loaded data.
        """
        scan_desc = scan_desc or {}

        data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert data_loc is not None
        if not Path(data_loc).exists():
            if DATA_PATH is not None:
                data_loc = Path(DATA_PATH) / data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)

        wave_note = shim_wave_note(data_loc)
        f = h5py.File(data_loc, "r")

        primary_dataset_name = next(iter(f))
        # This is bugged for the moment in h5py due to an inability to read fixed length unicode
        # strings

        # Use dimension labels instead of
        dimension_labels = list(f["/" + primary_dataset_name].attrs["IGORWaveDimensionLabels"][0])
        if any(not x for x in dimension_labels):
            logger.info(dimension_labels)

            if not robust_dimension_labels:
                msg = "Missing dimension labels. Use robust_dimension_labels=True to override"
                raise ValueError(
                    msg,
                )
            used_blanks = 0
            for i in range(len(dimension_labels)):
                if not dimension_labels[i]:
                    dimension_labels[i] = f"missing{used_blanks}"
                    used_blanks += 1

            logger.info(dimension_labels)

        scaling = f["/" + primary_dataset_name].attrs["IGORWaveScaling"][-len(dimension_labels) :]
        raw_data = f["/" + primary_dataset_name][:]

        scaling = [
            np.linspace(
                scale[1],
                scale[1] + scale[0] * raw_data.shape[i],
                raw_data.shape[i],
                dtype=np.float64,
            )
            for i, scale in enumerate(scaling)
        ]

        dataset_contents = {}
        attrs = scan_desc.pop("note", {})
        attrs.update(wave_note)

        built_coords = dict(zip(dimension_labels, scaling, strict=True))

        deg_to_rad_coords = {"theta", "beta", "phi", "alpha", "psi"}

        # the hemisphere axis is handled below
        built_coords = {
            k: np.deg2rad(c) if k in deg_to_rad_coords else c for k, c in built_coords.items()
        }

        deg_to_rad_attrs = {"theta", "beta", "alpha", "psi", "chi"}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in attrs:
                attrs[angle_attr] = np.deg2rad(float(attrs[angle_attr]))

        dataset_contents["spectrum"] = xr.DataArray(
            raw_data,
            coords=built_coords,
            dims=dimension_labels,
            attrs=attrs,
        )
        provenance_context: Provenance = {"what": "Loaded SES dataset from HDF5.", "by": "load_SES"}
        provenance_from_file(dataset_contents["spectrum"], str(data_loc), provenance_context)
        return xr.Dataset(
            dataset_contents,
            attrs={**scan_desc, "dataset_name": primary_dataset_name},
        )


class FITSEndstation(EndstationBase):
    """Loads data from the .fits format produced by the MAESTRO software and derivatives.

    This ends up being somewhat complicated, because the FITS export is written in LabView and
    does not conform to the standard specification for the FITS archive format.

    Many of the intricacies here are in fact those shared between MAESTRO's format
    and the Lanzara Lab's format. Conrad does not foresee this as an issue, because it is
    unlikely that many other ARPES labs will adopt this data format moving forward, in
    light of better options derivative of HDF like the NeXuS format.

    Memo: RA would not maintain this class.
    """

    PREPPED_COLUMN_NAMES: ClassVar[dict[str, str]] = {
        "time": "time",
        "Delay": "delay-var",  # these are named thus to avoid conflicts with the
        "Sample-X": "cycle-var",  # underlying coordinates
        "Mira": "pump_power",
        # insert more as needed
    }

    SKIP_COLUMN_NAMES: ClassVar[set[str]] = {
        "Phi",
        "null",
        "X",
        "Y",
        "Z",
        "mono_eV",
        "Slit Defl",
        "Optics Stage",
        "Scan X",
        "Scan Y",
        "Scan Z",
        # insert more as needed
    }

    SKIP_COLUMN_FORMULAS: ClassVar = {
        lambda name: bool("beamview" in name or "IMAQdx" in name),
    }

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "Phi": "chi",
        "Beta": "beta",
        "Azimuth": "chi",
        "Pump_energy_uJcm2": "pump_fluence",
        "T0_ps": "t0_nominal",
        "W_func": "workfunction",
        "Slit": "slit",
        "LMOTOR0": "x",
        "LMOTOR1": "y",
        "LMOTOR2": "z",
        "LMOTOR3": "theta",
        "LMOTOR4": "beta",
        "LMOTOR5": "chi",
        "LMOTOR6": "alpha",
    }

    def resolve_frame_locations(self, scan_desc: ScanDesc | None = None) -> list[Path]:
        """Determines all files associated with a given scan.

        This function resolves the file location(s) based on the provided `scan_desc` dictionary.
        It looks for the "path" or "file" key in the `scan_desc` to determine the file location.
        If the file does not exist at the provided location, it will attempt to find it in the
        `DATA_PATH` directory. If the file is still not found, a `RuntimeError` is raised.

        Args:
            scan_desc (ScanDesc | None): A dictionary containing scan metadata.
            It must include a "path" or "file" key specifying the location of the scan data file.

        Returns:
            list[Path]: A list containing the resolved file path(s).

        Raises:
            ValueError: If `scan_desc` is not provided or is `None`.
            RuntimeError: If the file cannot be found at the specified location or in the
                `DATA_PATH` directory.
        """
        if scan_desc is None:
            msg = "Must pass dictionary as file scan_desc to all endstation loading code."
            raise ValueError(
                msg,
            )
        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert original_data_loc
        if not Path(original_data_loc).exists():
            if DATA_PATH is not None:
                original_data_loc = Path(DATA_PATH) / original_data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)
        return [Path(original_data_loc)]

    def load_single_frame(  # noqa: PLR0915, PLR0912, C901
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Loads a scan from a single .fits file.

        This assumes the DAQ storage convention set by E. Rotenberg (possibly earlier authors)
        for the storage of ARPES data in FITS tables.

        This involves several complications:

        1. Hydrating/extracting coordinates from start/delta/n formats
        2. Extracting multiple scan regions
        3. Gracefully handling missing values
        4. Unwinding different scan conventions to common formats
        5. Handling early scan termination
        """
        if kwargs:
            logger.debug("load_single_frame: Any kwargs is not used at this level")
        # Use dimension labels instead of
        logger.debug("Opening FITS HDU list.")
        hdulist = fits.open(frame_path, ignore_missing_end=True)
        primary_dataset_name = None

        # Clean the header because sometimes out LabView produces improper FITS files
        for i in range(len(hdulist)):
            # This looks a little stupid, but because of confusing astropy internals actually works
            hdulist[i].header["UN_0_0"] = ""  # TODO: This card is broken, this is not a good fix
            del hdulist[i].header["UN_0_0"]
            hdulist[i].header["UN_0_0"] = ""
            if "TTYPE2" in hdulist[i].header and hdulist[i].header["TTYPE2"] == "Delay":
                logger.debug("Using ps delay units. This looks like an ALG main chamber scan.")
                hdulist[i].header["TUNIT2"] = ""
                del hdulist[i].header["TUNIT2"]
                hdulist[i].header["TUNIT2"] = "ps"

            logger.debug(f"HDU {i}: Attempting to fix FITS errors.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdulist[i].verify("fix+warn")
                hdulist[i].header.update()
            # This actually requires substantially more work because it is lossy to information
            # on the unit that was encoded

        hdu = hdulist[1]
        scan_desc = scan_desc or {}
        attrs = scan_desc.pop("note", scan_desc)
        attrs.update(dict(hdulist[0].header))  # type: ignore  # noqa: PGH003

        drop_attrs = ["COMMENT", "HISTORY", "EXTEND", "SIMPLE", "SCANPAR", "SFKE_0"]
        for dropped_attr in drop_attrs:
            if dropped_attr in attrs:
                del attrs[dropped_attr]  # type: ignore  # noqa: PGH003

        from arpes.utilities import rename_keys

        built_coords, dimensions, real_spectrum_shape = find_clean_coords(
            hdu,
            attrs,  # type: ignore  # noqa: PGH003
            mode="MC",
        )
        logger.debug("Recovered coordinates from FITS file.")

        attrs = rename_keys(attrs, self.RENAME_KEYS)  # type: ignore  # noqa: PGH003
        scan_desc = rename_keys(scan_desc, self.RENAME_KEYS)  # type: ignore  # noqa: PGH003

        def clean_key_name(k: str) -> str:
            if "#" in k:
                k = k.replace("#", "num")
            return k

        attrs = {clean_key_name(k): v for k, v in attrs.items()}
        scan_desc = {clean_key_name(k): v for k, v in scan_desc.items()}  # type: ignore  # noqa: PGH003

        # don't have phi because we need to convert pixels first
        deg_to_rad_coords = {"beta", "theta", "chi"}

        # convert angular attributes to radians
        for coord_name in deg_to_rad_coords:
            if coord_name in attrs:
                with contextlib.suppress(TypeError, ValueError):
                    attrs[coord_name] = np.deg2rad(float(attrs[coord_name]))

            if coord_name in scan_desc:
                with contextlib.suppress(TypeError, ValueError):
                    scan_desc[coord_name] = np.deg2rad(float(scan_desc[coord_name]))  # type: ignore  # noqa: PGH003

        data_vars = {}

        all_names = hdu.columns.names
        n_spectra = len([n for n in all_names if "Fixed_Spectra" in n or "Swept_Spectra" in n])
        for column_name in hdu.columns.names:
            # we skip some fixed set of the columns, such as the one dimensional axes, as well as
            # things that are too tricky to load at the moment, like the microscope images from
            # MAESTRO
            should_skip = False
            if column_name in self.SKIP_COLUMN_NAMES:
                should_skip = True

            for formula in self.SKIP_COLUMN_FORMULAS:
                if formula(column_name):
                    should_skip = True

            if should_skip:
                continue

            # the hemisphere axis is handled below
            dimension_for_column = dimensions[column_name]
            column_shape = real_spectrum_shape[column_name]

            column_display = self.PREPPED_COLUMN_NAMES.get(column_name, column_name)
            if "Fixed_Spectra" in column_display:
                if n_spectra == 1:
                    column_display = "spectrum"
                else:
                    column_display = "spectrum" + "-" + column_display.split("Fixed_Spectra")[1]

            if "Swept_Spectra" in column_display:
                if n_spectra == 1:
                    column_display = "spectrum"
                else:
                    column_display = "spectrum" + "-" + column_display.split("Swept_Spectra")[1]

            # sometimes if a scan is terminated early it can happen that the sizes do not match the
            # expected value as an example, if a beta map is supposed to have 401 slices, it might
            # end up having only 260 if it were terminated early
            # If we are confident in our parsing code above, we can handle this case and take a
            # subset of the coords so that the data matches
            try:
                resized_data = hdu.data.columns[column_name].array.reshape(column_shape)
            except ValueError:
                # if we could not resize appropriately, we will try to reify the shapes together
                rest_column_shape = column_shape[1:]
                n_per_slice = int(np.prod(rest_column_shape))
                total_shape = hdu.data.columns[column_name].array.shape
                total_n = np.prod(total_shape)

                n_slices = total_n // n_per_slice
                # if this isn't true, we can't recover
                data_for_resize = hdu.data.columns[column_name].array
                if total_n // n_per_slice != total_n / n_per_slice:
                    # the last slice was in the middle of writing when something hit the fan
                    # we need to infer how much of the data to read, and then repeat the above
                    # we need to cut the data

                    # This can happen when the labview crashes during data collection,
                    # we use column_shape[1] because of the row order that is used in the FITS file
                    data_for_resize = data_for_resize[
                        0 : (total_n // n_per_slice) * column_shape[1]
                    ]
                    warning_msg = "Column {} was in the middle of slice when DAQ stopped."
                    warning_msg += "Throwing out incomplete slice..."
                    warnings.warn(
                        warning_msg.format(
                            column_name,
                        ),
                        stacklevel=2,
                    )

                column_shape = list(column_shape)
                column_shape[0] = n_slices

                try:
                    resized_data = data_for_resize.reshape(column_shape)
                except Exception:
                    logger.exception(
                        "Found an error in resized_data=data_for_resize.rechape(column_shape)",
                    )
                    # sometimes for whatever reason FITS errors and cannot read the data
                    continue

                # we also need to adjust the coordinates
                altered_dimension = dimension_for_column[0]
                built_coords[altered_dimension] = built_coords[altered_dimension][:n_slices]

            data_vars[column_display] = xr.DataArray(
                resized_data,
                coords={k: c for k, c in built_coords.items() if k in dimension_for_column},
                dims=dimension_for_column,
                attrs=attrs,
            )

        def prep_spectrum(data: xr.DataArray) -> xr.DataArray:
            # don't do center pixel inference because the main chamber
            # at least consistently records the offset from the edge
            # of the recorded window
            if "pixel" in data.coords:
                phi_axis = data.coords["pixel"].values * np.deg2rad(1 / 10)

                if "pixel" in data.coords:
                    data = data.rename(pixel="phi")

                data = data.assign_coords(phi=phi_axis)

            # Always attach provenance
            provenance_context: Provenance = {
                "what": "Loaded MC dataset from FITS.",
                "by": "load_MC",
            }
            provenance_from_file(data, str(frame_path), provenance_context)

            return data

        if "spectrum" in data_vars:
            data_vars["spectrum"] = prep_spectrum(data_vars["spectrum"])

        # adjust angular coordinates
        built_coords = {
            k: np.deg2rad(c) if k in deg_to_rad_coords else c for k, c in built_coords.items()
        }

        logger.debug("Stitching together xr.Dataset.")
        return xr.Dataset(
            {
                f"safe-{name}" if name in data_var.coords else name: data_var
                for name, data_var in data_vars.items()
            },
            attrs={**scan_desc, "dataset_name": primary_dataset_name},
        )


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


def endstation_from_alias(alias: str) -> type[EndstationBase]:
    """Lookup the data loading class from an alias."""
    return _ENDSTATION_ALIASES[alias]


def endstation_name_from_alias(alias: str) -> str:
    """Lookup the data loading principal location from an alias."""
    return endstation_from_alias(alias).PRINCIPAL_NAME


def add_endstation(endstation_cls: type[EndstationBase]) -> None:
    """Registers a data loading plugin (Endstation class) together with its aliases.

    You can use this to add a plugin after the original search if it is defined in another
    module or in a notebook.
    """
    assert endstation_cls.PRINCIPAL_NAME is not None
    for alias in endstation_cls.ALIASES:
        if alias in _ENDSTATION_ALIASES:
            continue

        _ENDSTATION_ALIASES[alias] = endstation_cls

    _ENDSTATION_ALIASES[endstation_cls.PRINCIPAL_NAME] = endstation_cls


def resolve_endstation(*, retry: bool = True, **kwargs: Incomplete) -> type[EndstationBase]:
    """Tries to determine which plugin to use for loading a piece of data.

    Args:
        retry (bool): Whether to attempt to reload plugins and try again after failure.
          This is used as an import guard basiscally in case the user imported things
          very strangely to ensure plugins are loaded.
        kwargs: Contains the actual information required to identify the scan.

    Returns:
        The loading plugin that should be used for the data.
    """
    endstation_name = kwargs.get("location", kwargs.get("endstation"))

    # check if the user actually provided a plugin
    if isinstance(endstation_name, type):
        return endstation_name

    if endstation_name is None:
        warnings.warn("Endstation not provided. Using `fallback` plugin.", stacklevel=2)
        endstation_name = "fallback"
    logger.debug(f"_ENDSTATION_ALIASES is : {_ENDSTATION_ALIASES}")
    try:
        return endstation_from_alias(endstation_name)
    except KeyError as key_error:
        if retry:
            logger.debug("retry with `arpes.config.load_plugins()`")
            import arpes.config

            load_plugins()
            return resolve_endstation(retry=False, **kwargs)
        msg = "Could not identify endstation. Did you set the endstation or location?"
        msg += "Find a description of the available options in the endstations module."
        raise ValueError(msg) from key_error


def load_scan(
    scan_desc: ScanDesc,
    *,
    retry: bool = True,
    **kwargs: Incomplete,
) -> xr.Dataset:
    """Resolves a plugin and delegates loading a scan.

    This is used internally by `load_data` and should not be invoked directly
    by users.

    Determines which data loading class is appropriate for the data,
    shuffles a bit of metadata, and calls the .load function on the
    retrieved class to start the data loading process.

    Args:
        scan_desc: Information identifying the scan, typically the full path.
        retry: Used to attempt a reload of plugins and subsequent data load attempt.
        kwargs: pass to the endstation.load(scan_dec, **kwargs)

    Returns:
        Loaded and normalized ARPES scan data.
    """
    note: dict[str, str | float] | ScanDesc = scan_desc.get("note", scan_desc)
    full_note: ScanDesc = copy.deepcopy(scan_desc)
    assert isinstance(note, dict)
    full_note.update(note)

    endstation_cls = resolve_endstation(retry=retry, **full_note)
    logger.debug(f"Using plugin class {endstation_cls}")

    key: Literal["file", "path"] = "file" if "file" in scan_desc else "path"

    file = scan_desc[key]
    try:
        file_number: int = int(str(file))
        file = endstation_cls.find_first_file(file_number)
        scan_desc[key] = file
    except ValueError:
        pass

    logger.debug(f"Loading {scan_desc}")
    endstation = endstation_cls()
    return endstation.load(scan_desc, **kwargs)
