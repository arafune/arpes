"""Implements data loading for the IF UMCS Lublin ARPES group."""
import xarray as xr
import numpy as np
from pathlib import Path
from arpes.endstations import SCANDESC, HemisphericalEndstation, SingleFileEndstation

__all__ = ("IF_UMCS",)


class IF_UMCS(HemisphericalEndstation, SingleFileEndstation):
    """
    Implements loading xy text files from the Specs Prodigy software.
    """

    PRINCIPAL_NAME = 'IF_UMCS'
    ALIASES = ['IF_UMCS', 'LubARPES', 'LublinARPRES']

    _TOLERATED_EXTENSIONS = {'.xy'}

    RENAME_KEYS = {
        "Eff. Workfunction": "workfunction",
        "Analyzer Slit": "slit",
        "Pass Energy": "pass_energy",
        "Dwell Time": "dwell_time",
        "Analyzer Lens": "lens_mode",
        "Detector Voltage": "mcp_voltage",
    }

    MERGE_ATTRS = {
        "analyzer": "Specs PHOIBOS 150",
        "analyzer_name": "Specs PHOIBOS 150",
        "parallel_deflectors": False,
        "perpendicular_deflectors": False,
        "analyzer_radius": 150,
        "analyzer_type": "hemispherical",
        "mcp_voltage": None,
        # "probe_linewidth": 0.015,
    }

    def load_single_frame(
            self,
            frame_path: str | Path = "",
            scan_desc: SCANDESC | None = None,
            **kwargs: str | float,
    ) -> xr.Dataset:

        """ Load a single frame exported as xy files from Specs Lab Prodigy.   """

        # Read two column data from xy text file
        energy_counts = np.loadtxt(frame_path, comments="#")

        # Read the attributes
        attrs = {}
        with open(frame_path) as my_file:
            file_as_lines = my_file.readlines()
            # read the file header
            for line in file_as_lines:
                if line[0] == '#':
                    key, t, value = line[1:].partition(":")
                    key = key.strip()
                    value = value.strip()
                    if value.isnumeric():
                        attrs[key] = int(value)
                    else:
                        try:
                            attrs[key] = float(value)
                        except ValueError:
                            attrs[key] = value
                    if 'Cycle: 0' in line:
                        break

            num_of_en = attrs["Values/Curve"]

            # TODO count automatically number of energy channels for snapshot mode
            if attrs["Scan Mode"] == "SnapshotFAT":
                num_of_en = 105

            num_of_curves = attrs["Curves/Scan"]
            num_of_polar = energy_counts.size // (2 * num_of_en * num_of_curves)
            theta = np.zeros(num_of_polar)

            ind = 0
            for line in file_as_lines:
                if line[0:12] == '# Parameter:':
                    key, t, value = line.partition(" = ")
                    value = value.strip()
                    theta[ind] = float(value)
                    ind += 1

            kinetic_energy = energy_counts[0:num_of_en, 0]
            kinetic_ef_energy = kinetic_energy - attrs['Excitation Energy']

            counts = energy_counts[:, 1]
            loaded_data = counts.reshape((num_of_polar, num_of_curves, num_of_en))
            loaded_data = np.transpose(loaded_data, (2, 1, 0))

            dispersion_mode = True

            lens_mapping = {
                "HighAngularDispersion":    (np.deg2rad(3), True, None),
                "MediumAngularDispersion":  (np.deg2rad(4), True, None),
                "LowAngularDispersion":     (np.deg2rad(7), True, None),
                "WideAngleMode":            (np.deg2rad(13), True, None),
                "Magnification": (None, False, 9.5),
            }
            lens_mode = attrs["Analyzer Lens"].split(':')[0]

            if lens_mode in lens_mapping:
                phi_max, dispersion_mode, x_det = lens_mapping[lens_mode]
            else:
                raise ValueError("Unknown Analyzer Lens: {}".format(lens_mode))

            if dispersion_mode:
                phi = np.linspace(-phi_max, phi_max, num_of_curves, dtype='float')
                if num_of_polar > 1:
                    dims = ["eV", "phi", "theta"]
                    theta = theta * np.pi / 180
                    coords = {
                        dims[0]: kinetic_ef_energy,
                        dims[1]: phi,
                        dims[2]: theta
                    }
                else:
                    dims = ["eV", "phi"]
                    loaded_data = loaded_data[:, :, 0]
                    coords = {
                        dims[0]: kinetic_ef_energy,
                        dims[1]: phi
                    }
            else:
                x = np.linspace(-x_det, x_det, num_of_curves, dtype='float')
                dims = ["eV", "x"]
                loaded_data = loaded_data[:, :, 0]
                coords = {
                    dims[0]: kinetic_ef_energy,
                    dims[1]: x
                }

        return xr.Dataset(
            {'spectrum': xr.DataArray(
                loaded_data,
                coords=coords,
                dims=dims,
                attrs=attrs
            )
            }
        )

    def postprocess_final(self,
                          data: xr.Dataset,
                          scan_desc: SCANDESC | None = None,
                          ) -> xr.Dataset:
        """Add missing parameters """

        defaults = {
            "x": 78,
            "y": 0.5,
            "z": 2.5,
            "theta": 0,
            "beta": 0,
            "chi": 0,
            "psi": 0,
            "alpha": 0.5 * np.pi,
            "hv": 21.2,
        }
        for k, v in defaults.items():
            data.attrs[k] = v
            for s in data.S.spectra:
                s.attrs[k] = v

        return super().postprocess_final(data, scan_desc)
