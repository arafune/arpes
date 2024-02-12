"""Utilities related to handling Igor data ingestion quirks."""
from __future__ import annotations

import contextlib
import itertools
import subprocess
from pathlib import Path
from typing import Any

__all__ = ("shim_wave_note",)


def shim_wave_note(path: str | Path) -> dict[str, Any]:
    """Hack to read the corrupted wavenote out of the h5 files that Igor has been producing.

    h5 dump still produces the right value, so we use it from the command line in order to get the
    value of the note.

    This is not necessary unless you are trying to read HDF files exported from Igor (the
    preferred way before we developed an Igor data loading plugin).

    Args:
        path: Location of the file

    Returns:
        The header/wavenote contents.
    """
    wave_name = Path(path).stem
    cmd = f"h5dump -A --attribute /{wave_name}/IGORWaveNote {path}"
    h5_out = subprocess.getoutput(cmd)

    split_data = h5_out[h5_out.index("DATA {") :]
    assert len(split_data.split('"')) == 3  # noqa: PLR2004
    data = split_data.split('"')[1]

    # remove stuff below the end of the header
    with contextlib.suppress(ValueError):
        data = data[: data.index("ENDHEADER")]

    lines = [line.strip() for line in data.splitlines() if "=" in line]
    return dict([line.split("=") for line in itertools.chain(*[line.split(",") for line in lines])])
