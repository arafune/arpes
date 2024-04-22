"""Tools to get information about the running notebook and kernel."""

from __future__ import annotations

import datetime
import json
import urllib.request
import warnings
from datetime import UTC
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from os import SEEK_END
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from tqdm.notebook import tqdm
from traitlets.config import MultipleInstanceError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import Incomplete
__all__ = (
    "get_full_notebook_information",
    "get_notebook_name",
    "generate_logfile_path",
    "get_recent_logs",
    "get_recent_history",
    "wrap_tqdm",
)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def wrap_tqdm(
    x: Iterable[int],
    *args: Incomplete,
    interactive: bool = True,
    **kwargs: Incomplete,
) -> Iterable[int]:
    """Wraps with tqdm but supports disabling with a flag."""
    if not interactive:
        return x

    return tqdm(x, *args, **kwargs)


class ServerInfo(TypedDict, total=False):
    base_url: str
    password: bool
    pid: int
    port: int
    root_dir: str
    secure: bool
    sock: str
    token: str
    url: str
    version: str


class SessionInfo(TypedDict, total=False):
    id: str
    path: str
    name: str
    type: str
    kernel: dict[str, str | int]
    notebook: dict[str, str]


class NoteBookInfomation(TypedDict, total=False):
    server: ServerInfo
    session: SessionInfo


def get_full_notebook_information() -> NoteBookInfomation | None:
    """Javascriptless method to fetch current notebook sessions and the one matching this kernel.

    Returns:
        [TODO:description]

    Raises:
        ValueError: [TODO:description]
    """
    try:  # Respect those that opt not to use IPython
        import ipykernel
        from jupyter_server import serverapp
    except ImportError:
        msg = "Check your installation about ipykernel & jupyter_server (newer should be better)."
        warnings.warn(msg, stacklevel=2)
        return None
    try:
        connection_file = Path(ipykernel.get_connection_file()).stem
    except (MultipleInstanceError, RuntimeError):
        return None

    logger.debug(f"connection_file: {connection_file}")
    kernel_id = connection_file.split("-", 1)[1] if "-" in connection_file else connection_file

    servers = serverapp.list_running_servers()
    for server in servers:
        logger.debug(f"server: {server}")
        try:
            passwordless = not server["token"] and not server["password"]
            url = (
                server["url"]
                + "api/sessions"
                + ("" if passwordless else "?token={}".format(server["token"]))
            )
            if not url.startswith(("http:", "https:")):
                msg = "URL must start with 'http:' or 'https:'"
                raise ValueError(msg)
            sessions = json.load(urllib.request.urlopen(url))  # noqa: S310
            for sess in sessions:
                if sess["kernel"]["id"] == kernel_id:
                    return {
                        "server": server,
                        "session": sess,
                    }
        except (KeyError, TypeError):
            pass
    return None


def get_notebook_name() -> str:
    """Gets the unqualified name of the running Jupyter notebook if not password protected.

    As an example, if you were running a notebook called "Doping-Analysis.ipynb"
    this would return "Doping-Analysis".

    If no notebook is running for this kernel or the Jupyter session is password protected, we
    can only return None.
    """
    jupyter_info = get_full_notebook_information()
    if jupyter_info:
        return Path(jupyter_info["session"]["notebook"]["name"]).stem
    return ""


def generate_logfile_path() -> Path:
    """Generates a time and date qualified path for the notebook log file."""
    base_name = get_notebook_name() or "unnamed"
    full_name = "{}_{}_{}.log".format(
        base_name,
        datetime.datetime.now(tz=datetime.UTC).date().isoformat(),
        datetime.datetime.now(UTC).time().isoformat().split(".")[0].replace(":", "-"),
    )
    return Path("logs") / full_name


def get_recent_history(n_items: int = 10) -> list[str]:
    """Fetches recent cell evaluations for context on provenance outputs."""
    try:
        from IPython.core.getipython import get_ipython
        from IPython.core.interactiveshell import InteractiveShell

        ipython = get_ipython()
        assert isinstance(ipython, InteractiveShell)
        return [
            _[-1]
            for _ in list(
                ipython.history_manager.get_tail(  # type: ignore [union-attr]
                    n=n_items,
                    include_latest=True,
                ),
            )
        ]
    except (ImportError, AttributeError, AssertionError):
        return ["No accessible history."]


def get_recent_logs(n_bytes: int = 1000) -> list[str]:
    """Fetches a recent chunk of user logs. Used to populate a context on provenance outputs."""
    import arpes.config

    try:
        from IPython.core.getipython import get_ipython
        from IPython.core.interactiveshell import InteractiveShell

        ipython = get_ipython()
        assert isinstance(ipython, InteractiveShell)
        if arpes.config.CONFIG["LOGGING_STARTED"]:
            logging_file = arpes.config.CONFIG["LOGGING_FILE"]
            assert isinstance(logging_file, str | Path)
            with Path(logging_file).open("rb") as file:
                try:
                    file.seek(-n_bytes, SEEK_END)
                except OSError:
                    file.seek(0)

                lines = file.readlines()

            # ensure we get the most recent information
            final_cell = ipython.history_manager.get_tail(  # type: ignore [union-attr]
                n=1,
                include_latest=True,
            )[0][-1]
            return [_.decode() for _ in lines] + [final_cell]

    except (ImportError, AttributeError, AssertionError):
        pass

    return ["No logging available. Logging is only available inside Jupyter."]
