"""Tools to get information about the running notebook and kernel."""

from __future__ import annotations

import datetime
import json
import urllib.request
from datetime import UTC
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from os import SEEK_END
from pathlib import Path
from typing import TYPE_CHECKING, Required, TypedDict, TypeVar
from urllib.error import HTTPError

import ipykernel
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from jupyter_server import serverapp
from tqdm.notebook import tqdm
from traitlets.config import MultipleInstanceError

from arpes.config import CONFIG

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import Incomplete
__all__ = (
    "generate_logfile_path",
    "get_full_notebook_information",
    "get_notebook_name",
    "get_recent_history",
    "get_recent_logs",
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

T = TypeVar("T")


def wrap_tqdm(
    x: Iterable[T],
    *args: Incomplete,
    interactive: bool = True,
    **kwargs: Incomplete,
) -> Iterable[T]:
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
    notebook: Required[dict[str, str]]


class NoteBookInfomation(TypedDict, total=True):
    server: ServerInfo
    session: SessionInfo


def get_full_notebook_information() -> NoteBookInfomation | None:
    """Javascriptless method to fetch current notebook sessions and the one matching this kernel.

    Returns:
        [TODO:description]

    Raises:
        ValueError: [TODO:description]
    """
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
        except HTTPError:
            logger.debug("Could not read notebook information")
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
    except (AttributeError, AssertionError):
        return ["No accessible history."]


def get_recent_logs(n_bytes: int = 1000) -> list[str]:
    """Fetches a recent chunk of user logs. Used to populate a context on provenance outputs."""
    try:
        ipython = get_ipython()
        assert isinstance(ipython, InteractiveShell)
        if CONFIG["LOGGING_STARTED"]:
            logging_file = CONFIG["LOGGING_FILE"]
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

    except (AttributeError, AssertionError):
        pass

    return ["No logging available. Logging is only available inside Jupyter."]
