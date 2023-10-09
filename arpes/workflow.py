"""Utilities for managing analyses. Use of these is highly subjective and not core to PyARPES.

This module contains some extra utilities for managing scientific workflows, especially between
notebooks and workspaces.

Provides:
- go_to_figures
- go_to_workspace
- go_to_cwd

- publish_data
- read_data
- consume_data
- summarize_data

A core feature of this module is that you can export and import data from between notebooks.
`easy_pickle` also fulfills this to an extent, but with the tools included here,
there are some extra useful goodies, like tracking of publishing and consuming notebooks,
so that you get reminders about where your data comes from if you need to regenerate it.

This isn't dataflow for Jupyter notebooks, but it is at least more convenient than managing it
all yourself.

Another (better?) usage pattern is to turn data dependencies into code-dependencies (re-run
reproducible analyses) and share code between notebooks using a local module.
"""
from __future__ import annotations  # noqa: I001

import subprocess
import sys
from collections import defaultdict
from functools import wraps
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Any

from logging import INFO, Formatter, StreamHandler, getLogger

import dill

from arpes.config import WorkspaceManager
from arpes.plotting.utils import path_for_plot
from arpes.utilities.jupyter import get_notebook_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete

    from arpes._typing import WORKSPACETYPE

__all__ = (
    "go_to_figures",
    "go_to_workspace",
    "go_to_cwd",
    "publish_data",
    "read_data",
    "consume_data",
    "summarize_data",
)


LOGLEVEL = INFO
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def with_workspace(f: Callable) -> Callable:
    @wraps(f)
    def wrapped_with_workspace(*args, workspace=None, **kwargs: Incomplete):
        with WorkspaceManager(workspace=workspace):
            import arpes.config

            workspace: WORKSPACETYPE = arpes.config.CONFIG["WORKSPACE"]

        return f(*args, workspace=workspace, **kwargs)

    return wrapped_with_workspace


def _open_path(p: Path | str) -> None:
    """Attempts to open the path p in the filesystem explorer, or else prints the path.

    Args:
        p: The path to open.
    """
    if "win" in sys.platform:
        subprocess.Popen(rf"explorer {p}")
    else:
        print(p)  # noqa: T201


@with_workspace
def go_to_workspace(workspace: WORKSPACETYPE | None = None) -> None:
    """Opens the workspace folder, otherwise opens the location of the running notebook."""
    path = Path.cwd()

    from arpes.config import CONFIG

    workspace = workspace or CONFIG["WORKSPACE"]

    if workspace:
        path = Path(workspace["path"])

    _open_path(path)


def go_to_cwd() -> None:
    """Opens the current working directory in the OS file browser."""
    _open_path(Path.cwd())


def go_to_figures() -> None:
    """Opens the figures folder.

    If in a workspace, opens the figures folder for the current workspace and the current day,
    otherwise finds the most recent one and opens it.
    """
    path = path_for_plot("")
    if not Path(path).exists():
        path = sorted(Path(path).parent.glob("*"))[-1]

    _open_path(path)


def get_running_context() -> tuple[Incomplete, Path]:
    return get_notebook_name(), Path.cwd()


class DataProvider:
    workspace_name: str | None
    path: Path

    def _read_pickled(self, name: str, default=None):
        try:
            with Path(self.path / f"{name}.pickle").open("rb") as f:
                return dill.load(f)
        except FileNotFoundError:
            return default

    def _write_pickled(self, name: str, value: object) -> None:
        with Path(self.path / f"{name}.pickle").open("wb") as f:
            dill.dump(value, f)

    @property
    def publishers(self) -> dict[str, object]:
        return self._read_pickled("publishers", defaultdict(list))

    @publishers.setter
    def publishers(self, new_publishers: dict[str, object]) -> None:
        assert isinstance(new_publishers, dict)
        self._write_pickled("publishers", new_publishers)

    @property
    def consumers(self) -> dict[str, object]:
        return self._read_pickled("consumers", defaultdict(list))

    @consumers.setter
    def consumers(self, new_consumers: dict[str, object]) -> None:
        assert isinstance(new_consumers, dict)
        self._write_pickled("consumers", new_consumers)

    def __init__(self, path: Path, workspace_name: str | None = None) -> None:
        self.path = path / "data_provider"
        self.workspace_name = workspace_name

        if self.workspace_name is None:
            if not self.path.exists():
                msg = 'No detected workspace or "data_provider" folder.'
                msg += "Ensure you are in a workspace"
                msg += " or let PyARPES know this is safe by adding the"
                msg += ' "data_provider" folder yourself.'
                raise ValueError(
                    msg,
                )
        elif not self.path.exists():
            self.path.mkdir(parents=True)

        if not (self.path / "data").exists():
            (self.path / "data").mkdir(parents=True)

    def publish(self, key: str, data: object) -> None:
        context = get_running_context()
        publishers = self.publishers

        old_publisher = publishers[key]
        msg = f"{old_publisher} -> {[context]}"
        logger.warning(msg)

        publishers[key] = [context]
        self.publishers = publishers
        self.write_data(key, data)

        self.summarize_consumers(key=key)

    def consume(self, key: str, *, subscribe: bool = True) -> object:
        if subscribe:
            context = get_running_context()
            consumers = self.consumers

            if not any(c == context for c in consumers[key]):
                consumers[key].append(context)
                self.consumers = consumers

            self.summarize_clients(key if key != "*" else "")

        return self.read_data(key)

    @classmethod
    def from_workspace(
        cls: type[DataProvider],
        workspace: WORKSPACETYPE | None = None,
    ) -> DataProvider:
        if workspace is not None:
            return cls(path=Path(workspace["path"]), workspace_name=workspace["name"])

        return cls(path=Path(Path.cwd()), workspace_name=None)

    def summarize_clients(self, key: str = "") -> None:
        self.summarize_publishers(key=key)
        self.summarize_consumers(key=key)

    def summarize_publishers(self, key: str = "") -> None:
        if key == "*":
            key = ""

        publishers = self.publishers
        print(f'PUBLISHERS FOR {key or "ALL"}')
        if not key:
            pprint(dict(publishers))
        else:
            pprint({k: v for k, v in publishers.items() if k == key})

    def summarize_consumers(self, key: str = "") -> None:
        consumers = self.consumers
        print(f'CONSUMERS FOR {key or "ALL"}')
        if not key:
            pprint(dict(consumers))
        else:
            pprint({k: v for k, v in consumers.items() if k in {"*", key}})

    @property
    def data_keys(self) -> list[str]:
        return [p.stem for p in (self.path / "data").glob("*.pickle")]

    def read_data(self, key: str = "*") -> object:
        if key == "*":
            return {k: self.read_data(key=k) for k in self.data_keys}

        with Path(self.path / "data" / f"{key}.pickle").open("rb") as f:
            return dill.load(f)

    def write_data(self, key: str, data: object) -> None:
        with Path(self.path / "data" / f"{key}.pickle").open("wb") as f:
            return dill.dump(data, f)


@with_workspace
def publish_data(key: str, data: Incomplete, workspace: WORKSPACETYPE) -> None:
    """Publish/write data to a DataProvider."""
    provider = DataProvider.from_workspace(workspace)
    provider.publish(key, data)


@with_workspace
def read_data(key: str = "*", workspace: WORKSPACETYPE | None = None) -> Any:
    """Read/consume a summary of the available data from a DataProvider.

    Differs from consume_data in that it does not set up a dependency.
    """
    provider = DataProvider.from_workspace(workspace)
    return provider.consume(key, subscribe=False)


@with_workspace
def summarize_data(key: str = "", workspace: WORKSPACETYPE | None = None) -> None:
    """Give a summary of the available data from a DataProvider."""
    provider = DataProvider.from_workspace(workspace)
    provider.summarize_clients(key=key)


@with_workspace
def consume_data(key: str = "*", workspace: WORKSPACETYPE | None = None) -> object:
    """Read/consume data from a DataProvider in a given workspace."""
    provider = DataProvider.from_workspace(workspace)
    return provider.consume(key, subscribe=True)
