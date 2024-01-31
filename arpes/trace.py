"""Provides lightweight perf and tracing tools which also provide light logging functionality."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete

__all__ = [
    "traceable",
]
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


@dataclass
class Trace:
    silent: bool = False
    start_time: float = field(default_factory=time.time_ns)

    def __call__(self, message: str) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            message: [TODO:description]

        Returns:
            [TODO:description]
        """
        if self.silent:
            return

        now = time.time_ns()
        elapsed = (now - self.start_time) // 1000000  # to ms
        message = f"{elapsed} ms: {message}"
        logger.info(message)


def traceable(original: Callable) -> Callable:
    """A decorator which takes a function and feeds a trace instance through its parameters.

    The call API of the returned function is that there is a `trace=` parameter which expects
    a bool (feature gate).

    Internally, this decorator turns that into a `Trace` instance and silences it if tracing is
    to be disabled (the user passed trace=False or did not pass trace= by keyword).

    Args:
        original: The function to decorate

    Returns:
        The decorated function which accepts a trace= keyword argument.
    """

    @functools.wraps(original)
    def _inner(*args: Incomplete, **kwargs: bool) -> Callable:
        """[TODO:summary].

        [TODO:description]

        Args:
            args: [TODO:description]
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        trace = kwargs.get("trace", False)

        # this allows us to pass Trace instances into function calls
        if not isinstance(trace, Trace):
            trace = Trace(silent=not trace)

        kwargs["trace"] = trace
        return original(*args, **kwargs)

    return _inner
