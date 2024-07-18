"""A utility excepthook for Qt applications which ensures errors are visible in Jupyter."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from types import FrameType, TracebackType

__all__ = ("patched_excepthook",)


def patched_excepthook(
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    exc_tb: TracebackType | FakeTB | None,
) -> None:
    """Prints the traceback instead of dying silently. Useful for debugging Qt apps in Jupyter."""
    enriched_tb = _add_missing_frames(exc_tb) if exc_tb else exc_tb
    traceback.print_exception(exc_type, exc_value, enriched_tb)


def _add_missing_frames(tb: TracebackType | FakeTB) -> FakeTB:
    result = FakeTB(tb.tb_frame, tb.tb_lasti, tb.tb_lineno, tb.tb_next)
    frame = tb.tb_frame.f_back
    while frame:
        result = FakeTB(frame, frame.f_lasti, frame.f_lineno, result)
        frame = frame.f_back
    return result


class FakeTB(NamedTuple):
    tb_frame: FrameType
    tb_lasti: int
    tb_lineno: int
    tb_next: TracebackType | FakeTB | None
