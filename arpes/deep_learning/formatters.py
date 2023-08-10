"""Provides plotting formatters for different kinds of data and targets."""
from __future__ import annotations

from typing import Any

__all__ = [
    "SpectrumFormatter",
    "FloatTitleFormatter",
]


class SpectrumFormatter:
    """Knows how to plot an ARPES spectrum onto an interpretation plot."""

    def show(self, data, ax=None):
        """Just imshow the data for now with no other decoration."""
        spectrum, row = data
        ax.imshow(spectrum, origin="lower")


class FloatTitleFormatter:
    """Plots a floating point target as a title annotation onto a plot for its parent item."""

    context: dict[str, Any] = None
    title_formatter: str = r"{label}={data:.3f}"

    @property
    def computed_context(self) -> dict[str, Any]:
        """Annotate whether this is a ground truth or predicted value."""
        return {"label": "True" if self.context.get("is_ground_truth", False) else "Pred"}

    def show(self, data, ax=None):
        """Sets the title for the parent data axis to be the formatted float value."""
        title = ax.get_title()
        context = {
            **self.computed_context,
            "data": data,
        }

        ax.set_title(
            f"{title}; {self.title_formatter.format(**context)}",
        )
