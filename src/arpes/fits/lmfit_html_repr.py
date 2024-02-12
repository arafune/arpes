"""For now we monkeypatch lmfit to make it easier to work with in Jupyter.

We should consider forking or providing a pull at a later date after this settles down.

The end goal here is to allow pleasing and functional representations of curve fitting sessions
performed in Jupyter, so that they can be rapidly understood, and screencapped for simple purposes,
like including in group meeting notes.

Todo: To be deprecated (?)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# pylint: disable=protected-access
import numpy as np
from lmfit import model

if TYPE_CHECKING:
    from _typeshed import Incomplete


def repr_multiline_ModelResult(self: model.ModelResult, **kwargs: Incomplete) -> str:
    """Provides a text-based multiline representation used in Qt based interactive tools.

    Args:
        self: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]
    """
    template = "ModelResult\n  Converged: {success}\n  "
    template += "Components:\n {formatted_components}\n  Parameters:\n{parameters}"

    return template.format(
        success=self.success,
        formatted_components="\n".join(
            [(" " * 4) + c._repr_multiline_text_() for c in self.components],
        ),
        parameters="\n".join(
            f"    {l_item}" for l_item in self.params._repr_multiline_text_(**kwargs).split("\n")
        ),
    )


def repr_html_ModelResult(self: model.ModelResult, **kwargs: Incomplete) -> str:
    """Provides a better Jupyter representation of an `lmfit.ModelResult` instance."""
    template = """
        <div>
            <span><strong>Converged: </strong>{success}</span>
            <div>{formatted_components}<div>
            <div>{parameters}</div>
        </div>
        """
    return template.format(
        success=self.success,
        formatted_components="".join(f"<div>{c._repr_html_()}</div>" for c in self.components),
        parameters=self.params._repr_html_(**kwargs),
    )


def repr_html_Model(self: model.Model) -> str:
    """Better Jupyter representation of `lmfit.Model` instances."""
    template = """
    <div>
    <strong>{name}</strong>
    </div>
    """
    return template.format(name=self.name)


def repr_multiline_Model(self: model.Model) -> str:
    """Provides a text-based multiline representation used in Qt based interactive tools."""
    return self.name


ALL_PARAMETER_ATTRIBUTES = ["name", "value", "min", "max", "stderr", "vary", "expr", "brute_step"]
SKIP_ON_SHORT = {"min", "max", "vary", "expr", "brute_step"}


def repr_html_Parameters(
    self: model.Parameters,
    *,
    short: bool = False,
) -> str:
    """HTML representation for `lmfit.Parameters` instances."""
    keys = sorted(self.keys())
    template = """
    <table>
      <thead>
          <tr>
            {cols}
          </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """
    return template.format(
        cols="".join(
            f"<th>{c}</th>" for c in ALL_PARAMETER_ATTRIBUTES if not short or c not in SKIP_ON_SHORT
        ),
        rows="".join(self[p].to_table_row(short=short) for p in keys),
    )


def repr_multiline_Parameters(
    self: model.Parameters,
    *,
    short: bool = False,
) -> str:
    """Provides a text-based multiline representation used in Qt based interactive tools."""
    return "\n".join(self[k]._repr_multiline_text_(short=short) for k in self)


def repr_html_Parameter(
    self: model.Parameter,
    *,
    short: bool = False,
) -> str:
    """HTML representation for `lmfit.Parameter` instances."""
    if short:
        return f"""
            <tr>
                <th>{self.name}</th>
                <th>{self.value:.3f}</th>
                <th>{self.stderr:.3f}</th>
            </tr>
            """

    template = """
            <tr>
                <th>{name}</th>
                <th>{value:.3f}</th>
                <th>{min:.3f}</th>
                <th>{max:.3f}</th>
                <th>{stderr:.3f}</th>
                <th>{vary}</th>
                <th>{expr}</th>
                <th>{brute_step}</th>
            </tr>
            """
    return template.format(
        name=self.name,
        value=self.value,
        min=self.min,
        max=self.max,
        stderr=self.stderr or np.inf,
        vary=self.vary,
        expr=self.expr or "",
        brute_step=self.brute_step or "",
    )


def repr_multiline_Parameter(
    self: model.Parameter,
    *,
    short: bool = False,
) -> str:
    """Provides a text-based multiline representation used in Qt based interactive tools."""
    template = "{name}:\n{contents}"

    get_attrs = [a for a in ALL_PARAMETER_ATTRIBUTES if not short or a not in SKIP_ON_SHORT]

    def format_attr(value: float) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"

        return str(value)

    return template.format(
        name=self.name,
        contents="\n".join(
            f"  {attr_name}: {format_attr(getattr(self, attr_name))}"
            for attr_name in get_attrs
            if attr_name != "name"
        ),
    )


model.Model._repr_html_ = repr_html_Model
model.Model._repr_multiline_text_ = repr_multiline_Model  # unused ?
model.Parameters._repr_html_ = repr_html_Parameters
model.Parameters._repr_multiline_text_ = repr_multiline_Parameters  # unused ?
model.ModelResult._repr_html_ = repr_html_ModelResult
model.ModelResult._repr_multiline_text_ = repr_multiline_ModelResult  # unused?

model.Parameter.to_table_row = repr_html_Parameter
model.Parameter._repr_multiline_text_ = repr_multiline_Parameter

# we don't export anything, just monkey-patch
__all__ = ()
