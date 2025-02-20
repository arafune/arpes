"""Contains utility functions for decorating matplotlib look."""

from typing import Literal, Unpack

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import colorConverter
from matplotlib.image import AxesImage
from matplotlib.typing import ColorType

from arpes._typing import IMshowParam

__all__ = ("h_gradient_fill", "v_gradient_fill")


class GradientFillParam(
    IMshowParam,
    total=False,
):
    step: Literal["pre", "mid", "post"] | None


def h_gradient_fill(
    x1: float,
    x2: float,
    x_solid: float | None,
    fill_color: ColorType = "red",
    ax: Axes | None = None,
    **kwargs: Unpack[GradientFillParam],
) -> AxesImage:  # <== checkme!
    """Fills a gradient between x1 and x2.

    If x_solid is not None, the gradient will be extended
    at the maximum opacity from the closer limit towards x_solid.

    Args:
        x1(float): lower side of x
        x2(float): height side of x
        x_solid: If x_solid is not None, the gradient will be extended at the maximum opacity from
                 the closer limit towards x_solid.
        fill_color (str): Color name, pass it as "c" in mpl.colors.to_rgb
        ax(Axes): Axes on which to plot.
        **kwargs: Pass to imshow  (Z order can be set here.)

    Returns:
        The result of the inner imshow.

    Todo:
        Stop using imshow, move to plotting.decoration
    """
    if ax is None:
        ax = plt.gca()
    assert isinstance(ax, Axes)

    alpha = float(kwargs.get("alpha", 1.0))
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("origin", "lower")
    step = kwargs.pop("step", None)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert fill_color

    z = np.empty((1, 100, 4), dtype=float)

    rgb = colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[None, :]
    assert x1 < x2
    xmin, xmax, (ymin, ymax) = x1, x2, ylim
    kwargs.setdefault("extent", (xmin, xmax, ymin, ymax))

    im: AxesImage = ax.imshow(
        z,
        **kwargs,
    )

    if x_solid is not None:
        xlow, xhigh = (x2, x_solid) if x_solid > x2 else (x_solid, x1)
        ax.fill_betweenx(
            ylim,
            xlow,
            xhigh,
            color=fill_color,
            alpha=alpha,
            step=step,
        )

    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    return im


def v_gradient_fill(
    y1: float,
    y2: float,
    y_solid: float | None,
    fill_color: ColorType = "red",
    ax: Axes | None = None,
    **kwargs: Unpack[GradientFillParam],
) -> AxesImage:
    """Fills a gradient vertically between y1 and y2.

    If y_solid is not None, the gradient will be extended
    at the maximum opacity from the closer limit towards y_solid.

    Args:
        y1(float): Lower side for limit to fill.
        y2(float): Higher side for to fill.
        y_solid (float|solid): If y_solid is not None, the gradient will be extended at the maximum
            opacity from the closer limit towards y_solid.
        fill_color (str): Color name, pass it as "c" in mpl.colors.to_rgb  (Default "red")
        ax(Axes): matplotlib Axes object
        **kwargs: (str|float): pass to ax.imshow

    Returns:
        The result of the inner imshow call.

    Todo: Stop using imshow, move to decoration
    """
    if ax is None:
        ax = plt.gca()

    alpha = float(kwargs.get("alpha", 1.0))
    assert isinstance(ax, Axes)
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("origin", "lower")
    step = kwargs.pop("step", None)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert fill_color

    z = np.empty((100, 1, 4), dtype=float)

    rgb = colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]
    assert y1 < y2
    (xmin, xmax), ymin, ymax = xlim, y1, y2
    kwargs.setdefault("extent", (xmin, xmax, ymin, ymax))
    im: AxesImage = ax.imshow(
        z,
        **kwargs,
    )

    if y_solid is not None:
        ylow, yhigh = (y2, y_solid) if y_solid > y2 else (y_solid, y1)
        ax.fill_between(
            x=xlim,
            y1=ylow,
            y2=yhigh,
            color=fill_color,
            alpha=alpha,
            step=step,
        )

    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    return im
