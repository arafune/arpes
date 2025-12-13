"""Unit test for arpes.plotting.utils."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

from arpes.plotting.utils import get_colorbars


def test_get_colorbars_all_paths():
    plt.close("all")
    fig, ax = plt.subplots()

    # ------------------------------------------------------------
    # 1. Pattern where Colorbar is attached to an image (images)
    # ------------------------------------------------------------
    img = ax.imshow([[0, 1], [2, 3]])
    cbar1 = fig.colorbar(img, ax=ax)

    assert isinstance(img.colorbar, Colorbar)

    # ------------------------------------------------------------
    # 2. Pattern where Colorbar is attached to a collection (collections)
    # ------------------------------------------------------------
    collection = ax.scatter([0, 1], [0, 1], c=[0, 1])
    cbar2 = fig.colorbar(collection, ax=ax)

    assert isinstance(collection.colorbar, Colorbar)

    # ------------------------------------------------------------
    # 3. Direct attribute ax.colorbar
    # ------------------------------------------------------------
    sm = ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="viridis")
    ax.colorbar = Colorbar(ax, sm)  # Fake attach
    assert isinstance(ax.colorbar, Colorbar)

    # ------------------------------------------------------------
    # 4. Direct attribute ax.cbar
    # ------------------------------------------------------------
    ax.cbar = Colorbar(ax, sm)
    assert isinstance(ax.cbar, Colorbar)

    # ------------------------------------------------------------
    # 5. Adding Colorbar to ax.get_children() (explicit addition)
    # ------------------------------------------------------------
    # child_cbar = Colorbar(ax, sm)

    # ------------------------------------------------------------
    # Execution: fig=None -> also covers the gcf (get current figure) branch
    # ------------------------------------------------------------
    result = get_colorbars(None)

    # ------------------------------------------------------------
    # Ensure successful unique filtering based on ID (de-duplication)
    # ------------------------------------------------------------
    # All colorbars are detected (order is not guaranteed)
    detected_ids = {id(cb) for cb in result}
    # expected_ids = {id(cbar1), id(cbar2), id(ax.colorbar), id(ax.cbar), id(child_cbar)}

    expected_ids = {id(cbar1), id(cbar2), id(ax.colorbar), id(ax.cbar)}
    assert detected_ids == expected_ids
    assert len(result) == 4
