"""Utilities related to interpretation of model results.

This borrows ideas heavily from fastai which provides interpreter classes
for different kinds of models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import starmap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset

from arpes.utilities.jupyter import get_tqdm

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from _typeshed import Incomplete
    from matplotlib.axes import Axes
    from torch.utils.data import DataLoader
__all__ = [
    "Interpretation",
    "InterpretationItem",
]

tqdm = get_tqdm()


@dataclass
class InterpretationItem:
    """Provides tools to introspect model performance on a single item."""

    target: Any
    predicted_target: Any
    loss: float
    index: int
    parent_dataloader: DataLoader

    @property
    def dataset(self) -> Dataset:
        """Fetches the original dataset used to train and containing this item.

        We need to unwrap the dataset in case we are actually dealing
        with a Subset. We should obtain an indexed Dataset at the end
        of the day, and we will know this is the case because we use
        the sentinel attribute `is_indexed` to mark this.

        This may fail sometimes, but this is better than returning junk
        data which is what happens if we get a shuffled view over the
        dataset.
        """
        dset = self.parent_dataloader.dataset
        if isinstance(dset, Subset):
            dset = dset.dataset

        assert dset.is_indexed is True
        return dset

    def show(
        self,
        input_formatter: Incomplete,
        target_formatter: Incomplete,
        ax: Axes | None = None,
        *,
        pullback: bool = True,
    ) -> None:
        """Plots item onto the provided axes. See also the `show` method of `Interpretation`."""
        if ax is None:
            _, ax = plt.subplots()

        dset = self.dataset
        with dset.no_transforms():
            x = dset[self.index][0]

        if input_formatter is not None:
            input_formatter.show(x, ax)

        ax.set_title(
            f"Item {self.index}; loss={float(self.loss):.3f}\n",
        )

        if target_formatter is not None:
            if hasattr(target_formatter, "context"):
                target_formatter.context = {"is_ground_truth": True}

            target = self.decodes_target(self.target) if pullback else self.target
            target_formatter.show(target, ax)

            if hasattr(target_formatter, "context"):
                target_formatter.context = {"is_ground_truth": False}

            predicted = (
                self.decodes_target(self.predicted_target) if pullback else self.predicted_target
            )
            target_formatter.show(predicted, ax)

    def decodes_target(self, value: Incomplete) -> Incomplete:
        """Pulls the predicted target backwards through the transformation stack.

        Pullback continues until an irreversible transform is met in order
        to be able to plot targets and predictions in a natural space.
        """
        tfm = self.dataset.transforms
        if hasattr(tfm, "decodes_target"):
            return tfm.decodes_target(value)

        return value


@dataclass
class Interpretation:
    """Provides utilities to interpret predictions of a model.

    Importantly, this is not intended to provide any model introspection
    tools.
    """

    model: pl.LightningModule
    train_dataloader: DataLoader
    val_dataloaders: DataLoader

    train: bool = True
    val_index: int = 0

    train_items: list[InterpretationItem] = field(init=False, repr=False)
    val_item_lists: list[list[InterpretationItem]] = field(init=False, repr=False)

    @property
    def items(self) -> list[InterpretationItem]:
        """All of the ``InterpretationItem`` instances inside this instance."""
        if self.train:
            return self.train_items

        return self.val_item_lists[self.val_index]

    def top_losses(self, *, ascending: bool = False) -> list[InterpretationItem]:
        """Orders the items by loss."""

        def key(item: Incomplete) -> Incomplete:
            return item.loss if ascending else -item.loss

        return sorted(self.items, key=key)

    def show(
        self,
        n_items: int | tuple[int, int] = 9,
        items: list[InterpretationItem] | None = None,
        input_formatter: Incomplete = None,
        target_formatter: Incomplete = None,
    ) -> None:
        """Plots a subset of the interpreted items.

        For each item, we "plot" its data, its label, and model performance characteristics
        on this item.

        For example, on an image classification task this might mean to plot the image,
        the images class name as a label above it, the predicted class, and the numerical loss.
        """
        layout = None

        if items is None:
            if isinstance(n_items, tuple):
                layout = n_items
            else:
                n_rows = int(math.ceil(n_items**0.5))
                layout = (n_rows, n_rows)

            items = self.top_losses()[:n_items]
        else:
            n_items = len(items)
            n_rows = int(math.ceil(n_items**0.5))
            layout = (n_rows, n_rows)

        assert isinstance(n_items, int)
        _, axes = plt.subplots(*layout, figsize=(layout[0] * 3, layout[1] * 4))

        items_with_nones = list(items) + [None] * (np.prod(layout) - n_items)
        for item, ax in zip(items_with_nones, axes.ravel(), strict=True):
            if item is None:
                ax.axis("off")
            else:
                item.show(input_formatter, target_formatter, ax)

        plt.tight_layout()

    @classmethod
    def from_trainer(cls: type[Incomplete], trainer: pl.Trainer) -> list[InterpretationItem]:
        """Builds an interpreter from an instance of a `pytorch_lightning.Trainer`."""
        return cls(trainer.model, trainer.train_dataloader, trainer.val_dataloaders)

    def dataloader_to_item_list(self, dataloader: DataLoader) -> list[InterpretationItem]:
        """Converts data loader into a list of interpretation items corresponding to the data."""
        items = []

        for batch in tqdm(dataloader.iter_all()):
            x, y, indices = batch
            with torch.no_grad():
                y_hat = self.model(x).cpu()
                y_hats = torch.unbind(y_hat, axis=0)
                ys = torch.unbind(y, axis=0)

                losses = list(starmap(self.model.criterion, zip(y_hats, ys, strict=True)))

            for yi, yi_hat, loss, index in zip(
                ys,
                y_hats,
                losses,
                torch.unbind(indices, axis=0),
                strict=True,
            ):
                items.append(
                    InterpretationItem(
                        torch.squeeze(yi),
                        torch.squeeze(yi_hat),
                        int(index),
                        torch.squeeze(loss),
                        dataloader,
                    ),
                )

        return items

    def __post_init__(self) -> None:
        """Populates train_items and val_item_lists.

        This is done by iterating through the dataloaders and pushing data through the models.
        """
        self.train_items = self.dataloader_to_item_list(self.train_dataloader)
        self.val_item_lists = [self.dataloader_to_item_list(dl) for dl in self.val_dataloaders]
