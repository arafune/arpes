"""Implements transform pipelines for pytorch_lightning with basic inverse transform."""

from __future__ import annotations

from dataclasses import Field, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import Incomplete

__all__ = ["ComposeBoth", "ReversibleLambda", "Identity"]


class Identity:
    """Represents a reversible identity transform."""

    def encodes(self, x: Incomplete) -> Incomplete:
        return x

    def __call__(self, x: Incomplete) -> Incomplete:
        return x

    def decodes(self, x: Incomplete) -> Incomplete:
        return x

    def __repr__(self) -> str:
        return "Identity()"


_identity = Identity()


@dataclass
class ReversibleLambda:
    """A reversible anonymous function, so long as the caller supplies an inverse."""

    encodes: Field = field(repr=False)
    decodes: Field = field(default=lambda x: x, repr=False)

    def __call__(self, value: Incomplete) -> Field[Incomplete]:
        """Apply the inner lambda to the data in forward pass."""
        return self.encodes(value)


@dataclass
class ComposeBoth:
    """Like `torchvision.transforms.Compose` but it operates on data & target in each transform."""

    transforms: list[Any]

    def __post_init__(self) -> None:
        """Replace missing transforms with identities."""
        safe_transforms = []
        for t in self.transforms:
            if isinstance(t, tuple | list):
                xt, yt = t
                t = [xt or _identity, yt or _identity]

            safe_transforms.append(t)

        self.original_transforms = self.transforms
        self.transforms = safe_transforms

    def __call__(self, x: Incomplete, y: Incomplete) -> Incomplete:
        """If this transform has separate data and target functions, apply separately.

        Otherwise, we apply the single transform to both the data and the target.
        """
        for t in self.transforms:
            if isinstance(t, list | tuple):
                xt, yt = t
                x, y = xt(x), yt(y)
            else:
                x, y = t(x, y)

        return x, y

    def decodes_target(self, y: Incomplete) -> Incomplete:
        """Pull the target back in the transform stack as far as possible.

        This is necessary only for the predicted target because
        otherwise we can *always* push the ground truth target and input
        forward in the transform stack.

        This is imperfect because for some transforms we need X and Y
        in order to process the data.
        """
        for t in self.transforms[::-1]:
            if isinstance(t, list | tuple):
                _, yt = t

                y = yt.decodes(y)
            else:
                break

        return y

    def __repr__(self) -> str:
        """Show both of the constitutent parts of this transform."""
        return (
            self.__class__.__name__
            + "(\n\t"
            + "\n\t".join([str(t) for t in self.original_transforms])
            + "\n)"
        )
