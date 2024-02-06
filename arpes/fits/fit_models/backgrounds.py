"""Definitions of common backgrounds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lmfit.models import update_param_vals

from .functional_forms import affine_bkg
from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    import lmfit as lf
    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import NAN_POLICY

__all__ = ["AffineBackgroundModel"]


class AffineBackgroundModel(XModelMixin):
    """A model for an affine background."""

    def __init__(
        self,
        independent_vars: list[str] | None = None,
        prefix: str = "",
        nan_policy: NAN_POLICY = "raise",
        **kwargs: Incomplete,
    ) -> None:
        """Defer to lmfit for initialization."""
        if independent_vars is None:
            independent_vars = ["x"]
        assert isinstance(independent_vars, list)
        kwargs.update(
            {"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars},
        )
        super().__init__(affine_bkg, **kwargs)

    def guess(self, data: xr.DataArray | NDArray[np.float_], **kwargs: float) -> lf.Parameters:
        """Use the tenth percentile value for the slope and a zero offset.

        Generally this should converge well regardless.
        """
        pars = self.make_params()

        pars["%slin_bkg" % self.prefix].set(value=np.percentile(data, 10))
        pars["%sconst_bkg" % self.prefix].set(value=0)

        return update_param_vals(pars, self.prefix, **kwargs)
