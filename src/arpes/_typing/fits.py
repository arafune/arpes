from typing import Literal, TypedDict

from lmfit import Parameter

NAN_POLICY = Literal["raise", "propagate", "omit"]


class ModelArgs(TypedDict, total=False):
    """KWargs for lf.Model."""

    independent_vars: list[str]
    param_names: list[str]
    nan_policy: NAN_POLICY
    prefix: str
    name: str
    form: str


class ParametersArgs(TypedDict, total=False):
    """Class for arguments for Parameters."""

    name: str | Parameter
    value: float  # initial value
    vary: bool  # Whether the parameter is varied during the fit
    min: float  # Lower bound for value (default, -np.inf)
    max: float  # Upper bound for value (default np.inf)
    expr: str  # Mathematical expression to constrain the value.
    brute_step: float  # step size for grid points in the brute method.
