from ._interface import FittingMethod
from ._uld_ig_config import resolve_config, validate_anchor
from ._uld_ig_standard import fit_standard
from ._uld_ig_soft import fit as fit_soft


def default_config(config=None):
    return resolve_config(config, "5p")


def hard_config(config=None):
    return resolve_config(config, "4p")


def fit_default(data_sizes, losses, config=None):
    return fit_standard(data_sizes, losses, config, parameterization="5p")


def fit_hard(data_sizes, losses, config=None, *, l0):
    if l0 is None:
        raise ValueError("Hard fitting requires observed L0 (the D=0 loss)")
    l0, _ = validate_anchor(l0, 0.0)
    return fit_standard(data_sizes, losses, config, parameterization="4p", l0=l0)


FITTING_METHODS = {
    "default": FittingMethod(fit_default, default_config, 5),
    "hard": FittingMethod(fit_hard, hard_config, 4, requires_l0=True),
    "soft": FittingMethod(fit_soft, default_config, 5, requires_l0=True, uses_anchor_weight=True),
}
