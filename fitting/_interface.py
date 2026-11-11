from dataclasses import dataclass
from typing import Callable

from ._anchors import anchor_policy, resolve_weight


@dataclass(frozen=True)
class FittingMethod:
    fit: Callable
    resolve_config: Callable
    free_parameters: int
    requires_l0: bool = False
    uses_anchor_weight: bool = False

    @property
    def min_points(self):
        return self.free_parameters

    anchor_policy = staticmethod(anchor_policy)
    resolve_weight = staticmethod(resolve_weight)
