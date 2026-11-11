from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable


HALF_THEN_HALVE = "half_then_halve"


@dataclass(frozen=True)
class BudgetSelection:

    strategy: str
    budget_ratio: Decimal
    total_data_size: int
    budget: Decimal
    selected_data_sizes: tuple[int, ...]
    total_used_data: int

def _parse_budget(value: int | str | Decimal, total_data_size: int) -> int:
    try:
        budget = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid budget: {value!r}") from exc
    if (isinstance(value, bool) or not budget.is_finite()
            or budget != budget.to_integral_value()
            or budget <= 0 or budget > total_data_size):
        raise ValueError(
            f"budget must be a positive integer no larger than Dmax={total_data_size}, "
            f"got {value!r}"
        )
    return int(budget)


def select_half_then_halve_budget(
    available_data_sizes: Iterable[int],
    budget: int | str | Decimal,
) -> BudgetSelection:
    sizes = tuple(sorted(set(int(size) for size in available_data_sizes)))
    if not sizes or sizes[0] < 0:
        raise ValueError("available_data_sizes must contain nonnegative D values")
    positive_sizes = tuple(size for size in sizes if size > 0)
    if not positive_sizes:
        raise ValueError("At least one positive D is required")

    total_data_size = positive_sizes[-1]
    budget_int = _parse_budget(budget, total_data_size)
    if budget_int % 2:
        raise ValueError(f"half of budget={budget_int} is not an integer D")
    first_size = budget_int // 2
    available = set(positive_sizes)
    if first_size not in available:
        raise ValueError(
            f"The half-budget starting point is not present in the input data: D={first_size}"
        )

    selected: list[int] = []
    current = first_size
    while current in available:
        selected.append(current)
        if current % 2:
            break
        current //= 2

    total_used_data = sum(selected)
    if total_used_data > budget_int:
        raise RuntimeError("Budget selection exceeded its limit")
    return BudgetSelection(
        strategy=HALF_THEN_HALVE,
        budget_ratio=Decimal(budget_int) / Decimal(total_data_size),
        total_data_size=total_data_size,
        budget=Decimal(budget_int),
        selected_data_sizes=tuple(selected),
        total_used_data=total_used_data,
    )
