from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


SEED_COLUMNS = ("training seed", "training_seed", "seed")
IDENTITY_COLUMNS = {"ID", "model", "dataset", *SEED_COLUMNS}


@dataclass(frozen=True)
class CurveData:
    losses: dict[int, float]
    seed: str | None


def read_curves(
    csv_path: str | Path,
    *, allow_zero: bool = False,
) -> dict[tuple[str, str], CurveData]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        if not {"model", "dataset"}.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain model and dataset columns")

        data_sizes = read_available_data_sizes(path)
        values: dict[tuple[str, str], dict[int, float]] = {}
        seed_labels = {}
        for row_number, row in enumerate(reader, start=2):
            model = (row.get("model") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            if not model or not dataset:
                raise ValueError(
                    f"Missing model or dataset in {path} at row {row_number}"
                )
            key = (model, dataset)
            if key in values:
                raise ValueError(f"Duplicate curve {key}; average training seeds before fitting")
            values[key] = {}
            labels = {(row.get(name) or "").strip() for name in SEED_COLUMNS}
            labels.discard("")
            if len(labels) > 1:
                raise ValueError(f"Conflicting training seed labels at row {row_number}")
            seed_labels[key] = next(iter(labels), None)
            for data_size in data_sizes:
                raw_loss = (row.get(str(data_size)) or "").strip()
                if not raw_loss:
                    continue
                try:
                    loss = float(raw_loss)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid loss at row {row_number}, D={data_size}: "
                        f"{raw_loss!r}"
                    ) from exc
                if not (_is_finite_positive(loss) or (allow_zero and loss == 0.0)):
                    raise ValueError(
                        f"Loss must be finite and {'nonnegative' if allow_zero else 'positive'} at row {row_number}, "
                        f"D={data_size}, got {loss!r}"
                    )
                values[key][data_size] = loss

    if not values:
        raise ValueError(f"No curves found in {path}")

    curves: dict[tuple[str, str], CurveData] = {}
    for key, losses_by_size in sorted(values.items()):
        curve = dict(sorted(losses_by_size.items()))
        if not curve:
            raise ValueError(f"Curve {key} has no loss values")
        curves[key] = CurveData(losses=curve, seed=seed_labels[key])
    return curves


def read_available_data_sizes(csv_path: str | Path) -> tuple[int, ...]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {path}") from exc

    sizes: list[int] = []
    unknown_columns: list[str] = []
    for column in header:
        if column in IDENTITY_COLUMNS:
            continue
        try:
            size = int(column)
        except ValueError:
            unknown_columns.append(column)
            continue
        if size < 0:
            raise ValueError(f"D must be non-negative, got column {column!r}")
        sizes.append(size)

    if unknown_columns:
        names = ", ".join(repr(name) for name in unknown_columns)
        raise ValueError(f"Unrecognized non-D columns in {path}: {names}")
    if not sizes:
        raise ValueError(f"No D columns found in {path}")
    if len(sizes) != len(set(sizes)):
        raise ValueError(f"Duplicate D columns found in {path}")

    return tuple(sorted(sizes))


def _is_finite_positive(value: float) -> bool:
    return value > 0.0 and value != float("inf") and value != float("-inf")
