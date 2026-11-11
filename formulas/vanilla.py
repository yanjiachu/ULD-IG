from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


FORMULA = "(B / D**beta + E)**alpha"


PARAMETER_NAMES = ("B", "E", "alpha", "beta")


def _as_valid_1d_array(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _log_prediction_from_theta(log_d: np.ndarray, theta: np.ndarray) -> np.ndarray:
    log_b, log_e, log_alpha, log_beta = theta
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    return alpha * np.logaddexp(log_b - beta * log_d, log_e)


def predict_log_vanilla(
    data_sizes: Sequence[float], parameters: Mapping[str, float]
) -> np.ndarray:
    d = _as_valid_1d_array(data_sizes, "D")
    if np.any(d <= 0):
        raise ValueError("Vanilla prediction requires every D to be positive")

    missing = [name for name in PARAMETER_NAMES if name not in parameters]
    if missing:
        raise ValueError(f"Missing vanilla parameters: {', '.join(missing)}")
    values = np.asarray([parameters[name] for name in PARAMETER_NAMES], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All vanilla parameters must be finite and positive")

    return _log_prediction_from_theta(np.log(d), np.log(values))


def predict_vanilla(
    data_sizes: Sequence[float], parameters: Mapping[str, float]
) -> np.ndarray:
    log_prediction = predict_log_vanilla(data_sizes, parameters)
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.exp(log_prediction)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Vanilla prediction is not finite for the requested D")
    return prediction


predict = predict_vanilla
predict_log = predict_log_vanilla
