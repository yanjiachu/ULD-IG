from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


FORMULA = "B / (D_l + D**beta) + E"


PARAMETER_NAMES = ("B", "D_l", "beta", "E")


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
    log_c, b_s, log_beta, log_e = theta
    beta = np.exp(log_beta)
    log_denominator = np.logaddexp(0.0, b_s + beta * log_d)
    return np.logaddexp(log_c - log_denominator, log_e)


def predict_log_rectified(
    data_sizes: Sequence[float], parameters: Mapping[str, float]
) -> np.ndarray:
    d = _as_valid_1d_array(data_sizes, "D")
    if np.any(d <= 0):
        raise ValueError("Rectified prediction requires every D to be positive")

    missing = [name for name in PARAMETER_NAMES if name not in parameters]
    if missing:
        raise ValueError(f"Missing rectified parameters: {', '.join(missing)}")
    values = np.asarray([parameters[name] for name in PARAMETER_NAMES], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All rectified parameters must be finite and positive")

    log_b, log_d_l, log_beta, log_e = np.log(values)
    return np.logaddexp(
        log_b - np.logaddexp(log_d_l, np.exp(log_beta) * np.log(d)), log_e
    )


def predict_rectified(
    data_sizes: Sequence[float], parameters: Mapping[str, float]
) -> np.ndarray:
    log_prediction = predict_log_rectified(data_sizes, parameters)
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.exp(log_prediction)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Rectified prediction is not finite for the requested D")
    return prediction


predict = predict_rectified
predict_log = predict_log_rectified
