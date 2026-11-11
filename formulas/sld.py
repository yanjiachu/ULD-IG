from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


FORMULA = "B + A / (D + D0)**alpha"


NAMES = ("A", "alpha", "D0", "B")


def predict_log_sld(data_sizes, parameters):
    d = np.asarray(data_sizes, dtype=float)
    if d.ndim != 1 or not d.size or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("SLD prediction requires finite nonnegative 1D D")
    p = np.array([parameters[n] for n in NAMES], dtype=float)
    if not np.all(np.isfinite(p)) or np.any(p[:3] <= 0) or p[3] < 0:
        raise ValueError("SLD requires A, alpha, D0 > 0 and B >= 0")
    with np.errstate(divide='ignore'):
        log_shift = np.logaddexp(np.log(d), np.log(p[2]))
    return np.logaddexp(np.log(p[3]) if p[3] else -np.inf,
                        np.log(p[0])-p[1]*log_shift)


def predict_sld(data_sizes, parameters):
    with np.errstate(over='ignore', under='ignore'):
        values = np.exp(predict_log_sld(data_sizes, parameters))
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("SLD prediction outside representable positive loss range")
    return values


def _paper_prediction(theta, d):
    log_a, log_alpha, log_d0, b = theta
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        return b + np.exp(log_a)*np.power(d+np.exp(log_d0), -np.exp(log_alpha))


predict = predict_sld
predict_log = predict_log_sld
