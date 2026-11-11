from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .vanilla import predict_log_vanilla


FORMULA = "B / D**beta + E"


PARAMETER_NAMES = ("B", "E", "beta")


def predict_log_power(data_sizes: Sequence[float], parameters: Mapping[str, float]):
    return predict_log_vanilla(data_sizes, {**parameters, 'alpha': 1.0})


def predict_power(data_sizes: Sequence[float], parameters: Mapping[str, float]):
    with np.errstate(over='ignore', invalid='ignore'):
        prediction = np.exp(predict_log_power(data_sizes, parameters))
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Power prediction is not finite for the requested D")
    return prediction


predict = predict_power
predict_log = predict_log_power
