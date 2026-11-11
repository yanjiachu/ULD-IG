import math
from types import FunctionType

import numpy as np


def validate_anchor(l0, anchor_weight):
    if isinstance(anchor_weight, (bool, str)) or not np.isscalar(anchor_weight):
        raise ValueError('anchor_weight must be a finite nonnegative number')
    weight = float(anchor_weight)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('anchor_weight must be a finite nonnegative number')
    if l0 is not None:
        if isinstance(l0, (bool, str)) or not np.isscalar(l0):
            raise ValueError('Observed L0 must be finite and positive')
        l0 = float(l0)
        if not np.isfinite(l0) or l0 <= 0:
            raise ValueError('Observed L0 must be finite and positive')
    if weight > 0 and l0 is None:
        raise ValueError('Positive anchor_weight requires observed L0')
    return l0, weight


def huber_value(residual, delta):
    absolute = abs(residual)
    return float(residual**2/2 if absolute <= delta else delta*(absolute-delta/2))


def power_weight(ratio, c, p):
    for value in (ratio, c, p):
        if (isinstance(value, (bool, str)) or not np.isscalar(value)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("ratio, c and p must be finite positive numbers")
    try:
        weight = float(c) * float(ratio) ** (-float(p))
    except OverflowError as exc:
        raise ValueError("Scheduled weight must be finite and positive") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Scheduled weight must be finite and positive")
    return weight


def anchor_policy(settings, fixed_weight=None):
    if fixed_weight is not None:
        policy = {"type": "fixed", "weight": fixed_weight}
    else:
        policy = settings.get("soft_anchor", {})
    if not isinstance(policy, dict):
        raise ValueError("soft_anchor must be a mapping")
    if policy.get("type") == "fixed" and set(policy) == {"type", "weight"}:
        weight = policy["weight"]
        if (isinstance(weight, (bool, str)) or not np.isscalar(weight)
                or not np.isfinite(weight) or weight < 0):
            raise ValueError("Fixed anchor weight must be finite and nonnegative")
        return {"type": "fixed", "weight": float(weight)}
    if policy.get("type") == "inverse_power" and set(policy) == {"type", "c", "p"}:
        power_weight(1.0, policy["c"], policy["p"])
        return dict(policy)
    raise ValueError("soft_anchor requires inverse_power (c, p) or fixed (weight)")


def resolve_weight(policy, ratio):
    if policy["type"] == "fixed":
        return policy["weight"]
    return power_weight(ratio, policy["c"], policy["p"])


def log_start_from_theta(method, theta):
    if method == 'rectified':
        return float(np.logaddexp(theta[0], theta[3]))
    if method == 'sld':
        log_amplitude = theta[0]-np.exp(theta[1])*theta[2]
        log_floor = np.log(theta[3]) if theta[3] > 0 else -np.inf
        return float(np.logaddexp(log_amplitude, log_floor))
    raise ValueError('Unknown baseline method')


def theta_from_parameters(method, parameters):
    p = parameters
    if method == 'rectified':
        return np.array([np.log(p['B'])-np.log(p['D_l']), -np.log(p['D_l']),
                         np.log(p['beta']), np.log(p['E'])])
    if method == 'sld':
        return np.array([np.log(p['A']), np.log(p['alpha']), np.log(p['D0']), p['B']])
    raise ValueError('Unknown baseline method')


def decomposition(method, parameters):
    p = parameters
    if method == 'rectified':
        floor = p['E']
        log_amplitude = np.log(p['B'])-np.log(p['D_l'])
        shape = '1 / (1 + D**beta / D_l)'
    elif method == 'sld':
        floor = p['B']
        log_amplitude = np.log(p['A'])-p['alpha']*np.log(p['D0'])
        shape = '(1 + D / D0)**(-alpha)'
    else:
        raise ValueError('Unknown baseline method')
    with np.errstate(over='ignore', under='ignore'):
        amplitude = float(np.exp(log_amplitude))
    return dict(E=float(floor), A=amplitude if np.isfinite(amplitude) else None,
                log_A=float(log_amplitude), s=shape, s_at_zero=1.)


def anchor_diagnostics(method, parameters, l0, weight, delta):
    l0, weight = validate_anchor(l0, weight)
    log_start = log_start_from_theta(method, theta_from_parameters(method, parameters))
    residual = None if l0 is None else float(log_start-np.log(l0))
    loss = None if residual is None else huber_value(residual, delta)
    with np.errstate(over='ignore', under='ignore'):
        estimated = float(np.exp(log_start))
    return dict(space='log_loss', penalty_type='huber', huber_delta=delta, weight=weight,
                observed_L0=l0, estimated_L0=estimated if np.isfinite(estimated) else None,
                log_estimated_L0=log_start, log_residual=residual, huber_loss=loss,
                penalty=0. if weight == 0 else float(weight*loss))


def augmented_objective(method, theta, original_args, *, original_objective, l0, weight, delta):
    value = original_objective(theta, *original_args)
    if weight == 0 or not np.isfinite(value):
        return value
    residual = log_start_from_theta(method, theta)-np.log(l0)
    return value+weight*huber_value(residual, delta)


def data_objective(method, parameters, data_sizes, losses, config, *, namespace):
    original_objective = namespace["_objective"]
    theta = theta_from_parameters(method, parameters)
    d, y = np.asarray(data_sizes, dtype=float), np.asarray(losses, dtype=float)
    if method == 'rectified':
        config_type = namespace['RectifiedConfig']
        resolved = config if isinstance(config, config_type) else config_type.from_mapping(config)
        return original_objective(theta, np.log(d), np.log(y), resolved.huber_delta)
    d, y = np.maximum(d, namespace["EPS"]), np.maximum(y, namespace["EPS"])
    return original_objective(theta, d, y, max(float(y.max()-y.min()), namespace["EPS"]))


def fit_soft(method, original, data_sizes, losses, config=None, *, l0, weight, delta=.01):
    if method not in ('rectified', 'sld'):
        raise ValueError('Unknown baseline method')
    l0, weight = validate_anchor(l0, weight)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Anchor Huber delta must be finite and positive')
    original_namespace = original.__globals__
    original_objective = original_namespace["_objective"]
    if weight == 0:
        result = original(data_sizes, losses, config)
    else:
        def objective(theta, *args):
            return augmented_objective(method, theta, args, original_objective=original_objective, l0=l0, weight=weight, delta=delta)
        namespace = {**original.__globals__, '_objective': objective}
        fit = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        fit.__kwdefaults__ = original.__kwdefaults__
        result = fit(data_sizes, losses, config)
    parameters = result['parameters']
    anchor = anchor_diagnostics(method, parameters, l0, weight, delta)
    data_loss = data_objective(method, parameters, data_sizes, losses, config, namespace=original_namespace)
    opt = result['optimization']
    np.testing.assert_allclose(opt['objective'], data_loss+anchor['penalty'], rtol=1e-8, atol=1e-11)
    opt.update(data_objective=float(data_loss), anchor_objective=anchor['penalty'],
               data_objective_type=opt.get('objective_type', 'log_huber_sum'),
               total_objective_type='original_data_loss_plus_weighted_log_huber_anchor')
    result.update(anchor=anchor, decomposition=decomposition(method, parameters))
    return result

