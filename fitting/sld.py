from collections.abc import Mapping


from formulas.sld import (
    FORMULA, NAMES, _paper_prediction, predict_log_sld, predict_sld
)

import numpy as np


from scipy.optimize import minimize


EPS = 1e-8


DEFAULT_CONFIG = dict(ftol=1e-9, gtol=1e-9)


OBJECTIVE_TYPE = 'normalized_linear_plus_log_mse'


def _decode(theta):
    return dict(zip(NAMES, [*np.exp(theta[:3]).tolist(), float(theta[3])], strict=True))


def _objective(theta, d, y, y_range):
    prediction = _paper_prediction(theta, d)
    if np.any(prediction <= 0) or not np.all(np.isfinite(prediction)):
        return float('inf')
    with np.errstate(over='ignore', invalid='ignore'):
        linear_residual = (prediction-y)/y_range
        log_residual = np.log(prediction+EPS)-np.log(y+EPS)
        return float(np.mean(linear_residual**2+log_residual**2))


def fit_sld(data_sizes, losses, config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping) or set(config)-set(cfg):
            raise ValueError("SLD paper fitter accepts only ftol and gtol")
        cfg.update(config)
    for name in ('ftol', 'gtol'):
        cfg[name] = float(cfg[name])
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"Invalid {name}")
    d, y_raw = np.asarray(data_sizes, dtype=float), np.asarray(losses, dtype=float)
    if (d.ndim != 1 or y_raw.shape != d.shape or len(d) < 4
            or not np.all(np.isfinite(d)) or not np.all(np.isfinite(y_raw))
            or np.any(d <= 0) or np.any(y_raw <= 0) or len(np.unique(d)) != len(d)):
        raise ValueError("SLD requires four or more distinct positive D and finite positive losses")
    d_fit, y = np.maximum(d, EPS), np.maximum(y_raw, EPS)
    y_min, y_max = float(y.min()), float(y.max())
    y_range = max(y_max-y_min, EPS)
    bounds = np.array([[-20,20], [-10,10], [-20,20], [0,y_min]], dtype=float)
    b0 = max(.9*y_min, EPS)
    d0 = max(.1*d_fit.min(), EPS)
    shifted_y = np.clip(y-b0, EPS, None)
    valid = (d_fit > 0) & (shifted_y > 0)
    if np.sum(valid) >= 2:
        slope, intercept = np.polyfit(np.log(d_fit[valid]+d0), np.log(shifted_y[valid]), 1)
        alpha0, a0 = max(-slope, EPS), max(np.exp(intercept), EPS)
    else:
        alpha0, a0 = .5, max(y_max-b0, EPS)
    p0 = np.array([np.log(a0), np.log(alpha0), np.log(d0), b0], dtype=float)
    result = minimize(_objective, p0, args=(d_fit,y,y_range),
                      method='L-BFGS-B', bounds=bounds, options=cfg)
    finite = bool(np.isfinite(result.fun) and np.all(np.isfinite(result.x)))
    fallback = not bool(result.success)
    theta = p0 if fallback else np.asarray(result.x, dtype=float)
    value = _objective(theta,d_fit,y,y_range)
    if not np.all(np.isfinite(theta)) or not np.isfinite(value):
        raise RuntimeError("SLD paper fit did not produce finite returned parameters and objective")
    message = str(result.message)
    warnings = []
    if fallback:
        warnings.append('Optimizer did not converge; returned the appendix initialization: '+message)
    boundary = [n for i,n in enumerate(NAMES) if np.any(np.isclose(theta[i],bounds[i],rtol=0,atol=1e-6))]
    if boundary:
        warnings.append('parameters at bounds: '+', '.join(boundary))
    return dict(formula=FORMULA,status='warning' if warnings else 'success',parameters=_decode(theta),
                optimization=dict(objective=value,objective_type=OBJECTIVE_TYPE,
                                  optimizer_success=bool(result.success),message=message,
                                  fallback_used=fallback,best_start_index=0,
                                  finite_starts=int(finite),successful_starts=int(result.success),
                                  iterations=int(getattr(result,'nit',0)),
                                  function_evaluations=int(getattr(result,'nfev',0)),
                                  warnings=warnings,parameters_at_bounds=boundary,
                                  runs=[dict(start_index=0,success=bool(result.success),
                                             objective=float(result.fun) if finite else None,
                                             message=message)],
                                  config={**cfg,'optimizer':'L-BFGS-B',
                                          'gradient':'numerical','num_starts':1,
                                          'scheme':'paper_appendix_b3_2','epsilon':EPS,
                                          'iteration_limit':'scipy_default',
                                          'theta_bounds':bounds.tolist()},
                                  y_range=y_range,
                                  training_clipping=dict(D=int(np.sum(d<EPS)),L=int(np.sum(y_raw<EPS))),
                                  theta_names=['ln_A','ln_alpha','ln_D0','B'],
                                  initialization=dict(strategy='paper_single_data_driven',
                                                      initial_thetas=[p0.tolist()])))

from ._interface import FittingMethod
from ._anchors import fit_soft


def resolve_config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping) or set(config) - set(cfg):
            raise ValueError("SLD paper fitter accepts only ftol and gtol")
        cfg.update(config)
    for name in ("ftol", "gtol"):
        cfg[name] = float(cfg[name])
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"Invalid {name}")
    return cfg


def fit_sld_soft(data_sizes, losses, config=None, *, l0, anchor_weight):
    return fit_soft("sld", fit_sld, data_sizes, losses, config,
                    l0=l0, weight=anchor_weight)


FITTING_METHODS = {
    "default": FittingMethod(fit_sld, resolve_config, 4),
    "soft": FittingMethod(fit_sld_soft, resolve_config, 4,
                          requires_l0=True, uses_anchor_weight=True),
}
