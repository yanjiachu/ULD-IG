from __future__ import annotations


from collections.abc import Mapping, Sequence


from formulas.power import (
    FORMULA, PARAMETER_NAMES, predict_log_power, predict_power
)

import numpy as np


from scipy.optimize import minimize


from .vanilla import (
    VanillaConfig, _as_valid_1d_array, _objective_with_gradient as _vanilla_objective,
)


def _config(raw):
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError("Power configuration must be a mapping")
        bounds = raw.get('parameter_bounds', {})
        if not isinstance(bounds, Mapping) or set(bounds)-set(PARAMETER_NAMES):
            raise ValueError("Power bounds may define only B, E and beta")
    return VanillaConfig.from_mapping(raw)


def _objective_with_gradient(theta, log_d, log_observed, delta):
    expanded = np.array([theta[0], theta[1], 0.0, theta[2]])
    value, gradient = _vanilla_objective(expanded, log_d, log_observed, delta)
    return value, gradient[[0, 1, 3]]


def fit_power(data_sizes, losses, config=None):
    resolved = _config(config)
    d = _as_valid_1d_array(data_sizes, 'D')
    y = _as_valid_1d_array(losses, 'L')
    if d.shape != y.shape or d.size < 3:
        raise ValueError("Power requires matching D/L arrays and at least three fitting points")
    if np.any(d <= 0) or np.any(y <= 0) or np.unique(d).size != d.size:
        raise ValueError("Power requires positive losses and distinct positive D values")
    bounds = np.log([resolved.parameter_bounds[n] for n in PARAMETER_NAMES])
    rng = np.random.default_rng(resolved.random_seed)
    starts = rng.uniform(bounds[:, 0], bounds[:, 1], (resolved.num_random_starts, 3))
    candidates, failures = [], []
    for index, initial in enumerate(starts):
        try:
            result = minimize(_objective_with_gradient, initial, jac=True,
                args=(np.log(d), np.log(y), resolved.huber_delta),
                method=resolved.optimizer, bounds=[tuple(pair) for pair in bounds],
                options={n:getattr(resolved,n) for n in ('maxiter','ftol','gtol')})
        except Exception as exc:
            failures.append(f"start {index}: {type(exc).__name__}: {exc}")
            continue
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            candidates.append((float(result.fun), index, result))
        else:
            failures.append(f"start {index}: non-finite optimizer result")
    if not candidates:
        raise RuntimeError("All power random starts failed to produce a finite result. " + '; '.join(failures[-3:]))
    converged = [v for v in candidates if v[2].success]
    objective, best_index, best = min(converged or candidates, key=lambda v:v[0])
    parameters = dict(zip(PARAMETER_NAMES, map(float,np.exp(best.x))))
    boundary = [n for i,n in enumerate(PARAMETER_NAMES)
                if np.any(np.isclose(best.x[i], bounds[i], rtol=0, atol=1e-6))]
    warnings = []
    if not best.success:
        warnings.append('best optimizer run did not report convergence')
    if boundary:
        warnings.append('parameters at or near configured bounds: ' + ', '.join(boundary))
    saved_config = resolved.as_dict()
    saved_config['parameter_bounds'] = {n:saved_config['parameter_bounds'][n] for n in PARAMETER_NAMES}
    return dict(formula=FORMULA, status='warning' if warnings else 'success', parameters=parameters,
        optimization=dict(gradient='analytic', objective=objective, objective_type='log_huber_sum',
            optimizer_success=bool(best.success), message=str(best.message),
            best_start_index=best_index, finite_starts=len(candidates), successful_starts=len(converged),
            fallback_used=not bool(converged), iterations=int(getattr(best,'nit',0)),
            function_evaluations=int(getattr(best,'nfev',0)),
            parameters_at_bounds=boundary, warnings=warnings, config=saved_config))

from ._interface import FittingMethod


def resolve_config(config=None):
    resolved = _config(config).as_dict()
    resolved["parameter_bounds"] = {name: resolved["parameter_bounds"][name] for name in PARAMETER_NAMES}
    return resolved


FITTING_METHODS = {"default": FittingMethod(fit_power, resolve_config, 3)}
