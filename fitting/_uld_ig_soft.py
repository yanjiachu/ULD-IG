from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from formulas.uld_ig import expression, log_value_gradient, predict_log
from . import _uld_ig_config as defaults
from ._uld_ig_config import validate_anchor


REQUIRES_L0 = True
USES_ANCHOR_WEIGHT = True
FREE_PARAMETERS = 5
MIN_POINTS = 5


def resolve_config(config=None):
    return defaults.resolve_config(config, "5p")


def huber_value(residual, delta):
    absolute = abs(residual)
    return float(residual**2/2 if absolute <= delta else delta*(absolute-delta/2))


def anchor_terms(parameters, l0, anchor_weight, delta):
    estimated = parameters['E'] + parameters['A']
    residual = None if l0 is None else float(estimated-l0)
    log_zero = np.logaddexp(np.log(parameters['E']), np.log(parameters['A']))
    log_residual = None if l0 is None else float(log_zero-np.log(l0))
    loss = None if log_residual is None else huber_value(log_residual, delta)
    penalty = 0. if anchor_weight == 0 else float(anchor_weight*loss)
    return dict(space='log_loss', penalty_type='huber', huber_delta=delta,
                weight=anchor_weight, observed_L0=l0,
                estimated_L0=float(estimated), residual=residual,
                log_residual=log_residual, huber_loss=loss, penalty=penalty)


def objective(theta, log_d, log_y, delta, l0=None, anchor_weight=0., shape_bounds=None, stats=None):
    prediction, jacobian = log_value_gradient(theta, log_d, "5p", shape_bounds=shape_bounds, stats=stats)
    residual = prediction-log_y
    absolute = np.abs(residual)
    value = float(np.where(absolute <= delta, residual**2/2, delta*(absolute-delta/2)).sum())
    gradient = jacobian.T@np.clip(residual, -delta, delta)
    if anchor_weight == 0:
        return value, gradient
    log_zero = np.logaddexp(np.log(theta[0]), theta[4])
    residual = log_zero-np.log(l0)
    penalty = anchor_weight*huber_value(residual, delta)
    anchor_gradient = (anchor_weight*np.clip(residual, -delta, delta)
                       * np.array([np.exp(-log_zero), 0., 0., 0., np.exp(theta[4]-log_zero)]))
    return float(value+penalty), gradient+anchor_gradient


def fit(data_sizes, losses, config=None, *, l0=None, anchor_weight=None):
    started = perf_counter()
    l0, anchor_weight = validate_anchor(l0, anchor_weight)
    cfg = resolve_config(config)
    d, y = np.asarray(data_sizes, dtype=float), np.asarray(losses, dtype=float)
    if (d.ndim != 1 or y.shape != d.shape or d.size < 5 or not np.all(np.isfinite(d))
            or not np.all(np.isfinite(y)) or np.any(d <= 0) or np.any(y <= 0)
            or np.unique(d).size != d.size):
        raise ValueError('Require at least five distinct positive D and finite positive losses')
    scale = float(y.max())
    names = ('E', 'd0', 'beta', 'alpha', 'A')
    theta_names = ['E', 'ln_d0', 'ln_beta', 'ln_alpha', 'ln_A']

    def resolve(group):
        r = cfg[group]
        return np.array([np.array(r['E_fraction'])*scale,
                         np.log(r['d0_factors'])+np.log([d.min(), d.max()]),
                         np.log(r['beta']), np.log(r['alpha']),
                         np.log(r['A_factors'])+np.log(scale)])

    bounds, initial_bounds = resolve('parameter_bounds'), resolve('initial_ranges')
    bounds[0, 0] = max(bounds[0, 0], cfg['e_floor_fraction_of_min_fit_loss']*float(y.min()))
    if bounds[0, 0] >= bounds[0, 1]:
        raise ValueError('E lower bound meets or exceeds upper bound')
    starts = np.random.default_rng(cfg['random_seed']).uniform(
        initial_bounds[:, 0], initial_bounds[:, 1], size=(cfg['num_random_starts'], 5))
    candidates, runs = [], []
    stats = dict(shape_evaluations=0, shape_seconds=0.)
    args = (np.log(d), np.log(y), cfg['huber_delta'], l0, anchor_weight, bounds[2:4], stats)
    for index, initial in enumerate(starts):
        run_start = perf_counter()
        try:
            result = minimize(objective, np.clip(initial, bounds[:, 0], bounds[:, 1]), jac=True,
                              method='L-BFGS-B', bounds=bounds, args=args,
                              options={n: cfg[n] for n in ('maxiter', 'ftol', 'gtol')})
            finite = np.isfinite(result.fun) and np.all(np.isfinite(result.x))
            runs.append(dict(start_index=index, success=bool(result.success), message=str(result.message),
                             objective=float(result.fun) if finite else None,
                             parameters=defaults.decode_free(result.x) if finite else None,
                             nfev=int(result.nfev), nit=int(result.nit), elapsed_seconds=perf_counter()-run_start))
            if finite:
                candidates.append((float(result.fun), index, result))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            runs.append(dict(start_index=index, success=False, message=str(exc), objective=None,
                             parameters=None, elapsed_seconds=perf_counter()-run_start))
    if not candidates:
        raise RuntimeError('All starts failed to produce a finite result')
    converged = [c for c in candidates if c[2].success]
    value, index, best = min(converged or candidates, key=lambda c: c[0])
    parameters = defaults.decode_free(best.x)
    anchor = anchor_terms(parameters, l0, anchor_weight, cfg['huber_delta'])
    data_value = objective(best.x, np.log(d), np.log(y), cfg['huber_delta'],
                           shape_bounds=bounds[2:4])[0]
    boundary = [n for i, n in enumerate(names) if np.any(np.isclose(best.x[i], bounds[i], rtol=0, atol=1e-6))]
    warnings = []
    if not converged:
        warnings.append('No converged start; returning best finite result')
    if boundary:
        warnings.append('parameters at bounds: '+', '.join(boundary))
    _, jacobian = log_value_gradient(best.x, np.log(d), "5p", shape_bounds=bounds[2:4], stats=stats)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    if condition is not None and not np.isfinite(condition):
        condition = None
    tolerance = max(1e-8, .01*abs(value))
    near = [c for c in (converged or candidates) if c[0] <= value+tolerance]
    probes = np.array([d.max(), 10*d.max()])
    span = np.ptp([predict_log(probes, defaults.decode_free(c[2].x)) for c in near], axis=0)
    if condition is None or condition > 1e8:
        warnings.append('Ill-conditioned training Jacobian; parameters may not be identifiable')
    if span.max() > np.log(1.5):
        warnings.append('Near-optimal starts disagree on extrapolation')
    optimization = dict(objective=value,
                        objective_type='log_huber_sum_plus_weighted_l0_log_huber' if anchor_weight else 'log_huber_sum',
                        data_objective=data_value, anchor_objective=anchor['penalty'],
                        optimizer_success=bool(best.success), message=str(best.message),
                        best_start_index=index, finite_starts=len(candidates), successful_starts=len(converged),
                        parameters_at_bounds=boundary, warnings=warnings, runs=runs,
                        config={**cfg, 'optimizer': 'L-BFGS-B', 'gradient': 'analytic'},
                        fallback_used=not bool(converged), elapsed_seconds=perf_counter()-started,
                        shape_evaluations=stats['shape_evaluations'], shape_seconds=stats['shape_seconds'],
                        theta_names=theta_names, theta_bounds=bounds.tolist(), e_lower_bound=float(bounds[0, 0]),
                        loss_scale=scale,
                        initialization=dict(strategy='independent_uniform_in_theta', theta_bounds=initial_bounds.tolist(),
                                            initial_thetas=starts.tolist(), clipped_e_starts=int(np.sum(starts[:, 0] < bounds[0, 0])),
                                            projected_initial_thetas=np.clip(starts, bounds[:, 0], bounds[:, 1]).tolist()))
    return dict(formula=expression("5p"), status='warning' if warnings else 'success', parameters=parameters,
                optimization=optimization, anchor=anchor,
                diagnostics=dict(jacobian_singular_values=singular.tolist(), jacobian_condition=condition,
                                 jacobian_coordinates=theta_names, probe_D=probes.tolist(),
                                 near_optimal_tolerance=tolerance, near_optimal_starts=len(near), log_prediction_span=span.tolist()))
