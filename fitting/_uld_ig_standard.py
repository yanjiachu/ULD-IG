from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from formulas.uld_ig import expression, log_value_gradient, predict_log
from . import _uld_ig_config as defaults


def resolve_config(config, parameterization):
    if parameterization not in ('4p', '5p'):
        raise ValueError('Parameterization must be 4p or 5p')
    return defaults.resolve_config(config, parameterization)


def objective(theta, log_d, log_y, delta, parameterization, l0=None, shape_bounds=None, stats=None):
    prediction, jacobian = log_value_gradient(theta, log_d, parameterization,
                                              l0, shape_bounds, stats)
    residual = prediction-log_y
    absolute = np.abs(residual)
    value = np.where(absolute <= delta, residual**2/2, delta*(absolute-delta/2)).sum()
    return float(value), jacobian.T@np.clip(residual, -delta, delta)


def fit_standard(data_sizes, losses, config=None, *, parameterization, l0=None):
    started = perf_counter()
    cfg = resolve_config(config, parameterization)
    d, y = np.asarray(data_sizes, dtype=float), np.asarray(losses, dtype=float)
    count = 4 if parameterization == '4p' else 5
    if (d.ndim != 1 or y.shape != d.shape or d.size < count or not np.all(np.isfinite(d))
            or not np.all(np.isfinite(y)) or np.any(d <= 0) or np.any(y <= 0)
            or np.unique(d).size != d.size):
        raise ValueError(f'Require at least {count} distinct positive D and finite positive losses')
    if parameterization == '4p':
        l0 = float(l0)
        if not np.isfinite(l0) or l0 <= 0:
            raise ValueError('4p requires a finite positive observed L0')
        scale = l0
    else:
        if l0 is not None:
            raise ValueError('5p must not receive observed L0')
        scale = float(y.max())
    names = ('E', 'd0', 'beta', 'alpha') + (() if count == 4 else ('A',))
    theta_names = ['E', 'ln_d0', 'ln_beta', 'ln_alpha'] + ([] if count == 4 else ['ln_A'])

    def resolve(group):
        r = cfg[group]
        rows = [np.array(r['E_fraction'])*scale,
                np.log(r['d0_factors'])+np.log([d.min(), d.max()]),
                np.log(r['beta']), np.log(r['alpha'])]
        if count == 5:
            rows.append(np.log(r['A_factors'])+np.log(scale))
        return np.array(rows)

    def decode(theta):
        return defaults.decode_hard(theta, l0) if count == 4 else defaults.decode_free(theta)

    bounds, initial_bounds = resolve('parameter_bounds'), resolve('initial_ranges')
    bounds[0, 0] = max(bounds[0, 0], cfg['e_floor_fraction_of_min_fit_loss']*float(y.min()))
    if bounds[0, 0] >= bounds[0, 1]:
        raise ValueError('E lower bound meets or exceeds upper bound')
    starts = np.random.default_rng(cfg['random_seed']).uniform(
        initial_bounds[:, 0], initial_bounds[:, 1], size=(cfg['num_random_starts'], count))
    candidates, runs = [], []
    stats = dict(shape_evaluations=0, shape_seconds=0.)
    args = (np.log(d), np.log(y), cfg['huber_delta'], parameterization, l0, bounds[2:4], stats)
    for index, initial in enumerate(starts):
        run_start = perf_counter()
        try:
            result = minimize(objective, np.clip(initial, bounds[:, 0], bounds[:, 1]), jac=True,
                              method='L-BFGS-B', bounds=bounds, args=args,
                              options={n: cfg[n] for n in ('maxiter', 'ftol', 'gtol')})
            finite = np.isfinite(result.fun) and np.all(np.isfinite(result.x))
            runs.append(dict(start_index=index, success=bool(result.success), message=str(result.message),
                             objective=float(result.fun) if finite else None,
                             parameters=decode(result.x) if finite else None,
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
    boundary = [n for i, n in enumerate(names) if np.any(np.isclose(best.x[i], bounds[i], rtol=0, atol=1e-6))]
    warnings = []
    if not converged:
        warnings.append('No converged start; returning best finite result')
    if boundary:
        warnings.append('parameters at bounds: '+', '.join(boundary))
    if count == 4 and np.any(y > l0):
        warnings.append('Observed fitting losses exceed fixed L0; observations not clipped')
    _, jacobian = log_value_gradient(best.x, np.log(d), parameterization, l0, bounds[2:4], stats)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    if condition is not None and not np.isfinite(condition):
        condition = None
    tolerance = max(1e-8, .01*abs(value))
    near = [c for c in (converged or candidates) if c[0] <= value+tolerance]
    probes = np.array([d.max(), 10*d.max()])
    span = np.ptp([predict_log(probes, decode(c[2].x)) for c in near], axis=0)
    if condition is None or condition > 1e8:
        warnings.append('Ill-conditioned training Jacobian; parameters may not be identifiable')
    if span.max() > np.log(1.5):
        warnings.append('Near-optimal starts disagree on extrapolation')
    optimization = dict(objective=value, objective_type='log_huber_sum', optimizer_success=bool(best.success),
                        message=str(best.message), best_start_index=index, finite_starts=len(candidates),
                        successful_starts=len(converged), parameters_at_bounds=boundary, warnings=warnings,
                        runs=runs, config={**cfg, 'optimizer': 'L-BFGS-B', 'gradient': 'analytic'},
                        fallback_used=not bool(converged), elapsed_seconds=perf_counter()-started,
                        shape_evaluations=stats['shape_evaluations'], shape_seconds=stats['shape_seconds'],
                        theta_names=theta_names, theta_bounds=bounds.tolist(), e_lower_bound=float(bounds[0, 0]),
                        initialization=dict(strategy='independent_uniform_in_theta', theta_bounds=initial_bounds.tolist(),
                                            initial_thetas=starts.tolist(), clipped_e_starts=int(np.sum(starts[:, 0] < bounds[0, 0])),
                                            projected_initial_thetas=np.clip(starts, bounds[:, 0], bounds[:, 1]).tolist()))
    if count == 5:
        optimization['loss_scale'] = scale
    return dict(formula=expression(parameterization), status='warning' if warnings else 'success',
                parameters=decode(best.x), optimization=optimization,
                diagnostics=dict(jacobian_singular_values=singular.tolist(), jacobian_condition=condition,
                                 jacobian_coordinates=theta_names, probe_D=probes.tolist(),
                                 near_optimal_tolerance=tolerance, near_optimal_starts=len(near), log_prediction_span=span.tolist()))

