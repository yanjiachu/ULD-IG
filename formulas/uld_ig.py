import numpy as np
from scipy.special import gammainc


NAME = "ULD-IG"
SHAPE = "P(2,(D/d0)**(-beta))**(alpha/(2*beta))"


def expression(parameterization="5p"):
    if parameterization not in ("4p", "5p"):
        raise ValueError("Parameterization must be 4p or 5p")
    return ('E + (L0-E)*(' if parameterization == '4p' else 'E + A*(') + SHAPE + ')'


def log_shape_gradient(x, log_beta, log_alpha, shape_bounds=None, stats=None):
    x = np.asarray(x, dtype=float)
    a, b = np.exp(log_alpha), np.exp(log_beta)
    t = -b*x
    ell_base, k = np.empty_like(t),np.empty_like(t)
    small, large = t < -9, t > np.log(10000.)
    mid = ~(small|large)
    u = np.exp(t[small])
    correction = -2*u/3+u*u/4-u**3/15
    ell_base[small] = 2*t[small]-np.log(2.)+np.log1p(correction)
    k[small] = 2+(-2*u/3+u*u/2-u**3/5)/(1+correction)
    ell_base[large],k[large] = 0.,0.
    u = np.exp(t[mid])
    base = np.empty_like(u)
    lower = u <= 1
    base[lower] = np.log(gammainc(2.,u[lower]))
    base[~lower] = np.log1p(-np.exp(np.log1p(u[~lower])-u[~lower]))
    ell_base[mid] = base
    k[mid] = np.exp(2*t[mid]-u-ell_base[mid])
    ell = a/(2*b)*ell_base
    g = np.column_stack([a*k/2,-ell-a*x*k/2,ell])
    if not np.all(np.isfinite(ell)) or not np.all(np.isfinite(g)):
        raise FloatingPointError("Nonfinite log shape or gradient")
    return ell, g


def log_value_gradient(theta, log_d, parameterization, l0=None, shape_bounds=None, stats=None):
    e, log_d0, log_beta, log_alpha = theta[:4]
    ell, gradient = log_shape_gradient(np.asarray(log_d)-log_d0, log_beta, log_alpha,
                                       shape_bounds, stats)
    excess = (np.log(l0-e) if parameterization == '4p' else theta[4]) + ell
    value = np.logaddexp(np.log(e), excess)
    weight = np.exp(excess-value)
    e_gradient = (-np.expm1(ell) if parameterization == '4p' else 1.)*np.exp(-value)
    jacobian = np.column_stack([e_gradient, weight[:, None]*gradient])
    if parameterization == '5p':
        jacobian = np.column_stack([jacobian, weight])
    return value, jacobian


def predict_log(data_sizes, parameters):
    parameterization = "4p" if "L0" in parameters else "5p"
    d = np.asarray(data_sizes, dtype=float)
    if d.ndim != 1 or not d.size or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError('Prediction requires finite nonnegative 1D D')
    names = ('E', 'd0', 'beta', 'alpha', 'L0' if parameterization == '4p' else 'A')
    values = np.array([parameters[n] for n in names], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Require finite positive parameters')
    if parameterization == '4p' and values[0] >= values[-1]:
        raise ValueError('Require E < observed L0')
    zero = values[-1] if parameterization == '4p' else values[0]+values[-1]
    result = np.full(d.shape, np.log(zero))
    mask = d > 0
    if mask.any():
        theta = np.r_[values[0], np.log(values[1:4])]
        if parameterization == '5p':
            theta = np.r_[theta, np.log(values[4])]
        result[mask] = log_value_gradient(theta, np.log(d[mask]), parameterization,
                                           zero if parameterization == '4p' else None)[0]
    return result


def predict(data_sizes, parameters):
    parameterization = "4p" if "L0" in parameters else "5p"
    values = np.exp(predict_log(data_sizes, parameters))
    values[np.asarray(data_sizes) == 0] = (parameters['L0'] if parameterization == '4p'
                                         else parameters['E']+parameters['A'])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise FloatingPointError('Prediction outside representable positive loss range')
    return values

