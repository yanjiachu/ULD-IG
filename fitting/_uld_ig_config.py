from collections.abc import Mapping
from copy import deepcopy

import numpy as np

DEFAULT_4P = dict(
    huber_delta=0.01, e_floor_fraction_of_min_fit_loss=0.3,
    num_random_starts=30, random_seed=42,
    maxiter=10000, ftol=1e-12, gtol=1e-8,
    initial_ranges=dict(E_fraction=[.2,.8], d0_factors=[.1,10.],
                        beta=[1e-4,10.], alpha=[1e-4,10.]),
    parameter_bounds=dict(E_fraction=[1e-6,1-1e-6], d0_factors=[.1,10.],
                          beta=[1e-4,10.], alpha=[1e-4,10.]))

DEFAULT_5P = dict(
    huber_delta=0.01, e_floor_fraction_of_min_fit_loss=0.3,
    num_random_starts=30, random_seed=42,
    maxiter=10000, ftol=1e-12, gtol=1e-8,
    initial_ranges=dict(A_factors=[.1,10.], E_fraction=[.2,.8], d0_factors=[.1,10.],
                        beta=[1e-4,10.], alpha=[1e-4,10.]),
    parameter_bounds=dict(A_factors=[1e-6,1e3], E_fraction=[1e-6,1-1e-6], d0_factors=[.1,10.],
                          beta=[1e-4,10.], alpha=[1e-4,10.]))


def resolve_config(raw=None, parameterization="5p"):
    if parameterization not in ("4p", "5p"):
        raise ValueError("Parameterization must be 4p or 5p")
    cfg = deepcopy(DEFAULT_4P if parameterization == "4p" else DEFAULT_5P)
    if raw is not None:
        if not isinstance(raw, Mapping) or set(raw)-set(cfg):
            raise ValueError("Invalid inverse Gamma optimizer configuration")
        for key, value in raw.items():
            if key in ('initial_ranges','parameter_bounds'):
                if not isinstance(value, Mapping) or set(value)-set(cfg[key]):
                    raise ValueError(f"Invalid {key}")
                cfg[key].update(value)
            else:
                cfg[key] = value
    for key in ('num_random_starts','maxiter','random_seed'):
        v = cfg[key]
        if isinstance(v,bool) or not isinstance(v,int) or v < (0 if key=='random_seed' else 1):
            raise ValueError(f"Invalid {key}")
    for key in ('huber_delta','ftol','gtol'):
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    cfg['e_floor_fraction_of_min_fit_loss'] = float(cfg['e_floor_fraction_of_min_fit_loss'])
    if (not np.isfinite(cfg['e_floor_fraction_of_min_fit_loss'])
            or not 0 <= cfg['e_floor_fraction_of_min_fit_loss'] < 1):
        raise ValueError("Invalid e_floor_fraction_of_min_fit_loss")
    for group in ('initial_ranges','parameter_bounds'):
        for name, pair in cfg[group].items():
            values = np.asarray(pair,dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)) or not 0 < values[0] < values[1]:
                raise ValueError(f"Invalid {group}.{name}")
            if name=='E_fraction' and values[1]>=1:
                raise ValueError("E fractions must be strictly between zero and one")
            cfg[group][name] = values.tolist()
    for name, pair in cfg['initial_ranges'].items():
        low,high = cfg['parameter_bounds'][name]
        if pair[0]<low or pair[1]>high:
            raise ValueError(f"Initial range {name} must lie within bounds")
    return cfg


def decode_hard(theta, l0):
    return dict(E=float(theta[0]),d0=float(np.exp(theta[1])),
                beta=float(np.exp(theta[2])),alpha=float(np.exp(theta[3])),L0=float(l0))


def decode_free(theta):
    return dict(E=float(theta[0]),d0=float(np.exp(theta[1])),
                beta=float(np.exp(theta[2])),alpha=float(np.exp(theta[3])),A=float(np.exp(theta[4])),
                estimated_L0=float(theta[0]+np.exp(theta[4])))



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

