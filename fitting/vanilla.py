from __future__ import annotations


from dataclasses import dataclass, field


from typing import Mapping, Sequence


from formulas.vanilla import (
    FORMULA, PARAMETER_NAMES, _as_valid_1d_array, _log_prediction_from_theta, predict_log_vanilla, predict_vanilla
)

import numpy as np


from scipy.optimize import minimize


METHOD_ID = "vanilla"


DEFAULT_PARAMETER_BOUNDS = {
    "B": (1.0e-12, 1.0e12),
    "E": (1.0e-12, 1.0e3),
    "alpha": (1.0e-3, 10.0),
    "beta": (1.0e-3, 10.0),
}


@dataclass(frozen=True)
class VanillaConfig:

    huber_delta: float = 0.001
    num_random_starts: int = 30
    random_seed: int = 42
    optimizer: str = "L-BFGS-B"
    maxiter: int = 10_000
    ftol: float = 1.0e-12
    gtol: float = 1.0e-8
    parameter_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PARAMETER_BOUNDS)
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> "VanillaConfig":
        if raw is None:
            config = cls()
            config.validate()
            return config

        bounds = dict(DEFAULT_PARAMETER_BOUNDS)
        raw_bounds = raw.get("parameter_bounds", {})
        if not isinstance(raw_bounds, Mapping):
            raise ValueError("parameter_bounds must be a mapping")
        for name, value in raw_bounds.items():
            if name not in PARAMETER_NAMES:
                raise ValueError(f"Unknown vanilla parameter bound: {name!r}")
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"Bounds for {name} must contain [lower, upper]")
            if len(value) != 2:
                raise ValueError(f"Bounds for {name} must contain two values")
            bounds[name] = (float(value[0]), float(value[1]))

        config = cls(
            huber_delta=float(raw.get("huber_delta", 0.001)),
            num_random_starts=int(raw.get("num_random_starts", 30)),
            random_seed=int(raw.get("random_seed", 42)),
            optimizer=str(raw.get("optimizer", "L-BFGS-B")),
            maxiter=int(raw.get("maxiter", 10_000)),
            ftol=float(raw.get("ftol", 1.0e-12)),
            gtol=float(raw.get("gtol", 1.0e-8)),
            parameter_bounds=bounds,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not np.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be a finite positive number")
        if self.num_random_starts < 1:
            raise ValueError("num_random_starts must be a positive integer")
        if self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if not np.isfinite(self.ftol) or self.ftol <= 0:
            raise ValueError("ftol must be a finite positive number")
        if not np.isfinite(self.gtol) or self.gtol <= 0:
            raise ValueError("gtol must be a finite positive number")
        if self.optimizer != "L-BFGS-B":
            raise ValueError("Vanilla currently supports optimizer='L-BFGS-B' only")

        if set(self.parameter_bounds) != set(PARAMETER_NAMES):
            raise ValueError(
                "parameter_bounds must define exactly B, E, alpha and beta"
            )
        for name in PARAMETER_NAMES:
            lower, upper = self.parameter_bounds[name]
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"Bounds for {name} must be finite")
            if lower <= 0 or upper <= 0 or lower >= upper:
                raise ValueError(
                    f"Bounds for {name} must satisfy 0 < lower < upper"
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "huber_delta": self.huber_delta,
            "num_random_starts": self.num_random_starts,
            "random_seed": self.random_seed,
            "optimizer": self.optimizer,
            "maxiter": self.maxiter,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "parameter_bounds": {
                name: [
                    float(self.parameter_bounds[name][0]),
                    float(self.parameter_bounds[name][1]),
                ]
                for name in PARAMETER_NAMES
            },
        }


def _validate_fit_inputs(
    data_sizes: Sequence[float], losses: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    d = _as_valid_1d_array(data_sizes, "D")
    observed = _as_valid_1d_array(losses, "L")
    if d.shape != observed.shape:
        raise ValueError("D and L must have the same length")
    if d.size < len(PARAMETER_NAMES):
        raise ValueError("Vanilla requires at least four fitting points")
    if np.any(d <= 0):
        raise ValueError("Vanilla requires every D to be positive")
    if np.any(observed <= 0):
        raise ValueError("Vanilla requires every L to be positive")
    if np.unique(d).size != d.size:
        raise ValueError("Vanilla requires unique D values")
    return d, observed


def _decode_log_parameters(log_parameters: np.ndarray) -> dict[str, float]:
    values = np.exp(log_parameters)
    return {
        name: float(value)
        for name, value in zip(PARAMETER_NAMES, values, strict=True)
    }


def _huber_sum(residuals: np.ndarray, delta: float) -> float:
    absolute = np.abs(residuals)
    losses = np.where(
        absolute <= delta,
        0.5 * residuals**2,
        delta * (absolute - 0.5 * delta),
    )
    return float(np.sum(losses))


def _objective(
    theta: np.ndarray,
    log_d: np.ndarray,
    log_observed: np.ndarray,
    delta: float,
) -> float:
    log_predicted = _log_prediction_from_theta(log_d, theta)
    if not np.all(np.isfinite(log_predicted)):
        return float("inf")
    return _huber_sum(log_predicted - log_observed, delta)


def _objective_with_gradient(theta, log_d, log_observed, delta):
    log_b, log_e, log_alpha, log_beta = theta
    alpha, beta = np.exp(log_alpha), np.exp(log_beta)
    term = log_b - beta * log_d
    total = np.logaddexp(term, log_e)
    weight = np.exp(term - total)
    log_predicted = alpha * total
    residual = log_predicted - log_observed
    jac = np.column_stack([alpha * weight, alpha * np.exp(log_e-total),
                           log_predicted, -alpha * weight * beta * log_d])
    return _huber_sum(residual, delta), jac.T @ np.clip(residual, -delta, delta)


def fit_vanilla(
    data_sizes: Sequence[float],
    losses: Sequence[float],
    config: VanillaConfig | Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved = (
        config
        if isinstance(config, VanillaConfig)
        else VanillaConfig.from_mapping(config)
    )
    resolved.validate()
    d, observed = _validate_fit_inputs(data_sizes, losses)
    log_d = np.log(d)
    log_observed = np.log(observed)

    log_bounds = np.asarray(
        [
            [
                np.log(resolved.parameter_bounds[name][0]),
                np.log(resolved.parameter_bounds[name][1]),
            ]
            for name in PARAMETER_NAMES
        ],
        dtype=np.float64,
    )
    scipy_bounds = [tuple(bound) for bound in log_bounds]
    rng = np.random.default_rng(resolved.random_seed)
    initializations = rng.uniform(
        low=log_bounds[:, 0],
        high=log_bounds[:, 1],
        size=(resolved.num_random_starts, len(PARAMETER_NAMES)),
    )

    candidates: list[tuple[float, int, object]] = []
    successful_starts = 0
    failure_messages: list[str] = []
    for start_index, initial in enumerate(initializations):
        try:
            result = minimize(
                _objective_with_gradient,
                initial,
                jac=True,
                args=(log_d, log_observed, resolved.huber_delta),
                method=resolved.optimizer,
                bounds=scipy_bounds,
                options={
                    "maxiter": resolved.maxiter,
                    "ftol": resolved.ftol,
                    "gtol": resolved.gtol,
                },
            )
        except Exception as exc:
            failure_messages.append(f"start {start_index}: {type(exc).__name__}: {exc}")
            continue

        objective = float(result.fun)
        if np.isfinite(objective) and np.all(np.isfinite(result.x)):
            candidates.append((objective, start_index, result))
            if bool(result.success):
                successful_starts += 1
        else:
            failure_messages.append(
                f"start {start_index}: non-finite optimizer result"
            )

    if not candidates:
        details = "; ".join(failure_messages[-3:])
        suffix = f" Last failures: {details}" if details else ""
        raise RuntimeError(
            "All vanilla random starts failed to produce a finite result."
            + suffix
        )

    converged = [candidate for candidate in candidates if candidate[2].success]
    objective, best_start_index, best_result = min(
        converged or candidates, key=lambda candidate: candidate[0]
    )
    parameters = _decode_log_parameters(np.asarray(best_result.x, dtype=float))

    boundary_parameters = [
        name
        for index, name in enumerate(PARAMETER_NAMES)
        if np.isclose(
            best_result.x[index], log_bounds[index, 0], rtol=0.0, atol=1.0e-6
        )
        or np.isclose(
            best_result.x[index], log_bounds[index, 1], rtol=0.0, atol=1.0e-6
        )
    ]
    warnings: list[str] = []
    if not bool(best_result.success):
        warnings.append("best optimizer run did not report convergence")
    if boundary_parameters:
        warnings.append(
            "parameters at or near configured bounds: "
            + ", ".join(boundary_parameters)
        )

    return {
        "formula": FORMULA,
        "status": "warning" if warnings else "success",
        "parameters": parameters,
        "optimization": {
            "gradient": "analytic",
            "objective": objective,
            "optimizer_success": bool(best_result.success),
            "message": str(best_result.message),
            "best_start_index": best_start_index,
            "finite_starts": len(candidates),
            "successful_starts": successful_starts,
            "iterations": int(getattr(best_result, "nit", 0)),
            "function_evaluations": int(getattr(best_result, "nfev", 0)),
            "parameters_at_bounds": boundary_parameters,
            "warnings": warnings,
            "config": resolved.as_dict(),
        },
    }

from ._interface import FittingMethod


def resolve_config(config=None):
    return VanillaConfig.from_mapping(config).as_dict()


FITTING_METHODS = {"default": FittingMethod(fit_vanilla, resolve_config, 4)}
