from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from urllib.parse import quote

import numpy as np
import scipy
import yaml

from .config import ROOT, DEFAULT_CONFIG, load_config
from .data import read_available_data_sizes, read_curves
from fitting import FITTING_METHODS
from formulas import FORMULAS
from .selection import select_half_then_halve_budget


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _curve_id(identity):
    return "-".join(quote(part, safe="").replace("-", "%2D") for part in identity)


def _fingerprint(config_file, data_file):
    sources = [ROOT / "run.py"]
    for folder in ("common", "formulas", "fitting"):
        sources.extend(sorted((ROOT / folder).rglob("*.py")))
    return {
        "data_sha256": _sha256(data_file),
        "config_sha256": _sha256(config_file),
        "implementation_sha256": {
            p.relative_to(ROOT).as_posix(): _sha256(p)
            for p in sorted(sources)
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "pyyaml": yaml.__version__},
    }


def _save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(*, dataset="SFT-15", formula="ULD-IG", fit_method="default",
        config_path=DEFAULT_CONFIG, budgets=None, anchor_weight=None,
        output_dir=None, resume=False, overwrite=False):
    if resume and overwrite:
        raise ValueError("Choose resume or overwrite, not both")
    raw, config_file = load_config(config_path)
    if dataset not in raw["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset}; choose from {', '.join(raw['datasets'])}")
    if formula not in FORMULAS or formula not in raw["formulas"]:
        raise ValueError(f"Unsupported formula: {formula}")
    available = FITTING_METHODS.get(formula, {})
    if fit_method not in available:
        raise ValueError(f"{formula} does not support fitting method {fit_method}; choose from {', '.join(available)}")
    curve_formula, fitter = FORMULAS[formula], available[fit_method]
    if not fitter.uses_anchor_weight and anchor_weight is not None:
        raise ValueError("--anchor-weight requires a fitting method that supports soft anchoring")
    for value in (dataset, formula):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Dataset and formula names must contain only letters, digits, '_' or '-'")

    entry = raw["datasets"][dataset]
    formula_settings = raw["formulas"][formula]
    settings = formula_settings.get("fitting") if isinstance(formula_settings, dict) else None
    if not isinstance(entry, dict) or not isinstance(settings, dict):
        raise ValueError("Dataset and formula-specific fitting settings must be mappings")
    if not isinstance(entry.get("file"), str) or not entry["file"].strip():
        raise ValueError("Dataset file must be a nonempty path string")
    data_file = (config_file.parent / entry["file"]).resolve()
    configured = entry.get("budgets")
    if (not isinstance(configured, list) or not configured
            or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in configured)
            or len(set(configured)) != len(configured)):
        raise ValueError("Dataset budgets must be distinct positive integers")
    selected_budgets = list(configured if budgets is None else budgets)
    if (not selected_budgets or len(set(selected_budgets)) != len(selected_budgets)
            or any(isinstance(b, bool) or not isinstance(b, int) or b not in configured for b in selected_budgets)):
        raise ValueError(f"Choose distinct configured budgets from {configured}")
    optimizer = fitter.resolve_config(settings.get("optimizer"))
    policy = fitter.anchor_policy(settings, anchor_weight) if fitter.uses_anchor_weight else None
    output = Path(output_dir).resolve() if output_dir is not None else ROOT / "results"
    directory = output / dataset / formula / fit_method
    if policy is not None:
        directory /= ("dynamic" if policy["type"] == "inverse_power"
                      else f"w_{repr(policy['weight'])}")

    curves = read_curves(data_file)
    sizes = read_available_data_sizes(data_file)
    selections = [select_half_then_halve_budget(sizes, b) for b in selected_budgets]
    for selection in selections:
        minimum = fitter.min_points
        if len(selection.selected_data_sizes) < minimum:
            raise ValueError(f"budget={selection.budget} requires at least {minimum} positive-D fitting points")
        if policy is not None:
            fitter.resolve_weight(policy, float(selection.budget_ratio))
        for identity, curve in curves.items():
            if fitter.requires_l0 and 0 not in curve.losses:
                raise ValueError(
                    f"{dataset}: {fit_method} fitting requires observed L0 (D=0); "
                    f"missing for {identity}. Use --fit-method default for datasets without L0."
                )
            if any(d not in curve.losses for d in selection.selected_data_sizes):
                raise ValueError(f"Missing fitting observations for {identity}, budget={selection.budget}")
        path = directory / f"budget_{int(selection.budget)}.json"
        if path.exists() and not (resume or overwrite):
            raise FileExistsError(f"{path} exists; use --resume or --overwrite")

    fingerprint = _fingerprint(config_file, data_file)
    identities = {_curve_id(identity) for identity in curves}
    written, failed = [], []
    for job, selection in enumerate(selections, 1):
        weight = fitter.resolve_weight(policy, float(selection.budget_ratio)) if policy is not None else None
        path = directory / f"budget_{int(selection.budget)}.json"
        expected = {
            "schema_version": 1, "dataset_name": dataset, "formula": formula,
            "fit_method": fit_method, "free_parameters": fitter.free_parameters,
            "budget": int(selection.budget), "budget_ratio": float(selection.budget_ratio),
            "selection_strategy": selection.strategy, "total_data_size": selection.total_data_size,
            "actual_used_budget": selection.total_used_data,
            "selected_data_sizes": list(selection.selected_data_sizes),
            "optimizer": optimizer, "anchor_policy": policy, "anchor_weight": weight,
            "provenance": fingerprint,
        }
        if resume and path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if any(report.get(k) != v for k, v in expected.items()):
                raise ValueError("Cannot resume after changing data, configuration, code or runtime")
            if (not isinstance(report.get("curves"), dict) or not isinstance(report.get("failures"), dict)
                    or not set(report["curves"]).issubset(identities)
                    or not set(report["failures"]).issubset(identities)
                    or set(report["curves"]) & set(report["failures"])):
                raise ValueError("Invalid saved curve identities")
        else:
            report = {**expected, "curves": {}, "failures": {}}
        print(f"JOB {job}/{len(selections)}: {dataset}/{formula}/{fit_method}, "
              f"budget={selection.budget}, completed={len(report['curves'])}/{len(curves)}", flush=True)
        for identity, curve in curves.items():
            key = _curve_id(identity)
            if key in report["curves"]:
                continue
            report["failures"].pop(key, None)
            try:
                options = {}
                if fitter.requires_l0:
                    options["l0"] = curve.losses[0]
                if fitter.uses_anchor_weight:
                    options["anchor_weight"] = weight
                fitted = fitter.fit(
                    selection.selected_data_sizes,
                    [curve.losses[d] for d in selection.selected_data_sizes],
                    optimizer, **options,
                )
                prediction_sizes = sorted(d for d in curve.losses if d > 0)
                predictions = curve_formula.predict(prediction_sizes, fitted["parameters"])
                report["curves"][key] = {
                    "source": {"input_file": data_file.name, "model": identity[0],
                               "dataset": identity[1], "training seed": curve.seed},
                    "selected_points": [{"D": d, "L": curve.losses[d]} for d in selection.selected_data_sizes],
                    "fit": fitted,
                    "predictions": [{"D": d, "predicted_L": float(value)}
                                    for d, value in zip(prediction_sizes, predictions)],
                }
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                report["failures"][key] = {"status": "failed", "message": str(exc)}
            report["coverage"] = {"expected": len(curves), "completed": len(report["curves"]),
                                  "failed": len(report["failures"])}
            _save(report, path)
            print(f"  completed={len(report['curves'])}/{len(curves)}, "
                  f"failed={len(report['failures'])}", flush=True)
        if report["failures"]:
            failed.append(str(path))
        written.append(path)
        print(f"Saved {path}", flush=True)
    if failed:
        raise RuntimeError("Some fits failed; inspect the JSON and retry with --resume: " + ", ".join(failed))
    return written
