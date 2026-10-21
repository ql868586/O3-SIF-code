
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - environment error
    raise ImportError("xgboost is required") from exc


RANDOM_STATE = 42
SPATIOTEMPORAL_VARS = ["x", "y", "year", "month"]
MODEL_A_VARS = ["DEM", "lccs", "t2m", "ssrd", "tp", "u10", "v10", "sp", "stl1", "swvl1"]
MODEL_B_VARS = MODEL_A_VARS + [
    "stl2", "stl3", "stl4", "stl5",
    "swvl2", "swvl3", "swvl4", "swvl5",
    "ssr", "str", "src", "pev",
]
MODEL_C_VARS = MODEL_B_VARS + ["lai_hv", "lai_lv", "evavt", "e"]

CONTROL_VARS_BY_SET = {"A": MODEL_A_VARS, "B": MODEL_B_VARS, "C": MODEL_C_VARS}


# =============================================================================
# CLI
# =============================================================================
def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nested spatial-temporal XGBoost tuning for the blocked DML analysis."
    )
    p.add_argument("--input", required=True,
                   help="Raw long-format CSV path.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for checkpoints and frozen parameter JSON.")
    p.add_argument("--outcome", default="SIF")
    p.add_argument("--exposure", default="O3")
    p.add_argument("--control-set", choices=["A", "B", "C"], default="A")
    p.add_argument("--x-col", default="x")
    p.add_argument("--y-col", default="y")
    p.add_argument("--year-col", default="year")
    p.add_argument("--month-col", default="month")
    p.add_argument("--spatial-block-factor", type=int, default=20)
    p.add_argument("--time-block-years", type=int, default=1)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--force", action="store_true",
                   help="Ignore fold checkpoints and recompute from scratch.")
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================
def month_number(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s.astype(str).str.extract(r"(\d{1,2})$")[0], errors="coerce")
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().any() or not s.between(1, 12).all():
        raise ValueError("month must contain integers from 1 to 12")
    return s.astype(int)


def grid_spacing(data: pd.DataFrame, x_col: str, y_col: str):
    dx = float(np.nanmedian(np.diff(np.sort(data[x_col].dropna().unique()))))
    dy = float(np.nanmedian(np.diff(np.sort(data[y_col].dropna().unique()))))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError("cannot infer positive x/y spacing")
    return dx, dy, float(data[x_col].min()), float(data[y_col].min())


def add_blocks(data: pd.DataFrame, a: argparse.Namespace,
               dx: float, dy: float, x0: float, y0: float) -> None:
    fx = np.floor((data[a.x_col] - x0) / (dx * a.spatial_block_factor)).astype(int)
    fy = np.floor((data[a.y_col] - y0) / (dy * a.spatial_block_factor)).astype(int)
    data["_spatial_block"] = fx.astype(str) + "_" + fy.astype(str)
    data["_time_block"] = (
        (data[a.year_col] - data[a.year_col].min()) // a.time_block_years
    ).astype(int)
    # Fold labels are assigned to unique blocks, never individual observations.
    data["_space_fold"] = pd.factorize(data["_spatial_block"], sort=True)[0] % a.outer_folds
    data["_time_fold"] = pd.factorize(data["_time_block"], sort=True)[0] % a.outer_folds


def blocked_splits(panel: pd.DataFrame, folds: int, indices: np.ndarray):
    """Yield (fold_id, train_idx, test_idx) with spatial OR temporal holdout.

    Note: this yields the original fold id so the caller can preserve the
    correspondence between ``_space_fold`` / ``_time_fold`` labels and the
    checkpoint file names. The notebook version enumerated the yielded tuples
    instead, which could mislabel folds whenever a fold was skipped.
    """
    indices = np.asarray(indices, dtype=int)
    for fold in range(folds):
        test_mask = (
            (panel.loc[indices, "_space_fold"].to_numpy() == fold)
            | (panel.loc[indices, "_time_fold"].to_numpy() == fold)
        )
        test = indices[test_mask]
        train = indices[~test_mask]
        if len(test) and len(train):
            yield fold, train, test


def controls(data: pd.DataFrame, a: argparse.Namespace) -> list[str]:
    vars_by_set = CONTROL_VARS_BY_SET
    return [c for c in SPATIOTEMPORAL_VARS + vars_by_set[a.control_set] if c in data.columns]


def parameter_candidates(n: int, seed: int) -> list[dict]:
    """Generate a deterministic, auditable search design (no adaptive leakage)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append({
            "n_estimators": int(rng.integers(300, 1201)),
            "max_depth": int(rng.integers(3, 11)),
            "learning_rate": float(10 ** rng.uniform(math.log10(.01), math.log10(.15))),
            "min_child_weight": int(rng.integers(1, 31)),
            "gamma": float(rng.uniform(0, 5)),
            "subsample": float(rng.uniform(.65, 1.0)),
            "colsample_bytree": float(rng.uniform(.65, 1.0)),
            "reg_alpha": float(10 ** rng.uniform(-4, 0.5)),
            "reg_lambda": float(10 ** rng.uniform(-1, math.log10(30))),
        })
    return rows


def _cuda_available() -> bool:
    """Detect a usable CUDA GPU without hard-coding a version check."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        pass
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        pass
    return False


def model(params: dict, a: argparse.Namespace, seed: int):
    version_parts = tuple(
        int(part) for part in xgb.__version__.split(".")[:2] if part.isdigit()
    )

    if a.device == "cuda":
        use_cuda = True
    elif a.device == "cpu":
        use_cuda = False
    else:  # "auto"
        use_cuda = _cuda_available()

    kwargs = dict(
        params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=a.n_jobs,
        verbosity=0,
    )
    if use_cuda:
        if version_parts >= (2, 0):
            kwargs["device"] = "cuda"
        else:  # XGBoost 1.x compatibility
            kwargs.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)
    return xgb.XGBRegressor(**kwargs)


def score_candidate(X: np.ndarray, y: np.ndarray, panel: pd.DataFrame,
                    indices: np.ndarray, folds: int, params: dict,
                    a: argparse.Namespace, seed: int) -> dict:
    rmses, maes, r2s = [], [], []
    for _, train, test in blocked_splits(panel, folds, indices):
        m = model(params, a, seed)
        m.fit(X[train], y[train])
        pred = m.predict(X[test])
        rmses.append(math.sqrt(mean_squared_error(y[test], pred)))
        maes.append(mean_absolute_error(y[test], pred))
        r2s.append(r2_score(y[test], pred))
        del m, pred
        gc.collect()
    if not rmses:
        raise ValueError("blocked CV produced no non-empty folds; reduce fold counts")
    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "mae_mean": float(np.mean(maes)),
        "r2_mean": float(np.mean(r2s)),
        "n_folds": len(rmses),
    }


def tune(X, y, panel, indices, folds, a, label, seed):
    candidates = parameter_candidates(a.n_iter, seed)
    records = []
    print(f"{label}: evaluating {len(candidates)} candidates x {folds} blocked folds")
    for i, params in enumerate(candidates, 1):
        metrics = score_candidate(X, y, panel, indices, folds, params, a, seed + i)
        records.append({"candidate": i, **params, **metrics})
        print(f"  {i:03d}/{len(candidates)} "
              f"RMSE={metrics['rmse_mean']:.6g} +/- {metrics['rmse_std']:.3g}")
    # Primary criterion is mean RMSE; std is a transparent tie breaker.
    best = min(records, key=lambda r: (r["rmse_mean"], r["rmse_std"]))
    params = {k: best[k] for k in candidates[0]}
    return params, best, pd.DataFrame(records)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    a = arguments()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"XGBoost {xgb.__version__}; device={a.device}; seed={a.seed}")

    # Read only columns used by DML. Loading the full CSV first can create
    # several copies of a very large panel and trigger an OS-level OOM kill.
    header = pd.read_csv(a.input, nrows=0).columns.tolist()
    requested = list(dict.fromkeys(
        [a.outcome, a.exposure, a.x_col, a.y_col, a.year_col, a.month_col]
        + MODEL_C_VARS + SPATIOTEMPORAL_VARS
    ))
    usecols = [c for c in requested if c in header]
    missing = [c for c in [a.outcome, a.exposure, a.x_col, a.y_col, a.year_col, a.month_col]
               if c not in usecols]
    if missing:
        raise ValueError(f"required columns missing from input: {missing}")

    numeric_hint = set(MODEL_C_VARS + [a.outcome, a.exposure, a.x_col, a.y_col])
    dtype = {c: "float32" for c in usecols if c in numeric_hint}
    dtype[a.year_col] = "int32"
    data = pd.read_csv(a.input, usecols=usecols, dtype=dtype, low_memory=False)

    # Strict control-set check: refuse to silently drop declared controls.
    missing_controls = [c for c in CONTROL_VARS_BY_SET[a.control_set]
                        if c not in data.columns]
    if missing_controls:
        raise ValueError(
            f"control-set {a.control_set} requires columns missing from input: "
            f"{missing_controls}"
        )

    data[a.month_col] = month_number(data[a.month_col])
    data[a.year_col] = pd.to_numeric(data[a.year_col], errors="raise").astype(int)
    dx, dy, x0, y0 = grid_spacing(data, a.x_col, a.y_col)
    add_blocks(data, a, dx, dy, x0, y0)

    cs = controls(data, a)
    required = [a.outcome, a.exposure] + cs + ["_spatial_block", "_time_block"]
    work = data[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    del data
    gc.collect()

    if len(work) < 100:
        raise ValueError(f"only {len(work)} complete rows; at least 100 are required")

    X = np.ascontiguousarray(work[cs].to_numpy(dtype=np.float32, copy=True))
    targets = {
        "treatment": work[a.exposure].to_numpy(dtype=np.float32),
        "outcome": work[a.outcome].to_numpy(dtype=np.float32),
    }

    # Fold labels must be recomputed after dropping rows, while preserving block identity.
    work["_space_fold"] = pd.factorize(work["_spatial_block"], sort=True)[0] % a.outer_folds
    work["_time_fold"] = pd.factorize(work["_time_block"], sort=True)[0] % a.outer_folds

    all_idx = np.arange(len(work))
    outer = list(blocked_splits(work, a.outer_folds, all_idx))
    if len(outer) < 2:
        raise ValueError("fewer than two usable outer blocked folds")

    nested_rows = []
    started = time.time()
    for fold, outer_train, outer_test in outer:
        fold_no = fold + 1  # 1-based display id, consistent with `_space_fold` / `_time_fold`
        checkpoint = out / f"nested_fold_{fold_no}.json"
        if checkpoint.exists() and not a.force:
            nested_rows.append(json.loads(checkpoint.read_text(encoding="utf-8")))
            continue

        fold_result = {
            "fold": fold_no,
            "n_train": len(outer_train),
            "n_test": len(outer_test),
            "models": {},
        }
        for label, y in targets.items():
            params, best, table = tune(
                X, y, work, outer_train, a.inner_folds, a,
                f"outer {fold_no} {label}",
                a.seed + fold_no * 1000 + (label == "outcome"),
            )
            m = model(params, a, a.seed + fold_no)
            m.fit(X[outer_train], y[outer_train])
            pred = m.predict(X[outer_test])
            fold_result["models"][label] = {
                "params": params,
                "inner_best": best,
                "outer_rmse": math.sqrt(mean_squared_error(y[outer_test], pred)),
                "outer_mae": mean_absolute_error(y[outer_test], pred),
                "outer_r2": r2_score(y[outer_test], pred),
            }
            del m, pred
            gc.collect()
            table.to_csv(out / f"search_outer{fold_no}_{label}.csv", index=False)

        checkpoint.write_text(json.dumps(fold_result, indent=2), encoding="utf-8")
        nested_rows.append(fold_result)

    # Final global tuning on all complete rows (frozen parameters for the primary DML).
    final = {}
    for label, y in targets.items():
        params, best, table = tune(
            X, y, work, all_idx, a.inner_folds, a,
            f"final {label}",
            a.seed + 9000 + (label == "outcome"),
        )
        final[label] = {"params": params, "inner_best": best}
        table.to_csv(out / f"search_final_{label}.csv", index=False)

    nested_summary = {}
    for label in targets:
        values = [row["models"][label] for row in nested_rows]
        nested_summary[label] = {
            metric: {
                "mean": float(np.mean([v[metric] for v in values])),
                "std": float(np.std([v[metric] for v in values], ddof=1))
                       if len(values) > 1 else 0.0,
            }
            for metric in ("outer_rmse", "outer_mae", "outer_r2")
        }

    summary = {
        "script": Path(__file__).name,
        "input": str(a.input),
        "n_complete_rows": len(work),
        "controls": cs,
        "control_set": a.control_set,
        "outer_folds": a.outer_folds,
        "inner_folds": a.inner_folds,
        "n_iter": a.n_iter,
        "seed": a.seed,
        "device_requested": a.device,
        "xgboost_version": xgb.__version__,
        "nested_outer_results": nested_rows,
        "nested_outer_summary": nested_summary,
        "final_tuned": final,
        "elapsed_seconds": time.time() - started,
        "method_note": (
            "Outer predictions are never used for parameter selection; "
            "final_tuned is frozen before primary DML."
        ),
    }
    (out / "dml_hyperopt_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Compact parameter file consumable by the DML main program.
    fold_params = {
        str(row["fold"]): {label: row["models"][label]["params"] for label in targets}
        for row in nested_rows
    }
    dml_params = {
        "treatment": final["treatment"]["params"],
        "outcome": final["outcome"]["params"],
        "fold_params": fold_params,
        "metadata": {
            "source": Path(__file__).name,
            "seed": a.seed,
            "control_set": a.control_set,
            "fold_params_are_nested": True,
        },
    }
    (out / "dml_xgb_params.json").write_text(
        json.dumps(dml_params, indent=2), encoding="utf-8"
    )

    print("\nNested tuning complete. Results:", out / "dml_hyperopt_results.json")
    print("Frozen parameters for DML:", out / "dml_xgb_params.json")


if __name__ == "__main__":
    main()