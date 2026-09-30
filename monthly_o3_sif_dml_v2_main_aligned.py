#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calendar-month heterogeneity DML: O3 -> SIF
Aligned with reviewer-compact-dml-full-rerun-v6.0
====================================================

Scientific goal
---------------
Estimate how the contemporaneous adjusted O3 -> SIF slope differs by calendar
month while preserving the year-stratified, spatial-blocked DML logic of the
current main program.

Design
------
1. For each year (2000-2022) and calendar month (1-12), fit a separate
   5-fold spatial-blocked DML model using only that year-month cell.

   - nuisance learners: XGBoost for O3 and SIF
   - controls: main-program Model A, except month fixed effects are NOT added
     because calendar month is constant inside each year-month cell
   - LCCS: nominal categorical variable, one-hot encoded at model-matrix
     construction; raw class codes are never passed to XGBoost as an ordered
     predictor
   - cell-level inference: spatial-cluster robust

2. For each calendar month, pool orthogonal score components across valid years
   and estimate the month-specific effect using spatial x year two-way clustered
   inference.

3. Treat Webb wild YEAR-cluster inference as a small-year-cluster sensitivity
   for the pooled month-specific effects. The primary pooled inference remains
   spatial x year two-way clustered.

4. Export O3 identifying-variation/support diagnostics, including residual SD,
   residual quantiles, denominator per observation, and information
   concentration in the largest residuals.

5. Test calendar-month heterogeneity jointly and report all pairwise month
   contrasts with Holm multiplicity correction.

Important alignment decisions
-----------------------------
- Reference spatial block factor = 20, 5 spatial folds.
- Frozen nuisance hyperparameters are inherited from the current main program
  and are NOT re-tuned here.
- Model A is the primary covariate specification.
- No random-effects meta-analysis is used; the current main program explicitly
  removed random-effects meta-analysis from its evidence chain.
- Month fixed effects from the annual main model are not inserted here because
  each fitted year-month cell contains only one calendar month.

Main outputs
------------
1)  month_year_effects.csv
2)  month_year_fold_diagnostics.csv
3)  month_year_spatial_components.csv
4)  year_month_support_diagnostics.csv
5)  calendar_month_pooled_effects.csv
6)  calendar_month_support_summary.csv
7)  calendar_month_wild_year_sensitivity.csv
8)  calendar_month_heterogeneity_omnibus.csv
9)  calendar_month_pairwise_heterogeneity.csv
10) skipped_year_month_cells.csv
11) MONTHLY_DML_MASTER_SUMMARY.csv
12) MONTHLY_DML_METADATA.json
13) figures/Figure_Monthly_Year_Heatmap.png
14) figures/Figure_CalendarMonth_Pooled_Forest.png

Jupyter
-------
    %run /root/autodl-tmp/wql/0922/monthly_o3_sif_dml_v2_main_aligned.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install xgboost before running this script."
    ) from exc


# =============================================================================
# 1. CONFIG
# =============================================================================

INPUT_FILE = r"/root/autodl-tmp/wq/matched_data_albers_all1.csv"
OUTPUT_DIR = r"/root/autodl-tmp/wql/0922/monthly_o3_sif_dml_v2_main_aligned"

OUTCOME = "SIF"
EXPOSURE = "O3"
X_COL = "x"
Y_COL = "y"
YEAR_COL = "year"
MONTH_COL = "month"

START_YEAR = 2000
END_YEAR = 2022

SEED = 42
DEVICE = "cuda"
N_JOBS = 8
MAX_BIN = 256

N_SPATIAL_FOLDS = 5
SPATIAL_BLOCK_FACTOR = 20

# Pre-specified cell support checks retained from the previous monthly program.
MIN_ROWS_PER_CELL = 20_000
MIN_SPATIAL_BLOCKS_PER_CELL = 15
MIN_VALID_YEARS_PER_MONTH_POOL = 10

# Same bootstrap scale as the current main program.
WILD_REPS = 9999
WILD_SEED = 20260921

# True = recompute every year-month cell from scratch, matching the main program's
# FULL RERUN philosophy. If a long run is interrupted, changing this to False is
# safe because cache reuse is guarded by a configuration signature.
FORCE_RERUN = True
FIG_DPI = 600

# Current main-program Model A. Calendar-month fixed effects are intentionally
# absent here because month is fixed inside every year-month cell.
MODEL_A = [
    "DEM", "lccs", "t2m", "ssrd", "tp", "u10", "v10", "sp", "stl1", "swvl1"
]
CONTROL_VARS = [X_COL, Y_COL] + MODEL_A

# LCCS is configured globally from observed finite codes in load_data().
LCCS_LEVELS = tuple()
LCCS_REFERENCE = None

# Frozen nuisance-model hyperparameters from the same prior blocked tuning stage
# used by reviewer-compact-dml-full-rerun-v6.0.
HYPERPARAMS = {
    "treatment": {
        "n_estimators": 1053,
        "max_depth": 9,
        "learning_rate": 0.10216433662775991,
        "min_child_weight": 25,
        "gamma": 4.309033817797793,
        "subsample": 0.8908973518801342,
        "colsample_bytree": 0.8492767336660302,
        "reg_alpha": 0.003354690482344186,
        "reg_lambda": 2.033716357895456,
    },
    "outcome": {
        "n_estimators": 1131,
        "max_depth": 10,
        "learning_rate": 0.0266047322246523,
        "min_child_weight": 3,
        "gamma": 0.5110259054946181,
        "subsample": 0.7833701909759339,
        "colsample_bytree": 0.9168109399871214,
        "reg_alpha": 0.007222709294055777,
        "reg_lambda": 1.912111490897236,
    },
}

SCRIPT_VERSION = "monthly-o3-sif-dml-main-aligned-v2.0"
MAIN_PROGRAM_VERSION = "reviewer-compact-dml-full-rerun-v6.0"


# =============================================================================
# 2. BASIC HELPERS
# =============================================================================

def cleanup_gpu():
    gc.collect()
    try:
        import cupy as cp  # type: ignore
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def month_number(s):
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(
            s.astype(str).str.extract(r"(\d{1,2})$")[0],
            errors="coerce",
        )
    s = pd.to_numeric(s, errors="coerce")
    bad = s.isna() | (~s.between(1, 12))
    if bad.any():
        raise ValueError(
            f"{MONTH_COL} contains {int(bad.sum()):,} invalid values; month must be 1..12."
        )
    return s.astype(np.int16)


def safe_r2(y_true, y_pred):
    if len(y_true) < 2 or float(np.std(y_true)) == 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    residual = y_true.astype(np.float64) - y_pred.astype(np.float64)
    return {
        "r2": safe_r2(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(residual)),
    }


def t_pvalue(t_value, df):
    if not np.isfinite(t_value):
        return np.nan
    return float(2.0 * stats.t.sf(abs(t_value), df))


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if len(vals) == 0:
        return out

    order = np.argsort(vals)
    sorted_p = vals[order]
    m = len(sorted_p)
    adjusted_sorted = np.maximum.accumulate(
        [(m - i) * sorted_p[i] for i in range(m)]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    out[np.flatnonzero(ok)] = adjusted
    return out


def residual_support_metrics(dr):
    """Identifying-variation/support diagnostics copied from the main-program logic."""
    d = np.asarray(dr, dtype=np.float64)
    if len(d) < 2:
        raise ValueError("need at least two treatment residuals")

    q = np.quantile(d, [0.01, 0.05, 0.50, 0.95, 0.99])
    a = np.abs(d)
    den = float(d @ d)
    q95a, q99a = np.quantile(a, [0.95, 0.99])

    return {
        "residual_mean": float(d.mean()),
        "residual_sd": float(d.std(ddof=1)),
        "residual_q01": float(q[0]),
        "residual_q05": float(q[1]),
        "residual_median": float(q[2]),
        "residual_q95": float(q[3]),
        "residual_q99": float(q[4]),
        "denominator_raw": den,
        "denominator_per_observation": den / len(d),
        "top5pct_abs_residual_denominator_share": (
            float(np.square(d[a >= q95a]).sum() / den) if den > 0 else np.nan
        ),
        "top1pct_abs_residual_denominator_share": (
            float(np.square(d[a >= q99a]).sum() / den) if den > 0 else np.nan
        ),
    }


# =============================================================================
# 3. LCCS NOMINAL ENCODING — MATCH CURRENT MAIN PROGRAM
# =============================================================================

def _format_lccs_level(v):
    x = float(v)
    if np.isfinite(x) and x.is_integer():
        return str(int(x))
    return (f"{x:g}").replace("-", "m").replace(".", "p")


def configure_lccs_encoding(df):
    """
    Configure a deterministic global one-hot representation for nominal LCCS.

    The smallest observed finite code is used as the reference category solely
    to avoid a redundant dummy column. This does not impose any ordering.
    """
    global LCCS_LEVELS, LCCS_REFERENCE

    if "lccs" not in df.columns:
        LCCS_LEVELS = tuple()
        LCCS_REFERENCE = None
        return {
            "mode": "absent",
            "levels": [],
            "reference": None,
            "n_dummy_features": 0,
        }

    vals = pd.to_numeric(df["lccs"], errors="coerce")
    levels = np.sort(vals[np.isfinite(vals)].unique().astype(np.float64))
    if len(levels) == 0:
        LCCS_LEVELS = tuple()
        LCCS_REFERENCE = None
        return {
            "mode": "present_but_no_finite_levels",
            "levels": [],
            "reference": None,
            "n_dummy_features": 0,
        }

    LCCS_LEVELS = tuple(float(v) for v in levels)
    LCCS_REFERENCE = float(levels[0])
    return {
        "mode": "nominal_one_hot",
        "levels": [float(v) for v in LCCS_LEVELS],
        "reference": float(LCCS_REFERENCE),
        "n_dummy_features": max(0, len(LCCS_LEVELS) - 1),
        "missing_count": int(vals.isna().sum()),
        "note": "Raw LCCS codes are never used as a continuous/ordered predictor.",
    }


def expanded_control_names(controls):
    controls = list(controls)
    names = [c for c in controls if c != "lccs"]
    if "lccs" in controls:
        if LCCS_REFERENCE is None or len(LCCS_LEVELS) == 0:
            raise RuntimeError(
                "LCCS requested as a control before categorical encoding was configured."
            )
        names.extend(
            f"_lccs_cat_{_format_lccs_level(lv)}"
            for lv in LCCS_LEVELS
            if float(lv) != float(LCCS_REFERENCE)
        )
    return names


def make_design_matrix(work, controls):
    """Build the dense float32 XGBoost matrix with LCCS one-hot encoded."""
    controls = list(controls)
    numeric = [c for c in controls if c != "lccs"]
    feature_names = expanded_control_names(controls)

    X = np.empty((len(work), len(feature_names)), dtype=np.float32)
    j = 0

    if numeric:
        arr = work[numeric].to_numpy(np.float32)
        X[:, :len(numeric)] = arr
        j = len(numeric)
        del arr

    if "lccs" in controls:
        vals = pd.to_numeric(work["lccs"], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(vals).all():
            raise ValueError(
                "LCCS contains missing/non-finite values after complete-case filtering."
            )
        for lv in LCCS_LEVELS:
            if float(lv) == float(LCCS_REFERENCE):
                continue
            X[:, j] = (vals == float(lv))
            j += 1

    if j != X.shape[1]:
        raise RuntimeError("design-matrix feature count mismatch")

    return X, feature_names


def lccs_encoding_metadata():
    return {
        "mode": "nominal_one_hot" if LCCS_REFERENCE is not None else "absent",
        "levels": [float(v) for v in LCCS_LEVELS],
        "reference": None if LCCS_REFERENCE is None else float(LCCS_REFERENCE),
        "n_dummy_features": max(0, len(LCCS_LEVELS) - 1),
        "feature_names": (
            []
            if LCCS_REFERENCE is None
            else [
                f"_lccs_cat_{_format_lccs_level(v)}"
                for v in LCCS_LEVELS
                if float(v) != float(LCCS_REFERENCE)
            ]
        ),
        "interpretation": (
            "Nominal categorical adjustment; no numerical ordering of LCCS class codes is assumed."
        ),
    }


def config_signature():
    payload = {
        "version": SCRIPT_VERSION,
        "main_program_version": MAIN_PROGRAM_VERSION,
        "controls_raw": CONTROL_VARS,
        "controls_expanded": expanded_control_names(CONTROL_VARS),
        "lccs_encoding": lccs_encoding_metadata(),
        "hyperparameters": HYPERPARAMS,
        "years": [START_YEAR, END_YEAR],
        "folds": N_SPATIAL_FOLDS,
        "block_factor": SPATIAL_BLOCK_FACTOR,
        "min_rows": MIN_ROWS_PER_CELL,
        "min_spatial_blocks": MIN_SPATIAL_BLOCKS_PER_CELL,
        "min_valid_years_per_month": MIN_VALID_YEARS_PER_MONTH_POOL,
        "wild_reps": WILD_REPS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


# =============================================================================
# 4. DATA LOADING + GRID/BLOCKS
# =============================================================================

def infer_grid_spacing(df):
    xs = np.sort(df[X_COL].dropna().unique())
    ys = np.sort(df[Y_COL].dropna().unique())

    dxs = np.diff(xs.astype(np.float64))
    dys = np.diff(ys.astype(np.float64))
    dxs = dxs[np.isfinite(dxs) & (dxs > 0)]
    dys = dys[np.isfinite(dys) & (dys > 0)]

    if len(dxs) == 0 or len(dys) == 0:
        raise ValueError("Unable to infer positive grid spacing from x/y coordinates.")

    return (
        float(np.median(dxs)),
        float(np.median(dys)),
        float(df[X_COL].min()),
        float(df[Y_COL].min()),
    )


def spatial_block_ids(x, y, dx, dy, x0, y0):
    bx = np.floor(
        (x.astype(np.float64, copy=False) - x0) / (dx * SPATIAL_BLOCK_FACTOR)
    ).astype(np.int32)
    by = np.floor(
        (y.astype(np.float64, copy=False) - y0) / (dy * SPATIAL_BLOCK_FACTOR)
    ).astype(np.int32)
    return bx.astype(np.int64) * np.int64(10_000_000) + by.astype(np.int64)


def load_data():
    header = pd.read_csv(INPUT_FILE, nrows=0)
    available = set(header.columns)

    required = {OUTCOME, EXPOSURE, X_COL, Y_COL, YEAR_COL, MONTH_COL}
    missing_required = sorted(required - available)
    if missing_required:
        raise ValueError(f"Missing required columns in main file: {missing_required}")

    # Robust alias handling exactly in the spirit of the current main program.
    lccs_source = "lccs" if "lccs" in available else (
        "LCCS" if "LCCS" in available else None
    )
    if lccs_source is None:
        raise ValueError("Model A requires LCCS, but neither 'lccs' nor 'LCCS' exists.")

    source_controls = []
    for c in CONTROL_VARS:
        if c == "lccs":
            source_controls.append(lccs_source)
        elif c in available:
            source_controls.append(c)
        else:
            raise ValueError(
                f"Model A control '{c}' is missing from the raw panel; refusing to silently change the specification."
            )

    needed = list(dict.fromkeys(
        [OUTCOME, EXPOSURE, X_COL, Y_COL, YEAR_COL, MONTH_COL] + source_controls
    ))

    print(f"Reading {len(needed)} columns from raw panel...")
    t0 = time.time()
    df = pd.read_csv(INPUT_FILE, usecols=needed, low_memory=False)
    print(f"Rows read: {len(df):,}; time={(time.time()-t0)/60:.1f} min")

    if lccs_source == "LCCS" and "lccs" not in df.columns:
        df = df.rename(columns={"LCCS": "lccs"})

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="raise").astype(np.int16)
    df[MONTH_COL] = month_number(df[MONTH_COL])
    df = df.loc[df[YEAR_COL].between(START_YEAR, END_YEAR)].copy()

    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce").astype(np.float64)
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce").astype(np.float64)
    for c in df.columns:
        if c not in {X_COL, Y_COL, YEAR_COL, MONTH_COL}:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    lccs_info = configure_lccs_encoding(df)
    print(
        "LCCS encoding: nominal one-hot; "
        f"levels={len(LCCS_LEVELS)}, reference={LCCS_REFERENCE}, "
        f"dummy_features={max(0, len(LCCS_LEVELS)-1)}, "
        f"missing={lccs_info.get('missing_count', 0):,}"
    )

    # Same panel-key sanity check used by the main program.
    df["_time_index"] = (
        df[YEAR_COL].astype(np.int32) * 12 + df[MONTH_COL].astype(np.int32)
    )
    if df.duplicated([X_COL, Y_COL, "_time_index"]).any():
        raise ValueError("Duplicate x/y/month panel keys detected.")

    dx, dy, x0, y0 = infer_grid_spacing(df)
    df["_spatial_block"] = spatial_block_ids(
        df[X_COL].to_numpy(np.float64),
        df[Y_COL].to_numpy(np.float64),
        dx, dy, x0, y0,
    )

    df[X_COL] = df[X_COL].astype(np.float32)
    df[Y_COL] = df[Y_COL].astype(np.float32)

    print(
        f"Grid dx={dx:.3f}, dy={dy:.3f}; "
        f"reference block ≈ {dx * SPATIAL_BLOCK_FACTOR / 1000:.2f} km"
    )
    print(
        f"Model A raw controls={len(CONTROL_VARS)}; "
        f"expanded XGBoost features={len(expanded_control_names(CONTROL_VARS))}"
    )

    return df, (dx, dy, x0, y0)


# =============================================================================
# 5. GLOBAL SPATIAL FOLD MAP
# =============================================================================

def balanced_fold_map(blocks):
    counts = pd.Series(blocks, copy=False).value_counts(sort=False)
    items = [(int(k), int(v)) for k, v in counts.items()]

    rng = np.random.default_rng(SEED)
    rng.shuffle(items)
    items.sort(key=lambda kv: kv[1], reverse=True)

    totals = np.zeros(N_SPATIAL_FOLDS, dtype=np.int64)
    mapping = {}

    for block, count in items:
        f = int(np.argmin(totals))
        mapping[block] = f
        totals[f] += count

    print(
        "Global spatial fold totals: "
        + ", ".join([f"F{i+1}={int(x):,}" for i, x in enumerate(totals)])
    )
    print(f"Global spatial blocks: {len(mapping):,}")
    return mapping


# =============================================================================
# 6. XGBOOST + CELL-LEVEL DML
# =============================================================================

def make_model(stage, year, month, fold):
    params = dict(HYPERPARAMS[stage])
    params.update(
        {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": DEVICE,
            "max_bin": MAX_BIN,
            "n_jobs": N_JOBS,
            "random_state": SEED + int(year) * 100 + int(month) * 10 + int(fold),
            "verbosity": 0,
            "validate_parameters": True,
        }
    )
    return xgb.XGBRegressor(**params)


def cluster_meat(score, cluster):
    codes, uniques = pd.factorize(cluster, sort=False)
    G = len(uniques)
    if G < 2:
        return np.nan, G

    sums = np.bincount(
        codes,
        weights=np.asarray(score, dtype=np.float64),
        minlength=G,
    )
    meat = float(sums @ sums)
    meat *= G / (G - 1)
    return meat, G


def single_cluster_inference(dr, yr, block):
    """Spatial-cluster robust inference inside one fixed year-month cell."""
    d = np.asarray(dr, dtype=np.float64)
    y = np.asarray(yr, dtype=np.float64)

    numerator = float(d @ y)
    denominator = float(d @ d)
    if denominator <= 0:
        raise RuntimeError("Non-positive DML denominator in year-month cell.")

    theta = numerator / denominator
    score = d * (y - theta * d)

    meat, g = cluster_meat(score, block)
    if not np.isfinite(meat) or meat <= 0:
        raise RuntimeError("Failed to compute positive spatial-cluster variance.")

    var = meat / denominator**2
    se = math.sqrt(var)
    df = max(1, g - 1)
    t_stat = theta / se
    crit = float(stats.t.ppf(0.975, df))

    return {
        "estimate_raw": theta,
        "effect_per_10ug_m3": theta * 10.0,
        "se_spatial_cluster_raw": se,
        "se_spatial_cluster_per_10ug_m3": se * 10.0,
        "ci95_low_t_per_10ug_m3": (theta - crit * se) * 10.0,
        "ci95_high_t_per_10ug_m3": (theta + crit * se) * 10.0,
        "p_value_t": t_pvalue(t_stat, df),
        "t_df": df,
        "n_spatial_clusters": g,
        "dml_numerator_raw": numerator,
        "dml_denominator_raw": denominator,
    }


def fit_year_month_cell(work, year, month, fold_map):
    fold_series = work["_spatial_block"].map(fold_map)
    if fold_series.isna().any():
        raise RuntimeError("Unmapped spatial blocks found.")

    fold = fold_series.to_numpy(dtype=np.int8)
    if np.unique(fold).size < N_SPATIAL_FOLDS:
        raise RuntimeError("Not all five spatial folds are represented in this cell.")

    X, feature_names = make_design_matrix(work, CONTROL_VARS)
    t = work[EXPOSURE].to_numpy(dtype=np.float32, copy=False)
    y = work[OUTCOME].to_numpy(dtype=np.float32, copy=False)
    block = work["_spatial_block"].to_numpy(dtype=np.int64, copy=False)

    t_hat = np.full(len(work), np.nan, dtype=np.float32)
    y_hat = np.full(len(work), np.nan, dtype=np.float32)
    diag_rows = []

    for f in range(N_SPATIAL_FOLDS):
        test = np.flatnonzero(fold == f)
        train = np.flatnonzero(fold != f)

        if len(test) == 0 or len(train) == 0:
            raise RuntimeError(f"Invalid spatial fold {f+1} in {year}-{month:02d}.")

        for stage, target, pred_dest in [
            ("treatment", t, t_hat),
            ("outcome", y, y_hat),
        ]:
            mdl = make_model(stage, year, month, f)
            mdl.fit(X[train], target[train])
            pred = mdl.predict(X[test]).astype(np.float32)
            pred_dest[test] = pred

            q = regression_metrics(target[test], pred)
            diag_rows.append(
                {
                    "year": int(year),
                    "month": int(month),
                    "stage": stage,
                    "spatial_fold": f + 1,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_test_blocks": int(np.unique(block[test]).size),
                    "r2": q["r2"],
                    "rmse": q["rmse"],
                    "mae": q["mae"],
                    "mean_residual_bias": q["bias"],
                }
            )

            del mdl, pred
            cleanup_gpu()

    if not np.isfinite(t_hat).all() or not np.isfinite(y_hat).all():
        raise RuntimeError("Incomplete OOF predictions in this year-month cell.")

    dr = (t - t_hat).astype(np.float32)
    yr = (y - y_hat).astype(np.float32)

    infer = single_cluster_inference(dr, yr, block)
    support = residual_support_metrics(dr)
    tm = regression_metrics(t, t_hat)
    ym = regression_metrics(y, y_hat)

    diagnostics = pd.DataFrame(diag_rows)
    tdiag = diagnostics.loc[diagnostics["stage"] == "treatment"]
    ydiag = diagnostics.loc[diagnostics["stage"] == "outcome"]

    result = {
        "year": int(year),
        "month": int(month),
        "n": int(len(work)),
        "n_spatial_blocks": int(np.unique(block).size),
        "n_controls_raw": int(len(CONTROL_VARS)),
        "n_controls_expanded": int(len(feature_names)),
        "controls": ";".join(feature_names),
        "control_specification": (
            "Main-program Model A; LCCS nominal one-hot; month/year fixed by stratification"
        ),
        **infer,
        "treatment_oof_r2": tm["r2"],
        "treatment_oof_rmse": tm["rmse"],
        "treatment_oof_mae": tm["mae"],
        "treatment_residual_mean": float(dr.mean()),
        "treatment_residual_sd": float(dr.std(ddof=1)),
        "treatment_min_fold_r2": float(tdiag["r2"].min()),
        "treatment_max_abs_fold_bias": float(tdiag["mean_residual_bias"].abs().max()),
        "outcome_oof_r2": ym["r2"],
        "outcome_oof_rmse": ym["rmse"],
        "outcome_oof_mae": ym["mae"],
        "outcome_residual_mean": float(yr.mean()),
        "outcome_residual_sd": float(yr.std(ddof=1)),
        "outcome_min_fold_r2": float(ydiag["r2"].min()),
        "outcome_max_abs_fold_bias": float(ydiag["mean_residual_bias"].abs().max()),
        "treatment_residual_q01": support["residual_q01"],
        "treatment_residual_q05": support["residual_q05"],
        "treatment_residual_median": support["residual_median"],
        "treatment_residual_q95": support["residual_q95"],
        "treatment_residual_q99": support["residual_q99"],
        "denominator_per_observation": support["denominator_per_observation"],
        "top5pct_abs_residual_denominator_share": support[
            "top5pct_abs_residual_denominator_share"
        ],
        "top1pct_abs_residual_denominator_share": support[
            "top1pct_abs_residual_denominator_share"
        ],
    }

    # Save orthogonal numerator/denominator components by space block. These are
    # sufficient for the preferred cross-year month pooling and its clustered SE.
    comp = pd.DataFrame(
        {
            "year": int(year),
            "month": int(month),
            "spatial_block": block.astype(np.int64, copy=False),
            "numerator_raw": (
                dr.astype(np.float64, copy=False) * yr.astype(np.float64, copy=False)
            ),
            "denominator_raw": dr.astype(np.float64, copy=False) ** 2,
        }
    )
    comp = (
        comp.groupby(["year", "month", "spatial_block"], observed=True, sort=False)
        .agg(
            numerator_raw=("numerator_raw", "sum"),
            denominator_raw=("denominator_raw", "sum"),
            n=("numerator_raw", "size"),
        )
        .reset_index()
    )

    del X, t, y, t_hat, y_hat, dr, yr
    cleanup_gpu()

    return result, diagnostics, comp


# =============================================================================
# 7. MONTH-SPECIFIC POOLED DML + TWO-WAY CLUSTERING
# =============================================================================

def pooled_month_dml(components, month, month_year_results):
    sub = components.loc[components["month"] == month].copy()
    annual = month_year_results.loc[month_year_results["month"] == month].copy()

    numerator = float(sub["numerator_raw"].sum())
    denominator = float(sub["denominator_raw"].sum())
    n_total = int(sub["n"].sum())
    if denominator <= 0:
        raise RuntimeError(f"Month {month}: non-positive pooled DML denominator.")

    theta = numerator / denominator
    sub["score"] = (
        sub["numerator_raw"].astype(float)
        - theta * sub["denominator_raw"].astype(float)
    )

    space = (
        sub.groupby("spatial_block", observed=True)["score"]
        .sum()
        .to_numpy(float)
    )
    year = (
        sub.groupby("year", observed=True)["score"]
        .sum()
        .to_numpy(float)
    )
    inter = sub["score"].to_numpy(float)

    gs = len(space)
    gy = len(year)
    gi = len(inter)
    if gs < 2 or gy < 2 or gi < 2:
        raise RuntimeError(
            f"Month {month}: insufficient clusters for pooled two-way inference."
        )

    meat_space = float(space @ space) * gs / (gs - 1)
    meat_year = float(year @ year) * gy / (gy - 1)
    meat_inter = float(inter @ inter) * gi / (gi - 1)

    var = (meat_space + meat_year - meat_inter) / denominator**2
    fallback = False
    if not np.isfinite(var) or var <= 0:
        var = max(meat_space, meat_year) / denominator**2
        fallback = True

    se = math.sqrt(var)
    df = max(1, min(gs, gy) - 1)
    crit_t = float(stats.t.ppf(0.975, df))
    t_stat = theta / se

    negative_years = int((annual["effect_per_10ug_m3"] < 0).sum())
    valid_years = int(len(annual))

    return {
        "month": int(month),
        "month_name": calendar_month_name(month),
        "n": n_total,
        "n_valid_years": valid_years,
        "negative_years": negative_years,
        "annual_sign_test_p": (
            float(
                stats.binomtest(
                    negative_years,
                    valid_years,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if valid_years > 0
            else np.nan
        ),
        "median_yearly_effect_per_10ug_m3": float(
            annual["effect_per_10ug_m3"].median()
        ),
        "estimate_raw": theta,
        "effect_per_10ug_m3": theta * 10.0,
        "se_space_year_raw": se,
        "se_space_year_per_10ug_m3": se * 10.0,
        "ci95_low_normal_per_10ug_m3": (theta - 1.96 * se) * 10.0,
        "ci95_high_normal_per_10ug_m3": (theta + 1.96 * se) * 10.0,
        "p_value_normal": float(2 * stats.norm.sf(abs(t_stat))),
        "ci95_low_t_per_10ug_m3": (theta - crit_t * se) * 10.0,
        "ci95_high_t_per_10ug_m3": (theta + crit_t * se) * 10.0,
        "p_value_t": t_pvalue(t_stat, df),
        "t_df": df,
        "n_spatial_clusters": gs,
        "n_year_clusters": gy,
        "n_intersections": gi,
        "variance_fallback": fallback,
        "dml_numerator_raw": numerator,
        "dml_denominator_raw": denominator,
        "denominator_per_observation": denominator / n_total,
        "mean_treatment_oof_r2": float(annual["treatment_oof_r2"].mean()),
        "min_treatment_oof_r2": float(annual["treatment_oof_r2"].min()),
        "mean_outcome_oof_r2": float(annual["outcome_oof_r2"].mean()),
        "min_outcome_oof_r2": float(annual["outcome_oof_r2"].min()),
        "median_treatment_residual_sd": float(
            annual["treatment_residual_sd"].median()
        ),
        "min_treatment_residual_sd": float(
            annual["treatment_residual_sd"].min()
        ),
        "median_denominator_per_observation": float(
            annual["denominator_per_observation"].median()
        ),
        "min_denominator_per_observation": float(
            annual["denominator_per_observation"].min()
        ),
        "median_top5pct_abs_residual_denominator_share": float(
            annual["top5pct_abs_residual_denominator_share"].median()
        ),
        "median_top1pct_abs_residual_denominator_share": float(
            annual["top1pct_abs_residual_denominator_share"].median()
        ),
    }


# =============================================================================
# 8. POOLED WILD YEAR-CLUSTER SENSITIVITY — SAME MAIN-PROGRAM LOGIC
# =============================================================================

def wild_year_cluster(comp, label, seed):
    """
    Webb wild YEAR-cluster sensitivity for a pooled calendar-month effect.

    This follows the current main-program pooled wild-year routine. It is a
    sensitivity analysis only; primary pooled uncertainty remains spatial x year
    two-way clustered.
    """
    y = (
        comp.groupby("year", observed=True)
        .agg(num=("numerator_raw", "sum"), den=("denominator_raw", "sum"))
        .reset_index()
    )
    N = y["num"].to_numpy(float)
    D = y["den"].to_numpy(float)
    G = len(N)

    if G < 2 or float(D.sum()) <= 0:
        return {
            "analysis": label,
            "effect_per_10ug_m3": np.nan,
            "wild_cluster_p_value": np.nan,
            "wild_basic_ci95_low_per_10ug_m3": np.nan,
            "wild_basic_ci95_high_per_10ug_m3": np.nan,
            "n_year_clusters": G,
            "bootstrap_reps": WILD_REPS,
        }

    theta = float(N.sum() / D.sum())

    # Retained exactly from the main-program pooled wild-year sensitivity logic.
    R = N.copy()
    r_denom = math.sqrt((G / (G - 1)) * float(R @ R))
    obs = float(R.sum() / r_denom) if r_denom > 0 else np.nan
    U = N - theta * D

    vals = np.array(
        [
            -math.sqrt(1.5),
            -1.0,
            -math.sqrt(0.5),
            math.sqrt(0.5),
            1.0,
            math.sqrt(1.5),
        ],
        dtype=float,
    )

    rng = np.random.default_rng(int(seed))
    exc = 0
    delta = np.empty(WILD_REPS, dtype=float)
    done = 0

    while done < WILD_REPS:
        b = min(2000, WILD_REPS - done)
        W = vals[rng.integers(0, 6, size=(b, G))]
        WR = W * R
        denom_b = np.sqrt((G / (G - 1)) * np.sum(WR**2, axis=1))
        tb = np.divide(
            WR.sum(axis=1),
            denom_b,
            out=np.full(b, np.nan, dtype=float),
            where=denom_b > 0,
        )
        if np.isfinite(obs):
            exc += int((np.abs(tb[np.isfinite(tb)]) >= abs(obs)).sum())
        delta[done:done+b] = (W * U).sum(axis=1) / D.sum()
        done += b

    q = np.quantile(delta[np.isfinite(delta)], [0.025, 0.975])
    p = (1 + exc) / (WILD_REPS + 1) if np.isfinite(obs) else np.nan

    return {
        "analysis": label,
        "effect_per_10ug_m3": theta * 10.0,
        "wild_cluster_p_value": p,
        "wild_basic_ci95_low_per_10ug_m3": (theta - q[1]) * 10.0,
        "wild_basic_ci95_high_per_10ug_m3": (theta - q[0]) * 10.0,
        "n_year_clusters": G,
        "bootstrap_reps": WILD_REPS,
        "wild_cluster_weights": "Webb six-point",
        "wild_resampling_dimension": "calendar_year_only",
        "inference_role": (
            "small-year-cluster sensitivity; primary pooled inference is spatial x year two-way clustered"
        ),
    }


# =============================================================================
# 9. FORMAL CALENDAR-MONTH HETEROGENEITY
# =============================================================================

def month_covariance_matrix(components, pooled_df):
    eligible = pooled_df.loc[
        pooled_df["n_valid_years"] >= MIN_VALID_YEARS_PER_MONTH_POOL
    ].copy()
    months = eligible["month"].astype(int).tolist()

    if len(months) < 2:
        raise RuntimeError("Fewer than two calendar months are eligible for heterogeneity testing.")

    theta = (
        eligible.set_index("month")
        .loc[months, "estimate_raw"]
        .to_numpy(float)
    )
    D = (
        eligible.set_index("month")
        .loc[months, "dml_denominator_raw"]
        .to_numpy(float)
    )

    cells = pd.MultiIndex.from_frame(
        components[["spatial_block", "year"]].drop_duplicates()
    )
    cell_frame = cells.to_frame(index=False)
    cell_frame.columns = ["spatial_block", "year"]

    score_mat = np.zeros((len(cell_frame), len(months)), dtype=np.float64)

    for j, m in enumerate(months):
        sub = components.loc[
            components["month"] == m,
            ["spatial_block", "year", "numerator_raw", "denominator_raw"],
        ]
        merged = cell_frame.merge(
            sub,
            on=["spatial_block", "year"],
            how="left",
        ).fillna(0.0)

        score_mat[:, j] = (
            merged["numerator_raw"].to_numpy(float)
            - theta[j] * merged["denominator_raw"].to_numpy(float)
        )

    # Space-cluster meat.
    space_codes, space_uniques = pd.factorize(
        cell_frame["spatial_block"], sort=False
    )
    gs = len(space_uniques)
    space_sum = np.zeros((gs, len(months)), dtype=np.float64)
    for j in range(len(months)):
        space_sum[:, j] = np.bincount(
            space_codes,
            weights=score_mat[:, j],
            minlength=gs,
        )
    meat_space = space_sum.T @ space_sum
    if gs > 1:
        meat_space *= gs / (gs - 1)

    # Year-cluster meat.
    year_codes, year_uniques = pd.factorize(cell_frame["year"], sort=False)
    gy = len(year_uniques)
    year_sum = np.zeros((gy, len(months)), dtype=np.float64)
    for j in range(len(months)):
        year_sum[:, j] = np.bincount(
            year_codes,
            weights=score_mat[:, j],
            minlength=gy,
        )
    meat_year = year_sum.T @ year_sum
    if gy > 1:
        meat_year *= gy / (gy - 1)

    # Space x year intersection meat.
    gi = len(score_mat)
    meat_inter = score_mat.T @ score_mat
    if gi > 1:
        meat_inter *= gi / (gi - 1)

    Hinv = np.diag(1.0 / D)
    cov = Hinv @ (meat_space + meat_year - meat_inter) @ Hinv
    cov = 0.5 * (cov + cov.T)

    eigval, eigvec = np.linalg.eigh(cov)
    psd_adjusted = bool(np.any(eigval < -1e-14))
    if psd_adjusted:
        eigval = np.clip(eigval, 0.0, None)
        cov = (eigvec * eigval) @ eigvec.T
        cov = 0.5 * (cov + cov.T)

    return months, theta, cov, gs, gy, psd_adjusted


def heterogeneity_tests(components, pooled_df):
    months, theta, cov, gs, gy, psd_adjusted = month_covariance_matrix(
        components, pooled_df
    )

    M = len(months)
    R = np.zeros((M - 1, M), dtype=np.float64)
    for i in range(1, M):
        R[i - 1, i] = 1.0
        R[i - 1, 0] = -1.0

    diff = R @ theta
    V = R @ cov @ R.T
    q = M - 1
    F = float(diff.T @ np.linalg.pinv(V) @ diff / q)
    df2 = max(1, min(gs, gy) - 1)
    p_omnibus = float(stats.f.sf(F, q, df2))

    omnibus = pd.DataFrame(
        [
            {
                "test": "Calendar-month omnibus heterogeneity",
                "null_hypothesis": "All pooled calendar-month effects are equal",
                "n_months_tested": M,
                "months_tested": ";".join([str(m) for m in months]),
                "wald_F": F,
                "df1": q,
                "df2": df2,
                "p_value": p_omnibus,
                "covariance_psd_adjusted": psd_adjusted,
                "inference": "space + year clustered covariance across month-specific orthogonal scores",
            }
        ]
    )

    rows = []
    for i in range(M):
        for j in range(i + 1, M):
            w = np.zeros(M)
            w[i] = 1.0
            w[j] = -1.0

            est = float(w @ theta)
            var = float(w @ cov @ w)
            se = math.sqrt(max(var, 0.0))
            tstat = est / se if se > 0 else np.nan
            crit = float(stats.t.ppf(0.975, df2))

            rows.append(
                {
                    "month_1": months[i],
                    "month_1_name": calendar_month_name(months[i]),
                    "month_2": months[j],
                    "month_2_name": calendar_month_name(months[j]),
                    "difference_month1_minus_month2_per_10ug_m3": est * 10.0,
                    "se_difference_per_10ug_m3": se * 10.0,
                    "ci95_low_per_10ug_m3": (est - crit * se) * 10.0,
                    "ci95_high_per_10ug_m3": (est + crit * se) * 10.0,
                    "t_statistic": tstat,
                    "df": df2,
                    "p_value_raw": t_pvalue(tstat, df2),
                }
            )

    pairwise = pd.DataFrame(rows)
    pairwise["p_value_holm"] = holm_adjust(
        pairwise["p_value_raw"].to_numpy(float)
    )
    pairwise["significant_holm_0.05"] = pairwise["p_value_holm"] < 0.05

    return omnibus, pairwise


# =============================================================================
# 10. SUMMARIES
# =============================================================================

def calendar_month_name(m):
    names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    return names[int(m)]


def make_support_tables(results_df):
    support_cols = [
        "year",
        "month",
        "month_name",
        "n",
        "n_spatial_blocks",
        "treatment_oof_r2",
        "treatment_min_fold_r2",
        "treatment_residual_mean",
        "treatment_residual_sd",
        "treatment_residual_q01",
        "treatment_residual_q05",
        "treatment_residual_median",
        "treatment_residual_q95",
        "treatment_residual_q99",
        "dml_denominator_raw",
        "denominator_per_observation",
        "top5pct_abs_residual_denominator_share",
        "top1pct_abs_residual_denominator_share",
    ]
    cell_support = results_df[support_cols].copy()

    month_support = (
        cell_support.groupby(["month", "month_name"], observed=True)
        .agg(
            n_valid_years=("year", "nunique"),
            n_total=("n", "sum"),
            treatment_oof_r2_mean=("treatment_oof_r2", "mean"),
            treatment_oof_r2_min=("treatment_oof_r2", "min"),
            treatment_residual_sd_median=("treatment_residual_sd", "median"),
            treatment_residual_sd_min=("treatment_residual_sd", "min"),
            treatment_residual_sd_max=("treatment_residual_sd", "max"),
            denominator_per_observation_median=("denominator_per_observation", "median"),
            denominator_per_observation_min=("denominator_per_observation", "min"),
            denominator_per_observation_max=("denominator_per_observation", "max"),
            top5pct_denominator_share_median=("top5pct_abs_residual_denominator_share", "median"),
            top1pct_denominator_share_median=("top1pct_abs_residual_denominator_share", "median"),
        )
        .reset_index()
        .sort_values("month")
    )

    return cell_support, month_support


# =============================================================================
# 11. FIGURES
# =============================================================================

def plot_heatmap(month_year_df, figure_dir):
    heat = (
        month_year_df.pivot(
            index="month",
            columns="year",
            values="effect_per_10ug_m3",
        )
        .reindex(list(range(1, 13)))
    )

    values = heat.to_numpy(float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = float(finite.max()) if len(finite) else 1.0
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    im = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(
        heat.columns.astype(int),
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(
        [calendar_month_name(i) for i in range(1, 13)],
        fontsize=9,
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Calendar month", fontsize=11)
    ax.set_title("Year-specific calendar-month adjusted O$_3$-SIF slopes", fontsize=14)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        r"Adjusted SIF slope per +10 $\mu$g m$^{-3}$ O$_3$",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(
        figure_dir / "Figure_Monthly_Year_Heatmap.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / "Figure_Monthly_Year_Heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_month_forest(pooled_df, figure_dir):
    d = pooled_df.sort_values("month").reset_index(drop=True).copy()
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    ax.axvline(0.0, linestyle="--", linewidth=1.1, zorder=1)

    for i, r in d.iterrows():
        yi = y[i]
        if i % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, alpha=0.08, zorder=0)

        ax.hlines(
            yi,
            r["ci95_low_t_per_10ug_m3"],
            r["ci95_high_t_per_10ug_m3"],
            linewidth=1.8,
            zorder=2,
        )
        ax.scatter(
            r["effect_per_10ug_m3"],
            yi,
            s=48,
            zorder=3,
        )

        ax.text(
            1.02,
            yi,
            (
                f"{r['effect_per_10ug_m3']:.4f} "
                f"({r['ci95_low_t_per_10ug_m3']:.4f} to "
                f"{r['ci95_high_t_per_10ug_m3']:.4f}); "
                f"{int(r['negative_years'])}/{int(r['n_valid_years'])} years < 0"
            ),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9.0,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(d["month_name"], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(
        r"Adjusted SIF slope per +10 $\mu$g m$^{-3}$ O$_3$ (95% CI)",
        fontsize=11.5,
    )
    ax.set_title("Calendar-month adjusted O$_3$-SIF slopes", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.text(
        0.995,
        0.985,
        "Primary uncertainty: spatial × year two-way clustered",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
    )

    plt.subplots_adjust(left=0.12, right=0.77, top=0.92, bottom=0.08)
    fig.savefig(
        figure_dir / "Figure_CalendarMonth_Pooled_Forest.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / "Figure_CalendarMonth_Pooled_Forest.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    cache_dir = out / "cell_cache"
    cache_dir.mkdir(exist_ok=True)

    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)

    print("=" * 112)
    print("CALENDAR-MONTH HETEROGENEITY DML: O3 -> SIF — MAIN-PROGRAM-ALIGNED V2")
    print("=" * 112)
    print(f"Input           : {INPUT_FILE}")
    print(f"Output          : {OUTPUT_DIR}")
    print(f"Script version  : {SCRIPT_VERSION}")
    print(f"Aligned main    : {MAIN_PROGRAM_VERSION}")
    print(f"XGBoost         : {xgb.__version__}; device={DEVICE}")
    print(f"FORCE_RERUN     : {FORCE_RERUN}")
    print("=" * 112)

    data, grid = load_data()
    fold_map = balanced_fold_map(data["_spatial_block"].to_numpy(np.int64))
    signature = config_signature()

    sample_summary = (
        data.groupby([YEAR_COL, MONTH_COL], observed=True)
        .agg(
            n=(OUTCOME, "size"),
            n_spatial_blocks=("_spatial_block", "nunique"),
            o3_mean=(EXPOSURE, "mean"),
            o3_sd=(EXPOSURE, "std"),
            sif_mean=(OUTCOME, "mean"),
            sif_sd=(OUTCOME, "std"),
        )
        .reset_index()
        .sort_values([YEAR_COL, MONTH_COL])
    )
    sample_summary.to_csv(out / "year_month_sample_summary.csv", index=False)

    results = []
    diags = []
    comps = []
    skipped = []

    years = list(range(START_YEAR, END_YEAR + 1))
    months = list(range(1, 13))
    total_tasks = len(years) * len(months)
    task = 0
    t_all = time.time()

    cols_keep = list(dict.fromkeys(
        [OUTCOME, EXPOSURE, YEAR_COL, MONTH_COL, "_spatial_block"] + CONTROL_VARS
    ))

    for year in years:
        print("\n" + "#" * 112)
        print(f"YEAR {year}")
        print("#" * 112)

        for month in months:
            task += 1

            cache_json = cache_dir / f"year_{year}_month_{month}.json"
            cache_diag = cache_dir / f"year_{year}_month_{month}_diagnostics.csv"
            cache_comp = cache_dir / f"year_{year}_month_{month}_components.csv"

            if (
                not FORCE_RERUN
                and cache_json.exists()
                and cache_diag.exists()
                and cache_comp.exists()
            ):
                cached = json.loads(cache_json.read_text(encoding="utf-8"))
                if cached.get("config_signature") == signature:
                    print(
                        f"[resume {task}/{total_tasks}] {year}-{month:02d}: "
                        f"effect={cached['result']['effect_per_10ug_m3']:.6f}"
                    )
                    results.append(cached["result"])
                    diags.append(pd.read_csv(cache_diag))
                    comps.append(pd.read_csv(cache_comp))
                    continue

            work = data.loc[
                (data[YEAR_COL] == year) & (data[MONTH_COL] == month),
                cols_keep,
            ].copy()

            work = (
                work.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[OUTCOME, EXPOSURE] + CONTROL_VARS)
                .reset_index(drop=True)
            )

            n = len(work)
            n_blocks = int(work["_spatial_block"].nunique())

            if n < MIN_ROWS_PER_CELL or n_blocks < MIN_SPATIAL_BLOCKS_PER_CELL:
                reason = (
                    f"n={n:,}, blocks={n_blocks}; "
                    f"minimum rows={MIN_ROWS_PER_CELL:,}, "
                    f"minimum blocks={MIN_SPATIAL_BLOCKS_PER_CELL}"
                )
                print(f"[skip {task}/{total_tasks}] {year}-{month:02d}: {reason}")
                skipped.append(
                    {
                        "year": year,
                        "month": month,
                        "month_name": calendar_month_name(month),
                        "n": n,
                        "n_spatial_blocks": n_blocks,
                        "reason": reason,
                    }
                )
                del work
                continue

            represented_folds = int(
                work["_spatial_block"].map(fold_map).dropna().nunique()
            )
            if represented_folds < N_SPATIAL_FOLDS:
                reason = (
                    f"Only {represented_folds}/{N_SPATIAL_FOLDS} spatial folds represented."
                )
                print(f"[skip {task}/{total_tasks}] {year}-{month:02d}: {reason}")
                skipped.append(
                    {
                        "year": year,
                        "month": month,
                        "month_name": calendar_month_name(month),
                        "n": n,
                        "n_spatial_blocks": n_blocks,
                        "reason": reason,
                    }
                )
                del work
                continue

            print(
                f"[fit {task}/{total_tasks}] {year}-{month:02d}: "
                f"n={n:,}, blocks={n_blocks:,}, "
                f"elapsed={(time.time()-t_all)/60:.1f} min"
            )

            result, diag, comp = fit_year_month_cell(
                work, year, month, fold_map
            )
            results.append(result)
            diags.append(diag)
            comps.append(comp)

            diag.to_csv(cache_diag, index=False)
            comp.to_csv(cache_comp, index=False)
            cache_json.write_text(
                json.dumps(
                    {
                        "config_signature": signature,
                        "result": result,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            del work, diag, comp
            cleanup_gpu()

    if not results:
        raise RuntimeError("No year-month DML models were successfully fitted.")

    results_df = pd.DataFrame(results).sort_values(["year", "month"]).reset_index(drop=True)
    diag_df = pd.concat(diags, ignore_index=True)
    comp_df = pd.concat(comps, ignore_index=True)

    results_df["month_name"] = results_df["month"].map(calendar_month_name)
    diag_df["month_name"] = diag_df["month"].map(calendar_month_name)
    comp_df["month_name"] = comp_df["month"].map(calendar_month_name)

    results_df.to_csv(out / "month_year_effects.csv", index=False)
    diag_df.to_csv(out / "month_year_fold_diagnostics.csv", index=False)
    comp_df.to_csv(out / "month_year_spatial_components.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "skipped_year_month_cells.csv", index=False)

    # Support / identifying-variation diagnostics.
    cell_support, month_support = make_support_tables(results_df)
    cell_support.to_csv(out / "year_month_support_diagnostics.csv", index=False)
    month_support.to_csv(out / "calendar_month_support_summary.csv", index=False)

    # Preferred month-specific pooled DML effects.
    pooled_rows = []
    wild_rows = []

    for m in months:
        sub_results = results_df.loc[results_df["month"] == m]
        if len(sub_results) < MIN_VALID_YEARS_PER_MONTH_POOL:
            print(
                f"[pool skip] month={m}: valid years={len(sub_results)} < "
                f"{MIN_VALID_YEARS_PER_MONTH_POOL}"
            )
            continue

        pooled_rows.append(pooled_month_dml(comp_df, m, results_df))

        sub_comp = comp_df.loc[comp_df["month"] == m].copy()
        wr = wild_year_cluster(
            sub_comp,
            label=f"calendar_month_{m:02d}",
            seed=WILD_SEED + int(m),
        )
        wr["month"] = int(m)
        wr["month_name"] = calendar_month_name(m)
        wild_rows.append(wr)

    pooled_df = pd.DataFrame(pooled_rows).sort_values("month").reset_index(drop=True)
    wild_df = pd.DataFrame(wild_rows).sort_values("month").reset_index(drop=True)

    if len(pooled_df) == 0:
        raise RuntimeError("No calendar month has enough valid years for pooled inference.")

    pooled_df = pooled_df.merge(
        wild_df[
            [
                "month",
                "wild_cluster_p_value",
                "wild_basic_ci95_low_per_10ug_m3",
                "wild_basic_ci95_high_per_10ug_m3",
                "n_year_clusters",
                "bootstrap_reps",
            ]
        ].rename(columns={"n_year_clusters": "wild_n_year_clusters"}),
        on="month",
        how="left",
        validate="one_to_one",
    )

    pooled_df.to_csv(out / "calendar_month_pooled_effects.csv", index=False)
    wild_df.to_csv(out / "calendar_month_wild_year_sensitivity.csv", index=False)

    omnibus, pairwise = heterogeneity_tests(comp_df, pooled_df)
    omnibus.to_csv(out / "calendar_month_heterogeneity_omnibus.csv", index=False)
    pairwise.to_csv(out / "calendar_month_pairwise_heterogeneity.csv", index=False)

    master_cols = [
        "month",
        "month_name",
        "n",
        "n_valid_years",
        "negative_years",
        "median_yearly_effect_per_10ug_m3",
        "effect_per_10ug_m3",
        "ci95_low_t_per_10ug_m3",
        "ci95_high_t_per_10ug_m3",
        "p_value_t",
        "wild_cluster_p_value",
        "wild_basic_ci95_low_per_10ug_m3",
        "wild_basic_ci95_high_per_10ug_m3",
        "mean_treatment_oof_r2",
        "min_treatment_oof_r2",
        "mean_outcome_oof_r2",
        "min_outcome_oof_r2",
        "median_treatment_residual_sd",
        "min_treatment_residual_sd",
        "denominator_per_observation",
        "median_denominator_per_observation",
        "min_denominator_per_observation",
        "median_top5pct_abs_residual_denominator_share",
        "median_top1pct_abs_residual_denominator_share",
    ]
    master = pooled_df[master_cols].copy()
    master.to_csv(out / "MONTHLY_DML_MASTER_SUMMARY.csv", index=False)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "aligned_main_program_version": MAIN_PROGRAM_VERSION,
        "input_file": INPUT_FILE,
        "output_dir": OUTPUT_DIR,
        "years": [START_YEAR, END_YEAR],
        "primary_monthly_estimand": (
            "calendar-month-specific contemporaneous adjusted O3-SIF slope obtained from year-month spatial-blocked DML and pooled across years"
        ),
        "controls_raw": CONTROL_VARS,
        "controls_expanded": expanded_control_names(CONTROL_VARS),
        "control_note": (
            "Model A from the current main program; calendar month and calendar year are fixed by year-month stratification, so no month FE or year term is added inside a cell."
        ),
        "lccs_encoding": lccs_encoding_metadata(),
        "crossfit": "5-fold spatial blocked within each year-month cell",
        "reference_spatial_block_factor": SPATIAL_BLOCK_FACTOR,
        "approx_reference_block_km": grid[0] * SPATIAL_BLOCK_FACTOR / 1000.0,
        "cell_inference": (
            "single-cluster spatial-block robust inference within each fixed year-month cell"
        ),
        "pooled_inference": (
            "spatial + year two-way clustered orthogonal-score inference within each calendar month"
        ),
        "pooled_small_cluster_sensitivity": (
            "Webb wild year-cluster sensitivity copied from the current main-program pooled wild-year logic; not the primary pooled inference"
        ),
        "heterogeneity_test": (
            "multivariate space+year clustered covariance across eligible calendar-month orthogonal-score estimates; pairwise contrasts use Holm correction"
        ),
        "support_diagnostics": (
            "treatment OOF residual scale, quantiles, denominator per observation, and top-5%/top-1% information concentration"
        ),
        "random_effects_meta_analysis": (
            "not used; deliberately removed to match the current main-program evidence chain"
        ),
        "hyperparameters": HYPERPARAMS,
        "hyperparameter_role": (
            "frozen from prior blocked tuning; no retuning on final month-heterogeneity results"
        ),
        "force_rerun": FORCE_RERUN,
        "config_signature": signature,
    }
    (out / "MONTHLY_DML_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_heatmap(results_df, figure_dir)
    plot_month_forest(pooled_df, figure_dir)

    print("\n" + "=" * 112)
    print("MONTHLY CALENDAR-MONTH DML FINISHED")
    print("=" * 112)
    print("Main outputs:")
    for name in [
        "MONTHLY_DML_MASTER_SUMMARY.csv",
        "calendar_month_pooled_effects.csv",
        "calendar_month_wild_year_sensitivity.csv",
        "calendar_month_support_summary.csv",
        "calendar_month_heterogeneity_omnibus.csv",
        "calendar_month_pairwise_heterogeneity.csv",
        "month_year_effects.csv",
        "month_year_fold_diagnostics.csv",
        "month_year_spatial_components.csv",
        "year_month_support_diagnostics.csv",
        "skipped_year_month_cells.csv",
        "MONTHLY_DML_METADATA.json",
    ]:
        print(out / name)
    print("\nFigures:")
    print(figure_dir / "Figure_Monthly_Year_Heatmap.png")
    print(figure_dir / "Figure_CalendarMonth_Pooled_Forest.png")
    print("=" * 112)


if __name__ == "__main__":
    main()
