#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCCS vegetation heterogeneity extension aligned with the 2026-09-22
reviewer-compact O3 -> SIF DML main program (v6.0).

Primary design retained here
----------------------------
- 2000-2022 year-stratified, 5-fold spatial-blocked DML.
- Same frozen global XGBoost nuisance hyperparameters as the main primary model.
- Same reference spatial block factor = 20.
- Same flexible seasonality: 11 calendar-month fixed-effect indicators
  (January reference), not sine/cosine harmonics.
- Annual uncertainty: spatial x calendar-month two-way clustered inference.
- Small-month-cluster sensitivity: Webb six-point wild month-cluster bootstrap-t
  with the same two-way studentization as the main program.
- Pooled vegetation effects: orthogonal-score pooling with spatial x year
  two-way clustered inference; Webb wild year-cluster sensitivity is reported
  separately and does not replace the primary pooled inference.
- Formal LCCS heterogeneity: omnibus + pairwise clustered score contrasts.
- 10,000 observations per vegetation-year cell is the primary support threshold;
  5,000 and 20,000 are integrated sensitivity thresholds without refitting.
- Only one designated time-stability test is retained: a linear effect trend.

LCCS-specific design
--------------------
- The eight ecological groups are defined by group_definitions below.
- When a reported ecological group combines multiple raw LCCS subclasses,
  those raw subclasses are still adjusted as nominal one-hot indicators within
  that group (one reference subclass omitted). Single-code groups add no dummy.
- The spatial fold map is constructed from the full 2000-2022 panel BEFORE
  restricting to the eight target LCCS groups, matching the main program's
  nationwide fold construction.

Deliberately NOT repeated here
------------------------------
- strict within-pixel FE DML
- interannual pixel-month within-transformed DML
- future-O3 negative-control exposure
- block-factor 10/30 sensitivity
- A/B/C covariate sensitivity
- random-effects meta-analysis
- any post-hoc calendar breakpoint test, including 2017

These are either handled by the main DML evidence chain or intentionally removed
from the reviewer-facing specification.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

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
        "xgboost is required. Install a CUDA-enabled build when DEVICE='cuda'."
    ) from exc


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_FILE = r"/root/autodl-tmp/wq/matched_data_albers_all1.csv"
OUTPUT_DIR = r"/root/autodl-tmp/wql/0922/lccs_yearly_spatial_dml_v3_aligned"

OUTCOME = "SIF"
EXPOSURE = "O3"
X_COL = "x"
Y_COL = "y"
YEAR_COL = "year"
MONTH_COL = "month"
LCCS_COL = "lccs"

START_YEAR = 2000
END_YEAR = 2022

RANDOM_STATE = 42
DEVICE = "cuda"
N_JOBS = 8
MAX_BIN = 256

N_SPATIAL_FOLDS = 5
SPATIAL_BLOCK_FACTOR = 20

# First run: all group-years are fitted.
# Restart after interruption: completed group-years are automatically reused.
FORCE_RERUN = False

# Primary guardrail and integrated sensitivity thresholds.
# The DML fit floor is the smallest sensitivity threshold, so one model run
# supplies all threshold analyses without a second script or repeated fitting.
PRIMARY_MIN_ROWS = 10_000
SENSITIVITY_THRESHOLDS = [5_000, 10_000, 20_000]
FIT_MIN_ROWS = min(SENSITIVITY_THRESHOLDS)
MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR = 15
MIN_VALID_YEARS_FOR_POOL = 10
MIN_VALID_YEARS_FOR_TIME_TEST = 12

FIG_DPI = 600
WILD_REPS = 9999
WILD_SEED = 20260921
FDR_ALPHA = 0.05

# Paper-facing default: keep one main LCCS figure (pooled forest plot).
MAKE_ANNUAL_HEATMAP = False
MAKE_THRESHOLD_SENSITIVITY_FIGURE = False

lccs_names = {
    10: "Cultivated Land",
    50: "Evergreen Broadleaf Forest",
    60: "Deciduous Broadleaf Forest",
    70: "Evergreen Needleleaf Forest",
    80: "Deciduous Needleleaf Forest",
    90: "Mixed Forest",
    120: "Shrublands",
    130: "Grasslands",
}

group_definitions = {
    10: [10, 11, 12, 20],
    50: [50],
    60: [60, 61, 62],
    70: [70, 71, 72],
    80: [80, 81, 82],
    90: [90],
    120: [120, 121, 122],
    130: [130],
}

GROUP_ORDER = [10, 50, 60, 70, 80, 90, 120, 130]

# Primary year-specific Model A aligned with the new main program.
# LCCS itself is handled separately below as nominal within-group subclass dummies.
MONTH_FE = [f"_month_{m:02d}" for m in range(2, 13)]  # January reference
CONTROL_VARS = [
    X_COL,
    Y_COL,
] + MONTH_FE + [
    "DEM",
    "t2m",
    "ssrd",
    "tp",
    "u10",
    "v10",
    "sp",
    "stl1",
    "swvl1",
]

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

SCRIPT_VERSION = "lccs-main-v6-aligned-monthfe-v3.0"
RUN_VERSION = "lccs-reviewer-aligned-v3.0"


# =============================================================================
# 2. HELPERS
# =============================================================================

def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def cleanup_gpu():
    gc.collect()
    try:
        import cupy as cp  # type: ignore
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def safe_r2(y_true, y_pred):
    if len(y_true) < 2 or float(np.std(y_true)) == 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


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
            f"{MONTH_COL} contains {int(bad.sum()):,} invalid values."
        )
    return s.astype(np.int16)


def two_sided_normal_p(z):
    return float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan


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


def bh_adjust(pvalues):
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if len(vals) == 0:
        return out

    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)

    restored = np.empty(m, dtype=float)
    restored[order] = adj
    out[np.flatnonzero(ok)] = restored
    return out


def config_signature():
    payload = {
        "version": SCRIPT_VERSION,
        "groups": group_definitions,
        "base_controls": CONTROL_VARS,
        "month_fixed_effects": MONTH_FE,
        "lccs_subclass_adjustment": "nominal one-hot within merged ecological group; smallest listed raw code is reference",
        "fold_construction": "balanced nationwide block map built on full 2000-2022 panel before LCCS restriction",
        "hyperparameters": HYPERPARAMS,
        "folds": N_SPATIAL_FOLDS,
        "block_factor": SPATIAL_BLOCK_FACTOR,
        "years": [START_YEAR, END_YEAR],
        "wild_reps": WILD_REPS,
        "wild_seed": WILD_SEED,
        "primary_min_rows": PRIMARY_MIN_ROWS,
        "sensitivity_thresholds": SENSITIVITY_THRESHOLDS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

def infer_grid_spacing(data):
    xs = np.sort(data[X_COL].dropna().unique())
    ys = np.sort(data[Y_COL].dropna().unique())

    dxs = np.diff(xs.astype(np.float64))
    dys = np.diff(ys.astype(np.float64))
    dxs = dxs[np.isfinite(dxs) & (dxs > 0)]
    dys = dys[np.isfinite(dys) & (dys > 0)]

    return (
        float(np.median(dxs)),
        float(np.median(dys)),
        float(data[X_COL].min()),
        float(data[Y_COL].min()),
    )


def spatial_block_ids(x, y, dx, dy, x0, y0):
    bx = np.floor(
        (x.astype(np.float64, copy=False) - x0)
        / (dx * SPATIAL_BLOCK_FACTOR)
    ).astype(np.int32)

    by = np.floor(
        (y.astype(np.float64, copy=False) - y0)
        / (dy * SPATIAL_BLOCK_FACTOR)
    ).astype(np.int32)

    return (
        bx.astype(np.int64) * np.int64(10_000_000)
        + by.astype(np.int64)
    )


def build_lccs_mapping():
    mapping = {}
    for group_code, raw_codes in group_definitions.items():
        for raw in raw_codes:
            if raw in mapping:
                raise ValueError(f"Raw LCCS code {raw} assigned twice.")
            mapping[raw] = group_code
    return mapping


def subclass_feature_names(group_code):
    """Nominal raw-LCCS subclass dummies inside one merged ecological group."""
    levels = sorted(int(v) for v in group_definitions[group_code])
    if len(levels) <= 1:
        return []
    reference = levels[0]
    return [f"_lccs_subclass_{v}" for v in levels if v != reference]


def make_group_design_matrix(work, group_code):
    """
    Build the exact nuisance-model matrix for one LCCS group-year.

    Base columns match the main primary year-specific Model A except that the
    broad LCCS group is the stratum. If that broad group contains multiple raw
    LCCS subclasses, the subclasses are one-hot adjusted as nominal labels.
    """
    base = work[CONTROL_VARS].to_numpy(dtype=np.float32, copy=False)
    levels = sorted(int(v) for v in group_definitions[group_code])
    names = list(CONTROL_VARS)
    if len(levels) <= 1:
        return base, names

    raw = np.rint(work[LCCS_COL].to_numpy(dtype=np.float64, copy=False)).astype(np.int32)
    reference = levels[0]
    extra_levels = [v for v in levels if v != reference]
    X = np.empty((len(work), base.shape[1] + len(extra_levels)), dtype=np.float32)
    X[:, :base.shape[1]] = base
    j = base.shape[1]
    for lv in extra_levels:
        X[:, j] = (raw == lv).astype(np.float32)
        names.append(f"_lccs_subclass_{lv}")
        j += 1
    return X, names

def load_data():
    header = pd.read_csv(INPUT_FILE, nrows=0)
    available = set(header.columns)

    # Match the main program's robust LCCS alias handling.
    lccs_source = "lccs" if "lccs" in available else ("LCCS" if "LCCS" in available else None)
    if lccs_source is None:
        raise ValueError("Missing LCCS/lccs column.")

    raw_controls = [c for c in CONTROL_VARS if not c.startswith("_")]
    needed = [
        OUTCOME, EXPOSURE, X_COL, Y_COL, YEAR_COL, MONTH_COL, lccs_source,
    ] + raw_controls
    needed = list(dict.fromkeys(needed))
    missing = [c for c in needed if c not in available]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Reading {len(needed)} columns from the full panel...")
    t0 = time.time()
    data = pd.read_csv(INPUT_FILE, usecols=needed, low_memory=False)
    print(f"Rows read: {len(data):,}; time={(time.time()-t0)/60:.1f} min")

    if lccs_source == "LCCS" and LCCS_COL not in data.columns:
        data = data.rename(columns={"LCCS": LCCS_COL})

    data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
    data[MONTH_COL] = month_number(data[MONTH_COL])
    data = data.loc[data[YEAR_COL].between(START_YEAR, END_YEAR)].copy()
    data[YEAR_COL] = data[YEAR_COL].astype(np.int16)

    data[X_COL] = pd.to_numeric(data[X_COL], errors="coerce").astype(np.float64)
    data[Y_COL] = pd.to_numeric(data[Y_COL], errors="coerce").astype(np.float64)
    for c in data.columns:
        if c not in {X_COL, Y_COL, YEAR_COL, MONTH_COL}:
            data[c] = pd.to_numeric(data[c], errors="coerce").astype(np.float32)

    # IMPORTANT: infer the national grid and construct the balanced spatial-fold
    # map BEFORE restricting to target vegetation classes. This mirrors the main
    # program, so the same nationwide spatial blocks determine cross-fitting.
    grid = infer_grid_spacing(data)
    dx, dy, x0, y0 = grid
    full_blocks = spatial_block_ids(
        data[X_COL].to_numpy(dtype=np.float64),
        data[Y_COL].to_numpy(dtype=np.float64),
        dx, dy, x0, y0,
    )
    data["_spatial_block"] = full_blocks
    fold_map = balanced_fold_map(full_blocks)
    n_full_blocks = len(fold_map)
    print(f"Nationwide reference blocks (factor={SPATIAL_BLOCK_FACTOR}): {n_full_blocks:,}")

    # Map raw nominal LCCS classes into the eight reported ecological groups.
    mapping = build_lccs_mapping()
    raw_lccs = np.rint(data[LCCS_COL].to_numpy(dtype=np.float64))
    group = pd.Series(raw_lccs, index=data.index).map(mapping)
    data["_lccs_group"] = group
    before = len(data)
    data = data.loc[data["_lccs_group"].notna()].copy()
    data["_lccs_group"] = data["_lccs_group"].astype(np.int16)
    print(
        f"LCCS target groups retained: {len(data):,}/{before:,} "
        f"({len(data)/before:.1%})"
    )

    # Flexible calendar-month fixed effects: January reference, 11 indicators.
    month = data[MONTH_COL].to_numpy(dtype=np.int16)
    for mm in range(2, 13):
        data[f"_month_{mm:02d}"] = (month == mm).astype(np.float32)

    # x/y no longer require float64 after grid/block construction.
    data[X_COL] = data[X_COL].astype(np.float32)
    data[Y_COL] = data[Y_COL].astype(np.float32)
    return data, grid, fold_map, n_full_blocks

def balanced_fold_map(blocks):
    counts = pd.Series(blocks, copy=False).value_counts(sort=False)

    rng = np.random.default_rng(RANDOM_STATE)
    items = [(int(k), int(v)) for k, v in counts.items()]
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
        + ", ".join(
            f"F{i+1}={int(x):,}"
            for i, x in enumerate(totals)
        )
    )
    return mapping


# =============================================================================
# 5. DML
# =============================================================================

def make_model(stage, year, fold):
    params = dict(HYPERPARAMS[stage])
    params.update(
        {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": DEVICE,
            "max_bin": MAX_BIN,
            "n_jobs": N_JOBS,
            "random_state": RANDOM_STATE + year * 10 + fold,
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
        weights=score.astype(np.float64, copy=False),
        minlength=G,
    )

    meat = float(np.dot(sums, sums))
    meat *= G / (G - 1)
    return meat, G


def annual_inference(dr, yr, block, month):
    dr64 = dr.astype(np.float64, copy=False)
    yr64 = yr.astype(np.float64, copy=False)

    numerator = float(np.dot(dr64, yr64))
    denominator = float(np.dot(dr64, dr64))
    theta = numerator / denominator

    score = dr64 * (yr64 - theta * dr64)

    m_space, g_space = cluster_meat(score, block)
    m_month, g_month = cluster_meat(score, month)

    pair, _ = pd.factorize(
        pd.MultiIndex.from_arrays([block, month]),
        sort=False,
    )
    m_inter, g_inter = cluster_meat(score, pair)

    var = (
        m_space + m_month - m_inter
    ) / denominator**2

    fallback = False
    if not np.isfinite(var) or var <= 0:
        var = max(m_space, m_month) / denominator**2
        fallback = True

    se = math.sqrt(var)
    df = max(1, min(g_space, g_month) - 1)
    crit = float(stats.t.ppf(0.975, df))

    return {
        "estimate_raw": theta,
        "effect_per_10ug_m3": theta * 10.0,
        "se_two_way_per_10ug_m3": se * 10.0,
        "ci95_low_t_min_cluster_per_10ug_m3": (
            theta - crit * se
        ) * 10.0,
        "ci95_high_t_min_cluster_per_10ug_m3": (
            theta + crit * se
        ) * 10.0,
        "p_value_t_min_cluster_sensitivity": t_pvalue(theta / se, df),
        "t_df_min_cluster": df,
        "n_spatial_clusters": g_space,
        "n_month_clusters": g_month,
        "n_space_month_intersections": g_inter,
        "variance_fallback": fallback,
        "dml_numerator_raw": numerator,
        "dml_denominator_raw": denominator,
    }




def _two_way_cell_variance(cell_score, cell_den):
    sm = np.asarray(cell_score, dtype=np.float64)
    dm = np.asarray(cell_den, dtype=np.float64)
    gs, gt = sm.shape
    occupied = dm > 0
    gi = int(occupied.sum())
    den = float(dm.sum())
    ss = sm.sum(axis=1)
    st = sm.sum(axis=0)
    ms = float(ss @ ss) * (gs / (gs - 1) if gs > 1 else 1.0)
    mt = float(st @ st) * (gt / (gt - 1) if gt > 1 else 1.0)
    mi = float(np.square(sm[occupied]).sum()) * (gi / (gi - 1) if gi > 1 else 1.0)
    var = (ms + mt - mi) / (den * den)
    if not np.isfinite(var) or var <= 0:
        var = max(ms, mt) / (den * den)
    return float(var), gs, gt, gi


def wild_month_cluster_bootstrap_t(group_code, year, dr, yr, space, month,
                                   reps=WILD_REPS, seed=WILD_SEED, batch=128):
    """
    Webb six-point wild resampling on calendar-month clusters only, with the
    same spatial x month two-way studentization as the main program.
    This is a small-month-cluster sensitivity, not a multiway wild bootstrap.
    """
    d = np.asarray(dr, dtype=np.float64)
    y = np.asarray(yr, dtype=np.float64)
    sp = np.asarray(space)
    mo = np.asarray(month)

    den = float(d @ d)
    theta = float((d @ y) / den)
    score = d * (y - theta * d)

    su, si = pd.factorize(sp, sort=False)
    mu, mi = pd.factorize(mo, sort=True)
    gs = int(su.max() + 1)
    gm = int(mu.max() + 1)
    S = np.zeros((gs, gm), dtype=np.float64)
    D = np.zeros((gs, gm), dtype=np.float64)
    np.add.at(S, (su.astype(np.int64), mu.astype(np.int64)), score)
    np.add.at(D, (su.astype(np.int64), mu.astype(np.int64)), d * d)

    obs_var, gs, gm, gi = _two_way_cell_variance(S, D)
    obs_se = math.sqrt(obs_var)
    obs_t = theta / obs_se

    webb = np.array([
        -math.sqrt(1.5), -1.0, -math.sqrt(0.5),
        math.sqrt(0.5), 1.0, math.sqrt(1.5),
    ], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    tboot = np.empty(int(reps), dtype=np.float64)
    done = 0
    occ = D > 0
    fsp = gs / (gs - 1) if gs > 1 else 1.0
    fmt = gm / (gm - 1) if gm > 1 else 1.0
    fint = gi / (gi - 1) if gi > 1 else 1.0

    while done < reps:
        b = min(batch, reps - done)
        W = webb[rng.integers(0, len(webb), size=(b, gm))]
        weighted = S[None, :, :] * W[:, None, :]
        delta = weighted.sum(axis=(1, 2)) / den
        centered = weighted - delta[:, None, None] * D[None, :, :]
        s_space = centered.sum(axis=2)
        s_month = centered.sum(axis=1)
        ms = np.sum(s_space * s_space, axis=1) * fsp
        mt = np.sum(s_month * s_month, axis=1) * fmt
        mi2 = np.sum(centered[:, occ] * centered[:, occ], axis=1) * fint
        vb = (ms + mt - mi2) / (den * den)
        fallback = np.maximum(ms, mt) / (den * den)
        vb = np.where(np.isfinite(vb) & (vb > 0), vb, fallback)
        tboot[done:done+b] = delta / np.sqrt(vb)
        done += b

    crit = float(np.quantile(np.abs(tboot), 0.95))
    p = float((1 + np.sum(np.abs(tboot) >= abs(obs_t))) / (reps + 1))
    return {
        "lccs_group": int(group_code),
        "year": int(year),
        "wild_month_cluster_twoway_studentized_p_value": p,
        "wild_bootstrap_t_critical_95": crit,
        "wild_month_cluster_twoway_studentized_ci95_low_per_10ug_m3": (theta - crit * obs_se) * 10.0,
        "wild_month_cluster_twoway_studentized_ci95_high_per_10ug_m3": (theta + crit * obs_se) * 10.0,
        "wild_resampling_month_clusters": int(gm),
        "studentization_spatial_clusters": int(gs),
        "studentization_space_month_cells": int(gi),
        "wild_bootstrap_reps": int(reps),
        "wild_month_cluster_weights": "Webb six-point",
        "wild_resampling_dimension": "calendar_month_only",
        "studentization": "spatial_x_month_two_way_cluster_robust",
        "is_multiway_wild_cluster_bootstrap": False,
    }

def spatial_components(year, group_code, dr, yr, block):
    tmp = pd.DataFrame(
        {
            "spatial_block": block.astype(np.int64, copy=False),
            "num": (
                dr.astype(np.float64, copy=False)
                * yr.astype(np.float64, copy=False)
            ),
            "den": dr.astype(np.float64, copy=False) ** 2,
        }
    )

    g = (
        tmp.groupby("spatial_block", observed=True, sort=False)
        .agg(
            numerator_raw=("num", "sum"),
            denominator_raw=("den", "sum"),
            n=("num", "size"),
        )
        .reset_index()
    )

    g.insert(0, "year", int(year))
    g.insert(0, "lccs_group", int(group_code))
    return g


def fit_group_year(work, group_code, year, fold_map):
    fold_series = work["_spatial_block"].map(fold_map)
    if fold_series.isna().any():
        raise RuntimeError("Unmapped spatial blocks.")
    fold = fold_series.to_numpy(dtype=np.int8)
    if np.unique(fold).size < N_SPATIAL_FOLDS:
        raise RuntimeError("Not all five nationwide spatial folds are represented.")

    X, feature_names = make_group_design_matrix(work, group_code)
    t = work[EXPOSURE].to_numpy(dtype=np.float32, copy=False)
    y = work[OUTCOME].to_numpy(dtype=np.float32, copy=False)
    block = work["_spatial_block"].to_numpy(dtype=np.int64, copy=False)
    month = work[MONTH_COL].to_numpy(dtype=np.int16, copy=False)

    t_hat = np.full(len(work), np.nan, dtype=np.float32)
    y_hat = np.full(len(work), np.nan, dtype=np.float32)
    diag_rows = []

    for f in range(N_SPATIAL_FOLDS):
        test = np.flatnonzero(fold == f)
        train = np.flatnonzero(fold != f)
        if len(test) == 0 or len(train) == 0:
            raise RuntimeError(f"Invalid fold {f+1}: train={len(train)}, test={len(test)}")

        for stage, target, pred_dest in [
            ("treatment", t, t_hat),
            ("outcome", y, y_hat),
        ]:
            model = make_model(stage, year, f)
            model.fit(X[train], target[train])
            pred = model.predict(X[test]).astype(np.float32)
            pred_dest[test] = pred
            residual = target[test].astype(np.float64) - pred.astype(np.float64)
            diag_rows.append({
                "lccs_group": group_code,
                "lccs_name": lccs_names[group_code],
                "year": year,
                "stage": stage,
                "spatial_fold": f + 1,
                "n_train": len(train),
                "n_test": len(test),
                "n_test_blocks": int(np.unique(block[test]).size),
                "r2": safe_r2(target[test], pred),
                "rmse": float(np.sqrt(mean_squared_error(target[test], pred))),
                "mae": float(mean_absolute_error(target[test], pred)),
                "mean_residual_bias": float(np.mean(residual)),
            })
            del model, pred, residual
            cleanup_gpu()

    if not np.isfinite(t_hat).all() or not np.isfinite(y_hat).all():
        raise RuntimeError("Incomplete OOF predictions.")

    dr = (t - t_hat).astype(np.float32)
    yr = (y - y_hat).astype(np.float32)
    inference = annual_inference(dr, yr, block, month)
    wild = wild_month_cluster_bootstrap_t(
        group_code, year, dr, yr, block, month,
        reps=WILD_REPS,
        seed=WILD_SEED + int(group_code) * 1000 + int(year),
    )
    diag = pd.DataFrame(diag_rows)

    levels = sorted(int(v) for v in group_definitions[group_code])
    result = {
        "lccs_group": group_code,
        "lccs_name": lccs_names[group_code],
        "year": year,
        "n": len(work),
        "n_spatial_blocks": int(np.unique(block).size),
        "n_controls": len(feature_names),
        "controls": ";".join(feature_names),
        "month_control": "11 calendar-month fixed effects; January reference",
        "lccs_subclass_reference": int(levels[0]),
        "n_lccs_subclass_dummies": max(0, len(levels) - 1),
        **inference,
        **{k: v for k, v in wild.items() if k not in {"lccs_group", "year"}},
        "treatment_oof_r2": safe_r2(t, t_hat),
        "treatment_oof_rmse": float(np.sqrt(mean_squared_error(t, t_hat))),
        "treatment_residual_mean": float(np.mean(dr)),
        "treatment_residual_sd": float(np.std(dr, ddof=1)),
        "treatment_min_fold_r2": float(diag.loc[diag["stage"] == "treatment", "r2"].min()),
        "treatment_max_abs_fold_bias": float(
            diag.loc[diag["stage"] == "treatment", "mean_residual_bias"].abs().max()
        ),
        "outcome_oof_r2": safe_r2(y, y_hat),
        "outcome_oof_rmse": float(np.sqrt(mean_squared_error(y, y_hat))),
        "outcome_residual_mean": float(np.mean(yr)),
        "outcome_min_fold_r2": float(diag.loc[diag["stage"] == "outcome", "r2"].min()),
    }

    comp = spatial_components(year, group_code, dr, yr, block)
    del X, t, y, t_hat, y_hat, dr, yr
    cleanup_gpu()
    return result, diag, comp

def pooled_group_dml(components, group_code, yearly_df):
    c = components.loc[
        components["lccs_group"] == group_code
    ].copy()

    numerator = float(c["numerator_raw"].sum())
    denominator = float(c["denominator_raw"].sum())
    theta = numerator / denominator

    c["score"] = (
        c["numerator_raw"].astype(float)
        - theta * c["denominator_raw"].astype(float)
    )

    space = (
        c.groupby("spatial_block", observed=True)["score"]
        .sum()
        .to_numpy(float)
    )
    year = (
        c.groupby("year", observed=True)["score"]
        .sum()
        .to_numpy(float)
    )
    inter = c["score"].to_numpy(float)

    gs = len(space)
    gy = len(year)
    gi = len(inter)

    meat_space = np.dot(space, space) * gs / (gs - 1)
    meat_year = np.dot(year, year) * gy / (gy - 1)
    meat_inter = np.dot(inter, inter) * gi / (gi - 1)

    var = (
        meat_space + meat_year - meat_inter
    ) / denominator**2

    fallback = False
    if not np.isfinite(var) or var <= 0:
        var = max(meat_space, meat_year) / denominator**2
        fallback = True

    se = math.sqrt(var)
    df = max(1, min(gs, gy) - 1)
    crit = float(stats.t.ppf(0.975, df))

    yd = yearly_df.loc[
        yearly_df["lccs_group"] == group_code
    ].copy()

    n_negative = int((yd["effect_per_10ug_m3"] < 0).sum())
    n_years = len(yd)

    sign_p = (
        float(
            stats.binomtest(
                n_negative,
                n_years,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if n_years > 0
        else np.nan
    )

    return {
        "lccs_group": group_code,
        "lccs_name": lccs_names[group_code],
        "n_valid_years": n_years,
        "negative_years": n_negative,
        "annual_sign_test_p": sign_p,
        "median_annual_effect_per_10ug_m3": float(
            yd["effect_per_10ug_m3"].median()
        ),
        "estimate_raw": theta,
        "effect_per_10ug_m3": theta * 10.0,
        "se_space_year_per_10ug_m3": se * 10.0,
        "ci95_low_t_per_10ug_m3": (
            theta - crit * se
        ) * 10.0,
        "ci95_high_t_per_10ug_m3": (
            theta + crit * se
        ) * 10.0,
        "p_value_t": t_pvalue(theta / se, df),
        "t_df": df,
        "n_spatial_clusters": gs,
        "n_year_clusters": gy,
        "variance_fallback": fallback,
        "dml_numerator_raw": numerator,
        "dml_denominator_raw": denominator,
        "mean_treatment_oof_r2": float(
            yd["treatment_oof_r2"].mean()
        ),
        "min_treatment_oof_r2": float(
            yd["treatment_oof_r2"].min()
        ),
        "mean_outcome_oof_r2": float(
            yd["outcome_oof_r2"].mean()
        ),
        "min_outcome_oof_r2": float(
            yd["outcome_oof_r2"].min()
        ),
    }


# =============================================================================
# 7. META-ANALYSIS
# =============================================================================

def wild_year_cluster_group(components, group_code, seed):
    """Webb wild year-cluster sensitivity for one pooled LCCS effect."""
    c = components.loc[components["lccs_group"] == group_code].copy()
    y = c.groupby("year", observed=True).agg(
        num=("numerator_raw", "sum"), den=("denominator_raw", "sum")
    ).reset_index()
    N = y["num"].to_numpy(float)
    D = y["den"].to_numpy(float)
    G = len(N)
    if G < 2:
        return {
            "lccs_group": group_code,
            "lccs_name": lccs_names[group_code],
            "n_year_clusters": G,
            "wild_cluster_p_value": np.nan,
        }

    theta = float(N.sum() / D.sum())
    R = N.copy()
    obs = float(R.sum() / math.sqrt((G / (G - 1)) * float(R @ R)))
    U = N - theta * D
    vals = np.array([
        -math.sqrt(1.5), -1.0, -math.sqrt(0.5),
        math.sqrt(0.5), 1.0, math.sqrt(1.5),
    ])
    rng = np.random.default_rng(int(seed))
    exc = 0
    delta = np.empty(WILD_REPS)
    done = 0
    while done < WILD_REPS:
        b = min(2000, WILD_REPS - done)
        W = vals[rng.integers(0, 6, size=(b, G))]
        WR = W * R
        tb = WR.sum(1) / np.sqrt((G / (G - 1)) * np.sum(WR ** 2, axis=1))
        exc += int((np.abs(tb) >= abs(obs)).sum())
        delta[done:done+b] = (W * U).sum(1) / D.sum()
        done += b

    q = np.quantile(delta, [0.025, 0.975])
    return {
        "lccs_group": group_code,
        "lccs_name": lccs_names[group_code],
        "effect_per_10ug_m3": theta * 10.0,
        "wild_cluster_p_value": (1 + exc) / (WILD_REPS + 1),
        "wild_basic_ci95_low_per_10ug_m3": (theta - q[1]) * 10.0,
        "wild_basic_ci95_high_per_10ug_m3": (theta - q[0]) * 10.0,
        "n_year_clusters": G,
        "bootstrap_reps": WILD_REPS,
        "wild_cluster_weights": "Webb six-point",
        "inference_role": "pooled small-year-cluster sensitivity; primary inference remains space x year two-way clustered",
    }

def multivariate_group_covariance(components, pooled_df):
    eligible = pooled_df.loc[
        pooled_df["n_valid_years"] >= MIN_VALID_YEARS_FOR_POOL
    ].copy()

    groups = eligible["lccs_group"].astype(int).tolist()
    theta = (
        eligible.set_index("lccs_group")
        .loc[groups, "estimate_raw"]
        .to_numpy(float)
    )

    D = (
        eligible.set_index("lccs_group")
        .loc[groups, "dml_denominator_raw"]
        .to_numpy(float)
    )

    cells = pd.MultiIndex.from_frame(
        components[["spatial_block", "year"]].drop_duplicates()
    )

    score_matrix = np.zeros(
        (len(cells), len(groups)),
        dtype=np.float64,
    )

    cell_frame = cells.to_frame(index=False)
    cell_frame.columns = ["spatial_block", "year"]

    for j, g in enumerate(groups):
        sub = components.loc[
            components["lccs_group"] == g,
            [
                "spatial_block",
                "year",
                "numerator_raw",
                "denominator_raw",
            ],
        ]

        merged = cell_frame.merge(
            sub,
            on=["spatial_block", "year"],
            how="left",
        ).fillna(0.0)

        score_matrix[:, j] = (
            merged["numerator_raw"].to_numpy(float)
            - theta[j]
            * merged["denominator_raw"].to_numpy(float)
        )

    # Spatial-cluster meat.
    space_codes, space_uniques = pd.factorize(
        cell_frame["spatial_block"],
        sort=False,
    )
    gs = len(space_uniques)
    space_sum = np.zeros((gs, len(groups)), dtype=np.float64)

    for j in range(len(groups)):
        space_sum[:, j] = np.bincount(
            space_codes,
            weights=score_matrix[:, j],
            minlength=gs,
        )

    meat_space = space_sum.T @ space_sum
    meat_space *= gs / (gs - 1)

    # Year-cluster meat.
    year_codes, year_uniques = pd.factorize(
        cell_frame["year"],
        sort=False,
    )
    gy = len(year_uniques)
    year_sum = np.zeros((gy, len(groups)), dtype=np.float64)

    for j in range(len(groups)):
        year_sum[:, j] = np.bincount(
            year_codes,
            weights=score_matrix[:, j],
            minlength=gy,
        )

    meat_year = year_sum.T @ year_sum
    meat_year *= gy / (gy - 1)

    # Intersection meat: spatial block x year cell.
    gi = len(score_matrix)
    meat_inter = score_matrix.T @ score_matrix
    meat_inter *= gi / (gi - 1)

    Hinv = np.diag(1.0 / D)

    covariance = (
        Hinv
        @ (meat_space + meat_year - meat_inter)
        @ Hinv
    )

    covariance = 0.5 * (covariance + covariance.T)

    # PSD correction only if numerical subtraction causes a negative eigenvalue.
    eigval, eigvec = np.linalg.eigh(covariance)
    psd_adjusted = bool(np.any(eigval < -1e-14))
    if psd_adjusted:
        eigval = np.clip(eigval, 0.0, None)
        covariance = (
            eigvec * eigval
        ) @ eigvec.T
        covariance = 0.5 * (covariance + covariance.T)

    return groups, theta, covariance, gs, gy, psd_adjusted


def heterogeneity_tests(components, pooled_df):
    groups, theta, covariance, gs, gy, psd_adjusted = (
        multivariate_group_covariance(
            components,
            pooled_df,
        )
    )

    G = len(groups)

    # Omnibus equality test using first eligible group as reference.
    R = np.zeros((G - 1, G), dtype=np.float64)
    for i in range(1, G):
        R[i - 1, i] = 1.0
        R[i - 1, 0] = -1.0

    diff = R @ theta
    V = R @ covariance @ R.T

    q = G - 1
    F = float(
        diff.T @ np.linalg.pinv(V) @ diff / q
    )
    df2 = max(1, min(gs, gy) - 1)
    p_omnibus = float(stats.f.sf(F, q, df2))

    omnibus = pd.DataFrame(
        [
            {
                "test": "LCCS omnibus heterogeneity",
                "null_hypothesis": "All pooled LCCS effects are equal",
                "n_groups_tested": G,
                "groups_tested": ";".join(map(str, groups)),
                "wald_F": F,
                "df1": q,
                "df2": df2,
                "p_value": p_omnibus,
                "covariance_psd_adjusted": psd_adjusted,
            }
        ]
    )

    # Pairwise contrasts.
    rows = []

    for i in range(G):
        for j in range(i + 1, G):
            w = np.zeros(G)
            w[i] = 1.0
            w[j] = -1.0

            estimate = float(w @ theta)
            var = float(w @ covariance @ w)
            se = math.sqrt(max(var, 0.0))

            tstat = estimate / se if se > 0 else np.nan
            p = t_pvalue(tstat, df2)
            crit = float(stats.t.ppf(0.975, df2))

            rows.append(
                {
                    "group_1": groups[i],
                    "group_1_name": lccs_names[groups[i]],
                    "group_2": groups[j],
                    "group_2_name": lccs_names[groups[j]],
                    "difference_group1_minus_group2_per_10ug_m3": estimate * 10.0,
                    "se_difference_per_10ug_m3": se * 10.0,
                    "ci95_low_per_10ug_m3": (
                        estimate - crit * se
                    ) * 10.0,
                    "ci95_high_per_10ug_m3": (
                        estimate + crit * se
                    ) * 10.0,
                    "t_statistic": tstat,
                    "df": df2,
                    "p_value_raw": p,
                }
            )

    pairwise = pd.DataFrame(rows)
    pairwise["p_value_holm"] = holm_adjust(
        pairwise["p_value_raw"].to_numpy(float)
    )
    pairwise["significant_holm_0.05"] = (
        pairwise["p_value_holm"] < 0.05
    )

    return omnibus, pairwise


# =============================================================================
# 9. GROUP-SPECIFIC FORMAL TIME TESTS
# =============================================================================

def cluster_meat_matrix(scores, cluster):
    codes, uniques = pd.factorize(cluster, sort=False)
    G = len(uniques)
    p = scores.shape[1]

    totals = np.zeros((G, p), dtype=np.float64)
    for j in range(p):
        totals[:, j] = np.bincount(
            codes,
            weights=scores[:, j],
            minlength=G,
        )

    meat = totals.T @ totals
    if G > 1:
        meat *= G / (G - 1)

    return meat, G


def score_regression(components, design, cols):
    d = components.merge(
        design,
        on="year",
        how="left",
        validate="many_to_one",
    )

    Z = d[cols].to_numpy(float)
    N = d["numerator_raw"].to_numpy(float)
    D = d["denominator_raw"].to_numpy(float)

    H = Z.T @ (D[:, None] * Z)
    rhs = Z.T @ N
    beta = np.linalg.solve(H, rhs)

    moment = N - D * (Z @ beta)
    score = Z * moment[:, None]

    m_space, gs = cluster_meat_matrix(
        score,
        d["spatial_block"].to_numpy(),
    )
    m_year, gy = cluster_meat_matrix(
        score,
        d["year"].to_numpy(),
    )

    gi = len(score)
    m_inter = score.T @ score
    if gi > 1:
        m_inter *= gi / (gi - 1)

    Hinv = np.linalg.inv(H)
    cov = (
        Hinv
        @ (m_space + m_year - m_inter)
        @ Hinv
    )
    cov = 0.5 * (cov + cov.T)

    eigval, eigvec = np.linalg.eigh(cov)
    if np.any(eigval < -1e-14):
        eigval = np.clip(eigval, 0.0, None)
        cov = (eigvec * eigval) @ eigvec.T

    return beta, cov, max(1, min(gs, gy) - 1)


def lincomb(beta, cov, df, weights, label):
    w = np.asarray(weights, dtype=float)

    estimate = float(w @ beta)
    var = float(w @ cov @ w)
    se = math.sqrt(max(var, 0.0))
    tstat = estimate / se if se > 0 else np.nan
    crit = float(stats.t.ppf(0.975, df))

    return {
        "test": label,
        "estimate_per_10ug_m3": estimate * 10.0,
        "se_per_10ug_m3": se * 10.0,
        "ci95_low_per_10ug_m3": (
            estimate - crit * se
        ) * 10.0,
        "ci95_high_per_10ug_m3": (
            estimate + crit * se
        ) * 10.0,
        "t_statistic": tstat,
        "df": df,
        "p_value": t_pvalue(tstat, df),
    }


def group_time_tests(components, group_code):
    """Single designated temporal-stability test: linear trend in LCCS effect."""
    c = components.loc[components["lccs_group"] == group_code].copy()
    years = np.array(sorted(c["year"].unique()), dtype=int)
    if len(years) < MIN_VALID_YEARS_FOR_TIME_TEST:
        return pd.DataFrame([{
            "lccs_group": group_code,
            "lccs_name": lccs_names[group_code],
            "test": "SKIPPED",
            "reason": f"Only {len(years)} valid years; minimum={MIN_VALID_YEARS_FOR_TIME_TEST}",
        }])

    center = float(years.mean())
    design = pd.DataFrame({
        "year": years,
        "intercept": 1.0,
        "time": years - center,
    })
    beta, cov, df = score_regression(c, design, ["intercept", "time"])
    row = lincomb(
        beta, cov, df, [0, 1],
        "linear_effect_trend_per_calendar_year",
    )
    row["lccs_group"] = group_code
    row["lccs_name"] = lccs_names[group_code]
    return pd.DataFrame([row])

def forest_plot(pooled_df, out_dir):
    d = (
        pooled_df.set_index("lccs_group")
        .loc[[g for g in GROUP_ORDER if g in pooled_df["lccs_group"].values]]
        .reset_index()
    )
    y = np.arange(len(d))[::-1]
    est = d["effect_per_10ug_m3"].to_numpy(float)
    lo = d["ci95_low_t_per_10ug_m3"].to_numpy(float)
    hi = d["ci95_high_t_per_10ug_m3"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axvline(0.0, linestyle="--", linewidth=1)
    for i in range(len(d)):
        ax.hlines(y[i], lo[i], hi[i], linewidth=2)
        ax.scatter(est[i], y[i], s=55, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(d["lccs_name"], fontsize=10)
    ax.set_xlabel(r"Adjusted O$_3$–SIF slope per +10 $\mu$g m$^{-3}$ O$_3$ (95% CI)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    lim = np.nanmax(np.abs(np.r_[lo, hi])) if len(d) else 0.03
    lim = max(0.03, float(np.ceil(lim / 0.005) * 0.005))
    ax.set_xlim(-lim, lim)

    for i, r in d.iterrows():
        ax.text(
            1.02, y[i],
            f"{r['effect_per_10ug_m3']:.4f} ({r['ci95_low_t_per_10ug_m3']:.4f} – {r['ci95_high_t_per_10ug_m3']:.4f})",
            transform=ax.get_yaxis_transform(), va="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "Figure_LCCS_Pooled_Forest.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_dir / "Figure_LCCS_Pooled_Forest.pdf", bbox_inches="tight")
    plt.close(fig)

def annual_heatmap(yearly_df, out_dir):
    pivot = (
        yearly_df.pivot(
            index="lccs_group",
            columns="year",
            values="effect_per_10ug_m3",
        )
        .reindex(GROUP_ORDER)
    )

    fig, ax = plt.subplots(figsize=(14, 5.6))

    values = pivot.to_numpy(float)

    vmax = np.nanmax(np.abs(values))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        pivot.columns.astype(int),
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        [lccs_names[int(g)] for g in pivot.index],
        fontsize=9,
    )

    ax.set_title(
        "Annual LCCS-specific DML Effects, 2000–2022",
        fontweight="bold",
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        "Effect on SIF per +10 μg/m³ O₃"
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / "Figure_LCCS_Annual_Heatmap.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "Figure_LCCS_Annual_Heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 11. MAIN
# =============================================================================

def restrict_cells(df, cells):
    """Keep rows belonging to the supplied LCCS-group x year cells."""
    if df.empty or cells.empty:
        return df.iloc[0:0].copy()
    key = cells[["lccs_group", "year"]].drop_duplicates().copy()
    return df.merge(key, on=["lccs_group", "year"], how="inner")


def pool_at_threshold(all_yearly, all_components, threshold):
    """Re-pool already-fitted orthogonal scores under one sample-size floor."""
    yd = all_yearly.loc[
        (all_yearly["n"] >= threshold)
        & (all_yearly["n_spatial_blocks"] >= MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR)
    ].copy()
    cells = yd[["lccs_group", "year"]].drop_duplicates()
    comp = restrict_cells(all_components, cells)

    rows = []
    for g in GROUP_ORDER:
        n_years = int(yd.loc[yd["lccs_group"] == g, "year"].nunique())
        if n_years >= MIN_VALID_YEARS_FOR_POOL:
            r = pooled_group_dml(comp, g, yd)
            r["sample_threshold"] = int(threshold)
            rows.append(r)

    pooled = pd.DataFrame(rows)
    if not pooled.empty:
        pooled["q_value_BH"] = bh_adjust(pooled["p_value_t"].to_numpy(float))
        pooled["fdr_significant_0.05"] = pooled["q_value_BH"] < 0.05
    return yd, comp, pooled


def run_threshold_sensitivity(all_yearly, all_diag, all_comp, support_audit, out):
    """Integrated 5k/10k/20k threshold sensitivity; no DML is refitted here."""
    sens_dir = out / "sample_threshold_sensitivity"
    fig_dir = sens_dir / "figures"
    sens_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # Cells fitted only because the sensitivity floor is below the 10,000 primary floor.
    extra_low = all_yearly.loc[
        (all_yearly["n"] >= FIT_MIN_ROWS)
        & (all_yearly["n"] < PRIMARY_MIN_ROWS)
        & (all_yearly["n_spatial_blocks"] >= MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR),
        ["lccs_group", "lccs_name", "year", "n", "n_spatial_blocks"],
    ].copy()
    extra_low.to_csv(sens_dir / "additional_low_threshold_cells_needed.csv", index=False)

    eligibility = support_audit.copy()
    for threshold in SENSITIVITY_THRESHOLDS:
        eligibility[f"eligible_n_ge_{threshold}"] = (
            (eligibility["n"] >= threshold)
            & (eligibility["n_spatial_blocks"] >= MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR)
            & (eligibility["represented_folds"] >= N_SPATIAL_FOLDS)
        )
    eligibility.to_csv(sens_dir / "threshold_cell_eligibility.csv", index=False)

    pooled_parts, support_rows, diag_rows = [], [], []
    for threshold in SENSITIVITY_THRESHOLDS:
        yd, _, pooled = pool_at_threshold(all_yearly, all_comp, threshold)
        cells = yd[["lccs_group", "year"]].drop_duplicates()
        diag = restrict_cells(all_diag, cells)

        if not pooled.empty:
            pooled_parts.append(pooled)

        support_rows.append({
            "sample_threshold": threshold,
            "eligible_group_year_cells": len(yd),
            "pooled_groups_with_at_least_10_years": len(pooled),
            "minimum_actual_cell_n": yd["n"].min() if len(yd) else np.nan,
            "median_actual_cell_n": yd["n"].median() if len(yd) else np.nan,
            "minimum_spatial_blocks": yd["n_spatial_blocks"].min() if len(yd) else np.nan,
            "negative_pooled_groups": int((pooled["effect_per_10ug_m3"] < 0).sum()) if len(pooled) else 0,
            "fdr_significant_groups": int(pooled["fdr_significant_0.05"].sum()) if len(pooled) else 0,
        })

        for stage, d in diag.groupby("stage", observed=True):
            diag_rows.append({
                "sample_threshold": threshold,
                "stage": stage,
                "n_fold_fits": len(d),
                "mean_r2": d["r2"].mean(),
                "median_r2": d["r2"].median(),
                "p05_r2": d["r2"].quantile(0.05),
                "min_r2": d["r2"].min(),
                "mean_rmse": d["rmse"].mean(),
                "mean_mae": d["mae"].mean(),
                "mean_abs_residual_bias": d["mean_residual_bias"].abs().mean(),
                "min_test_blocks_per_fold": d["n_test_blocks"].min(),
                "median_train_n": d["n_train"].median(),
                "median_test_n": d["n_test"].median(),
                "minimum_train_n": d["n_train"].min(),
                "minimum_test_n": d["n_test"].min(),
            })

    pooled_all = pd.concat(pooled_parts, ignore_index=True)
    support = pd.DataFrame(support_rows)
    nuisance = pd.DataFrame(diag_rows)

    pooled_all.to_csv(sens_dir / "threshold_pooled_effects.csv", index=False)
    support.to_csv(sens_dir / "threshold_support_summary.csv", index=False)
    nuisance.to_csv(sens_dir / "threshold_nuisance_diagnostics.csv", index=False)

    ref = pooled_all.loc[
        pooled_all["sample_threshold"] == PRIMARY_MIN_ROWS,
        [
            "lccs_group", "lccs_name", "effect_per_10ug_m3",
            "se_space_year_per_10ug_m3", "q_value_BH",
            "fdr_significant_0.05", "n_valid_years",
        ],
    ].rename(columns={
        "effect_per_10ug_m3": "effect_10000",
        "se_space_year_per_10ug_m3": "se_10000",
        "q_value_BH": "q_10000",
        "fdr_significant_0.05": "fdr_sig_10000",
        "n_valid_years": "n_years_10000",
    })

    comparison = pooled_all.merge(ref, on=["lccs_group", "lccs_name"], how="left")
    comparison["delta_effect_vs_10000"] = comparison["effect_per_10ug_m3"] - comparison["effect_10000"]
    comparison["abs_delta_effect_vs_10000"] = comparison["delta_effect_vs_10000"].abs()
    comparison["relative_abs_change_vs_10000"] = (
        comparison["abs_delta_effect_vs_10000"]
        / comparison["effect_10000"].abs().replace(0, np.nan)
    )
    comparison["delta_se_vs_10000"] = comparison["se_space_year_per_10ug_m3"] - comparison["se_10000"]
    comparison["same_sign_as_10000"] = (
        np.sign(comparison["effect_per_10ug_m3"]) == np.sign(comparison["effect_10000"])
    )
    comparison["same_fdr_decision_as_10000"] = (
        comparison["fdr_significant_0.05"] == comparison["fdr_sig_10000"]
    )
    comparison.to_csv(sens_dir / "threshold_comparison_vs_10000.csv", index=False)

    stability_rows = []
    for threshold, d in comparison.groupby("sample_threshold", observed=True):
        d = d.loc[d["effect_10000"].notna()].copy()
        rel = d["relative_abs_change_vs_10000"].replace([np.inf, -np.inf], np.nan)
        stability_rows.append({
            "sample_threshold": threshold,
            "n_common_groups": len(d),
            "max_abs_delta_effect_vs_10000": d["abs_delta_effect_vs_10000"].max(),
            "median_abs_delta_effect_vs_10000": d["abs_delta_effect_vs_10000"].median(),
            "max_relative_abs_change_vs_10000": rel.max(),
            "median_relative_abs_change_vs_10000": rel.median(),
            "max_abs_delta_se_vs_10000": d["delta_se_vs_10000"].abs().max(),
            "all_signs_same_as_10000": bool(d["same_sign_as_10000"].all()),
            "all_fdr_decisions_same_as_10000": bool(d["same_fdr_decision_as_10000"].all()),
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(sens_dir / "threshold_stability_summary_vs_10000.csv", index=False)

    if MAKE_THRESHOLD_SENSITIVITY_FIGURE:
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        for g in GROUP_ORDER:
            d = pooled_all.loc[pooled_all["lccs_group"] == g].sort_values("sample_threshold")
            if not d.empty:
                ax.plot(d["sample_threshold"], d["effect_per_10ug_m3"], marker="o", linewidth=1.5, label=lccs_names[g])
        ax.axvline(PRIMARY_MIN_ROWS, linestyle="--", linewidth=1.1, label="Primary threshold = 10,000")
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Minimum observations per vegetation-year cell")
        ax.set_ylabel(r"Adjusted O$_3$–SIF slope per +10 $\mu$g m$^{-3}$ O$_3$")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "Figure_sample_threshold_sensitivity.png", dpi=FIG_DPI, bbox_inches="tight")
        fig.savefig(fig_dir / "Figure_sample_threshold_sensitivity.pdf", bbox_inches="tight")
        plt.close(fig)

    print("\n" + "=" * 108)
    print("INTEGRATED SAMPLE-SIZE THRESHOLD SENSITIVITY")
    print("=" * 108)
    print(support.to_string(index=False))
    print("\nStability relative to 10,000:")
    print(stability.to_string(index=False))
    return support, stability, nuisance


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "year_cache"
    cache_dir.mkdir(exist_ok=True)
    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)

    if PRIMARY_MIN_ROWS not in SENSITIVITY_THRESHOLDS:
        raise ValueError("PRIMARY_MIN_ROWS must be included in SENSITIVITY_THRESHOLDS.")

    print("=" * 112)
    print("LCCS VEGETATION HETEROGENEITY — ALIGNED WITH REVIEWER-COMPACT MAIN DML v6")
    print("=" * 112)
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print("Seasonality: 11 calendar-month fixed effects (January reference)")
    print("Spatial folds: nationwide block map constructed before LCCS restriction")
    print("Within merged LCCS groups: raw subclasses adjusted as nominal one-hot indicators")
    print(f"Primary threshold: {PRIMARY_MIN_ROWS:,}")
    print(f"Sensitivity thresholds: {SENSITIVITY_THRESHOLDS}")
    print(f"Webb wild bootstrap reps: {WILD_REPS:,}")
    print("=" * 112)

    data, grid, fold_map, n_full_blocks = load_data()
    signature = config_signature()

    sample_summary = (
        data.groupby("_lccs_group", observed=True)
        .agg(
            n=(OUTCOME, "size"),
            year_min=(YEAR_COL, "min"),
            year_max=(YEAR_COL, "max"),
            n_years=(YEAR_COL, "nunique"),
            n_spatial_blocks=("_spatial_block", "nunique"),
        )
        .reset_index().rename(columns={"_lccs_group": "lccs_group"})
    )
    sample_summary["lccs_name"] = sample_summary["lccs_group"].map(lccs_names)
    sample_summary.to_csv(out / "lccs_sample_summary.csv", index=False)

    yearly_results, diagnostics_all, components_all, audit_rows = [], [], [], []
    total_tasks = len(GROUP_ORDER) * (END_YEAR - START_YEAR + 1)
    task = 0
    t_all = time.time()

    cols = list(dict.fromkeys([
        OUTCOME, EXPOSURE, YEAR_COL, MONTH_COL, LCCS_COL, "_spatial_block",
    ] + CONTROL_VARS))

    for group_code in GROUP_ORDER:
        print("\n" + "#" * 112)
        print(f"LCCS {group_code}: {lccs_names[group_code]}")
        print("#" * 112)
        for year in range(START_YEAR, END_YEAR + 1):
            task += 1
            cache_json = cache_dir / f"lccs_{group_code}_year_{year}.json"
            cache_diag = cache_dir / f"lccs_{group_code}_year_{year}_diagnostics.csv"
            cache_comp = cache_dir / f"lccs_{group_code}_year_{year}_components.csv"

            if not FORCE_RERUN and cache_json.exists() and cache_diag.exists() and cache_comp.exists():
                cached = json.loads(cache_json.read_text(encoding="utf-8"))
                if cached.get("config_signature") == signature:
                    result = cached["result"]
                    yearly_results.append(result)
                    diagnostics_all.append(pd.read_csv(cache_diag))
                    components_all.append(pd.read_csv(cache_comp))
                    audit_rows.append({
                        "lccs_group": group_code,
                        "lccs_name": lccs_names[group_code],
                        "year": year,
                        "n": int(result["n"]),
                        "n_spatial_blocks": int(result["n_spatial_blocks"]),
                        "represented_folds": N_SPATIAL_FOLDS,
                        "fit_status": "reused_cache",
                    })
                    print(f"[resume {task}/{total_tasks}] {year}: n={int(result['n']):,}, effect={result['effect_per_10ug_m3']:.6f}")
                    continue

            work = data.loc[
                (data["_lccs_group"] == group_code) & (data[YEAR_COL] == year), cols,
            ].copy()
            work = (
                work.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[OUTCOME, EXPOSURE, MONTH_COL, LCCS_COL] + CONTROL_VARS)
                .reset_index(drop=True)
            )

            n = len(work)
            n_blocks = int(work["_spatial_block"].nunique())
            represented_folds = int(work["_spatial_block"].map(fold_map).dropna().nunique())

            if (
                n < FIT_MIN_ROWS
                or n_blocks < MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR
                or represented_folds < N_SPATIAL_FOLDS
            ):
                audit_rows.append({
                    "lccs_group": group_code,
                    "lccs_name": lccs_names[group_code],
                    "year": year,
                    "n": n,
                    "n_spatial_blocks": n_blocks,
                    "represented_folds": represented_folds,
                    "fit_status": "not_fitted_below_sensitivity_support",
                })
                print(f"[skip {task}/{total_tasks}] {year}: n={n:,}, blocks={n_blocks}, folds={represented_folds}")
                del work
                continue

            print(
                f"[fit {task}/{total_tasks}] {year}: n={n:,}, blocks={n_blocks:,}, "
                f"elapsed={(time.time()-t_all)/60:.1f} min"
            )
            result, diag, comp = fit_group_year(work, group_code, year, fold_map)
            yearly_results.append(result)
            diagnostics_all.append(diag)
            components_all.append(comp)
            audit_rows.append({
                "lccs_group": group_code,
                "lccs_name": lccs_names[group_code],
                "year": year,
                "n": n,
                "n_spatial_blocks": n_blocks,
                "represented_folds": represented_folds,
                "fit_status": "fitted",
            })

            diag.to_csv(cache_diag, index=False)
            comp.to_csv(cache_comp, index=False)
            cache_json.write_text(
                json.dumps({
                    "config_signature": signature,
                    "result": json_safe(result),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            del work, diag, comp
            cleanup_gpu()

    if not yearly_results:
        raise RuntimeError("No LCCS group-year DML model was successfully fitted.")

    all_yearly = (
        pd.DataFrame(yearly_results)
        .sort_values(["lccs_group", "year"])
        .drop_duplicates(["lccs_group", "year"])
        .reset_index(drop=True)
    )
    all_diag = pd.concat(diagnostics_all, ignore_index=True).drop_duplicates(
        ["lccs_group", "year", "stage", "spatial_fold"]
    )
    all_comp = pd.concat(components_all, ignore_index=True).drop_duplicates(
        ["lccs_group", "year", "spatial_block"]
    )
    support_audit = (
        pd.DataFrame(audit_rows)
        .sort_values(["lccs_group", "year"])
        .drop_duplicates(["lccs_group", "year"], keep="last")
    )

    # BH adjustment of the wild month-cluster sensitivity within each LCCS group.
    all_yearly["wild_month_q_value_BH_within_lccs"] = np.nan
    all_yearly["wild_month_fdr_significant_0.05_within_lccs"] = False
    for g, idx in all_yearly.groupby("lccs_group", observed=True).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        q = bh_adjust(all_yearly.loc[ii, "wild_month_cluster_twoway_studentized_p_value"].to_numpy(float))
        all_yearly.loc[ii, "wild_month_q_value_BH_within_lccs"] = q
        all_yearly.loc[ii, "wild_month_fdr_significant_0.05_within_lccs"] = np.isfinite(q) & (q <= FDR_ALPHA)

    # Primary 10,000-observation analysis. Lower-threshold cells remain sensitivity-only.
    primary_yearly = all_yearly.loc[
        (all_yearly["n"] >= PRIMARY_MIN_ROWS)
        & (all_yearly["n_spatial_blocks"] >= MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR)
    ].copy()
    primary_cells = primary_yearly[["lccs_group", "year"]].drop_duplicates()
    primary_diag = restrict_cells(all_diag, primary_cells)
    primary_comp = restrict_cells(all_comp, primary_cells)

    primary_yearly.to_csv(out / "lccs_yearly_effects.csv", index=False)
    primary_diag.to_csv(out / "lccs_fold_diagnostics.csv", index=False)
    primary_comp.to_csv(out / "lccs_spatial_score_components.csv", index=False)

    primary_skipped = support_audit.loc[
        (support_audit["n"] < PRIMARY_MIN_ROWS)
        | (support_audit["n_spatial_blocks"] < MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR)
        | (support_audit["represented_folds"] < N_SPATIAL_FOLDS)
    ].copy()
    if primary_skipped.empty:
        primary_skipped["reason"] = pd.Series(dtype=str)
    else:
        primary_skipped["reason"] = primary_skipped.apply(
            lambda r: (
                f"n={int(r['n']):,}, blocks={int(r['n_spatial_blocks'])}, folds={int(r['represented_folds'])}; "
                f"primary minimum rows={PRIMARY_MIN_ROWS:,}, blocks={MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR}, "
                f"folds={N_SPATIAL_FOLDS}"
            ),
            axis=1,
        )
    primary_skipped.to_csv(out / "lccs_skipped_group_years.csv", index=False)

    _, _, pooled_primary = pool_at_threshold(all_yearly, all_comp, PRIMARY_MIN_ROWS)
    pooled_df = pooled_primary.drop(columns=["sample_threshold"], errors="ignore")
    pooled_df.to_csv(out / "lccs_pooled_effects.csv", index=False)

    if len(pooled_df) >= 2:
        omnibus, pairwise = heterogeneity_tests(primary_comp, pooled_df)
    else:
        omnibus = pd.DataFrame([{
            "test": "LCCS omnibus heterogeneity",
            "status": "SKIPPED",
            "reason": "Fewer than two pooled LCCS groups met the primary support requirement.",
        }])
        pairwise = pd.DataFrame()
    omnibus.to_csv(out / "lccs_heterogeneity_omnibus.csv", index=False)
    pairwise.to_csv(out / "lccs_pairwise_heterogeneity.csv", index=False)

    # Single designated time-stability test only; no 2017 breakpoint tests.
    trend_parts = []
    for g in GROUP_ORDER:
        if (primary_comp["lccs_group"] == g).any():
            trend_parts.append(group_time_tests(primary_comp, g))
    time_tests = pd.concat(trend_parts, ignore_index=True) if trend_parts else pd.DataFrame()
    time_tests.to_csv(out / "lccs_linear_time_trends.csv", index=False)

    # Pooled wild year-cluster sensitivity; primary pooled inference remains space x year clustered.
    wild_year_rows = []
    for g in pooled_df.get("lccs_group", pd.Series(dtype=int)).astype(int).tolist():
        wild_year_rows.append(
            wild_year_cluster_group(primary_comp, g, WILD_SEED + 100_000 + int(g))
        )
    wild_year_df = pd.DataFrame(wild_year_rows)
    if not wild_year_df.empty:
        wild_year_df["q_value_BH"] = bh_adjust(wild_year_df["wild_cluster_p_value"].to_numpy(float))
        wild_year_df["fdr_significant_0.05"] = wild_year_df["q_value_BH"] <= FDR_ALPHA
    wild_year_df.to_csv(out / "lccs_pooled_wild_year_cluster_sensitivity.csv", index=False)

    # One main paper figure by default.
    if not pooled_df.empty:
        forest_plot(pooled_df, figure_dir)
    if MAKE_ANNUAL_HEATMAP:
        annual_heatmap(primary_yearly, figure_dir)

    master_cols = [
        "lccs_group", "lccs_name", "n_valid_years", "negative_years",
        "median_annual_effect_per_10ug_m3", "effect_per_10ug_m3",
        "ci95_low_t_per_10ug_m3", "ci95_high_t_per_10ug_m3",
        "p_value_t", "q_value_BH", "fdr_significant_0.05",
        "mean_treatment_oof_r2", "min_treatment_oof_r2",
        "mean_outcome_oof_r2", "min_outcome_oof_r2",
    ]
    pooled_df[master_cols].to_csv(out / "LCCS_DML_MASTER_SUMMARY.csv", index=False)

    support_sens, stability_sens, nuisance_sens = run_threshold_sensitivity(
        all_yearly, all_diag, all_comp, support_audit, out
    )

    metadata = {
        "run_version": RUN_VERSION,
        "cache_signature_version": SCRIPT_VERSION,
        "main_program_alignment": "reviewer-compact-dml-full-rerun-v6.0 (2026-09-22)",
        "input_file": INPUT_FILE,
        "output_dir": OUTPUT_DIR,
        "primary_estimand": "year-stratified LCCS-specific spatial-blocked DML pooled across years",
        "primary_model": "Model A",
        "seasonality": "11 calendar-month fixed-effect indicators; January reference",
        "national_fold_construction": "balanced block-factor-20 map from full panel before LCCS restriction",
        "nationwide_reference_spatial_blocks": n_full_blocks,
        "lccs_subclass_adjustment": "nominal one-hot within merged ecological group; smallest raw code as reference",
        "primary_min_rows": PRIMARY_MIN_ROWS,
        "sensitivity_thresholds": SENSITIVITY_THRESHOLDS,
        "fit_min_rows": FIT_MIN_ROWS,
        "min_spatial_blocks_per_group_year": MIN_SPATIAL_BLOCKS_PER_GROUP_YEAR,
        "lccs_names": lccs_names,
        "group_definitions": group_definitions,
        "base_controls": CONTROL_VARS,
        "crossfit": "year-specific 5-fold spatial-blocked cross-fitting using nationwide fold map",
        "annual_primary_inference": "spatial + month two-way clustered orthogonal-score inference",
        "annual_small_cluster_sensitivity": "Webb wild month-cluster bootstrap-t with spatial x month two-way studentization",
        "pooled_inference": "spatial + year two-way clustered orthogonal-score inference",
        "pooled_small_cluster_sensitivity": "Webb wild year-cluster bootstrap",
        "heterogeneity_test": "multivariate spatial+year clustered orthogonal-score covariance; omnibus + Holm pairwise contrasts",
        "time_test": "single designated linear effect trend only",
        "removed": [
            "random-effects meta-analysis",
            "all 2017 pre/post and segmented breakpoint tests",
            "harmonic month sine/cosine seasonality",
        ],
        "threshold_note": "10,000 is the primary conservative support guardrail; 5,000 and 20,000 are integrated sensitivity thresholds.",
        "hyperparameters": HYPERPARAMS,
        "wild_reps": WILD_REPS,
        "config_signature": signature,
    }
    (out / "LCCS_DML_METADATA.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 112)
    print("LCCS ALIGNED ANALYSIS FINISHED")
    print("=" * 112)
    for name in [
        "LCCS_DML_MASTER_SUMMARY.csv",
        "lccs_yearly_effects.csv",
        "lccs_pooled_effects.csv",
        "lccs_heterogeneity_omnibus.csv",
        "lccs_pairwise_heterogeneity.csv",
        "lccs_linear_time_trends.csv",
        "lccs_pooled_wild_year_cluster_sensitivity.csv",
        "lccs_fold_diagnostics.csv",
        "figures/Figure_LCCS_Pooled_Forest.png",
        "sample_threshold_sensitivity/threshold_support_summary.csv",
        "sample_threshold_sensitivity/threshold_stability_summary_vs_10000.csv",
        "sample_threshold_sensitivity/threshold_nuisance_diagnostics.csv",
    ]:
        print(out / name)
    print("=" * 112)

if __name__ == "__main__":
    main()
