#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer-resistant, compact O3 -> SIF DML evidence chain, FULL RERUN.

Scientific sequence
-------------------
00. Full-period all-data space x time blocked DML (single 2000-2022 nuisance architecture)
01. Primary year-stratified spatial-blocked DML + O3 identifying-variation/support diagnostics
02. Strict within-pixel, within-year (pixel-year FE) DML
03. Interannual pixel-month within-transformed estimand + exact paired raw-vs-within design check
04. Conditional future-O3 negative-control exposure (future O3 tested jointly with current O3)
05. Spatial block-scale sensitivity
06. Covariate-set sensitivity (A/B/C; Model C is over-adjustment sensitivity only)
07. Small-cluster robust inference: two-way cluster-robust SE + Webb wild month-cluster bootstrap-t sensitivity + pooled wild year-cluster bootstrap
08. Temporal stability (single designated linear trend test only)

Design principles
-----------------
- The primary estimand is chosen from the scientific question, not from statistical significance.
- EXP00 transparently restores the full-period all-data DML as a distinct specification: one nuisance architecture is fit over 2000-2022 with 5 spatial x 5 contiguous temporal cross-fitting. It is reported regardless of significance and is not used to select the primary estimand.
- "Year-stratified" is used instead of claiming the primary model is pure within-pixel temporal DML.
- The interannual within-transformed model is treated as a distinct estimand, not a stricter version of the primary model.
- To isolate the anomaly transformation itself, raw and anomaly models are additionally fit on identical rows,
  identical spatial folds, identical control columns, and the same learner architecture.
- The future-O3 placebo is conditional: current and future O3 are residualized on the same X and entered
  jointly, so the placebo coefficient tests whether future O3 adds information after contemporaneous O3.
- Cheap support/information-concentration diagnostics are retained; leverage-trimming refits are removed.
- Seasonality is controlled flexibly with 11 calendar-month fixed-effect indicators rather than a single harmonic.
- A lightweight empirical spatial correlogram of OOF residuals/orthogonal scores is reported to contextualize the reference block scale; it does not select the model post hoc.
- Annual primary uncertainty is spatial x month two-way cluster robust. Webb wild resampling is applied only to month clusters as a small-cluster sensitivity and is explicitly NOT described as a multiway wild-cluster bootstrap.
- Frozen nuisance hyperparameters are not retuned on the final results; an OOF performance sanity table is exported for the final month-FE specification.
- LCCS is treated as a nominal land-cover variable and one-hot encoded at model-matrix construction; the raw class codes are never used as an ordered/continuous predictor.

Deliberately removed
--------------------
- LCCS/month/spatial heterogeneity analyses (already handled elsewhere)
- pre/post 2017 2D reruns
- 3-fold vs 5-fold temporal sensitivity
- random-effects meta-analysis
- leverage trimming refits
- duplicated strict-FE formal time tests
- all post-hoc calendar breakpoint tests (including 2017)

The script reads the raw panel once and reruns every retained model from scratch.
It does NOT reuse previous result files.

Jupyter:
    %run /path/to/dml_reviewer_compact_v6_1_full_data.py
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
INPUT_FILE = r"/root/autodl-tmp/wq/matched_data_albers_all1.csv"
OUTPUT_DIR = r"/root/autodl-tmp/wql/0922/dml_reviewer_compact_v6_1_full_data"

OUTCOME = "SIF"
EXPOSURE = "O3"
XCOL, YCOL, YEAR, MONTH = "x", "y", "year", "month"
START_YEAR, END_YEAR = 2000, 2022

SEED = 42
DEVICE = "cuda"
N_JOBS = 8
MAX_BIN = 256
N_SFOLD = 5
FULL_TFOLD = 5
REF_BLOCK_FACTOR = 20
BLOCK_FACTORS = [10, 20, 30]
STRICT_MIN_MONTHS = 6
MATCHED_MIN_BASELINE_YEARS = 5
WILD_REPS = 9999
WILD_SEED = 20260921
FDR_ALPHA = 0.05

# Lightweight spatial-dependence diagnostic; this is NOT an additional DML experiment.
SPATIAL_DIAG_SAMPLE_PER_YEAR_MONTH = 1500
SPATIAL_DIAG_MAX_PAIRS_PER_YEAR_MONTH = 100000
SPATIAL_DIAG_DISTANCE_BINS_KM = [0, 25, 50, 75, 100, 125, 150, 200]
SPATIAL_DIAG_NEAR_ZERO_ABS = 0.05

SCRIPT_VERSION = "reviewer-compact-dml-full-rerun-v6.1-full-data"

# Every retained experiment is rerun from scratch.
FORCE_RERUN = True

# EXP00: transparently retain the full-period all-data DML.
# This is computationally expensive and is ON by default because it is part of the
# manuscript-facing evidence chain in v6.1.
RUN_FULL_PERIOD_ALL_DATA = True

# Configured from the observed raw LCCS codes in load_data().  The raw codes are
# nominal labels, not ordered measurements.  One reference category is omitted
# solely to avoid redundant dummy columns; XGBoost receives only binary indicators.
LCCS_LEVELS = tuple()
LCCS_REFERENCE = None

MODEL_A = [
    "DEM", "lccs", "t2m", "ssrd", "tp", "u10", "v10", "sp", "stl1", "swvl1"
]
MODEL_B = MODEL_A + [
    "stl2", "stl3", "stl4", "stl5",
    "swvl2", "swvl3", "swvl4", "swvl5",
    "ssr", "str", "src", "pev"
]
MODEL_C = MODEL_B + ["lai_hv", "lai_lv", "evavt", "e"]

TIME_VARYING_A = ["t2m", "ssrd", "tp", "u10", "v10", "sp", "stl1", "swvl1"]
# January is the reference category. Full calendar-month indicators avoid imposing
# a sinusoidal seasonal shape on either O3 or SIF.
MONTH_FE = [f"_month_{m:02d}" for m in range(2, 13)]
FULL_TIME = [XCOL, YCOL, "_year_centered"] + MONTH_FE
YEAR_TIME = [XCOL, YCOL] + MONTH_FE

# Frozen nuisance-model hyperparameters from the previous nested/block-tuning stage.
HP = {
    "treatment": {
        "n_estimators": 1053, "max_depth": 9, "learning_rate": 0.10216433662775991,
        "min_child_weight": 25, "gamma": 4.309033817797793,
        "subsample": 0.8908973518801342, "colsample_bytree": 0.8492767336660302,
        "reg_alpha": 0.003354690482344186, "reg_lambda": 2.033716357895456,
    },
    "outcome": {
        "n_estimators": 1131, "max_depth": 10, "learning_rate": 0.0266047322246523,
        "min_child_weight": 3, "gamma": 0.5110259054946181,
        "subsample": 0.7833701909759339, "colsample_bytree": 0.9168109399871214,
        "reg_alpha": 0.007222709294055777, "reg_lambda": 1.912111490897236,
    },
    "fold": {
        1: {
            "treatment": {"n_estimators":1093,"max_depth":10,"learning_rate":0.05177114960937403,"min_child_weight":5,"gamma":4.087096472536959,"subsample":0.9741365591234228,"colsample_bytree":0.8626229875841074,"reg_alpha":0.0001795196009889072,"reg_lambda":0.6116188039009353},
            "outcome": {"n_estimators":821,"max_depth":10,"learning_rate":0.027431924170236294,"min_child_weight":23,"gamma":0.2652235628561056,"subsample":0.9155232846946602,"colsample_bytree":0.8085067963737109,"reg_alpha":0.0002508736288208214,"reg_lambda":0.8558959423479916},
        },
        2: {
            "treatment": {"n_estimators":465,"max_depth":10,"learning_rate":0.1084879663535185,"min_child_weight":22,"gamma":4.7789762199331856,"subsample":0.9731243010223389,"colsample_bytree":0.8046099456572131,"reg_alpha":0.1815903805353822,"reg_lambda":3.0896123510443676},
            "outcome": {"n_estimators":640,"max_depth":10,"learning_rate":0.09305457914202489,"min_child_weight":12,"gamma":0.06731179180203606,"subsample":0.9321482076519811,"colsample_bytree":0.6570816737624354,"reg_alpha":0.063388836802,"reg_lambda":0.3339171631340325},
        },
        3: {
            "treatment": {"n_estimators":856,"max_depth":10,"learning_rate":0.07088991197186671,"min_child_weight":22,"gamma":3.4695596681423746,"subsample":0.730305769485771,"colsample_bytree":0.8633577549465097,"reg_alpha":1.6081251890506334,"reg_lambda":1.3156844506060563},
            "outcome": {"n_estimators":694,"max_depth":8,"learning_rate":0.08865027271736356,"min_child_weight":17,"gamma":0.8555933076230421,"subsample":0.9893067125456055,"colsample_bytree":0.7872612725128207,"reg_alpha":0.0014882953649653075,"reg_lambda":1.3382091592127645},
        },
        4: {
            "treatment": {"n_estimators":973,"max_depth":9,"learning_rate":0.13927351714582462,"min_child_weight":8,"gamma":4.250842262998735,"subsample":0.7093629292447272,"colsample_bytree":0.9236533356882819,"reg_alpha":0.003774035416122741,"reg_lambda":0.6016154625724639},
            "outcome": {"n_estimators":434,"max_depth":9,"learning_rate":0.03453565453001152,"min_child_weight":15,"gamma":0.05838358184313053,"subsample":0.7458750631012042,"colsample_bytree":0.657744187493326,"reg_alpha":2.3024752173695977,"reg_lambda":0.27082872805683933},
        },
        5: {
            "treatment": {"n_estimators":1119,"max_depth":10,"learning_rate":0.03369745858341406,"min_child_weight":10,"gamma":3.5582544045908997,"subsample":0.7078989845483684,"colsample_bytree":0.6523944010534592,"reg_alpha":0.2237474907268608,"reg_lambda":2.4851882030793533},
            "outcome": {"n_estimators":344,"max_depth":10,"learning_rate":0.05356438720014123,"min_child_weight":29,"gamma":0.33306377256122455,"subsample":0.6731722235580344,"colsample_bytree":0.9951500209194727,"reg_alpha":1.0174540039767859,"reg_lambda":21.95556710775531},
        },
    },
}

# =============================================================================
# BASIC HELPERS
# =============================================================================
def mkdir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_gpu():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def month_num(s):
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s.astype(str).str.extract(r"(\d{1,2})$")[0], errors="coerce")
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().any() or not s.between(1, 12).all():
        raise ValueError("month must be 1..12")
    return s.astype(np.int16)


def add_time_features(d):
    """Add a centered year term and flexible calendar-month fixed-effect indicators."""
    y = d[YEAR].to_numpy(np.float32)
    m = d[MONTH].to_numpy(np.int16)
    d["_year_centered"] = (y - START_YEAR).astype(np.float32)
    for mm in range(2, 13):
        d[f"_month_{mm:02d}"] = (m == mm).astype(np.float32)


def grid_info(d):
    xs = np.sort(d[XCOL].dropna().unique())
    ys = np.sort(d[YCOL].dropna().unique())
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    if dx <= 0 or dy <= 0:
        raise ValueError("invalid grid spacing")
    return dx, dy, float(d[XCOL].min()), float(d[YCOL].min())


def block_id(x, y, grid, factor):
    dx, dy, x0, y0 = grid
    bx = np.floor((x.astype(np.float64) - x0) / (dx * factor)).astype(np.int32)
    by = np.floor((y.astype(np.float64) - y0) / (dy * factor)).astype(np.int32)
    return bx.astype(np.int64) * 10_000_000 + by.astype(np.int64)


def pixel_id(x, y, grid):
    dx, dy, x0, y0 = grid
    ix = np.rint((x.astype(np.float64) - x0) / dx).astype(np.int32)
    iy = np.rint((y.astype(np.float64) - y0) / dy).astype(np.int32)
    return ix.astype(np.int64) * 10_000_000 + iy.astype(np.int64)


def safe_r2(y, p):
    return np.nan if len(y) < 2 or np.std(y) == 0 else float(r2_score(y, p))


def metrics(y, p):
    r = y.astype(np.float64) - p.astype(np.float64)
    return dict(
        r2=safe_r2(y, p),
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        mae=float(mean_absolute_error(y, p)),
        bias=float(r.mean()),
    )


def residual_support_metrics(dr):
    """Cheap overlap/information-concentration diagnostics from treatment residuals."""
    d = np.asarray(dr, dtype=np.float64)
    if len(d) < 2:
        raise ValueError("need at least two residuals")
    q = np.quantile(d, [0.01, 0.05, 0.50, 0.95, 0.99])
    a = np.abs(d)
    den = float(d @ d)
    q95a, q99a = np.quantile(a, [0.95, 0.99])
    return dict(
        n=int(len(d)),
        residual_mean=float(d.mean()),
        residual_sd=float(d.std(ddof=1)),
        residual_q01=float(q[0]),
        residual_q05=float(q[1]),
        residual_median=float(q[2]),
        residual_q95=float(q[3]),
        residual_q99=float(q[4]),
        denominator_raw=den,
        denominator_per_observation=den / len(d),
        top5pct_abs_residual_denominator_share=float(np.square(d[a >= q95a]).sum() / den) if den > 0 else np.nan,
        top1pct_abs_residual_denominator_share=float(np.square(d[a >= q99a]).sum() / den) if den > 0 else np.nan,
    )



def spatial_dependence_diagnostic(w, dr, yr, theta, year):
    """
    Lightweight empirical spatial correlogram on OOF quantities.

    Pairs are formed only within the same calendar month and year, so the
    diagnostic targets residual spatial dependence rather than seasonality.
    To keep the calculation cheap on the ~83M-row panel, at most
    SPATIAL_DIAG_SAMPLE_PER_YEAR_MONTH observations are sampled per year-month,
    and at most SPATIAL_DIAG_MAX_PAIRS_PER_YEAR_MONTH local pairs are retained.

    Two quantities are examined:
      1) treatment residual, O3 - E[O3|X]
      2) orthogonal score, d * (y_res - theta*d)

    This diagnostic is descriptive and is not used to choose the primary
    block scale after looking at the effect estimate.
    """
    x = w[XCOL].to_numpy(np.float64)
    ycoord = w[YCOL].to_numpy(np.float64)
    months = w[MONTH].to_numpy(np.int16)
    d = np.asarray(dr, dtype=np.float64)
    yres = np.asarray(yr, dtype=np.float64)
    score = d * (yres - float(theta) * d)

    edges_km = np.asarray(SPATIAL_DIAG_DISTANCE_BINS_KM, dtype=float)
    edges_m = edges_km * 1000.0
    max_dist_m = float(edges_m[-1])
    rows = []

    for mm in range(1, 13):
        idx = np.flatnonzero(months == mm)
        if len(idx) < 20:
            continue
        rng = np.random.default_rng(SEED + int(year) * 100 + mm)
        if len(idx) > SPATIAL_DIAG_SAMPLE_PER_YEAR_MONTH:
            idx = rng.choice(idx, size=SPATIAL_DIAG_SAMPLE_PER_YEAR_MONTH, replace=False)

        coords = np.column_stack([x[idx], ycoord[idx]])
        valid = np.isfinite(coords).all(axis=1)
        idx = idx[valid]
        coords = coords[valid]
        if len(idx) < 20:
            continue

        tree = cKDTree(coords)
        try:
            pairs = tree.query_pairs(max_dist_m, output_type="ndarray")
        except TypeError:
            pairs = np.asarray(list(tree.query_pairs(max_dist_m)), dtype=np.int64)
        if pairs.size == 0:
            continue
        if pairs.ndim == 1:
            pairs = pairs.reshape(-1, 2)
        if len(pairs) > SPATIAL_DIAG_MAX_PAIRS_PER_YEAR_MONTH:
            take = rng.choice(len(pairs), size=SPATIAL_DIAG_MAX_PAIRS_PER_YEAR_MONTH, replace=False)
            pairs = pairs[take]

        delta = coords[pairs[:, 0]] - coords[pairs[:, 1]]
        dist = np.sqrt(np.sum(delta * delta, axis=1))

        for variable, values in [
            ("treatment_residual", d[idx]),
            ("orthogonal_score", score[idx]),
        ]:
            values = np.asarray(values, dtype=np.float64)
            sd = float(values.std(ddof=1))
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (values - float(values.mean())) / sd
            products = z[pairs[:, 0]] * z[pairs[:, 1]]

            for b in range(len(edges_m) - 1):
                lo = edges_m[b]
                hi = edges_m[b + 1]
                if b == len(edges_m) - 2:
                    mask = (dist >= lo) & (dist <= hi)
                else:
                    mask = (dist >= lo) & (dist < hi)
                n_pairs = int(mask.sum())
                if n_pairs == 0:
                    continue
                prod_sum = float(products[mask].sum())
                rows.append(dict(
                    year=int(year),
                    month=int(mm),
                    variable=variable,
                    distance_low_km=float(edges_km[b]),
                    distance_high_km=float(edges_km[b + 1]),
                    distance_mid_km=float((edges_km[b] + edges_km[b + 1]) / 2),
                    n_pairs=n_pairs,
                    standardized_product_sum=prod_sum,
                    mean_standardized_product=prod_sum / n_pairs,
                ))

    return pd.DataFrame(rows)


def summarize_spatial_dependence(diag, reference_block_km, out):
    """Aggregate the sampled correlogram and save a compact table/figure."""
    if diag is None or len(diag) == 0:
        empty = pd.DataFrame()
        empty.to_csv(out / "exp05_spatial_dependence_correlogram.csv", index=False)
        return empty, pd.DataFrame()

    d = diag.copy()
    d.to_csv(out / "exp05_spatial_dependence_correlogram_by_year_month.csv", index=False)
    agg = (
        d.groupby(
            ["variable", "distance_low_km", "distance_high_km", "distance_mid_km"],
            observed=True,
            as_index=False,
        )
        .agg(
            n_pairs=("n_pairs", "sum"),
            standardized_product_sum=("standardized_product_sum", "sum"),
        )
    )
    agg["mean_standardized_product"] = (
        agg["standardized_product_sum"] / agg["n_pairs"]
    )
    agg["reference_block_km"] = float(reference_block_km)
    agg.to_csv(out / "exp05_spatial_dependence_correlogram.csv", index=False)

    summary_rows = []
    for variable, g in agg.groupby("variable", observed=True):
        g = g.sort_values("distance_mid_km").reset_index(drop=True)
        near_zero = np.nan
        vals = np.abs(g["mean_standardized_product"].to_numpy(float))
        highs = g["distance_high_km"].to_numpy(float)
        # Require two consecutive bins below the threshold to avoid a noisy crossing.
        for i in range(max(0, len(vals) - 1)):
            if vals[i] <= SPATIAL_DIAG_NEAR_ZERO_ABS and vals[i + 1] <= SPATIAL_DIAG_NEAR_ZERO_ABS:
                near_zero = float(highs[i])
                break
        j = int(np.argmin(np.abs(g["distance_mid_km"].to_numpy(float) - reference_block_km)))
        summary_rows.append(dict(
            variable=variable,
            reference_block_km=float(reference_block_km),
            nearest_bin_mid_km=float(g.loc[j, "distance_mid_km"]),
            standardized_product_near_reference=float(g.loc[j, "mean_standardized_product"]),
            first_sustained_abs_product_below_threshold_km=near_zero,
            near_zero_abs_threshold=float(SPATIAL_DIAG_NEAR_ZERO_ABS),
            interpretation=(
                "Descriptive OOF spatial-dependence diagnostic only; block scale was not selected "
                "from this diagnostic or from the estimated O3-SIF effect."
            ),
        ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "exp05_spatial_dependence_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for variable, g in agg.groupby("variable", observed=True):
        g = g.sort_values("distance_mid_km")
        ax.plot(
            g["distance_mid_km"], g["mean_standardized_product"],
            marker="o", label=variable.replace("_", " ")
        )
    ax.axhline(0, ls="--", lw=1)
    ax.axhline(SPATIAL_DIAG_NEAR_ZERO_ABS, ls=":", lw=1)
    ax.axhline(-SPATIAL_DIAG_NEAR_ZERO_ABS, ls=":", lw=1)
    ax.axvline(reference_block_km, ls="--", lw=1)
    ax.set_xlabel("Pairwise distance (km)")
    ax.set_ylabel("Mean standardized pair product")
    ax.set_title("OOF spatial-dependence diagnostic")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "Figure_Exp05_spatial_dependence_diagnostic.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    return agg, summary


def _format_lccs_level(v):
    """Stable, filesystem/column-safe label for an observed LCCS code."""
    x = float(v)
    if np.isfinite(x) and x.is_integer():
        return str(int(x))
    return (f"{x:g}").replace("-", "m").replace(".", "p")


def configure_lccs_encoding(d):
    """
    Configure a deterministic one-hot representation for nominal LCCS codes.

    The smallest observed finite code is used as the reference category.  The
    choice of reference does not impose an ordering: all non-reference classes
    enter XGBoost as separate binary indicators.  Rows with missing LCCS remain
    missing and are removed by the same complete-case filtering used elsewhere.
    """
    global LCCS_LEVELS, LCCS_REFERENCE
    if "lccs" not in d.columns:
        LCCS_LEVELS = tuple()
        LCCS_REFERENCE = None
        return dict(mode="absent", levels=[], reference=None, n_dummy_features=0)

    vals = pd.to_numeric(d["lccs"], errors="coerce")
    levels = np.sort(vals[np.isfinite(vals)].unique().astype(np.float64))
    if len(levels) == 0:
        LCCS_LEVELS = tuple()
        LCCS_REFERENCE = None
        return dict(mode="present_but_no_finite_levels", levels=[], reference=None, n_dummy_features=0)

    LCCS_LEVELS = tuple(float(v) for v in levels)
    LCCS_REFERENCE = float(levels[0])
    return dict(
        mode="nominal_one_hot",
        levels=[float(v) for v in LCCS_LEVELS],
        reference=float(LCCS_REFERENCE),
        n_dummy_features=max(0, len(LCCS_LEVELS) - 1),
        missing_count=int(vals.isna().sum()),
        note="Raw LCCS codes are not used as a continuous/ordered predictor.",
    )


def expanded_control_names(controls):
    """Return feature names in the exact column order used by make_design_matrix()."""
    controls = list(controls)
    names = [c for c in controls if c != "lccs"]
    if "lccs" in controls:
        if LCCS_REFERENCE is None or len(LCCS_LEVELS) == 0:
            raise RuntimeError("LCCS requested as a control before categorical encoding was configured")
        names.extend(
            f"_lccs_cat_{_format_lccs_level(lv)}"
            for lv in LCCS_LEVELS
            if float(lv) != float(LCCS_REFERENCE)
        )
    return names


def make_design_matrix(w, controls):
    """
    Build a dense float32 model matrix while treating LCCS as nominal.

    This expands LCCS only when a model matrix is constructed instead of adding
    many dummy columns to the ~83M-row master DataFrame, limiting persistent RAM
    overhead.  All folds/estimands use the same globally configured LCCS levels.
    """
    controls = list(controls)
    numeric = [c for c in controls if c != "lccs"]
    feature_names = expanded_control_names(controls)
    n = len(w)
    X = np.empty((n, len(feature_names)), dtype=np.float32)

    j = 0
    if numeric:
        a = w[numeric].to_numpy(np.float32)
        X[:, :len(numeric)] = a
        j = len(numeric)
        del a

    if "lccs" in controls:
        vals = pd.to_numeric(w["lccs"], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(vals).all():
            raise ValueError("LCCS contains missing/non-finite values after complete-case filtering")
        for lv in LCCS_LEVELS:
            if float(lv) == float(LCCS_REFERENCE):
                continue
            X[:, j] = (vals == float(lv))
            j += 1

    if j != X.shape[1]:
        raise RuntimeError("design-matrix feature count mismatch")
    return X, feature_names


def lccs_encoding_metadata():
    return dict(
        mode=("nominal_one_hot" if LCCS_REFERENCE is not None else "absent"),
        levels=[float(v) for v in LCCS_LEVELS],
        reference=(None if LCCS_REFERENCE is None else float(LCCS_REFERENCE)),
        n_dummy_features=max(0, len(LCCS_LEVELS) - 1),
        feature_names=(
            [] if LCCS_REFERENCE is None else
            [f"_lccs_cat_{_format_lccs_level(v)}" for v in LCCS_LEVELS if float(v) != float(LCCS_REFERENCE)]
        ),
        interpretation="Nominal categorical adjustment; no numerical ordering of LCCS class codes is assumed.",
    )


def load_data():
    hdr = pd.read_csv(INPUT_FILE, nrows=0)
    cols = set(hdr.columns)

    # Robust LCCS alias handling.
    lccs_source = "lccs" if "lccs" in cols else ("LCCS" if "LCCS" in cols else None)
    model_c_src = []
    for c in MODEL_C:
        if c == "lccs":
            if lccs_source is not None:
                model_c_src.append(lccs_source)
        elif c in cols:
            model_c_src.append(c)

    req = {OUTCOME, EXPOSURE, XCOL, YCOL, YEAR, MONTH}
    if req - cols:
        raise ValueError(f"missing required columns: {sorted(req - cols)}")

    use = [OUTCOME, EXPOSURE, XCOL, YCOL, YEAR, MONTH] + model_c_src
    use = list(dict.fromkeys(use))
    print(f"Reading {len(use)} columns from raw panel...")
    t0 = time.time()
    d = pd.read_csv(INPUT_FILE, usecols=use, low_memory=False)
    print(f"Rows={len(d):,}; read time={(time.time()-t0)/60:.1f} min")

    if lccs_source == "LCCS" and "lccs" not in d.columns:
        d = d.rename(columns={"LCCS": "lccs"})

    d[YEAR] = pd.to_numeric(d[YEAR], errors="raise").astype(np.int16)
    d[MONTH] = month_num(d[MONTH])
    d = d[d[YEAR].between(START_YEAR, END_YEAR)].copy()

    d[XCOL] = pd.to_numeric(d[XCOL], errors="coerce").astype(np.float64)
    d[YCOL] = pd.to_numeric(d[YCOL], errors="coerce").astype(np.float64)
    for c in d.columns:
        if c not in {XCOL, YCOL, YEAR, MONTH}:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(np.float32)

    lccs_info = configure_lccs_encoding(d)
    if "lccs" in d.columns:
        print(
            "LCCS encoding: nominal one-hot; "
            f"levels={len(LCCS_LEVELS)}, reference={LCCS_REFERENCE}, "
            f"dummy_features={max(0, len(LCCS_LEVELS)-1)}, missing={lccs_info.get('missing_count', 0):,}"
        )

    d["_time_index"] = d[YEAR].astype(np.int32) * 12 + d[MONTH].astype(np.int32)
    if d.duplicated([XCOL, YCOL, "_time_index"]).any():
        raise ValueError("duplicate x/y/month panel keys detected")

    add_time_features(d)
    grid = grid_info(d)
    x = d[XCOL].to_numpy(np.float64)
    y = d[YCOL].to_numpy(np.float64)
    d["_block20"] = block_id(x, y, grid, REF_BLOCK_FACTOR)
    d["_pixel"] = pixel_id(x, y, grid)
    d["_time_block"] = (d[YEAR].to_numpy(np.int32) - START_YEAR).astype(np.int16)
    d[XCOL] = d[XCOL].astype(np.float32)
    d[YCOL] = d[YCOL].astype(np.float32)

    full = {
        "A": [c for c in FULL_TIME + MODEL_A if c in d.columns],
        "B": [c for c in FULL_TIME + MODEL_B if c in d.columns],
        "C": [c for c in FULL_TIME + MODEL_C if c in d.columns],
    }
    yearly = {
        "A": [c for c in YEAR_TIME + MODEL_A if c in d.columns],
        "B": [c for c in YEAR_TIME + MODEL_B if c in d.columns],
        "C": [c for c in YEAR_TIME + MODEL_C if c in d.columns],
    }
    print("Controls full A/B/C:", *(len(full[k]) for k in "ABC"))
    print("Controls yearly A/B/C:", *(len(yearly[k]) for k in "ABC"))
    print(f"Grid dx={grid[0]:.3f}, dy={grid[1]:.3f}")
    return d, full, yearly, grid


# =============================================================================
# FOLDS + XGBOOST
# =============================================================================
def rr_map(values, k, seed=SEED):
    v = np.asarray(sorted(np.unique(values)))
    rng = np.random.default_rng(seed)
    rng.shuffle(v)
    if len(v) < k:
        raise ValueError("too few groups for requested folds")
    return {int(g): int(i % k) for i, g in enumerate(v)}


def time_map(values, k):
    v = np.asarray(sorted(np.unique(values)), dtype=int)
    if len(v) < k:
        raise ValueError("too few temporal groups for requested folds")
    out = {}
    for f, ch in enumerate(np.array_split(v, k)):
        for g in ch:
            out[int(g)] = f
    return out


def balanced_map(blocks, k=None):
    if k is None:
        k = N_SFOLD
    counts = pd.Series(blocks).value_counts(sort=False)
    items = [(int(a), int(b)) for a, b in counts.items()]
    rng = np.random.default_rng(SEED)
    rng.shuffle(items)
    items.sort(key=lambda z: z[1], reverse=True)
    totals = np.zeros(k, np.int64)
    out = {}
    for b, n in items:
        f = int(np.argmin(totals))
        out[b] = f
        totals[f] += n
    return out


def assign_2d(w, k):
    sm = rr_map(w["_block20"].to_numpy(), N_SFOLD)
    tm = time_map(w["_time_block"].to_numpy(), k)
    w["_sf"] = w["_block20"].map(sm).astype(np.int8)
    w["_tf"] = w["_time_block"].map(tm).astype(np.int8)


def model(stage, seed_key, tfold=None, use_fold_hp=False):
    if use_fold_hp and tfold is not None and (tfold + 1) in HP["fold"]:
        pars = dict(HP["fold"][tfold + 1][stage])
    else:
        pars = dict(HP[stage])
    pars.update(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device=DEVICE,
        max_bin=MAX_BIN,
        n_jobs=N_JOBS,
        random_state=SEED + int(seed_key),
        verbosity=0,
        validate_parameters=True,
    )
    return xgb.XGBRegressor(**pars)


def splits_spatial(fold):
    pos = np.arange(len(fold))
    for f in range(N_SFOLD):
        tr = pos[fold != f]
        te = pos[fold == f]
        if len(te) == 0:
            continue
        if len(tr) == 0:
            raise RuntimeError(f"spatial fold {f+1} has no training observations")
        yield f, None, tr, te


def splits_2d(sf, tf, kt):
    pos = np.arange(len(sf))
    cover = np.zeros(len(sf), np.int8)
    for s in range(N_SFOLD):
        for t in range(kt):
            te = (sf == s) & (tf == t)
            if not te.any():
                continue
            tr = (sf != s) & (tf != t)
            test = pos[te]
            train = pos[tr]
            cover[test] += 1
            yield s, t, train, test
    if not np.all(cover == 1):
        raise RuntimeError("2D OOF coverage failure")


# =============================================================================
# INFERENCE / COMPONENTS
# =============================================================================
def meat(score, cluster, small_sample):
    codes, u = pd.factorize(cluster, sort=False)
    G = len(u)
    sums = np.bincount(codes, weights=score, minlength=G)
    m = float(sums @ sums)
    if small_sample and G > 1:
        m *= G / (G - 1)
    return m, G


def infer(dr, yr, c1, c2, small_sample):
    d = dr.astype(np.float64)
    y = yr.astype(np.float64)
    num = float(d @ y)
    den = float(d @ d)
    if den <= 0:
        raise RuntimeError("non-positive DML denominator")
    theta = num / den
    score = d * (y - theta * d)

    m1, g1 = meat(score, c1, small_sample)
    m2, g2 = meat(score, c2, small_sample)
    pc, _ = pd.factorize(pd.MultiIndex.from_arrays([c1, c2]), sort=False)
    m12, g12 = meat(score, pc, small_sample)

    var = (m1 + m2 - m12) / den**2
    if not np.isfinite(var) or var <= 0:
        var = max(m1, m2) / den**2
    se = math.sqrt(var)
    z = theta / se
    df = max(1, min(g1, g2) - 1)
    crit = float(stats.t.ppf(0.975, df))

    return dict(
        estimate_raw=theta,
        effect_per_10ug_m3=theta * 10,
        se_two_way_raw=se,
        se_two_way_per_10ug_m3=se * 10,
        ci95_low_per_10ug_m3=(theta - 1.96 * se) * 10,
        ci95_high_per_10ug_m3=(theta + 1.96 * se) * 10,
        p_value_normal=float(2 * stats.norm.sf(abs(z))),
        t_df_min_cluster=df,
        ci95_low_t_min_cluster_per_10ug_m3=(theta - crit * se) * 10,
        ci95_high_t_min_cluster_per_10ug_m3=(theta + crit * se) * 10,
        p_value_t_min_cluster_sensitivity=float(2 * stats.t.sf(abs(z), df)),
        n_cluster1=g1,
        n_cluster2=g2,
        n_intersections=g12,
        dml_numerator_raw=num,
        dml_denominator_raw=den,
    )


def components(year, dr, yr, block):
    d = dr.astype(np.float64)
    y = yr.astype(np.float64)
    tmp = pd.DataFrame({
        "spatial_block": block.astype(np.int64),
        "num": d * y,
        "den": d * d,
        "n": np.ones(len(d), dtype=np.int32),
    })
    g = tmp.groupby("spatial_block", observed=True, sort=False).agg(
        numerator_raw=("num", "sum"),
        denominator_raw=("den", "sum"),
        n=("n", "sum"),
    ).reset_index()
    g.insert(0, "year", int(year))
    return g


def yearly_diag(years, t, th, y, yh):
    rows = []
    for yy in sorted(np.unique(years)):
        m = years == yy
        for stage, a, b in [("treatment", t, th), ("outcome", y, yh)]:
            z = metrics(a[m], b[m])
            rows.append(dict(
                stage=stage,
                year=int(yy),
                n=int(m.sum()),
                r2=z["r2"],
                rmse=z["rmse"],
                mae=z["mae"],
                bias_mean_residual=z["bias"],
                target_mean=float(a[m].mean()),
                prediction_mean=float(b[m].mean()),
            ))
    return pd.DataFrame(rows)


def time_contrib(years, tf, dr, yr, theta):
    d = dr.astype(np.float64)
    y = yr.astype(np.float64)
    total = float(d @ d)
    rows = []
    for f in sorted(np.unique(tf)):
        m = tf == f
        num = float(d[m] @ y[m])
        den = float(d[m] @ d[m])
        rows.append(dict(
            time_fold=int(f + 1),
            test_year_min=int(years[m].min()),
            test_year_max=int(years[m].max()),
            n=int(m.sum()),
            denominator_weight=den / total,
            local_effect_per_10ug_m3=(num / den) * 10,
            mean_treatment_residual=float(d[m].mean()),
        ))
    return pd.DataFrame(rows)


def pooled(comp, label):
    num = float(comp.numerator_raw.sum())
    den = float(comp.denominator_raw.sum())
    theta = num / den
    c = comp.copy()
    c["score"] = c.numerator_raw - theta * c.denominator_raw

    s = c.groupby("spatial_block", observed=True).score.sum().to_numpy(float)
    y = c.groupby("year", observed=True).score.sum().to_numpy(float)
    sy = c.score.to_numpy(float)
    gs, gy, gsy = len(s), len(y), len(sy)

    ms = float(s @ s) * (gs / (gs - 1) if gs > 1 else 1.0)
    my = float(y @ y) * (gy / (gy - 1) if gy > 1 else 1.0)
    msy = float(sy @ sy) * (gsy / (gsy - 1) if gsy > 1 else 1.0)
    var = (ms + my - msy) / den**2
    if not np.isfinite(var) or var <= 0:
        var = max(ms, my) / den**2

    se = math.sqrt(var)
    df = max(1, min(gs, gy) - 1)
    crit = float(stats.t.ppf(0.975, df))
    tval = theta / se

    return dict(
        analysis=label,
        effect_per_10ug_m3=theta * 10,
        se_two_way_space_year_per_10ug_m3=se * 10,
        ci95_low_t_min_cluster_per_10ug_m3=(theta - crit * se) * 10,
        ci95_high_t_min_cluster_per_10ug_m3=(theta + crit * se) * 10,
        p_value_t_min_cluster_sensitivity=float(2 * stats.t.sf(abs(tval), df)),
        t_df_min_cluster=df,
        n_spatial_clusters=gs,
        n_year_clusters=gy,
        dml_numerator_raw=num,
        dml_denominator_raw=den,
    )


def score_reg(comp, design, names):
    d = comp.merge(design, on="year", how="left", validate="many_to_one")
    Z = d[names].to_numpy(float)
    n = d.numerator_raw.to_numpy(float)
    den = d.denominator_raw.to_numpy(float)
    H = Z.T @ (den[:, None] * Z)
    beta = np.linalg.solve(H, Z.T @ n)
    r = n - den * (Z @ beta)
    sc = Z * r[:, None]

    def mm(cluster):
        codes, u = pd.factorize(cluster, sort=False)
        G = len(u)
        S = np.zeros((G, len(names)))
        for j in range(len(names)):
            S[:, j] = np.bincount(codes, weights=sc[:, j], minlength=G)
        return (S.T @ S) * (G / (G - 1) if G > 1 else 1.0), G

    ms, gs = mm(d.spatial_block)
    my, gy = mm(d.year)
    fac = len(d) / (len(d) - 1) if len(d) > 1 else 1.0
    msy = (sc.T @ sc) * fac
    hi = np.linalg.inv(H)
    cov = hi @ (ms + my - msy) @ hi
    cov = (cov + cov.T) / 2
    vals, vec = np.linalg.eigh(cov)
    cov = (vec * np.clip(vals, 0, None)) @ vec.T
    return beta, cov, max(1, min(gs, gy) - 1)


def linear_contrast(beta, cov, df, w, name):
    w = np.asarray(w, float)
    est = float(w @ beta)
    se = math.sqrt(max(0.0, float(w @ cov @ w)))
    tval = est / se if se > 0 else np.nan
    crit = float(stats.t.ppf(0.975, df))
    return dict(
        test=name,
        estimate_per_10ug_m3=est * 10,
        se_per_10ug_m3=se * 10,
        p_value_t=float(2 * stats.t.sf(abs(tval), df)) if np.isfinite(tval) else np.nan,
        ci95_low_per_10ug_m3=(est - crit * se) * 10,
        ci95_high_per_10ug_m3=(est + crit * se) * 10,
        t_df=df,
    )


def linear_time_trend_test(comp):
    """Single designated temporal-stability test: linear trend in annual DML effects."""
    yrs = np.array(sorted(comp.year.unique()), int)
    if len(yrs) < 3:
        raise ValueError("at least three years are required for the linear trend test")
    center = float(yrs.mean())
    des = pd.DataFrame({"year": yrs, "i": 1.0, "t": yrs - center})
    b, c, df = score_reg(comp, des, ["i", "t"])
    return pd.DataFrame([
        linear_contrast(b, c, df, [0, 1], "linear_effect_trend_per_calendar_year")
    ])


def two_way_contrast_from_influence(estimate, contrib, space, time_cluster):
    c = np.asarray(contrib, dtype=np.float64)
    sp = np.asarray(space)
    tt = np.asarray(time_cluster)
    sc, sv = pd.factorize(sp, sort=False)
    tc, tv = pd.factorize(tt, sort=False)
    gs, gt = len(sv), len(tv)

    ss = np.bincount(sc, weights=c, minlength=gs)
    st = np.bincount(tc, weights=c, minlength=gt)
    inter = sc.astype(np.int64) * gt + tc.astype(np.int64)
    si = np.bincount(inter, weights=c, minlength=gs * gt)
    occupied = np.bincount(inter, minlength=gs * gt) > 0
    gi = int(occupied.sum())

    ms = float(ss @ ss) * (gs / (gs - 1) if gs > 1 else 1.0)
    mt = float(st @ st) * (gt / (gt - 1) if gt > 1 else 1.0)
    mi = float(si[occupied] @ si[occupied]) * (gi / (gi - 1) if gi > 1 else 1.0)
    var = ms + mt - mi
    if not np.isfinite(var) or var <= 0:
        var = max(ms, mt)
    se = math.sqrt(var)
    df = max(1, min(gs, gt) - 1)
    crit = float(stats.t.ppf(0.975, df))
    tstat = estimate / se

    return dict(
        estimate_raw=float(estimate),
        estimate_per_10ug_m3=float(estimate * 10),
        se_per_10ug_m3=se * 10,
        ci95_low_per_10ug_m3=(estimate - crit * se) * 10,
        ci95_high_per_10ug_m3=(estimate + crit * se) * 10,
        p_value_t=float(2 * stats.t.sf(abs(tstat), df)),
        t_df=int(df),
        n_spatial_clusters=int(gs),
        n_time_clusters=int(gt),
        n_intersections=int(gi),
    )




def _cluster_meat_matrix(score_matrix, cluster, small_sample=True):
    """Cluster meat for vector scores; rows are observations, columns are parameters."""
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    codes, groups = pd.factorize(cluster, sort=False)
    G = len(groups)
    S = np.zeros((G, score_matrix.shape[1]), dtype=np.float64)
    for j in range(score_matrix.shape[1]):
        S[:, j] = np.bincount(codes, weights=score_matrix[:, j], minlength=G)
    meat = S.T @ S
    if small_sample and G > 1:
        meat *= G / (G - 1)
    return meat, G


def infer_multitreatment(D, yres, c1, c2, names, small_sample=True):
    """
    Orthogonal second stage for multiple residualized treatments with two-way clustered inference.

    D: n x k residualized treatment matrix.
    yres: residualized outcome.
    c1/c2: clustering variables (here space and year).
    """
    D = np.asarray(D, dtype=np.float64)
    y = np.asarray(yres, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != len(y):
        raise ValueError("D must be n x k and aligned with yres")
    if D.shape[1] != len(names):
        raise ValueError("names length does not match number of treatments")

    H = D.T @ D
    if np.linalg.matrix_rank(H) < H.shape[0]:
        raise RuntimeError("joint treatment residual matrix is rank deficient")
    Hinv = np.linalg.inv(H)
    beta = Hinv @ (D.T @ y)
    resid = y - D @ beta
    score = D * resid[:, None]

    m1, g1 = _cluster_meat_matrix(score, c1, small_sample)
    m2, g2 = _cluster_meat_matrix(score, c2, small_sample)
    inter, _ = pd.factorize(pd.MultiIndex.from_arrays([c1, c2]), sort=False)
    m12, g12 = _cluster_meat_matrix(score, inter, small_sample)
    meat = m1 + m2 - m12
    cov = Hinv @ meat @ Hinv
    cov = (cov + cov.T) / 2
    vals, vec = np.linalg.eigh(cov)
    cov = (vec * np.clip(vals, 0, None)) @ vec.T

    df = max(1, min(g1, g2) - 1)
    crit = float(stats.t.ppf(0.975, df))
    rows = []
    for j, name in enumerate(names):
        se = math.sqrt(max(0.0, float(cov[j, j])))
        tval = float(beta[j] / se) if se > 0 else np.nan
        rows.append(dict(
            treatment=name,
            estimate_raw=float(beta[j]),
            effect_per_10ug_m3=float(beta[j] * 10),
            se_two_way_raw=se,
            se_two_way_per_10ug_m3=se * 10,
            ci95_low_t_per_10ug_m3=(float(beta[j]) - crit * se) * 10,
            ci95_high_t_per_10ug_m3=(float(beta[j]) + crit * se) * 10,
            p_value_t=float(2 * stats.t.sf(abs(tval), df)) if np.isfinite(tval) else np.nan,
            t_df=df,
            n_cluster1=g1,
            n_cluster2=g2,
            n_intersections=g12,
        ))
    return pd.DataFrame(rows), beta, cov, resid


def fit_prepared_spatial_panel(w, controls, label, out, exposure_col=EXPOSURE):
    """
    Fit a full-period panel with SPATIAL-BLOCKED cross-fitting only.

    Rationale for Exp03:
    - The estimand is interannual variation within pixel x calendar-month cells,
      not forecasting performance in unseen contiguous time periods.
    - Entire spatial blocks are held out, so all years for held-out pixels are
      excluded from nuisance-model training.
    - Calendar year remains available to the nuisance learner through
      `_year_centered`, allowing flexible nonlinear temporal adjustment without
      forcing tree models to extrapolate across held-out time blocks.
    - Final uncertainty is two-way clustered by spatial block and calendar year.

    The point estimate remains an orthogonal DML slope.  No temporal-fold
    hyperparameters are used in this Exp03 design.
    """
    req = list(dict.fromkeys([
        OUTCOME, exposure_col, YEAR, MONTH, "_block20", "_sf"
    ] + controls))
    z = w.replace([np.inf, -np.inf], np.nan).dropna(subset=req).copy().reset_index(drop=True)
    if len(z) != len(w):
        raise RuntimeError(f"{label}: prepared matched panel unexpectedly lost rows")

    block = z["_block20"].to_numpy(np.int64)
    fold = z["_sf"].to_numpy(np.int8)
    years = z[YEAR].to_numpy(np.int16)

    res, fd, yd, _, dr, yr = fit_engine(
        z, controls, label, "spatial", block,
        fold=fold,
        use_fold_hp=False,
        cluster2=years,
        small_sample=True,
        return_resid=True,
        exposure_col=exposure_col,
    )

    pd.DataFrame([res]).to_csv(out / f"{label}_estimate.csv", index=False)
    fd.to_csv(out / f"{label}_fold_diagnostics.csv", index=False)
    yd.to_csv(out / f"{label}_yearly_nuisance.csv", index=False)
    return res, dr, yr, z


# =============================================================================
# GENERIC DML FITTERS
# =============================================================================
def fit_engine(
    w,
    controls,
    label,
    mode,
    block,
    fold=None,
    kt=None,
    use_fold_hp=False,
    cluster2=None,
    small_sample=False,
    return_resid=False,
    exposure_col=EXPOSURE,
):
    X, feature_names = make_design_matrix(w, controls)
    t = w[exposure_col].to_numpy(np.float32)
    y = w[OUTCOME].to_numpy(np.float32)
    years = w[YEAR].to_numpy(np.int16)
    months = w[MONTH].to_numpy(np.int16)

    th = np.full(len(w), np.nan, np.float32)
    yh = np.full(len(w), np.nan, np.float32)
    rows = []

    if mode == "spatial":
        split_iter = splits_spatial(fold)
    else:
        split_iter = splits_2d(w["_sf"].to_numpy(np.int8), w["_tf"].to_numpy(np.int8), kt)

    for s, tfd, tr, te in split_iter:
        sid = f"s{s+1}" if tfd is None else f"s{s+1}_t{tfd+1}"
        print(f"  {label}: {sid} train={len(tr):,} test={len(te):,}")

        for stage, target, dest in [("treatment", t, th), ("outcome", y, yh)]:
            seed_key = (int(years[te][0]) if mode == "spatial" else 0) * 10 + s + (tfd or 0)
            md = model(stage, seed_key, tfd, use_fold_hp)
            md.fit(X[tr], target[tr])
            p = md.predict(X[te]).astype(np.float32)
            dest[te] = p
            q = metrics(target[te], p)
            rows.append(dict(
                analysis=label,
                stage=stage,
                split=sid,
                space_fold=s + 1,
                time_fold=(tfd + 1 if tfd is not None else np.nan),
                n_train=len(tr),
                n_test=len(te),
                r2=q["r2"],
                rmse=q["rmse"],
                mae=q["mae"],
                mean_residual_bias=q["bias"],
                test_year_min=int(years[te].min()),
                test_year_max=int(years[te].max()),
            ))
            del md, p
            clean_gpu()

    if not np.isfinite(th).all() or not np.isfinite(yh).all():
        raise RuntimeError(f"{label}: incomplete OOF predictions")

    dr = (t - th).astype(np.float32)
    yr = (y - yh).astype(np.float32)
    c2 = cluster2 if cluster2 is not None else months
    inf = infer(dr, yr, block, c2, small_sample)

    fd = pd.DataFrame(rows)
    tm = metrics(t, th)
    ym = metrics(y, yh)
    ft = fd[fd.stage == "treatment"]
    fy = fd[fd.stage == "outcome"]

    res = dict(
        analysis=label,
        exposure=exposure_col,
        n=len(w),
        year_min=int(years.min()),
        year_max=int(years.max()),
        n_controls=len(feature_names),
        controls=";".join(feature_names),
        control_specification="LCCS nominal one-hot when present; all other listed controls retain numeric representation",
        **inf,
        treatment_oof_r2=tm["r2"],
        treatment_oof_rmse=tm["rmse"],
        treatment_oof_mae=tm["mae"],
        treatment_residual_mean=float(dr.mean()),
        treatment_residual_sd=float(dr.std(ddof=1)),
        treatment_min_fold_r2=float(ft.r2.min()),
        treatment_max_abs_fold_bias=float(ft.mean_residual_bias.abs().max()),
        outcome_oof_r2=ym["r2"],
        outcome_oof_rmse=ym["rmse"],
        outcome_oof_mae=ym["mae"],
        outcome_residual_mean=float(yr.mean()),
        outcome_residual_sd=float(yr.std(ddof=1)),
        outcome_min_fold_r2=float(fy.r2.min()),
        outcome_max_abs_fold_bias=float(fy.mean_residual_bias.abs().max()),
    )

    yd = yearly_diag(years, t, th, y, yh)
    tc = time_contrib(years, w["_tf"].to_numpy(np.int8), dr, yr, inf["estimate_raw"]) if mode == "2d" else pd.DataFrame()
    return res, fd, yd, tc, (dr if return_resid else None), (yr if return_resid else None)


def fit_spatial(w, controls, label, block, fold, return_resid=True, exposure_col=EXPOSURE):
    months = w[MONTH].to_numpy(np.int16)
    res, fd, yd, tc, dr, yr = fit_engine(
        w, controls, label, "spatial", block,
        fold=fold, cluster2=months, small_sample=True,
        return_resid=return_resid, exposure_col=exposure_col,
    )
    comp = components(int(w[YEAR].iloc[0]), dr, yr, block) if return_resid else pd.DataFrame()
    return res, fd, comp, dr, yr


def fit_2d(panel, controls, label, out, exposure_col=EXPOSURE, use_fold_hp=True):
    req = list(dict.fromkeys([
        OUTCOME, exposure_col, YEAR, MONTH, "_block20", "_time_block"
    ] + controls))
    w = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=req).copy().reset_index(drop=True)
    assign_2d(w, FULL_TFOLD)
    block = w["_block20"].to_numpy(np.int64)
    c2 = w["_time_block"].to_numpy(np.int16)
    res, fd, yd, tc, dr, yr = fit_engine(
        w, controls, label, "2d", block,
        kt=FULL_TFOLD, use_fold_hp=use_fold_hp,
        cluster2=c2, small_sample=False, return_resid=True,
        exposure_col=exposure_col,
    )
    pd.DataFrame([res]).to_csv(out / f"{label}_estimate.csv", index=False)
    fd.to_csv(out / f"{label}_fold_diagnostics.csv", index=False)
    yd.to_csv(out / f"{label}_yearly_nuisance.csv", index=False)
    tc.to_csv(out / f"{label}_timefold_contributions.csv", index=False)
    return res, dr, yr, w


# =============================================================================
# EXP 00: FULL-PERIOD ALL-DATA DML
# =============================================================================
def run_full_period_all_data_dml(data, full, root):
    """
    Full-period DML using all 2000-2022 observations in one nuisance architecture.

    Design
    ------
    - Model A controls for the full panel, including centered calendar year and
      11 calendar-month indicators.
    - 5 spatial folds x 5 contiguous temporal folds.
    - A test observation is predicted only by models trained outside BOTH its
      spatial fold and temporal fold.
    - Final uncertainty is two-way clustered by spatial block and calendar year.
    - This experiment is reported as a distinct full-period specification and
      is NOT used to select the primary year-stratified estimand.

    The experiment is intentionally retained even if its estimate is attenuated
    or statistically compatible with zero.
    """
    out = mkdir(root / "00_full_period_all_data_DML")
    print("\n" + "=" * 100)
    print("EXP 00: full-period ALL-DATA space x time blocked DML")
    print("=" * 100)

    label = "exp00_full_period_all_data_2D_DML"
    res, dr, yr, w = fit_2d(
        data,
        full["A"],
        label,
        out,
        exposure_col=EXPOSURE,
        use_fold_hp=True,
    )

    support = residual_support_metrics(dr)
    support_row = dict(
        analysis=label,
        n=int(len(w)),
        **support,
        crossfitting=(
            f"{N_SFOLD} spatial folds x {FULL_TFOLD} contiguous temporal folds; "
            "train excludes both the held-out spatial fold and held-out temporal fold"
        ),
        inference="spatial-block x calendar-year two-way clustered",
        interpretation=(
            "Distinct full-period all-data specification. One nuisance architecture spans "
            "2000-2022; reported transparently regardless of statistical significance and "
            "not used to choose the primary year-stratified estimand."
        ),
    )
    pd.DataFrame([support_row]).to_csv(
        out / "exp00_full_period_support_diagnostics.csv", index=False
    )

    summary = pd.DataFrame([dict(
        experiment=0,
        analysis=label,
        effect_per_10ug_m3=res["effect_per_10ug_m3"],
        report_p=res["p_value_t_min_cluster_sensitivity"],
        notes=(
            "FULL-PERIOD ALL-DATA DML; one 2000-2022 nuisance architecture; "
            f"{N_SFOLD} spatial x {FULL_TFOLD} contiguous temporal folds; "
            "distinct specification, not a significance-based model-selection comparator"
        ),
    )])
    summary.to_csv(out / "EXP00_SUMMARY.csv", index=False)

    # Preserve the compact estimator-facing row for manuscript/source-data tables.
    estimator_row = dict(
        design="full_period_all_data_2D_DML",
        n=int(res["n"]),
        effect_per_10ug_m3=float(res["effect_per_10ug_m3"]),
        ci95_low_t_min_cluster_per_10ug_m3=float(res["ci95_low_t_min_cluster_per_10ug_m3"]),
        ci95_high_t_min_cluster_per_10ug_m3=float(res["ci95_high_t_min_cluster_per_10ug_m3"]),
        p_value_t_min_cluster_sensitivity=float(res["p_value_t_min_cluster_sensitivity"]),
        treatment_oof_r2=float(res["treatment_oof_r2"]),
        outcome_oof_r2=float(res["outcome_oof_r2"]),
        treatment_residual_sd=float(res["treatment_residual_sd"]),
        dml_denominator_raw=float(res["dml_denominator_raw"]),
        denominator_per_observation=float(res["dml_denominator_raw"] / res["n"]),
        crossfitting=f"{N_SFOLD} spatial x {FULL_TFOLD} contiguous temporal folds",
        nuisance_architecture="single full-period nuisance architecture spanning 2000-2022",
        interpretation=(
            "Alternative full-period all-data specification; not formally ranked against "
            "the year-stratified primary estimand because nuisance architecture and temporal "
            "cross-fitting differ."
        ),
    )
    pd.DataFrame([estimator_row]).to_csv(
        out / "exp00_full_period_estimator_row.csv", index=False
    )

    del dr, yr, w
    clean_gpu()
    return summary, res, support_row


# =============================================================================
# STRICT PIXEL-YEAR FE
# =============================================================================
def strict_year(d):
    tv = [c for c in TIME_VARYING_A if c in d.columns]
    keep = [
        OUTCOME, EXPOSURE, XCOL, YCOL, YEAR, MONTH,
        "_pixel", "_block20"
    ] + MONTH_FE + tv
    w = d[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()

    cnt = w.groupby("_pixel", observed=True, sort=False)["_pixel"].transform("size")
    w = w[cnt >= STRICT_MIN_MONTHS].copy()
    g = w.groupby("_pixel", observed=True, sort=False)

    names = {}
    for c in [OUTCOME, EXPOSURE] + MONTH_FE + tv:
        n = "_w_" + c
        w[n] = (w[c].astype(np.float32) - g[c].transform("mean").astype(np.float32)).astype(np.float32)
        names[c] = n

    w[OUTCOME] = w[names[OUTCOME]]
    w[EXPOSURE] = w[names[EXPOSURE]]
    controls = [XCOL, YCOL] + [names[c] for c in MONTH_FE] + [names[c] for c in tv]
    return w, controls


# =============================================================================
# SMALL-CLUSTER / WILD BOOTSTRAP
# =============================================================================
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


def wild_month_cluster_bootstrap_t(year, dr, yr, space, month, reps=WILD_REPS, seed=WILD_SEED, batch=128):
    """
    Small-month-cluster sensitivity for a single year.

    Important: Webb six-point wild weights are applied ONLY to calendar-month
    clusters. Each bootstrap draw is then studentized with the same
    spatial x month two-way cluster-robust variance formula used for the
    annual estimate. This is therefore deliberately named
    "wild month-cluster bootstrap-t with two-way studentization" and is NOT
    claimed to be a multiway wild-cluster bootstrap.

    Primary annual uncertainty remains the spatial x month two-way clustered
    inference returned by fit_spatial(); this procedure is an additional
    sensitivity analysis for the small number (<=12) of month clusters.
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
        math.sqrt(0.5), 1.0, math.sqrt(1.5)
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
    return dict(
        year=int(year),
        effect_per_10ug_m3=theta * 10,
        wild_month_cluster_twoway_studentized_se_per_10ug_m3=obs_se * 10,
        wild_month_cluster_twoway_studentized_ci95_low_per_10ug_m3=(theta - crit * obs_se) * 10,
        wild_month_cluster_twoway_studentized_ci95_high_per_10ug_m3=(theta + crit * obs_se) * 10,
        wild_month_cluster_twoway_studentized_p_value=p,
        wild_bootstrap_t_critical_95=crit,
        wild_resampling_month_clusters=gm,
        studentization_spatial_clusters=gs,
        studentization_space_month_cells=gi,
        wild_bootstrap_reps=int(reps),
        wild_month_cluster_weights="Webb six-point",
        wild_resampling_dimension="calendar_month_only",
        studentization="spatial_x_month_two_way_cluster_robust",
        is_multiway_wild_cluster_bootstrap=False,
        inference_role="small-month-cluster sensitivity; primary annual inference is spatial x month two-way clustered",
    )


def bh_fdr(pvals, alpha=FDR_ALPHA):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q, np.zeros(len(p), dtype=bool)
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qq = ranked * m / np.arange(1, m + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    qq = np.clip(qq, 0, 1)
    back = np.empty(m, dtype=float)
    back[order] = qq
    q[ok] = back
    return q, np.isfinite(q) & (q <= alpha)


def wild_year_cluster(comp, label, seed):
    y = comp.groupby("year", observed=True).agg(
        num=("numerator_raw", "sum"), den=("denominator_raw", "sum")
    ).reset_index()
    N = y.num.to_numpy(float)
    D = y.den.to_numpy(float)
    G = len(N)
    theta = float(N.sum() / D.sum())

    # Webb wild cluster bootstrap-t on year clusters.
    R = N.copy()
    obs = float(R.sum() / math.sqrt((G / (G - 1)) * float(R @ R)))
    U = N - theta * D
    vals = np.array([
        -math.sqrt(1.5), -1, -math.sqrt(0.5),
        math.sqrt(0.5), 1, math.sqrt(1.5)
    ])
    rng = np.random.default_rng(seed)
    exc = 0
    delta = np.empty(WILD_REPS)
    done = 0
    while done < WILD_REPS:
        b = min(2000, WILD_REPS - done)
        W = vals[rng.integers(0, 6, size=(b, G))]
        WR = W * R
        tb = WR.sum(1) / np.sqrt((G / (G - 1)) * np.sum(WR**2, axis=1))
        exc += int((np.abs(tb) >= abs(obs)).sum())
        delta[done:done+b] = (W * U).sum(1) / D.sum()
        done += b

    q = np.quantile(delta, [0.025, 0.975])
    return dict(
        analysis=label,
        effect_per_10ug_m3=theta * 10,
        wild_cluster_p_value=(1 + exc) / (WILD_REPS + 1),
        wild_basic_ci95_low_per_10ug_m3=(theta - q[1]) * 10,
        wild_basic_ci95_high_per_10ug_m3=(theta - q[0]) * 10,
        n_year_clusters=G,
        bootstrap_reps=WILD_REPS,
    )


# =============================================================================
# EXP 03: INTERANNUAL ESTIMAND + EXACT PAIRED RAW-vs-ANOMALY DESIGN CHECK
# =============================================================================
def build_matched_interannual_anomaly(data, full_A):
    """
    Build a pixel-month WITHIN-transformed interannual panel.

    Key design change relative to V5
    --------------------------------
    V5 used contiguous temporal folds both to construct leave-time-fold
    baselines and to cross-fit XGBoost nuisance models.  That turned Exp03 into
    an out-of-time extrapolation problem; tree models can perform poorly at the
    temporal edges even when the causal nuisance problem itself is estimable.

    V6 instead defines the interannual estimand directly by within-transforming
    outcome, O3, and time-varying controls within each pixel x calendar-month
    cell.  Cross-fitting is then done by SPATIAL blocks only.  Because an entire
    spatial block is held out, all years for each held-out pixel are excluded
    from nuisance-model training.  Temporal dependence is handled in inference
    by the second clustering dimension (calendar year), while `_year_centered`
    remains a nuisance feature.

    Static/spatial controls are retained in raw form.  The resulting anomaly
    panel is paired row-for-row with the raw observations through `_rowpos`.
    """
    tv = [c for c in TIME_VARYING_A if c in data.columns]
    req = list(dict.fromkeys([
        OUTCOME, EXPOSURE, XCOL, YCOL, YEAR, MONTH,
        "_pixel", "_block20", "_time_index"
    ] + full_A + tv))

    w = data[req].copy()
    w["_rowpos"] = np.arange(len(w), dtype=np.int64)
    complete = list(dict.fromkeys([OUTCOME, EXPOSURE] + full_A + tv))
    w = w.replace([np.inf, -np.inf], np.nan).dropna(subset=complete).reset_index(drop=True)

    pm = ["_pixel", MONTH]
    pm_n = w.groupby(pm, observed=True)[YEAR].transform("count").to_numpy(np.int16)
    keep = pm_n >= MATCHED_MIN_BASELINE_YEARS
    print(
        "Building matched interannual within-transformation: "
        f"rows before support filter={len(w):,}; "
        f"coverage={keep.mean():.3%}; "
        f"minimum years/cell={MATCHED_MIN_BASELINE_YEARS}"
    )
    w = w.loc[keep].copy().reset_index(drop=True)

    # One common spatial-fold assignment for BOTH raw and anomaly panels.
    # Every row from a held-out spatial block, across all years, is kept out of
    # nuisance-model training.
    fmap = balanced_map(w["_block20"].to_numpy(np.int64), N_SFOLD)
    w["_sf"] = w["_block20"].map(fmap).astype(np.int8)

    transform_cols = [OUTCOME, EXPOSURE] + tv
    g = w.groupby(pm, observed=True, sort=False)
    for c in transform_cols:
        mu = g[c].transform("mean").to_numpy(np.float64)
        w[c] = (w[c].to_numpy(np.float64) - mu).astype(np.float32)
        del mu

    # Numerical audit: the within-transformed variables should have cell means
    # approximately zero.  This is a diagnostic only, not a selection rule.
    audit_rows = []
    for c in [OUTCOME, EXPOSURE]:
        cell_means = w.groupby(pm, observed=True, sort=False)[c].mean().to_numpy(np.float64)
        audit_rows.append(dict(
            variable=c,
            max_abs_pixel_month_mean=float(np.max(np.abs(cell_means))) if len(cell_means) else np.nan,
            mean_abs_pixel_month_mean=float(np.mean(np.abs(cell_means))) if len(cell_means) else np.nan,
            n_pixel_month_cells=int(len(cell_means)),
        ))
    w.attrs["within_audit"] = pd.DataFrame(audit_rows)

    # Recreate deterministic time features after transformation.  The time
    # features themselves are not demeaned; they are nuisance predictors.
    add_time_features(w)
    return w, list(full_A)


def _append_estimand_row(rows, result, name, interpretation):
    if result is None:
        return
    rows.append(dict(
        estimand=name,
        effect_per_10ug_m3=result["effect_per_10ug_m3"],
        ci95_low_per_10ug_m3=result["ci95_low_t_min_cluster_per_10ug_m3"],
        ci95_high_per_10ug_m3=result["ci95_high_t_min_cluster_per_10ug_m3"],
        p_value=result["p_value_t_min_cluster_sensitivity"],
        interpretation=interpretation,
    ))


def run_interannual_design_comparison(
    data, full, root, primary_result=None, strict_result=None
):
    out = mkdir(root / "03_interannual_design_comparison")
    print("\n" + "=" * 100)
    print("EXP 03: interannual pixel-month within-transformed DML")
    print("        spatial-blocked cross-fitting + space x year inference")
    print("=" * 100)

    wa, common_controls = build_matched_interannual_anomaly(data, full["A"])
    if "within_audit" in wa.attrs:
        wa.attrs["within_audit"].to_csv(out / "exp03_within_transform_audit.csv", index=False)

    rowpos = wa["_rowpos"].to_numpy(np.int64)

    raw_cols = list(dict.fromkeys([
        OUTCOME, EXPOSURE, XCOL, YCOL, YEAR, MONTH,
        "_pixel", "_block20", "_year_centered"
    ] + common_controls))
    wr = data.iloc[rowpos][raw_cols].copy().reset_index(drop=True)

    # Exact paired folds: same observations and same spatial fold labels.
    wr["_sf"] = wa["_sf"].to_numpy(np.int8)

    if not (
        np.array_equal(wr[YEAR].to_numpy(), wa[YEAR].to_numpy())
        and np.array_equal(wr[MONTH].to_numpy(), wa[MONTH].to_numpy())
        and np.array_equal(wr["_pixel"].to_numpy(), wa["_pixel"].to_numpy())
        and np.array_equal(wr["_sf"].to_numpy(), wa["_sf"].to_numpy())
    ):
        raise RuntimeError("paired raw/within alignment or spatial-fold identity failure")

    # Descriptive O3 variation before nuisance residualization.
    raw_o3 = wr[EXPOSURE].to_numpy(np.float64)
    simple_pm_mean = wr.groupby(["_pixel", MONTH], observed=True)[EXPOSURE].transform("mean").to_numpy(np.float64)
    simple_pm_anom = raw_o3 - simple_pm_mean
    within_pm_anom = wa[EXPOSURE].to_numpy(np.float64)

    descriptive = pd.DataFrame([
        dict(metric="raw_O3_SD_matched", value=float(np.std(raw_o3, ddof=1)), unit="ug m-3"),
        dict(metric="pixel_month_within_O3_SD", value=float(np.std(within_pm_anom, ddof=1)), unit="ug m-3"),
        dict(
            metric="max_abs_difference_manual_vs_stored_within_O3",
            value=float(np.max(np.abs(simple_pm_anom - within_pm_anom))),
            unit="ug m-3",
        ),
    ])
    descriptive.to_csv(out / "exp03_raw_O3_variation_diagnostic.csv", index=False)
    del simple_pm_mean, simple_pm_anom, raw_o3
    gc.collect()

    # ------------------------------------------------------------------
    # 03A. RAW matched spatial-blocked model.
    #      Not the primary estimand; only a paired design benchmark.
    # ------------------------------------------------------------------
    print("\n--- EXP03A: paired raw spatial-blocked benchmark ---")
    rr, dr_r, yr_r, wr_fit = fit_prepared_spatial_panel(
        wr, common_controls, "exp03A_paired_raw_spatial", out,
        exposure_col=EXPOSURE
    )

    # ------------------------------------------------------------------
    # 03B. Interannual pixel-month WITHIN-transformed model.
    #      Same rows, same spatial folds, same control names/learner.
    # ------------------------------------------------------------------
    print("\n--- EXP03B: paired interannual pixel-month within-transformed model ---")
    ra, dr_a, yr_a, wa_fit = fit_prepared_spatial_panel(
        wa, common_controls, "exp03B_interannual_within_spatial", out,
        exposure_col=EXPOSURE
    )

    # Explicit nuisance-performance gate for the revised Exp03.  This does not
    # select the effect estimate; it only checks whether the nuisance learners
    # behave acceptably after removing the out-of-time forecasting requirement.
    sanity_rows = []
    for label, res in [
        ("paired_raw_spatial", rr),
        ("interannual_within_spatial", ra),
    ]:
        yd = pd.read_csv(out / (
            "exp03A_paired_raw_spatial_yearly_nuisance.csv"
            if label == "paired_raw_spatial"
            else "exp03B_interannual_within_spatial_yearly_nuisance.csv"
        ))
        for stage, overall_r2, min_fold_r2, max_bias in [
            ("treatment", res["treatment_oof_r2"], res["treatment_min_fold_r2"], res["treatment_max_abs_fold_bias"]),
            ("outcome", res["outcome_oof_r2"], res["outcome_min_fold_r2"], res["outcome_max_abs_fold_bias"]),
        ]:
            ys = yd[yd.stage == stage]
            nneg = int((ys.r2 < 0).sum())
            sanity_rows.append(dict(
                design=label,
                stage=stage,
                overall_oof_r2=float(overall_r2),
                min_spatial_fold_oof_r2=float(min_fold_r2),
                min_year_oof_r2=float(ys.r2.min()),
                median_year_oof_r2=float(ys.r2.median()),
                n_years_negative_oof_r2=nneg,
                max_abs_fold_residual_bias=float(max_bias),
                # Gate is based on the actual cross-fitting units (spatial folds),
                # not on requiring positive R2 separately in every calendar year.
                # Year-specific R2 counts are descriptive diagnostics only.
                status=("PASS" if float(overall_r2) > 0 and float(min_fold_r2) >= 0 else "REVIEW"),
            ))
    sanity = pd.DataFrame(sanity_rows)
    sanity.to_csv(out / "exp03_nuisance_sanity_gate.csv", index=False)

    # Exact paired direct contrast: same rows, same spatial folds, same control
    # column set and learner architecture; only the within transformation differs.
    den_r = float(dr_r.astype(np.float64) @ dr_r.astype(np.float64))
    den_a = float(dr_a.astype(np.float64) @ dr_a.astype(np.float64))
    th_r = float((dr_r.astype(np.float64) @ yr_r.astype(np.float64)) / den_r)
    th_a = float((dr_a.astype(np.float64) @ yr_a.astype(np.float64)) / den_a)

    infl_r = (
        dr_r.astype(np.float64)
        * (yr_r.astype(np.float64) - th_r * dr_r.astype(np.float64))
        / den_r
    )
    infl_a = (
        dr_a.astype(np.float64)
        * (yr_a.astype(np.float64) - th_a * dr_a.astype(np.float64))
        / den_a
    )
    contrast = two_way_contrast_from_influence(
        th_r - th_a,
        infl_r - infl_a,
        wa["_block20"].to_numpy(np.int64),
        wa[YEAR].to_numpy(np.int16),
    )
    contrast.update(
        analysis="paired_raw_spatial_minus_interannual_within",
        raw_spatial_effect_per_10ug_m3=th_r * 10,
        interannual_within_effect_per_10ug_m3=th_a * 10,
        design_note=(
            "Same rows, same spatial folds, same control-column names and learner architecture; "
            "outcome, O3 and time-varying controls are pixel-month within-transformed in the interannual design. "
            "Inference is spatial x year two-way clustered."
        ),
    )
    pd.DataFrame([contrast]).to_csv(
        out / "exp03_paired_raw_minus_anomaly_contrast.csv", index=False
    )

    # Residual variation/support diagnostics for the exact paired comparison.
    sr = residual_support_metrics(dr_r)
    sa = residual_support_metrics(dr_a)
    variation = pd.DataFrame([
        dict(
            design="paired_raw_spatial",
            effect_per_10ug_m3=th_r * 10,
            treatment_oof_r2=rr["treatment_oof_r2"],
            outcome_oof_r2=rr["outcome_oof_r2"],
            treatment_min_fold_r2=rr["treatment_min_fold_r2"],
            outcome_min_fold_r2=rr["outcome_min_fold_r2"],
            **sr,
        ),
        dict(
            design="paired_interannual_within_spatial",
            effect_per_10ug_m3=th_a * 10,
            treatment_oof_r2=ra["treatment_oof_r2"],
            outcome_oof_r2=ra["outcome_oof_r2"],
            treatment_min_fold_r2=ra["treatment_min_fold_r2"],
            outcome_min_fold_r2=ra["outcome_min_fold_r2"],
            **sa,
        ),
    ])
    sd_ratio = sa["residual_sd"] / sr["residual_sd"]
    info_ratio = sa["denominator_per_observation"] / sr["denominator_per_observation"]
    variation["interannual_to_paired_raw_residual_sd_ratio"] = sd_ratio
    variation["interannual_to_paired_raw_information_ratio"] = info_ratio
    variation.to_csv(out / "exp03_paired_variation_support_diagnostic.csv", index=False)

    # Scientific estimand table.  Primary and interannual estimands answer
    # different questions, so no formal P value is assigned to their difference.
    estimand_rows = []
    _append_estimand_row(
        estimand_rows,
        primary_result,
        "primary_year_stratified_spatial_blocked",
        "Within each year; uses spatial + seasonal contrasts. Primary estimand.",
    )
    _append_estimand_row(
        estimand_rows,
        strict_result,
        "strict_within_pixel_within_year",
        "Within-pixel, within-year seasonal contrast after pixel-year demeaning.",
    )
    estimand_rows.append(dict(
        estimand="interannual_pixel_month_within",
        effect_per_10ug_m3=ra["effect_per_10ug_m3"],
        ci95_low_per_10ug_m3=ra["ci95_low_t_min_cluster_per_10ug_m3"],
        ci95_high_per_10ug_m3=ra["ci95_high_t_min_cluster_per_10ug_m3"],
        p_value=ra["p_value_t_min_cluster_sensitivity"],
        interpretation=(
            "Same pixel and calendar month across years after within transformation; "
            "spatial-blocked cross-fitting with space x year clustered inference."
        ),
    ))
    estimand_table = pd.DataFrame(estimand_rows)
    estimand_table.to_csv(out / "exp03_scientific_estimand_comparison.csv", index=False)

    if primary_result is not None:
        descriptive_difference = float(
            primary_result["effect_per_10ug_m3"] - ra["effect_per_10ug_m3"]
        )
        pd.DataFrame([dict(
            comparison="primary_year_stratified_minus_interannual_within",
            difference_per_10ug_m3=descriptive_difference,
            p_value=np.nan,
            note=(
                "Descriptive only: no formal P value because the primary and interannual "
                "estimands answer different scientific questions."
            ),
        )]).to_csv(
            out / "exp03_cross_estimand_difference_DESCRIPTIVE_ONLY.csv", index=False
        )

    summary = pd.DataFrame([
        dict(
            experiment=3,
            analysis="interannual_pixel_month_within",
            effect_per_10ug_m3=ra["effect_per_10ug_m3"],
            report_p=ra["p_value_t_min_cluster_sensitivity"],
            notes=(
                "Distinct complementary estimand; spatial-only cross-fitting; "
                f"residual-SD ratio vs paired raw={sd_ratio:.4f}; "
                f"denominator/N ratio={info_ratio:.4f}; "
                f"treatment min-fold OOF R2={ra['treatment_min_fold_r2']:.4f}; "
                f"outcome min-fold OOF R2={ra['outcome_min_fold_r2']:.4f}"
            ),
        ),
        dict(
            experiment=3,
            analysis="paired_raw_spatial_design_check",
            effect_per_10ug_m3=rr["effect_per_10ug_m3"],
            report_p=rr["p_value_t_min_cluster_sensitivity"],
            notes="Not the primary estimand; exact-row paired benchmark for the within transformation.",
        ),
        dict(
            experiment=3,
            analysis="paired_raw_minus_interannual_within",
            effect_per_10ug_m3=contrast["estimate_per_10ug_m3"],
            report_p=contrast["p_value_t"],
            notes="Inferential paired contrast: identical rows/spatial folds/control names/learner architecture.",
        ),
    ])
    summary.to_csv(out / "EXP03_SUMMARY.csv", index=False)

    # Compact identification-scale figure.
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    labels = ["Paired raw", "Interannual within"]
    vals = variation["residual_sd"].to_numpy()
    ax.bar(labels, vals)
    ax.set_ylabel(r"OOF residual O$_3$ SD ($\mu$g m$^{-3}$)")
    ax.set_title("O3 identifying variation in the exact paired design")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        out / "Figure_Exp03_paired_O3_identifying_variation.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    del wr, wa, wr_fit, wa_fit, dr_r, yr_r, dr_a, yr_a, infl_r, infl_a
    clean_gpu()
    return summary, variation, ra


# =============================================================================
# EXP 01/02/05/06/07/08: YEARLY CORE LOOP
# =============================================================================
def run_yearly_core(data, yearly, grid, root):
    out = mkdir(root / "01_02_05_06_07_08_yearly_core")
    print("\n" + "=" * 100)
    print("EXP 01/02/05/06/07/08: primary + strict FE + robustness + inference")
    print("=" * 100)

    xall = data[XCOL].to_numpy(np.float32)
    yall = data[YCOL].to_numpy(np.float32)
    maps = {}
    for f in BLOCK_FACTORS:
        b = data["_block20"].to_numpy(np.int64) if f == REF_BLOCK_FACTOR else block_id(xall, yall, grid, f)
        maps[f] = balanced_map(b)
        print(f"Block factor {f}: {len(maps[f]):,} spatial blocks")
        del b

    annual = []
    annual_wild = []
    primary_comp = []
    primary_support = []
    spatial_diag_rows = []
    strict_rows = []
    strict_comp = []
    block_rows = []
    block_comp = {f: [] for f in BLOCK_FACTORS}
    cov_rows = []
    cov_comp = {k: [] for k in "ABC"}

    # Running exact pooled residual-scale diagnostics without storing 83M residuals.
    dr_n = 0
    dr_sum = 0.0
    dr_sumsq = 0.0

    start = time.time()
    for i, yy in enumerate(range(START_YEAR, END_YEAR + 1), 1):
        print(f"\n=== YEAR {yy} ({i}/{END_YEAR-START_YEAR+1}), elapsed={(time.time()-start)/60:.1f} min ===")
        yd = data[data[YEAR] == yy].copy()

        # ---------------- EXP 01: PRIMARY YEAR-STRATIFIED MODEL A ----------------
        A = yearly["A"]
        reqA = list(dict.fromkeys([OUTCOME, EXPOSURE, MONTH] + A))
        wA = yd.replace([np.inf, -np.inf], np.nan).dropna(subset=reqA).copy()
        b20 = wA["_block20"].to_numpy(np.int64)
        f20 = wA["_block20"].map(maps[20]).to_numpy(np.int8)
        r, fd, comp, dr, yr = fit_spatial(
            wA, A, f"exp01_year_{yy}", b20, f20, True
        )
        annual.append(dict(year=yy, **r))
        primary_comp.append(comp)
        block_rows.append(dict(year=yy, block_factor=20, approx_block_km=grid[0]*20/1000, **r))
        block_comp[20].append(comp)

        sm = residual_support_metrics(dr)
        primary_support.append(dict(year=int(yy), **sm))

        # Lightweight spatial-dependence diagnostic on OOF quantities.
        # This does not refit DML and is not used to select the block scale post hoc.
        sdg = spatial_dependence_diagnostic(
            wA, dr, yr, r["estimate_raw"], yy
        )
        if len(sdg):
            spatial_diag_rows.append(sdg)

        d64 = dr.astype(np.float64)
        dr_n += len(d64)
        dr_sum += float(d64.sum())
        dr_sumsq += float(d64 @ d64)

        # ---------------- EXP 07: YEAR-SPECIFIC SMALL-CLUSTER INFERENCE ----------------
        mw = wild_month_cluster_bootstrap_t(
            yy, dr, yr, b20, wA[MONTH].to_numpy(np.int16),
            reps=WILD_REPS, seed=WILD_SEED + yy
        )
        annual_wild.append(mw)

        # ---------------- EXP 05: BLOCK-SCALE SENSITIVITY ----------------
        for bf in [10, 30]:
            b = block_id(
                wA[XCOL].to_numpy(np.float32),
                wA[YCOL].to_numpy(np.float32),
                grid, bf
            )
            ff = pd.Series(b).map(maps[bf]).to_numpy(np.int8)
            rb, _, cb, db, yb = fit_spatial(
                wA, A, f"exp05_year_{yy}_block_{bf}", b, ff, True
            )
            block_rows.append(dict(
                year=yy, block_factor=bf, approx_block_km=grid[0]*bf/1000, **rb
            ))
            block_comp[bf].append(cb)
            del b, ff, db, yb
            clean_gpu()

        # ---------------- EXP 06: A/B/C ON COMMON COMPLETE-CASE SAMPLE ----------------
        C = yearly["C"]
        reqC = list(dict.fromkeys([OUTCOME, EXPOSURE, MONTH] + C))
        wC = yd.replace([np.inf, -np.inf], np.nan).dropna(subset=reqC).copy()
        bc = wC["_block20"].to_numpy(np.int64)
        fc = wC["_block20"].map(maps[20]).to_numpy(np.int8)

        for k in "ABC":
            rc, _, cc, dc, yc = fit_spatial(
                wC, yearly[k], f"exp06_year_{yy}_model_{k}", bc, fc, True
            )
            cov_rows.append(dict(year=yy, model=k, paired_n=len(wC), **rc))
            cov_comp[k].append(cc)
            del dc, yc
            clean_gpu()

        # ---------------- EXP 02: STRICT WITHIN-PIXEL, WITHIN-YEAR ----------------
        ws, cs = strict_year(yd)
        bs = ws["_block20"].to_numpy(np.int64)
        fs = ws["_block20"].map(maps[20]).to_numpy(np.int8)
        rs, _, ccs, ds, ys = fit_spatial(
            ws, cs, f"exp02_year_{yy}_strict_FE", bs, fs, True
        )
        strict_rows.append(dict(year=yy, **rs))
        strict_comp.append(ccs)

        del yd, wA, wC, ws, dr, yr, ds, ys, b20, f20, bc, fc, bs, fs
        clean_gpu()

    # ---------------- EXP 01: ANNUAL + POOLED PRIMARY ----------------
    annual = pd.DataFrame(annual).sort_values("year")
    annual.to_csv(out / "exp01_yearly_primary_effects.csv", index=False)

    # Frozen-hyperparameter sanity check under the CURRENT month-FE specification.
    # No hyperparameters are reselected here; this is a transparent OOF performance audit.
    nuisance_summary = pd.DataFrame([dict(
        n_years=int(len(annual)),
        treatment_oof_r2_mean=float(annual.treatment_oof_r2.mean()),
        treatment_oof_r2_median=float(annual.treatment_oof_r2.median()),
        treatment_oof_r2_min=float(annual.treatment_oof_r2.min()),
        treatment_oof_r2_max=float(annual.treatment_oof_r2.max()),
        treatment_years_negative_oof_r2=int((annual.treatment_oof_r2 < 0).sum()),
        outcome_oof_r2_mean=float(annual.outcome_oof_r2.mean()),
        outcome_oof_r2_median=float(annual.outcome_oof_r2.median()),
        outcome_oof_r2_min=float(annual.outcome_oof_r2.min()),
        outcome_oof_r2_max=float(annual.outcome_oof_r2.max()),
        outcome_years_negative_oof_r2=int((annual.outcome_oof_r2 < 0).sum()),
        feature_specification=(
            "11 calendar-month fixed effects (January reference) + Model A covariates + "
            "nominal one-hot LCCS"
        ),
        lccs_encoding="nominal_one_hot",
        lccs_reference=(None if LCCS_REFERENCE is None else float(LCCS_REFERENCE)),
        lccs_dummy_features=max(0, len(LCCS_LEVELS) - 1),
        hyperparameter_status="Frozen from prior blocked tuning; not reselected on these results",
        interpretation=(
            "Sanity check only. OOF nuisance performance is reported to show how the frozen learner behaves "
            "under the final month-FE feature specification; it is not used to choose the causal estimate."
        ),
    )])
    nuisance_summary.to_csv(out / "exp01_primary_nuisance_performance_sanity_check.csv", index=False)

    ac = pd.concat(primary_comp, ignore_index=True)
    ac.to_csv(out / "exp01_primary_space_year_components.csv", index=False)

    p1 = pooled(ac, "exp01_year_stratified_pooled_DML")
    neg = int((annual.effect_per_10ug_m3 < 0).sum())
    p1["negative_years"] = neg
    pd.DataFrame([p1]).to_csv(out / "exp01_pooled_primary_DML.csv", index=False)

    primary_resid_mean = dr_sum / dr_n
    primary_resid_var = (dr_sumsq - dr_n * primary_resid_mean**2) / (dr_n - 1)
    primary_resid_sd = math.sqrt(max(primary_resid_var, 0.0))
    pd.DataFrame([dict(
        n=dr_n,
        treatment_residual_mean=primary_resid_mean,
        treatment_residual_sd=primary_resid_sd,
        dml_denominator_raw=dr_sumsq,
        denominator_per_observation=dr_sumsq / dr_n,
    )]).to_csv(out / "exp01_primary_O3_residual_scale.csv", index=False)

    # Cheap overlap / information-concentration diagnostics, no refitting.
    ps = pd.DataFrame(primary_support).sort_values("year")
    ps.to_csv(out / "exp01_primary_support_diagnostics_by_year.csv", index=False)
    support_summary = pd.DataFrame([dict(
        years=len(ps),
        residual_sd_median=float(ps.residual_sd.median()),
        residual_sd_min=float(ps.residual_sd.min()),
        residual_sd_max=float(ps.residual_sd.max()),
        top1pct_denominator_share_median=float(ps.top1pct_abs_residual_denominator_share.median()),
        top1pct_denominator_share_max=float(ps.top1pct_abs_residual_denominator_share.max()),
        top5pct_denominator_share_median=float(ps.top5pct_abs_residual_denominator_share.median()),
        top5pct_denominator_share_max=float(ps.top5pct_abs_residual_denominator_share.max()),
        note="Exact within-year diagnostics; no leverage-trimming refits are performed."
    )])
    support_summary.to_csv(out / "exp01_primary_support_diagnostics_summary.csv", index=False)

    # ---------------- EXP 02: STRICT FE POOLED ----------------
    strict_df = pd.DataFrame(strict_rows).sort_values("year")
    strict_df.to_csv(out / "exp02_strict_yearly_effects.csv", index=False)
    sc = pd.concat(strict_comp, ignore_index=True)
    p2 = pooled(sc, "exp02_strict_pixel_year_FE_pooled_DML")
    p2["negative_years"] = int((strict_df.effect_per_10ug_m3 < 0).sum())
    pd.DataFrame([p2]).to_csv(out / "exp02_strict_pooled_DML.csv", index=False)

    # ---------------- EXP 05: BLOCK SCALE ----------------
    br = pd.DataFrame(block_rows).sort_values(["block_factor", "year"])
    br.to_csv(out / "exp05_block_scale_yearly.csv", index=False)
    bp = []
    for bf in BLOCK_FACTORS:
        pp = pooled(pd.concat(block_comp[bf], ignore_index=True), f"exp05_block_{bf}_pooled")
        sub = br[br.block_factor == bf]
        pp.update(
            block_factor=bf,
            approx_block_km=grid[0] * bf / 1000,
            negative_years=int((sub.effect_per_10ug_m3 < 0).sum()),
        )
        bp.append(pp)
    bp = pd.DataFrame(bp)
    bp.to_csv(out / "exp05_block_scale_pooled.csv", index=False)

    # Empirical residual/score spatial dependence, reported as a diagnostic rather
    # than an additional experiment or a post-hoc block-selection rule.
    spatial_diag = (
        pd.concat(spatial_diag_rows, ignore_index=True)
        if spatial_diag_rows else pd.DataFrame()
    )
    spatial_corr, spatial_corr_summary = summarize_spatial_dependence(
        spatial_diag, grid[0] * REF_BLOCK_FACTOR / 1000.0, out
    )

    # ---------------- EXP 06: COVARIATE SETS ----------------
    cr = pd.DataFrame(cov_rows).sort_values(["model", "year"])
    cr.to_csv(out / "exp06_covariate_yearly.csv", index=False)
    cp = []
    for k in "ABC":
        pp = pooled(pd.concat(cov_comp[k], ignore_index=True), f"exp06_model_{k}_pooled")
        sub = cr[cr.model == k]
        pp.update(
            model=k,
            paired_n_median=float(sub.paired_n.median()),
            negative_years=int((sub.effect_per_10ug_m3 < 0).sum()),
            mean_treatment_oof_r2=float(sub.treatment_oof_r2.mean()),
            mean_outcome_oof_r2=float(sub.outcome_oof_r2.mean()),
            interpretation=(
                "primary" if k == "A" else
                "extended pre-treatment adjustment" if k == "B" else
                "over-adjustment/post-treatment sensitivity only"
            ),
        )
        cp.append(pp)
    cp = pd.DataFrame(cp)
    cp.to_csv(out / "exp06_covariate_pooled.csv", index=False)

    # ---------------- EXP 07: SMALL-CLUSTER INFERENCE ----------------
    # Primary annual uncertainty: spatial x month two-way clustered t inference (already in `annual`).
    # Additional sensitivity: Webb wild MONTH-cluster bootstrap-t, studentized by the same two-way variance.
    # BH-FDR is retained only as a supplementary multiplicity diagnostic across the 23 annual sensitivities.
    aw = pd.DataFrame(annual_wild).sort_values("year")
    aw["supplementary_fdr_q_bh"], aw["supplementary_fdr_reject_0p05"] = bh_fdr(
        aw["wild_month_cluster_twoway_studentized_p_value"].to_numpy(), FDR_ALPHA
    )
    annual_sc = annual.merge(
        aw.drop(columns=["effect_per_10ug_m3"]), on="year", how="left", validate="one_to_one"
    )
    annual_sc.to_csv(out / "exp07_yearly_small_cluster_inference.csv", index=False)

    wb = pd.DataFrame([
        wild_year_cluster(ac, "exp07_primary_wild_year_cluster", WILD_SEED),
        wild_year_cluster(sc, "exp07_strict_FE_wild_year_cluster", WILD_SEED + 1),
    ])
    wb.to_csv(out / "exp07_pooled_wild_year_cluster_bootstrap.csv", index=False)

    # ---------------- EXP 08: TEMPORAL STABILITY ----------------
    # One designated test only; no post-hoc calendar breakpoint search.
    ft = linear_time_trend_test(ac)
    ft.to_csv(out / "exp08_linear_time_trend.csv", index=False)

    # Annual effect figure.
    fig, ax = plt.subplots(figsize=(10, 5.6))
    yv = annual.effect_per_10ug_m3.to_numpy()
    lo = annual.ci95_low_t_min_cluster_per_10ug_m3.to_numpy()
    hi = annual.ci95_high_t_min_cluster_per_10ug_m3.to_numpy()
    xerr = np.vstack([yv - lo, hi - yv])
    ax.errorbar(annual.year, yv, yerr=xerr, fmt="o", ms=4, capsize=2, lw=1)
    ax.axhline(0, ls="--", lw=1)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Adjusted O$_3$-SIF slope per +10 $\mu$g m$^{-3}$ O$_3$")
    ax.set_title("Year-specific spatial-blocked DML effects")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "Figure_Exp01_yearly_DML_effects.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    rows = [
        dict(experiment=1, analysis=p1["analysis"], effect_per_10ug_m3=p1["effect_per_10ug_m3"], report_p=p1["p_value_t_min_cluster_sensitivity"], notes=f"PRIMARY; year-stratified spatial+seasonal contrast; {neg}/{len(annual)} annual effects < 0; O3 residual SD={primary_resid_sd:.4f}"),
        dict(experiment=2, analysis=p2["analysis"], effect_per_10ug_m3=p2["effect_per_10ug_m3"], report_p=p2["p_value_t_min_cluster_sensitivity"], notes=f"Within-pixel, within-year validation; {p2['negative_years']}/{len(strict_df)} annual effects < 0"),
    ]
    for _, z in bp.iterrows():
        rows.append(dict(experiment=5, analysis=f"block_{int(z.block_factor)}", effect_per_10ug_m3=z.effect_per_10ug_m3, report_p=z.p_value_t_min_cluster_sensitivity, notes=f"~{z.approx_block_km:.1f} km"))
    for _, z in cp.iterrows():
        rows.append(dict(experiment=6, analysis=f"Model_{z.model}", effect_per_10ug_m3=z.effect_per_10ug_m3, report_p=z.p_value_t_min_cluster_sensitivity, notes=z.interpretation))
    for _, z in wb.iterrows():
        rows.append(dict(experiment=7, analysis=z.analysis, effect_per_10ug_m3=z.effect_per_10ug_m3, report_p=z.wild_cluster_p_value, notes="Webb wild year-cluster bootstrap"))
    z = ft.iloc[0]
    rows.append(dict(experiment=8, analysis=z.test, effect_per_10ug_m3=z.estimate_per_10ug_m3, report_p=z.p_value_t, notes="Single designated temporal-stability test; no calendar breakpoint tests"))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "EXP01_02_05_06_07_08_SUMMARY.csv", index=False)
    return summary, annual, p1, p2, ac, primary_resid_sd, support_summary


# =============================================================================
# EXP 04: CONDITIONAL FUTURE-O3 NEGATIVE CONTROL
# =============================================================================
def build_future_o3_panel(data, yearly_A):
    """
    Same-pixel, same-calendar-month +1-year future exposure.
    Current row y receives O3 from y+1, so the current-year range is 2000-2021.
    """
    fut = data[["_pixel", YEAR, MONTH, EXPOSURE]].copy()
    fut[YEAR] = (fut[YEAR].astype(np.int32) - 1).astype(np.int16)
    fut = fut.rename(columns={EXPOSURE: "O3_future"})

    base_cols = list(dict.fromkeys([
        OUTCOME, EXPOSURE, XCOL, YCOL, YEAR, MONTH,
        "_pixel", "_block20"
    ] + yearly_A))
    base = data[data[YEAR] < END_YEAR][base_cols].copy()
    p = base.merge(
        fut,
        on=["_pixel", YEAR, MONTH],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    p = p.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(dict.fromkeys([OUTCOME, EXPOSURE, "O3_future"] + yearly_A))
    ).reset_index(drop=True)
    p["_joint_pos"] = np.arange(len(p), dtype=np.int64)
    return p


def fit_joint_current_future_placebo(p, controls, out):
    """
    Year-stratified spatial cross-fitting for Y, current O3 and future O3.
    The orthogonal second stage includes residualized current and future O3 jointly.
    The future coefficient is therefore a conditional negative-control coefficient.
    """
    fmap = balanced_map(p["_block20"].to_numpy(), N_SFOLD)
    n = len(p)
    ph_cur = np.full(n, np.nan, np.float32)
    ph_fut = np.full(n, np.nan, np.float32)
    ph_y = np.full(n, np.nan, np.float32)
    fold_rows = []

    for yy in sorted(p[YEAR].unique()):
        mask = p[YEAR].to_numpy() == yy
        w = p.loc[mask].copy()
        gpos = w["_joint_pos"].to_numpy(np.int64)
        X, feature_names = make_design_matrix(w, controls)
        tcur = w[EXPOSURE].to_numpy(np.float32)
        tfut = w["O3_future"].to_numpy(np.float32)
        y = w[OUTCOME].to_numpy(np.float32)
        fold = w["_block20"].map(fmap).to_numpy(np.int8)

        for f, _, tr, te in splits_spatial(fold):
            sid = f"y{int(yy)}_s{f+1}"
            print(f"  exp04_conditional_future_placebo: {sid} train={len(tr):,} test={len(te):,}")
            targets = [
                ("current_O3", "treatment", tcur, ph_cur, 0),
                ("future_O3", "treatment", tfut, ph_fut, 100000),
                ("outcome", "outcome", y, ph_y, 200000),
            ]
            for stage_name, hp_stage, target, dest, offset in targets:
                seed_key = int(yy) * 10 + f + offset
                md = model(hp_stage, seed_key, None, False)
                md.fit(X[tr], target[tr])
                pred = md.predict(X[te]).astype(np.float32)
                dest[gpos[te]] = pred
                q = metrics(target[te], pred)
                fold_rows.append(dict(
                    analysis="exp04_conditional_future_placebo",
                    year=int(yy),
                    stage=stage_name,
                    split=sid,
                    space_fold=f + 1,
                    n_train=len(tr),
                    n_test=len(te),
                    r2=q["r2"],
                    rmse=q["rmse"],
                    mae=q["mae"],
                    mean_residual_bias=q["bias"],
                ))
                del md, pred
                clean_gpu()
        del w, X, tcur, tfut, y, fold
        clean_gpu()

    if not (np.isfinite(ph_cur).all() and np.isfinite(ph_fut).all() and np.isfinite(ph_y).all()):
        raise RuntimeError("conditional placebo OOF coverage incomplete")

    cur = p[EXPOSURE].to_numpy(np.float32)
    fut = p["O3_future"].to_numpy(np.float32)
    y = p[OUTCOME].to_numpy(np.float32)
    dr_cur = (cur - ph_cur).astype(np.float32)
    dr_fut = (fut - ph_fut).astype(np.float32)
    yr = (y - ph_y).astype(np.float32)
    space = p["_block20"].to_numpy(np.int64)
    years = p[YEAR].to_numpy(np.int16)

    # Current-only estimate on the exact placebo sample, with the same nuisance Y/current residuals.
    current_only = infer(dr_cur, yr, space, years, True)
    current_only.update(analysis="exp04_current_O3_only_same_sample")

    # Joint orthogonal second stage: future coefficient is conditional on current O3.
    D = np.column_stack([dr_cur, dr_fut])
    joint, beta, cov, joint_resid = infer_multitreatment(
        D, yr, space, years,
        names=["current_O3_conditional_on_future", "future_O3_conditional_on_current"],
        small_sample=True,
    )

    raw_corr = float(np.corrcoef(cur.astype(np.float64), fut.astype(np.float64))[0, 1])
    resid_corr = float(np.corrcoef(dr_cur.astype(np.float64), dr_fut.astype(np.float64))[0, 1])
    vif = float(1.0 / max(1e-12, 1.0 - resid_corr**2))
    treatment_corr = np.corrcoef(D.astype(np.float64).T)
    cond_number = float(np.linalg.cond(treatment_corr))

    diagnostics = pd.DataFrame([dict(
        n=len(p),
        year_min=int(p[YEAR].min()),
        year_max=int(p[YEAR].max()),
        raw_current_future_O3_correlation=raw_corr,
        residual_current_future_O3_correlation=resid_corr,
        residualized_two_treatment_VIF=vif,
        residualized_treatment_correlation_condition_number=cond_number,
        current_residual_sd=float(dr_cur.std(ddof=1)),
        future_residual_sd=float(dr_fut.std(ddof=1)),
        current_denominator_per_observation=float((dr_cur.astype(np.float64) @ dr_cur.astype(np.float64)) / len(p)),
        future_denominator_per_observation=float((dr_fut.astype(np.float64) @ dr_fut.astype(np.float64)) / len(p)),
        interpretation="Future-O3 coefficient is tested conditional on contemporaneous O3; correlation/VIF quantify remaining collinearity."
    )])

    pd.DataFrame(fold_rows).to_csv(out / "exp04_joint_placebo_fold_diagnostics.csv", index=False)
    pd.DataFrame([current_only]).to_csv(out / "exp04_current_only_same_sample.csv", index=False)
    joint.to_csv(out / "exp04_joint_current_future_coefficients.csv", index=False)
    diagnostics.to_csv(out / "exp04_conditional_placebo_diagnostics.csv", index=False)

    return current_only, joint, diagnostics, dr_cur, dr_fut, yr


def run_future_placebo(data, yearly, root):
    out = mkdir(root / "04_conditional_future_O3_placebo")
    print("\n" + "=" * 100)
    print("EXP 04: conditional future-O3 (+1 year) negative-control exposure")
    print("=" * 100)

    p = build_future_o3_panel(data, yearly["A"])
    current_only, joint, diagnostics, dr_cur, dr_fut, yr = fit_joint_current_future_placebo(
        p, yearly["A"], out
    )

    cur_joint = joint[joint.treatment == "current_O3_conditional_on_future"].iloc[0]
    fut_joint = joint[joint.treatment == "future_O3_conditional_on_current"].iloc[0]

    summary = pd.DataFrame([
        dict(
            experiment=4,
            analysis="placebo_current_O3_only_same_sample",
            effect_per_10ug_m3=current_only["effect_per_10ug_m3"],
            report_p=current_only["p_value_t_min_cluster_sensitivity"],
            notes="Current-O3 estimate on exact 2000-2021 placebo sample using the same OOF nuisance residuals",
        ),
        dict(
            experiment=4,
            analysis="placebo_current_O3_joint",
            effect_per_10ug_m3=float(cur_joint.effect_per_10ug_m3),
            report_p=float(cur_joint.p_value_t),
            notes="Contemporaneous O3 coefficient in the joint residualized current+future model",
        ),
        dict(
            experiment=4,
            analysis="future_O3_conditional_on_current",
            effect_per_10ug_m3=float(fut_joint.effect_per_10ug_m3),
            report_p=float(fut_joint.p_value_t),
            notes=(
                "PRIMARY PLACEBO TEST: future O3 coefficient after contemporaneous O3 is included jointly; "
                f"residual current-future correlation={float(diagnostics.iloc[0].residual_current_future_O3_correlation):.3f}, "
                f"VIF={float(diagnostics.iloc[0].residualized_two_treatment_VIF):.2f}"
            ),
        ),
    ])
    summary.to_csv(out / "EXP04_SUMMARY.csv", index=False)

    del p, dr_cur, dr_fut, yr
    clean_gpu()
    return summary, diagnostics


# =============================================================================
# FINAL SUMMARY + FIGURES
# =============================================================================
def make_master_summary(root, pieces):
    master = pd.concat(pieces, ignore_index=True)
    master.to_csv(root / "DML_CORE_EXPERIMENTS_MASTER_SUMMARY.csv", index=False)
    return master


def make_core_forest(master, root):
    wanted = [
        "exp00_full_period_all_data_2D_DML",
        "exp01_year_stratified_pooled_DML",
        "exp02_strict_pixel_year_FE_pooled_DML",
        "interannual_pixel_month_anomaly",
        "future_O3_conditional_on_current",
        "block_10",
        "block_20",
        "block_30",
        "Model_A",
        "Model_B",
        "Model_C",
    ]
    d = master[master.analysis.isin(wanted)].copy()
    if len(d) == 0:
        return
    d["order"] = d.analysis.map({k: i for i, k in enumerate(wanted)})
    d = d.sort_values("order")

    # Compact overview only. Exact estimator-specific CIs remain in experiment CSVs.
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.45 * len(d) + 1.5)))
    yy = np.arange(len(d))[::-1]
    ax.scatter(d.effect_per_10ug_m3, yy, s=35)
    ax.axvline(0, ls="--", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(d.analysis)
    ax.set_xlabel(r"Adjusted O$_3$-SIF slope per +10 $\mu$g m$^{-3}$ O$_3$")
    ax.set_title("Compact DML evidence chain")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "Figure_Core_DML_estimates.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    root = mkdir(OUTPUT_DIR)
    print("=" * 110)
    print("REVIEWER-RESISTANT COMPACT O3-SIF DML — FULL RERUN")
    print("=" * 110)
    print("Input:", INPUT_FILE)
    print("Output:", OUTPUT_DIR)
    print("XGBoost:", xgb.__version__, "device=", DEVICE)
    print("FORCE_RERUN:", FORCE_RERUN)

    data, full, yearly, grid = load_data()

    metadata = {
        "script_version": SCRIPT_VERSION,
        "input": INPUT_FILE,
        "output": OUTPUT_DIR,
        "years": [START_YEAR, END_YEAR],
        "primary_estimand": (
            "year-stratified spatial-blocked DML pooled across years; within each year it uses spatial + seasonal contrasts"
        ),
        "full_period_all_data_estimand": (
            "single 2000-2022 nuisance architecture with 5 spatial x 5 contiguous temporal cross-fitting; "
            "reported as a distinct all-data specification regardless of significance"
        ),
        "strict_validation_estimand": "within-pixel, within-year pixel-year-demeaned DML",
        "complementary_interannual_estimand": (
            "pixel-month within-transformed interannual DML with spatial-blocked cross-fitting and spatial x year clustered inference"
        ),
        "paired_design_check": (
            "raw vs within-transformed interannual models fitted on identical observations, identical spatial folds, identical control-column names and learner architecture"
        ),
        "full_controls": full,
        "yearly_controls": yearly,
        "block_factors": BLOCK_FACTORS,
        "hyperparameters": HP,
        "lccs_encoding": lccs_encoding_metadata(),
        "experiments": {
            "00": "Full-period all-data DML: single 2000-2022 nuisance architecture with 5 spatial x 5 contiguous temporal folds",
            "01": "Primary 2000-2022 year-stratified spatial-blocked DML + identifying-variation/support diagnostics",
            "02": "Strict within-pixel, within-year pixel-year FE DML",
            "03": "Interannual pixel-month within-transformed estimand + exact paired raw-vs-within design check",
            "04": "Conditional future-O3 negative-control exposure: current and future O3 entered jointly after orthogonalization",
            "05": "Spatial block-scale sensitivity + empirical OOF spatial-dependence diagnostic (diagnostic does not select the model)",
            "06": "Covariate-set A/B/C sensitivity",
            "07": "Small-cluster robust inference: two-way cluster-robust annual SE + Webb wild month-cluster bootstrap-t sensitivity + pooled wild year-cluster bootstrap",
            "08": "Temporal stability: single designated linear trend test only",
        },
        "removed": [
            "heterogeneity analyses",
            "pre/post 2017 2D reruns",
            "temporal-fold-count sensitivity",
            "random-effects meta-analysis",
            "leverage trimming refits",
            "strict-FE duplicate formal time tests",
            "all post-hoc calendar breakpoint tests, including 2017",
        ],
        "interpretation_notes": {
            "primary_selection": "The year-stratified estimand is designated as primary because it directly addresses the contemporaneous physiological question; this designation is not based on statistical significance or effect magnitude.",
            "full_period_all_data": (
                "EXP00 is retained and reported regardless of significance. It fits one nuisance architecture over 2000-2022 "
                "and uses both spatial and contiguous temporal holdout. Because this changes nuisance architecture and cross-fitting "
                "relative to the primary year-stratified model, the two estimates are compared descriptively rather than treated as "
                "a formal same-estimand significance contest."
            ),
            "terminology": "Use 'year-stratified DML' for the primary model; reserve 'within-pixel, within-year' for the strict FE design.",
            "interannual_null": "The interannual estimate is interpreted as a distinct within-pixel-month estimand; residual identifying variation and nuisance-model diagnostics are reported explicitly.",
            "paired_contrast": "Only the exact-row paired raw-vs-within design receives a formal direct contrast; primary vs interannual estimands are compared descriptively because they answer different questions.",
            "interannual_crossfitting": "Exp03 uses spatial-blocked cross-fitting only. Entire held-out spatial blocks contain all years for those pixels; calendar-year dependence is handled by the second clustering dimension rather than by forcing tree nuisance models to extrapolate across contiguous held-out time blocks.",
            "placebo": "The primary placebo coefficient is future O3 conditional on contemporaneous O3 in a joint orthogonal second stage; residual current-future correlation and VIF are reported.",
            "Model_C": "over-adjustment/post-treatment sensitivity only",
            "causal_limit": "DML adjusts measured confounding but does not by itself eliminate unmeasured confounding; causal language requires the stated identification assumptions.",
            "seasonality": "Primary and sensitivity nuisance models use 11 calendar-month indicators (January reference), not a single harmonic seasonal form.",
            "lccs_encoding": "LCCS is a nominal class label. It is expanded to binary one-hot indicators at model-matrix construction; raw numeric class codes are never passed to XGBoost as an ordered predictor.",
            "spatial_scale": "The empirical OOF spatial correlogram is descriptive context for the reference block scale and is not used to choose the primary model after seeing the effect estimate.",
            "hyperparameters": "Nuisance-model hyperparameters remain frozen from the prior blocked tuning stage and are not reselected from these rerun results; an explicit OOF nuisance-performance sanity table is exported for the final month-FE primary specification, and fold/design diagnostics remain available elsewhere.",
            "annual_small_cluster_inference": "Primary annual uncertainty is spatial x month two-way cluster robust. The Webb procedure resamples calendar-month clusters only and uses spatial x month two-way studentization; it is a small-month-cluster sensitivity, not a multiway wild-cluster bootstrap.",
            "measurement_error_boundary": "Differences between the primary and interannual estimands are not attributed automatically to biology or to measurement error; exposure variation, nuisance diagnostics and the distinct identifying contrast are reported explicitly.",
        },
    }
    (root / "DML_CORE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 1/2/5/6/7/8: establish the prespecified year-stratified primary evidence chain.
    s_core, annual, p1, p2, primary_comp, primary_resid_sd, support_summary = run_yearly_core(
        data, yearly, grid, root
    )

    # 00: transparent full-period ALL-DATA experiment.
    # Executed after the yearly core so that primary outputs are already saved if the
    # computationally heavy full-period fit encounters a memory/runtime failure.
    if RUN_FULL_PERIOD_ALL_DATA:
        s0, full_period_result, full_period_support = run_full_period_all_data_dml(
            data, full, root
        )
    else:
        s0 = pd.DataFrame(columns=["experiment", "analysis", "effect_per_10ug_m3", "report_p", "notes"])
        full_period_result = None
        full_period_support = None

    # 3: distinct interannual within-pixel-month estimand + exact paired spatial design check.
    s3, paired_variation, interannual_result = run_interannual_design_comparison(
        data, full, root, p1, p2
    )

    # 4: conditional negative-control exposure.
    s4, placebo_diagnostics = run_future_placebo(data, yearly, root)

    master = make_master_summary(root, [s0, s_core, s3, s4])

    # Transparent descriptive comparison: full-period all-data vs year-stratified primary.
    # No formal direct contrast is reported because the nuisance architecture and temporal
    # cross-fitting differ between these two specifications.
    full_primary_n = int(annual["n"].sum())
    if full_period_result is not None:
        full_vs_primary = pd.DataFrame([
            dict(
                design="full_period_all_data_2D_DML",
                n=int(full_period_result["n"]),
                effect_per_10ug_m3=float(full_period_result["effect_per_10ug_m3"]),
                ci95_low_per_10ug_m3=float(full_period_result["ci95_low_t_min_cluster_per_10ug_m3"]),
                ci95_high_per_10ug_m3=float(full_period_result["ci95_high_t_min_cluster_per_10ug_m3"]),
                p_value=float(full_period_result["p_value_t_min_cluster_sensitivity"]),
                treatment_oof_r2=float(full_period_result["treatment_oof_r2"]),
                outcome_oof_r2=float(full_period_result["outcome_oof_r2"]),
                treatment_residual_sd=float(full_period_result["treatment_residual_sd"]),
                denominator_per_observation=float(
                    full_period_result["dml_denominator_raw"] / full_period_result["n"]
                ),
                nuisance_architecture="single full-period 2000-2022",
                crossfitting=f"{N_SFOLD} spatial x {FULL_TFOLD} contiguous temporal folds",
                comparison_status="descriptive only; not an exact-paired formal contrast",
            ),
            dict(
                design="year_stratified_primary_pooled",
                n=full_primary_n,
                effect_per_10ug_m3=float(p1["effect_per_10ug_m3"]),
                ci95_low_per_10ug_m3=float(p1["ci95_low_t_min_cluster_per_10ug_m3"]),
                ci95_high_per_10ug_m3=float(p1["ci95_high_t_min_cluster_per_10ug_m3"]),
                p_value=float(p1["p_value_t_min_cluster_sensitivity"]),
                treatment_oof_r2=np.nan,
                outcome_oof_r2=np.nan,
                treatment_residual_sd=float(primary_resid_sd),
                denominator_per_observation=float(p1["dml_denominator_raw"] / full_primary_n),
                nuisance_architecture="separate nuisance architecture within each calendar year",
                crossfitting=f"{N_SFOLD} spatial folds within each year",
                comparison_status="prespecified primary estimand; descriptive comparison with EXP00",
            ),
        ])
        full_vs_primary.to_csv(
            root / "FULL_PERIOD_VS_YEAR_STRATIFIED_DESCRIPTIVE.csv", index=False
        )

    # Final identifying-variation table: primary full sample vs exact paired raw/within.
    primary_info = p1["dml_denominator_raw"] / full_primary_n
    extra = pd.DataFrame([dict(
        design="full_primary_year_stratified",
        n=full_primary_n,
        effect_per_10ug_m3=p1["effect_per_10ug_m3"],
        treatment_oof_r2=np.nan,
        outcome_oof_r2=np.nan,
        residual_mean=np.nan,
        residual_sd=primary_resid_sd,
        residual_q01=np.nan,
        residual_q05=np.nan,
        residual_median=np.nan,
        residual_q95=np.nan,
        residual_q99=np.nan,
        denominator_raw=p1["dml_denominator_raw"],
        denominator_per_observation=primary_info,
        top5pct_abs_residual_denominator_share=float(support_summary.iloc[0].top5pct_denominator_share_median),
        top1pct_abs_residual_denominator_share=float(support_summary.iloc[0].top1pct_denominator_share_median),
        interannual_to_paired_raw_residual_sd_ratio=np.nan,
        interannual_to_paired_raw_information_ratio=np.nan,
    )])
    variation_final = pd.concat([extra, paired_variation], ignore_index=True, sort=False)

    inter = variation_final[variation_final.design == "paired_interannual_within_spatial"].iloc[0]
    variation_final["interannual_to_FULL_PRIMARY_residual_sd_ratio"] = float(inter.residual_sd / primary_resid_sd)
    variation_final["interannual_to_FULL_PRIMARY_information_ratio"] = float(inter.denominator_per_observation / primary_info)
    variation_final.to_csv(root / "O3_IDENTIFYING_VARIATION_FINAL_TABLE.csv", index=False)

    make_core_forest(master, root)

    print("\n" + "=" * 110)
    print("ALL RETAINED EXPERIMENTS FINISHED")
    print("Master summary:", root / "DML_CORE_EXPERIMENTS_MASTER_SUMMARY.csv")
    if RUN_FULL_PERIOD_ALL_DATA:
        print("Full-period vs year-stratified:", root / "FULL_PERIOD_VS_YEAR_STRATIFIED_DESCRIPTIVE.csv")
    print("Identifying-variation table:", root / "O3_IDENTIFYING_VARIATION_FINAL_TABLE.csv")
    print("=" * 110)

    del data
    clean_gpu()


if __name__ == "__main__":
    main()
