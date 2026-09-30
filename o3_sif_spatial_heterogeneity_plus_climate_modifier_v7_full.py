#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONTHLY SPATIAL KERNEL ORTHOGONAL DML + CLIMATE MODIFIER — v6-ALIGNED FULL RECOMPUTATION
=====================================================================

This program is the spatial-heterogeneity companion to the current
reviewer-compact O3 -> SIF DML main program (v6.0).

Stage 1 — exactly aligned nuisance DML
--------------------------------------
Re-read the original 2000-2022 panel and redo the PRIMARY year-stratified,
5-fold spatial-blocked Model-A nuisance cross-fitting used by the v6 main
program:

    T_res = O3  - E[O3 | W]
    Y_res = SIF - E[SIF | W]

Alignment with the v6 main program:
- reference spatial block factor = 20;
- globally balanced 5-fold spatial cross-fitting;
- frozen global XGBoost nuisance hyperparameters (no retuning);
- Model A environmental controls;
- 11 calendar-month fixed-effect indicators (January reference), NOT sin/cos;
- LCCS treated as nominal and one-hot encoded, NOT as an ordered number;
- x/y block and pixel IDs constructed with the same formulas as the main program.

Stage 2 — orthogonal sufficient statistics
------------------------------------------
For each year x month x observed pixel save

    B = sum(T_res * Y_res)
    A = sum(T_res^2)

Stage 3 — monthly spatial-kernel orthogonal DML
-----------------------------------------------
For each calendar month and every observed target pixel s estimate

    theta_m(s) = sum_i K_h(d_is) * B_i / sum_i K_h(d_is) * A_i

Primary bandwidth = 75 km.
Sensitivity bandwidths = 50 and 100 km.

Stage 4 — local uncertainty and support
---------------------------------------
Calculate year-cluster robust local standard errors, local treatment
information, effective neighbor count, local year coverage, and the
recommended support mask.

Important
---------
- This script intentionally uses a NEW output/cache directory. Old caches from
  pre-v6 programs are methodologically incompatible and must not be reused.
- No second-stage XGBoost R-learner.
- No ordinary GWR.
- No post-hoc spatial interpolation.
- The Gaussian spatial kernel is part of the effect estimator.
- Output remains one effect estimate for each observed pixel and month.

Stage 5 — pre-specified climate-state effect modification
---------------------------------------------------------
For April-September only, retain t2m in the v6 orthogonal cache, construct
same-pixel x same-calendar-month interannual t2m anomalies, and fit the
pre-specified R-loss / orthogonal-moment effect model:

    theta(i,y,m) = theta_m + gamma * t2m_anomaly(i,y,m)

where theta_m are month-specific baseline O3-SIF slopes and gamma is the ONE
climate-state modifier. Primary inference is two-way spatial-block x year
clustered. No climate-variable scanning or post-hoc month-window search.
"""

from __future__ import annotations

import gc
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
from scipy.ndimage import convolve1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = r"/root/autodl-tmp/wq/matched_data_albers_all1.csv"

# New directory: DO NOT reuse the old 0902 cache because the v6 nuisance design
# changed (month FE + nominal LCCS encoding).
OUTPUT_DIR = Path(
    r"/root/autodl-tmp/wql/0925/monthly_spatial_kernel_dml_v6_climate_modifier_full"
)
CACHE_DIR = OUTPUT_DIR / "year_month_pixel_cache_v6"

OUTCOME = "SIF"
TREATMENT = "O3"
XCOL = "x"
YCOL = "y"
YEAR = "year"
MONTH = "month"

START_YEAR = 2000
END_YEAR = 2022

RANDOM_STATE = 42
DEVICE = "cuda"
N_JOBS = 8
MAX_BIN = 256

N_FOLDS = 5
SPATIAL_BLOCK_FACTOR = 20

SCRIPT_VERSION = "monthly-spatial-kernel-dml-v6-climate-modifier-full-1.0"
ALIGNED_MAIN_VERSION = "reviewer-compact-dml-full-rerun-v6.0"

# Fresh full run for reporting. Keep False for the first v6-aligned run.
# Set True only after THIS v6-aligned program has already created compatible caches.
RESUME = False

PRIMARY_BANDWIDTH_KM = 75.0
RUN_BANDWIDTH_SENSITIVITY = True
SENSITIVITY_BANDWIDTHS_KM = [50.0, 100.0]
KERNEL_TRUNCATE = 2.0

MIN_LOCAL_YEAR_CLUSTERS = 10
MIN_LOCAL_MEAN_YEARS = 10.0
MIN_EFFECTIVE_NEIGHBOR_PIXELS = 50.0
LOCAL_DENOMINATOR_Q = 0.05
LOCAL_SE_Q = 0.95
MIN_LOCAL_KERNEL_WEIGHT = 1e-10

# -----------------------------------------------------------------------------
# Pre-specified NCC climate-state modifier stage
# -----------------------------------------------------------------------------
RUN_CLIMATE_STATE_MODIFIER = True
CLIMATE_OUTPUT_DIR = OUTPUT_DIR / "climate_state_modifier"
CLIMATE_ACTIVE_MONTHS = (4, 5, 6, 7, 8, 9)
CLIMATE_MIN_CLIMATOLOGY_YEARS = 10
CLIMATE_ALPHA = 0.05
CLIMATE_MULTIPLIER_BOOTSTRAP_REPS = 4999
CLIMATE_CURVE_Q_LOW = 0.05
CLIMATE_CURVE_Q_HIGH = 0.95
CLIMATE_CURVE_POINTS = 121

# Same Model A as the v6 main program. Seasonality is appended separately as
# 11 month fixed-effect indicators (January reference).
MODEL_A = [
    "DEM", "lccs", "t2m", "ssrd", "tp", "u10", "v10", "sp", "stl1", "swvl1"
]
MONTH_FE = [f"_month_{m:02d}" for m in range(2, 13)]
YEAR_TIME = [XCOL, YCOL] + MONTH_FE

# Configured from observed raw LCCS codes in _load_raw_data().
LCCS_LEVELS = tuple()
LCCS_REFERENCE = None

# Same frozen GLOBAL nuisance hyperparameters used by the primary year-stratified
# model in the v6 main program. Per-time-fold HP are not used for this estimator.
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

# =============================================================================
# SPATIAL KERNEL ORTHOGONAL DML
# =============================================================================

def _normalize_component_columns(df):
    """Normalize cache column names from earlier program variants."""
    rename = {}
    if "block" in df.columns and "spatial_block" not in df.columns:
        rename["block"] = "spatial_block"
    if rename:
        df = df.rename(columns=rename)

    required = [
        "year", "month", "pixel_id", "spatial_block",
        "x", "y", "numerator_raw", "denominator_raw", "n_obs",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cache/components missing columns: {missing}")
    return df


def _read_component_file(path, columns=None):
    df = pd.read_parquet(path, columns=columns)
    return _normalize_component_columns(df)


def _inspect_cache_grid(cache_files):
    """Infer the global regular Albers grid from cached pixel coordinates."""
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    dx = dy = None

    for i, p in enumerate(cache_files):
        z = pd.read_parquet(p, columns=["x", "y"])
        x = pd.to_numeric(z["x"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(z["y"], errors="coerce").to_numpy(np.float64)

        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]

        xmin = min(xmin, float(np.min(x)))
        xmax = max(xmax, float(np.max(x)))
        ymin = min(ymin, float(np.min(y)))
        ymax = max(ymax, float(np.max(y)))

        if dx is None:
            ux = np.unique(x)
            uy = np.unique(y)
            ddx = np.diff(np.sort(ux))
            ddy = np.diff(np.sort(uy))
            ddx = ddx[ddx > 0]
            ddy = ddy[ddy > 0]
            if len(ddx) and len(ddy):
                dx = float(np.median(ddx))
                dy = float(np.median(ddy))

        del z, x, y
        gc.collect()

    if dx is None or dy is None:
        raise RuntimeError("Could not infer source-grid spacing from cache.")

    nx = int(round((xmax - xmin) / dx)) + 1
    ny = int(round((ymax - ymin) / dy)) + 1

    return {
        "x0": xmin,
        "y0": ymin,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "xmax": xmax,
        "ymax": ymax,
    }


def _grid_indices(x, y, grid):
    ix = np.rint((np.asarray(x, np.float64) - grid["x0"]) / grid["dx"]).astype(np.int32)
    iy = np.rint((np.asarray(y, np.float64) - grid["y0"]) / grid["dy"]).astype(np.int32)

    ok = (
        (ix >= 0) & (ix < grid["nx"]) &
        (iy >= 0) & (iy < grid["ny"])
    )
    return ix, iy, ok


def _gaussian_kernel_1d(sigma_px, truncate=2.0):
    sigma_px = float(max(sigma_px, 1e-6))
    radius = max(1, int(truncate * sigma_px + 0.5))
    u = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (u / sigma_px) ** 2)
    k /= k.sum()
    return k


def _make_kernel(grid, bandwidth_km):
    kx = _gaussian_kernel_1d(
        bandwidth_km * 1000.0 / grid["dx"],
        truncate=KERNEL_TRUNCATE,
    )
    ky = _gaussian_kernel_1d(
        bandwidth_km * 1000.0 / grid["dy"],
        truncate=KERNEL_TRUNCATE,
    )
    return kx, ky


def _smooth2d(a, kx, ky):
    z = convolve1d(
        np.asarray(a, dtype=np.float64),
        kx,
        axis=1,
        mode="constant",
        cval=0.0,
    )
    z = convolve1d(
        z,
        ky,
        axis=0,
        mode="constant",
        cval=0.0,
    )
    return z


def _smooth2d_sqweights(mask, kx, ky):
    """sum of squared 2-D kernel weights over observed pixels."""
    z = convolve1d(
        np.asarray(mask, dtype=np.float64),
        kx ** 2,
        axis=1,
        mode="constant",
        cval=0.0,
    )
    z = convolve1d(
        z,
        ky ** 2,
        axis=0,
        mode="constant",
        cval=0.0,
    )
    return z


def _accumulate_totals(cache_files, grid):
    """
    First cache pass:
      - total orthogonal numerator/denominator by month x pixel
      - target observation/year support
      - exact pixel/block id raster
      - block-year components for national pooled DML
    """
    shape = (12, grid["ny"], grid["nx"])

    total_A = np.zeros(shape, dtype=np.float64)
    total_B = np.zeros(shape, dtype=np.float64)
    total_nobs = np.zeros(shape, dtype=np.int32)
    target_years = np.zeros(shape, dtype=np.uint8)

    pixel_id_map = np.full(
        (grid["ny"], grid["nx"]),
        -1,
        dtype=np.int64,
    )
    block_map = np.full(
        (grid["ny"], grid["nx"]),
        -1,
        dtype=np.int64,
    )

    block_year_parts = []

    cols = [
        "year", "month", "pixel_id", "spatial_block",
        "x", "y", "numerator_raw", "denominator_raw", "n_obs",
    ]

    for p in cache_files:
        print(f"[kernel pass 1] {p.name}")
        d = _read_component_file(p, columns=cols)

        d["month"] = pd.to_numeric(d["month"], errors="coerce").astype(np.int16)
        d["year"] = pd.to_numeric(d["year"], errors="coerce").astype(np.int16)

        ix, iy, ok = _grid_indices(d["x"], d["y"], grid)
        if not np.all(ok):
            d = d.loc[ok].reset_index(drop=True)
            ix = ix[ok]
            iy = iy[ok]

        m0 = d["month"].to_numpy(np.int16) - 1
        A = d["denominator_raw"].to_numpy(np.float64)
        B = d["numerator_raw"].to_numpy(np.float64)
        nobs = d["n_obs"].to_numpy(np.int32)

        for m in range(12):
            sel = (m0 == m)
            if not np.any(sel):
                continue

            np.add.at(total_A[m], (iy[sel], ix[sel]), A[sel])
            np.add.at(total_B[m], (iy[sel], ix[sel]), B[sel])
            np.add.at(total_nobs[m], (iy[sel], ix[sel]), nobs[sel])
            np.add.at(
                target_years[m],
                (iy[sel], ix[sel]),
                np.ones(int(sel.sum()), dtype=np.uint8),
            )

        pixel_id_map[iy, ix] = d["pixel_id"].to_numpy(np.int64)
        block_map[iy, ix] = d["spatial_block"].to_numpy(np.int64)

        by = (
            d.groupby(
                ["year", "month", "spatial_block"],
                observed=True,
                sort=False,
            )
            .agg(
                numerator_raw=("numerator_raw", "sum"),
                denominator_raw=("denominator_raw", "sum"),
            )
            .reset_index()
        )
        block_year_parts.append(by)

        del d, by, ix, iy, ok, m0, A, B, nobs
        gc.collect()

    block_year = pd.concat(block_year_parts, ignore_index=True)

    return {
        "A": total_A,
        "B": total_B,
        "nobs": total_nobs,
        "target_years": target_years,
        "pixel_id_map": pixel_id_map,
        "block_map": block_map,
        "block_year": block_year,
    }


def _national_pooled_months(block_year):
    """Two-way spatial-block + year clustered pooled DML, month by month."""
    rows = []

    for month in range(1, 13):
        d = block_year.loc[block_year["month"] == month].copy()

        num = float(d["numerator_raw"].sum())
        den = float(d["denominator_raw"].sum())
        theta = num / den

        d["score"] = d["numerator_raw"] - theta * d["denominator_raw"]

        s_space = (
            d.groupby("spatial_block", observed=True)["score"]
            .sum()
            .to_numpy(np.float64)
        )
        s_year = (
            d.groupby("year", observed=True)["score"]
            .sum()
            .to_numpy(np.float64)
        )
        s_inter = d["score"].to_numpy(np.float64)

        gs, gy, gi = len(s_space), len(s_year), len(s_inter)

        ms = (s_space @ s_space) * gs / (gs - 1)
        my = (s_year @ s_year) * gy / (gy - 1)
        mi = (s_inter @ s_inter) * gi / (gi - 1)

        var = (ms + my - mi) / den**2
        fallback = False
        if not np.isfinite(var) or var <= 0:
            var = max(ms, my) / den**2
            fallback = True

        se = math.sqrt(var)
        df = max(1, min(gs, gy) - 1)
        crit = stats.t.ppf(0.975, df)

        rows.append({
            "month": month,
            "effect_per_10ug_m3": theta * 10.0,
            "se_per_10ug_m3": se * 10.0,
            "ci95_low_per_10ug_m3": (theta - crit * se) * 10.0,
            "ci95_high_per_10ug_m3": (theta + crit * se) * 10.0,
            "p_value_t": float(2 * stats.t.sf(abs(theta / se), df)),
            "t_df": df,
            "n_spatial_blocks": gs,
            "n_years": gy,
            "variance_fallback": fallback,
            "dml_numerator_raw": num,
            "dml_denominator_raw": den,
        })

    return pd.DataFrame(rows)


def _build_year_month_rasters(d, grid):
    """Create 12 sparse year-specific A/B/presence rasters."""
    shape = (12, grid["ny"], grid["nx"])
    A = np.zeros(shape, dtype=np.float32)
    B = np.zeros(shape, dtype=np.float32)
    P = np.zeros(shape, dtype=np.uint8)

    ix, iy, ok = _grid_indices(d["x"], d["y"], grid)
    if not np.all(ok):
        d = d.loc[ok].reset_index(drop=True)
        ix = ix[ok]
        iy = iy[ok]

    m0 = d["month"].to_numpy(np.int16) - 1
    den = d["denominator_raw"].to_numpy(np.float32)
    num = d["numerator_raw"].to_numpy(np.float32)

    for m in range(12):
        sel = (m0 == m)
        if not np.any(sel):
            continue

        np.add.at(A[m], (iy[sel], ix[sel]), den[sel])
        np.add.at(B[m], (iy[sel], ix[sel]), num[sel])
        P[m, iy[sel], ix[sel]] = 1

    return A, B, P


def _kernel_stage(cache_files, output_dir, grid):
    """
    Main kernel estimator.

    For target location s and month m:
        theta_m(s) =
          sum_i K_h(d_is) * B_i
          ---------------------------------
          sum_i K_h(d_is) * A_i

    where:
        B_i = T_res * Y_res orthogonal moment
        A_i = T_res^2 treatment information

    Local uncertainty uses year-clustered kernel scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    totals = _accumulate_totals(cache_files, grid)

    national = _national_pooled_months(totals["block_year"])
    national.to_csv(
        output_dir / "calendar_month_pooled_dml_sanity_check.csv",
        index=False,
    )

    bandwidths = [PRIMARY_BANDWIDTH_KM]
    if RUN_BANDWIDTH_SENSITIVITY:
        bandwidths = sorted(set(
            [PRIMARY_BANDWIDTH_KM] + list(SENSITIVITY_BANDWIDTHS_KM)
        ))

    kernels = {}
    theta = {}
    local_A = {}
    score2 = {}
    local_year_clusters = {}

    print("\nPreparing kernel point estimates...")
    for bw in bandwidths:
        kx, ky = _make_kernel(grid, bw)
        kernels[bw] = (kx, ky)

        theta[bw] = np.full(
            (12, grid["ny"], grid["nx"]),
            np.nan,
            dtype=np.float32,
        )
        local_A[bw] = np.zeros(
            (12, grid["ny"], grid["nx"]),
            dtype=np.float32,
        )
        score2[bw] = np.zeros(
            (12, grid["ny"], grid["nx"]),
            dtype=np.float64,
        )
        local_year_clusters[bw] = np.zeros(
            (12, grid["ny"], grid["nx"]),
            dtype=np.uint8,
        )

        for m in range(12):
            Ak = _smooth2d(totals["A"][m], kx, ky)
            Bk = _smooth2d(totals["B"][m], kx, ky)

            valid = Ak > 0
            th = np.full_like(Ak, np.nan, dtype=np.float64)
            th[valid] = Bk[valid] / Ak[valid]

            theta[bw][m] = th.astype(np.float32)
            local_A[bw][m] = Ak.astype(np.float32)

            del Ak, Bk, th, valid
            gc.collect()

    print("\nYear-clustered local uncertainty...")
    cols = [
        "year", "month", "pixel_id", "spatial_block",
        "x", "y", "numerator_raw", "denominator_raw", "n_obs",
    ]

    for p in cache_files:
        print(f"[kernel pass 2] {p.name}")
        d = _read_component_file(p, columns=cols)
        A_y, B_y, P_y = _build_year_month_rasters(d, grid)

        for bw in bandwidths:
            kx, ky = kernels[bw]

            for m in range(12):
                Ak_y = _smooth2d(A_y[m], kx, ky)
                Bk_y = _smooth2d(B_y[m], kx, ky)

                th = theta[bw][m].astype(np.float64)
                valid = np.isfinite(th) & (local_A[bw][m] > 0)

                sc = np.zeros_like(Ak_y, dtype=np.float64)
                sc[valid] = Bk_y[valid] - th[valid] * Ak_y[valid]
                score2[bw][m][valid] += sc[valid] ** 2

                pk = _smooth2d(P_y[m], kx, ky)
                local_year_clusters[bw][m] += (
                    pk > MIN_LOCAL_KERNEL_WEIGHT
                ).astype(np.uint8)

                del Ak_y, Bk_y, th, valid, sc, pk

        del d, A_y, B_y, P_y
        gc.collect()

    print("\nWriting month-specific kernel results...")
    summary_rows = []
    primary_parts = []

    x_axis = grid["x0"] + np.arange(grid["nx"], dtype=np.float64) * grid["dx"]
    y_axis = grid["y0"] + np.arange(grid["ny"], dtype=np.float64) * grid["dy"]

    for bw in bandwidths:
        bw_dir = output_dir / f"kernel_{int(round(bw)):03d}km"
        bw_dir.mkdir(exist_ok=True)

        kx, ky = kernels[bw]

        for m in range(12):
            target = (
                (totals["target_years"][m] > 0)
                & (totals["pixel_id_map"] >= 0)
                & np.isfinite(theta[bw][m])
                & (local_A[bw][m] > 0)
            )

            iy, ix = np.where(target)
            if len(ix) == 0:
                continue

            th = theta[bw][m][iy, ix].astype(np.float64)
            A_loc = local_A[bw][m][iy, ix].astype(np.float64)
            G = local_year_clusters[bw][m][iy, ix].astype(np.int16)

            meat = score2[bw][m][iy, ix]
            corr = np.where(G > 1, G / (G - 1), np.nan)
            var = corr * meat / (A_loc ** 2)
            se = np.sqrt(np.maximum(var, 0.0))

            df_t = np.maximum(G - 1, 1)
            crit = stats.t.ppf(0.975, df_t)

            effect10 = th * 10.0
            se10 = se * 10.0
            ci_lo = (th - crit * se) * 10.0
            ci_hi = (th + crit * se) * 10.0

            tstat = np.divide(
                th,
                se,
                out=np.full_like(th, np.nan),
                where=np.isfinite(se) & (se > 0),
            )
            pval = 2.0 * stats.t.sf(np.abs(tstat), df_t)

            # Kernel support diagnostics.
            any_pixel = (totals["target_years"][m] > 0).astype(np.float64)
            sumw = _smooth2d(any_pixel, kx, ky)[iy, ix]
            sumw2 = _smooth2d_sqweights(any_pixel, kx, ky)[iy, ix]

            neff_pixels = np.divide(
                sumw ** 2,
                sumw2,
                out=np.zeros_like(sumw),
                where=sumw2 > 0,
            )

            weighted_years_num = _smooth2d(
                totals["target_years"][m].astype(np.float64),
                kx,
                ky,
            )[iy, ix]
            local_mean_years = np.divide(
                weighted_years_num,
                sumw,
                out=np.zeros_like(sumw),
                where=sumw > 0,
            )

            weighted_nobs_num = _smooth2d(
                totals["nobs"][m].astype(np.float64),
                kx,
                ky,
            )[iy, ix]
            local_mean_nobs = np.divide(
                weighted_nobs_num,
                sumw,
                out=np.zeros_like(sumw),
                where=sumw > 0,
            )

            approx_eff_pixel_years = neff_pixels * local_mean_years

            # Month-specific support thresholds.
            finite_A = A_loc[np.isfinite(A_loc) & (A_loc > 0)]
            den_cut = float(np.quantile(finite_A, LOCAL_DENOMINATOR_Q))

            finite_se = se10[np.isfinite(se10)]
            se_cut = float(np.quantile(finite_se, LOCAL_SE_Q))

            recommended = (
                (G >= MIN_LOCAL_YEAR_CLUSTERS)
                & (local_mean_years >= MIN_LOCAL_MEAN_YEARS)
                & (neff_pixels >= MIN_EFFECTIVE_NEIGHBOR_PIXELS)
                & (A_loc >= den_cut)
                & np.isfinite(se10)
                & (se10 <= se_cut)
            )

            pixel_ids = totals["pixel_id_map"][iy, ix]
            block_ids = totals["block_map"][iy, ix]

            out = pd.DataFrame({
                "month": m + 1,
                "bandwidth_km": bw,
                "pixel_id": pixel_ids,
                "spatial_block": block_ids,
                "x": x_axis[ix],
                "y": y_axis[iy],
                "target_n_obs": totals["nobs"][m][iy, ix],
                "target_n_years": totals["target_years"][m][iy, ix],
                "effect_raw": th,
                "effect_per_10ug_m3": effect10,
                "se_per_10ug_m3": se10,
                "ci95_low_per_10ug_m3": ci_lo,
                "ci95_high_per_10ug_m3": ci_hi,
                "p_value_t": pval,
                "t_df": df_t,
                "local_denominator": A_loc,
                "local_year_clusters": G,
                "local_kernel_weight_sum": sumw,
                "effective_neighbor_pixels": neff_pixels,
                "local_mean_years": local_mean_years,
                "local_mean_nobs": local_mean_nobs,
                "approx_effective_pixel_years": approx_eff_pixel_years,
                "recommended_map_mask": recommended,
            })

            month_path = bw_dir / f"month_{m+1:02d}_kernel_dml.parquet"
            out.to_parquet(month_path, index=False)

            valid_map = recommended & np.isfinite(effect10)
            w = A_loc[valid_map]

            if valid_map.any():
                weighted_mean_effect = float(
                    np.sum(effect10[valid_map] * w) / np.sum(w)
                )
                spatial_sd = float(np.std(effect10[valid_map], ddof=1))
                negative_fraction = float(np.mean(effect10[valid_map] < 0))
                median_se = float(np.median(se10[valid_map]))
                median_neff = float(np.median(neff_pixels[valid_map]))
            else:
                weighted_mean_effect = np.nan
                spatial_sd = np.nan
                negative_fraction = np.nan
                median_se = np.nan
                median_neff = np.nan

            pooled_row = national.loc[national["month"] == m + 1].iloc[0]

            summary_rows.append({
                "bandwidth_km": bw,
                "month": m + 1,
                "n_target_pixels": len(out),
                "n_recommended_pixels": int(recommended.sum()),
                "fraction_recommended": float(recommended.mean()),
                "kernel_weighted_mean_effect_per_10ug_m3": weighted_mean_effect,
                "national_pooled_effect_per_10ug_m3": float(
                    pooled_row["effect_per_10ug_m3"]
                ),
                "median_kernel_effect_per_10ug_m3": float(
                    np.nanmedian(effect10[valid_map])
                ) if valid_map.any() else np.nan,
                "spatial_sd_effect_per_10ug_m3": spatial_sd,
                "negative_pixel_fraction": negative_fraction,
                "median_local_se_per_10ug_m3": median_se,
                "median_effective_neighbor_pixels": median_neff,
                "local_denominator_p05_cutoff": den_cut,
                "local_se_p95_cutoff": se_cut,
            })

            if bw == PRIMARY_BANDWIDTH_KM:
                primary_parts.append(out)

            del (
                out, target, iy, ix, th, A_loc, G, meat, corr, var, se,
                df_t, crit, effect10, se10, ci_lo, ci_hi, tstat, pval,
                any_pixel, sumw, sumw2, neff_pixels, weighted_years_num,
                local_mean_years, weighted_nobs_num, local_mean_nobs,
                approx_eff_pixel_years, recommended
            )
            gc.collect()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "bandwidth_month_sensitivity_summary.csv",
        index=False,
    )

    if primary_parts:
        master = pd.concat(primary_parts, ignore_index=True)
        master.to_parquet(
            output_dir / "MONTHLY_SPATIAL_KERNEL_DML_MASTER.parquet",
            index=False,
        )
        master.to_csv(
            output_dir / "MONTHLY_SPATIAL_KERNEL_DML_MASTER.csv.gz",
            index=False,
            compression="gzip",
        )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "aligned_main_version": ALIGNED_MAIN_VERSION,
        "nuisance_alignment": (
            "Same primary year-stratified spatial-blocked Model-A nuisance residualization as v6: "
            "11 calendar-month FE indicators and nominal one-hot LCCS."
        ),
        "method": (
            "year-specific spatial-blocked nuisance DML followed by "
            "month-specific Gaussian spatial-kernel orthogonal DML"
        ),
        "kernel": "separable Gaussian kernel on projected Albers coordinates",
        "kernel_truncate_sigma": KERNEL_TRUNCATE,
        "primary_bandwidth_km": PRIMARY_BANDWIDTH_KM,
        "bandwidths_run_km": bandwidths,
        "effect_formula": (
            "theta_m(s) = sum_i K_h(d_is) * [T_res_i * Y_res_i] / "
            "sum_i K_h(d_is) * [T_res_i^2]"
        ),
        "local_inference": "year-cluster robust kernel score variance",
        "spatial_interpolation": None,
        "effect_learner": None,
        "old_pre_v6_cache_reused": False,
        "primary_map_column": "effect_per_10ug_m3",
        "recommended_map_filter": "recommended_map_mask == True",
        "support_rules": {
            "minimum_local_year_clusters": MIN_LOCAL_YEAR_CLUSTERS,
            "minimum_local_mean_years": MIN_LOCAL_MEAN_YEARS,
            "minimum_effective_neighbor_pixels": MIN_EFFECTIVE_NEIGHBOR_PIXELS,
            "local_denominator_quantile": LOCAL_DENOMINATOR_Q,
            "local_se_quantile": LOCAL_SE_Q,
        },
    }

    (output_dir / "SPATIAL_KERNEL_DML_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nKernel analysis finished.")
    print(output_dir / "MONTHLY_SPATIAL_KERNEL_DML_MASTER.parquet")
    print(output_dir / "bandwidth_month_sensitivity_summary.csv")
    print(output_dir / "calendar_month_pooled_dml_sanity_check.csv")


# =============================================================================
# PRE-SPECIFIED CLIMATE-STATE EFFECT MODIFICATION (NCC SCOPE TEST)
# =============================================================================


def _climate_safe_inverse(a):
    try:
        return np.linalg.inv(a), "inverse"
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a, rcond=1e-12), "pinv"


def _climate_weighted_mean_sd(x, w):
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return np.nan, np.nan
    x = np.asarray(x[good], np.float64)
    w = np.asarray(w[good], np.float64)
    sw = float(w.sum())
    mu = float(np.sum(w * x) / sw)
    var = float(np.sum(w * (x - mu) ** 2) / sw)
    return mu, math.sqrt(max(var, 0.0))


def _climate_t_summary(estimate, variance, df, alpha=0.05):
    if not np.isfinite(variance) or variance <= 0:
        return {
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t": np.nan,
            "p": np.nan,
            "df": int(df),
        }
    se = math.sqrt(float(variance))
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    t = float(estimate / se)
    p = float(2 * stats.t.sf(abs(t), df))
    return {
        "se": se,
        "ci_low": float(estimate - crit * se),
        "ci_high": float(estimate + crit * se),
        "t": t,
        "p": p,
        "df": int(df),
    }


def _climate_cluster_covariance(H_inv, cluster_scores):
    scores = np.asarray(cluster_scores, dtype=np.float64)
    keep = np.any(np.abs(scores) > 0, axis=1)
    scores = scores[keep]
    G = int(len(scores))
    if G < 2:
        raise RuntimeError("Too few non-empty clusters for climate-modifier covariance")
    meat = (scores.T @ scores) * (G / (G - 1.0))
    cov = H_inv @ meat @ H_inv
    return cov, G


def _climate_two_way_covariance(
    H_inv,
    block_scores,
    year_scores,
    intersection_meat_uncorrected,
    n_intersections,
):
    block_keep = np.any(np.abs(block_scores) > 0, axis=1)
    year_keep = np.any(np.abs(year_scores) > 0, axis=1)
    bs = block_scores[block_keep]
    ys = year_scores[year_keep]
    Gs = int(len(bs))
    Gy = int(len(ys))
    Gi = int(n_intersections)
    if min(Gs, Gy, Gi) < 2:
        raise RuntimeError("Too few clusters for climate-modifier two-way covariance")

    M_space = (bs.T @ bs) * (Gs / (Gs - 1.0))
    M_year = (ys.T @ ys) * (Gy / (Gy - 1.0))
    M_inter = intersection_meat_uncorrected * (Gi / (Gi - 1.0))
    meat = M_space + M_year - M_inter
    cov = H_inv @ meat @ H_inv
    eig = np.linalg.eigvalsh((cov + cov.T) / 2.0)
    return cov, {
        "n_spatial_blocks": Gs,
        "n_years": Gy,
        "n_block_year_intersections": Gi,
        "min_cov_eigenvalue": float(np.min(eig)),
        "covariance_psd_with_tolerance": bool(np.min(eig) >= -1e-12),
    }


def _climate_validate_cache(cache_files):
    required = {
        "year", "month", "pixel_id", "spatial_block",
        "numerator_raw", "denominator_raw", "n_obs", "t2m",
    }
    for p in cache_files:
        cols = set(pd.read_parquet(p).columns)
        missing = required - cols
        if missing:
            raise RuntimeError(
                f"{p.name} is missing climate-stage columns {sorted(missing)}. "
                "These are old caches. Set RESUME=False and rerun this FULL program "
                "so the v6 cache is rebuilt with t2m retained."
            )


def _climate_build_climatology(cache_files):
    print("\n[climate] Building same-pixel x same-month t2m climatology...")
    unique_parts = []
    for p in cache_files:
        z = pd.read_parquet(p, columns=["pixel_id", "month"])
        z = z.loc[z["month"].isin(CLIMATE_ACTIVE_MONTHS)]
        unique_parts.append(np.unique(z["pixel_id"].to_numpy(np.int64)))
        del z
    pixel_ids = np.unique(np.concatenate(unique_parts))
    del unique_parts
    gc.collect()

    n_pix = len(pixel_ids)
    temp_sum = np.zeros((n_pix, 12), dtype=np.float64)
    temp_count = np.zeros((n_pix, 12), dtype=np.int16)

    for p in cache_files:
        z = pd.read_parquet(p, columns=["pixel_id", "month", "t2m"])
        z = z.loc[z["month"].isin(CLIMATE_ACTIVE_MONTHS)].copy()
        pid = z["pixel_id"].to_numpy(np.int64)
        idx = np.searchsorted(pixel_ids, pid)
        if not np.array_equal(pixel_ids[idx], pid):
            raise RuntimeError("Climate climatology pixel lookup mismatch")
        m = z["month"].to_numpy(np.int16) - 1
        t = z["t2m"].to_numpy(np.float64)
        np.add.at(temp_sum, (idx, m), t)
        np.add.at(temp_count, (idx, m), 1)
        del z, pid, idx, m, t
        gc.collect()

    clim_mean = np.divide(
        temp_sum,
        temp_count,
        out=np.full_like(temp_sum, np.nan, dtype=np.float64),
        where=temp_count > 0,
    )
    del temp_sum

    support_rows = []
    long_parts = []
    for month in CLIMATE_ACTIVE_MONTHS:
        c = temp_count[:, month - 1]
        valid = c > 0
        support_rows.append({
            "month": month,
            "n_pixels_with_any_year": int(np.sum(valid)),
            "n_pixels_ge_min_climatology_years": int(
                np.sum(c >= CLIMATE_MIN_CLIMATOLOGY_YEARS)
            ),
            "fraction_ge_min_climatology_years": float(
                np.mean(c[valid] >= CLIMATE_MIN_CLIMATOLOGY_YEARS)
            ) if np.any(valid) else np.nan,
            "min_years": int(np.min(c[valid])) if np.any(valid) else 0,
            "median_years": float(np.median(c[valid])) if np.any(valid) else np.nan,
            "max_years": int(np.max(c[valid])) if np.any(valid) else 0,
        })
        keep = c > 0
        long_parts.append(pd.DataFrame({
            "pixel_id": pixel_ids[keep],
            "month": np.full(int(np.sum(keep)), month, dtype=np.int8),
            "t2m_climatology": clim_mean[keep, month - 1].astype(np.float32),
            "n_climatology_years": c[keep].astype(np.int16),
            "meets_min_climatology_years": (
                c[keep] >= CLIMATE_MIN_CLIMATOLOGY_YEARS
            ),
        }))

    pd.DataFrame(support_rows).to_csv(
        CLIMATE_OUTPUT_DIR / "temperature_climatology_support.csv",
        index=False,
    )
    pd.concat(long_parts, ignore_index=True).to_parquet(
        CLIMATE_OUTPUT_DIR / "pixel_month_t2m_climatology.parquet",
        index=False,
    )
    del long_parts
    gc.collect()
    return pixel_ids, clim_mean, temp_count


def _climate_design_matrix(month, temp_anom):
    n = len(month)
    Q = np.zeros((n, len(CLIMATE_ACTIVE_MONTHS) + 1), dtype=np.float64)
    for j, m in enumerate(CLIMATE_ACTIVE_MONTHS):
        Q[:, j] = (month == m)
    Q[:, -1] = temp_anom
    return Q


def _climate_load_arrays(path, pixel_ids, clim_mean, clim_count):
    z = pd.read_parquet(
        path,
        columns=[
            "year", "month", "pixel_id", "spatial_block",
            "numerator_raw", "denominator_raw", "n_obs", "t2m",
        ],
    )
    z = z.loc[z["month"].isin(CLIMATE_ACTIVE_MONTHS)].copy()
    pid = z["pixel_id"].to_numpy(np.int64)
    pidx = np.searchsorted(pixel_ids, pid)
    if not np.array_equal(pixel_ids[pidx], pid):
        raise RuntimeError(f"{path.name}: climate pixel lookup mismatch")

    month = z["month"].to_numpy(np.int16)
    midx = month - 1
    count = clim_count[pidx, midx]
    mu = clim_mean[pidx, midx]
    t = z["t2m"].to_numpy(np.float64)
    anom = t - mu
    A = z["denominator_raw"].to_numpy(np.float64)
    B = z["numerator_raw"].to_numpy(np.float64)
    block = z["spatial_block"].to_numpy(np.int64)
    year = z["year"].to_numpy(np.int16)
    nobs = z["n_obs"].to_numpy(np.int32)

    keep = (
        (count >= CLIMATE_MIN_CLIMATOLOGY_YEARS)
        & np.isfinite(anom)
        & np.isfinite(A)
        & np.isfinite(B)
        & (A > 0)
    )
    out = {
        "year": year[keep],
        "month": month[keep],
        "pixel_id": pid[keep],
        "spatial_block": block[keep],
        "A": A[keep],
        "B": B[keep],
        "t2m_anomaly": anom[keep],
        "n_climatology_years": count[keep],
        "n_obs": nobs[keep],
    }
    del z, pid, pidx, month, midx, count, mu, t, A, B, block, year, nobs, keep
    return out


def _climate_fit_modifier(cache_files, pixel_ids, clim_mean, clim_count):
    p_dim = len(CLIMATE_ACTIVE_MONTHS) + 1
    H = np.zeros((p_dim, p_dim), dtype=np.float64)
    rhs = np.zeros(p_dim, dtype=np.float64)

    month_rows = {m: 0 for m in CLIMATE_ACTIVE_MONTHS}
    month_A = {m: 0.0 for m in CLIMATE_ACTIVE_MONTHS}
    years_seen = set()
    blocks_seen = set()
    pixels_seen = set()
    anomaly_parts = []
    anomaly_weight_parts = []
    climatology_year_parts = []
    total_n_obs = 0

    print("[climate pass 1] orthogonal normal equations...")
    for path in cache_files:
        d = _climate_load_arrays(path, pixel_ids, clim_mean, clim_count)
        if len(d["A"]) == 0:
            continue
        Q = _climate_design_matrix(d["month"], d["t2m_anomaly"])
        A = d["A"]
        B = d["B"]
        H += Q.T @ (A[:, None] * Q)
        rhs += Q.T @ B

        for m in CLIMATE_ACTIVE_MONTHS:
            sel = d["month"] == m
            month_rows[m] += int(np.sum(sel))
            month_A[m] += float(np.sum(A[sel]))

        years_seen.update(np.unique(d["year"]).astype(int).tolist())
        blocks_seen.update(np.unique(d["spatial_block"]).astype(int).tolist())
        pixels_seen.update(np.unique(d["pixel_id"]).astype(int).tolist())
        total_n_obs += int(np.sum(d["n_obs"], dtype=np.int64))
        anomaly_parts.append(d["t2m_anomaly"].astype(np.float32))
        anomaly_weight_parts.append(A.astype(np.float32))
        climatology_year_parts.append(d["n_climatology_years"].astype(np.int16))
        del d, Q, A, B
        gc.collect()

    if not anomaly_parts:
        raise RuntimeError("No supported Apr-Sep rows for climate-state modifier")

    H_inv, inverse_method = _climate_safe_inverse(H)
    beta = H_inv @ rhs
    condition_number = float(np.linalg.cond(H))

    anomalies = np.concatenate(anomaly_parts).astype(np.float64)
    anomaly_weights = np.concatenate(anomaly_weight_parts).astype(np.float64)
    climatology_years = np.concatenate(climatology_year_parts)
    del anomaly_parts, anomaly_weight_parts, climatology_year_parts

    block_ids = np.array(sorted(blocks_seen), dtype=np.int64)
    year_ids = np.array(sorted(years_seen), dtype=np.int16)
    block_scores = np.zeros((len(block_ids), p_dim), dtype=np.float64)
    year_scores = np.zeros((len(year_ids), p_dim), dtype=np.float64)
    inter_meat = np.zeros((p_dim, p_dim), dtype=np.float64)
    n_intersections = 0

    print("[climate pass 2] spatial-block x year clustered scores...")
    for path in cache_files:
        d = _climate_load_arrays(path, pixel_ids, clim_mean, clim_count)
        if len(d["A"]) == 0:
            continue
        Q = _climate_design_matrix(d["month"], d["t2m_anomaly"])
        moment_resid = d["B"] - d["A"] * (Q @ beta)
        score = Q * moment_resid[:, None]

        for y in np.unique(d["year"]):
            yi = int(np.searchsorted(year_ids, y))
            year_scores[yi] += score[d["year"] == y].sum(axis=0)

        bidx = np.searchsorted(block_ids, d["spatial_block"])
        if not np.array_equal(block_ids[bidx], d["spatial_block"]):
            raise RuntimeError("Climate spatial-block lookup mismatch")
        local_block_score = np.zeros((len(block_ids), p_dim), dtype=np.float64)
        np.add.at(local_block_score, bidx, score)
        block_scores += local_block_score
        used = np.any(np.abs(local_block_score) > 0, axis=1)
        g = local_block_score[used]
        inter_meat += g.T @ g
        n_intersections += int(np.sum(used))

        del d, Q, moment_resid, score, bidx, local_block_score, used, g
        gc.collect()

    cov_space, Gs = _climate_cluster_covariance(H_inv, block_scores)
    cov_year, Gy = _climate_cluster_covariance(H_inv, year_scores)
    cov_two, two_info = _climate_two_way_covariance(
        H_inv,
        block_scores,
        year_scores,
        inter_meat,
        n_intersections,
    )
    diag_two = np.diag(cov_two)
    if np.all(np.isfinite(diag_two)) and np.all(diag_two > 0):
        cov_primary = cov_two
        primary_covariance = "two_way_spatial_block_x_year"
        primary_df = max(1, min(two_info["n_spatial_blocks"], two_info["n_years"]) - 1)
    else:
        cov_primary = cov_year
        primary_covariance = "year_cluster_fallback_due_invalid_two_way_diagonal"
        primary_df = max(1, Gy - 1)

    rng = np.random.default_rng(RANDOM_STATE)
    R = int(CLIMATE_MULTIPLIER_BOOTSTRAP_REPS)
    weights = rng.choice(np.array([-1.0, 1.0]), size=(R, len(year_ids)), replace=True)
    delta_beta = (weights @ year_scores) @ H_inv.T
    gamma_delta = delta_beta[:, -1]
    gamma_hat = float(beta[-1])
    boot_p = float((1 + np.sum(np.abs(gamma_delta) >= abs(gamma_hat))) / (R + 1))
    boot_ci = np.quantile(
        gamma_hat + gamma_delta,
        [CLIMATE_ALPHA / 2, 1 - CLIMATE_ALPHA / 2],
    )

    names = [f"month_{m:02d}_baseline" for m in CLIMATE_ACTIVE_MONTHS] + [
        "t2m_anomaly_modifier"
    ]
    rows = []
    for j, name in enumerate(names):
        scale = 10.0
        est = float(beta[j] * scale)
        pri = _climate_t_summary(
            est, cov_primary[j, j] * scale**2, primary_df, CLIMATE_ALPHA
        )
        yr = _climate_t_summary(
            est, cov_year[j, j] * scale**2, max(1, Gy - 1), CLIMATE_ALPHA
        )
        sp = _climate_t_summary(
            est, cov_space[j, j] * scale**2, max(1, Gs - 1), CLIMATE_ALPHA
        )
        row = {
            "parameter": name,
            "estimate_per_10ug_m3": est,
            "unit": (
                "SIF per +10 ug m-3 O3 per +1 K t2m anomaly"
                if j == len(names) - 1
                else "SIF per +10 ug m-3 O3 at t2m anomaly = 0 K"
            ),
            "primary_covariance": primary_covariance,
            "primary_df": primary_df,
            "primary_se": pri["se"],
            "primary_ci95_low": pri["ci_low"],
            "primary_ci95_high": pri["ci_high"],
            "primary_t": pri["t"],
            "primary_p": pri["p"],
            "year_cluster_se": yr["se"],
            "year_cluster_ci95_low": yr["ci_low"],
            "year_cluster_ci95_high": yr["ci_high"],
            "year_cluster_p": yr["p"],
            "spatial_cluster_se": sp["se"],
            "spatial_cluster_ci95_low": sp["ci_low"],
            "spatial_cluster_ci95_high": sp["ci_high"],
            "spatial_cluster_p": sp["p"],
        }
        if j == len(names) - 1:
            row.update({
                "year_multiplier_bootstrap_reps": R,
                "year_multiplier_bootstrap_p": boot_p,
                "year_multiplier_bootstrap_ci95_low": float(boot_ci[0] * 10.0),
                "year_multiplier_bootstrap_ci95_high": float(boot_ci[1] * 10.0),
            })
        rows.append(row)

    parameters = pd.DataFrame(rows)
    parameters.to_csv(
        CLIMATE_OUTPUT_DIR / "climate_modifier_parameters.csv",
        index=False,
    )

    q_probs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    q_vals = np.quantile(anomalies[np.isfinite(anomalies)], q_probs)
    wmean, wsd = _climate_weighted_mean_sd(anomalies, anomaly_weights)
    support_summary = {
        "n_supported_year_month_pixel_rows": int(len(anomalies)),
        "sum_original_n_obs": int(total_n_obs),
        "n_unique_pixels": int(len(pixels_seen)),
        "n_spatial_blocks": int(len(blocks_seen)),
        "n_years": int(len(years_seen)),
        "active_months": list(CLIMATE_ACTIVE_MONTHS),
        "min_climatology_years": int(CLIMATE_MIN_CLIMATOLOGY_YEARS),
        "t2m_anomaly_mean": float(np.mean(anomalies)),
        "t2m_anomaly_sd": float(np.std(anomalies, ddof=1)),
        "t2m_anomaly_A_weighted_mean": wmean,
        "t2m_anomaly_A_weighted_sd": wsd,
        "t2m_anomaly_quantiles": {str(q): float(v) for q, v in zip(q_probs, q_vals)},
        "climatology_years_min": int(np.min(climatology_years)),
        "climatology_years_median": float(np.median(climatology_years)),
        "climatology_years_max": int(np.max(climatology_years)),
        "orthogonal_information_matrix_condition_number": condition_number,
        "inverse_method": inverse_method,
        "two_way_covariance": two_info,
        "primary_covariance": primary_covariance,
    }
    (CLIMATE_OUTPUT_DIR / "climate_modifier_support_summary.json").write_text(
        json.dumps(support_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    month_support_rows = []
    total_A = float(sum(month_A.values()))
    for m in CLIMATE_ACTIVE_MONTHS:
        month_support_rows.append({
            "month": m,
            "n_supported_rows": month_rows[m],
            "sum_treatment_information_A": month_A[m],
            "fraction_of_supported_rows": month_rows[m] / max(len(anomalies), 1),
            "fraction_of_treatment_information": month_A[m] / max(total_A, 1e-300),
        })
    pd.DataFrame(month_support_rows).to_csv(
        CLIMATE_OUTPUT_DIR / "climate_modifier_month_support.csv",
        index=False,
    )

    lo = float(np.quantile(anomalies, CLIMATE_CURVE_Q_LOW))
    hi = float(np.quantile(anomalies, CLIMATE_CURVE_Q_HIGH))
    grid = np.linspace(lo, hi, CLIMATE_CURVE_POINTS)
    curve_rows = []
    equal_month_w = np.full(len(CLIMATE_ACTIVE_MONTHS), 1.0 / len(CLIMATE_ACTIVE_MONTHS))
    for a in grid:
        c = np.zeros(p_dim, dtype=np.float64)
        c[:len(CLIMATE_ACTIVE_MONTHS)] = equal_month_w
        c[-1] = a
        est_raw = float(c @ beta)
        var_raw = float(c @ cov_primary @ c)
        sm = _climate_t_summary(
            est_raw * 10.0, var_raw * 100.0, primary_df, CLIMATE_ALPHA
        )
        curve_rows.append({
            "t2m_anomaly_K": float(a),
            "equal_month_mean_effect_per_10ug_m3": est_raw * 10.0,
            "ci95_low": sm["ci_low"],
            "ci95_high": sm["ci_high"],
            "primary_covariance": primary_covariance,
        })
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(
        CLIMATE_OUTPUT_DIR / "climate_modifier_equal_month_curve.csv",
        index=False,
    )

    month_curve_rows = []
    for m_idx, m in enumerate(CLIMATE_ACTIVE_MONTHS):
        for a in grid:
            c = np.zeros(p_dim, dtype=np.float64)
            c[m_idx] = 1.0
            c[-1] = a
            est_raw = float(c @ beta)
            var_raw = float(c @ cov_primary @ c)
            sm = _climate_t_summary(
                est_raw * 10.0, var_raw * 100.0, primary_df, CLIMATE_ALPHA
            )
            month_curve_rows.append({
                "month": m,
                "t2m_anomaly_K": float(a),
                "effect_per_10ug_m3": est_raw * 10.0,
                "ci95_low": sm["ci_low"],
                "ci95_high": sm["ci_high"],
            })
    pd.DataFrame(month_curve_rows).to_csv(
        CLIMATE_OUTPUT_DIR / "climate_modifier_month_specific_curves.csv",
        index=False,
    )

    gamma_row = parameters.loc[
        parameters["parameter"] == "t2m_anomaly_modifier"
    ].iloc[0]
    if (
        gamma_row["primary_ci95_high"] < 0
        and gamma_row["year_cluster_ci95_high"] < 0
        and boot_p < CLIMATE_ALPHA
    ):
        classification = "clear_negative_temperature_modification"
    elif (
        gamma_row["primary_ci95_low"] > 0
        and gamma_row["year_cluster_ci95_low"] > 0
        and boot_p < CLIMATE_ALPHA
    ):
        classification = "clear_positive_temperature_modification"
    else:
        classification = "no_clear_temperature_modification"

    decision = {
        "classification": classification,
        "pre_specified_primary_parameter": "t2m_anomaly_modifier",
        "estimate_per_10ug_m3_per_1K": float(gamma_row["estimate_per_10ug_m3"]),
        "primary_ci95": [
            float(gamma_row["primary_ci95_low"]),
            float(gamma_row["primary_ci95_high"]),
        ],
        "primary_p": float(gamma_row["primary_p"]),
        "year_cluster_p": float(gamma_row["year_cluster_p"]),
        "year_multiplier_bootstrap_p": boot_p,
        "interpretation_guardrail": (
            "Climate-state modification of an adjusted O3-SIF association; "
            "not proof that warming causally amplifies ozone damage."
        ),
        "analysis_expansion_rule": (
            "Do not scan extra climate variables or alternative month windows based on this result."
        ),
    }
    (CLIMATE_OUTPUT_DIR / "CLIMATE_MODIFIER_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "parameters": parameters,
        "curve": curve,
        "decision": decision,
        "anomalies": anomalies,
        "support": support_summary,
    }


def _climate_plot_panel(result):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[climate plot] skipped: matplotlib unavailable ({exc})")
        return

    curve = result["curve"]
    anomalies = result["anomalies"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.fill_between(
        curve["t2m_anomaly_K"],
        curve["ci95_low"],
        curve["ci95_high"],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        curve["t2m_anomaly_K"],
        curve["equal_month_mean_effect_per_10ug_m3"],
        linewidth=1.8,
    )
    ax.axhline(0.0, linewidth=0.9, linestyle="--")
    ax.axvline(0.0, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Same-pixel, same-month t2m anomaly (K)")
    ax.set_ylabel("Adjusted SIF slope per +10 μg m$^{-3}$ O$_3$")
    ax.tick_params(direction="out")

    if len(anomalies):
        rng = np.random.default_rng(RANDOM_STATE)
        k = min(4000, len(anomalies))
        idx = rng.choice(len(anomalies), size=k, replace=False)
        ymin, ymax = ax.get_ylim()
        rug_y = ymin + 0.015 * (ymax - ymin)
        ax.plot(anomalies[idx], np.full(k, rug_y), "|", alpha=0.08, markersize=3)

    fig.tight_layout()
    fig.savefig(
        CLIMATE_OUTPUT_DIR / "climate_modifier_fig3c.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        CLIMATE_OUTPUT_DIR / "climate_modifier_fig3c.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _climate_state_modifier_stage(cache_files):
    if not RUN_CLIMATE_STATE_MODIFIER:
        print("[climate] RUN_CLIMATE_STATE_MODIFIER=False; skipped")
        return

    CLIMATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _climate_validate_cache(cache_files)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "aligned_main_version": ALIGNED_MAIN_VERSION,
        "analysis": "pre-specified active-season climate-state effect modification",
        "active_months": list(CLIMATE_ACTIVE_MONTHS),
        "modifier": "same-pixel same-calendar-month interannual t2m anomaly",
        "minimum_climatology_years": CLIMATE_MIN_CLIMATOLOGY_YEARS,
        "effect_model": (
            "theta(i,y,m) = month-specific baseline theta_m + gamma * t2m_anomaly(i,y,m)"
        ),
        "primary_parameter": "gamma",
        "primary_inference": "two-way spatial-block x year clustered sandwich",
        "sensitivity": [
            "year-cluster sandwich",
            "spatial-block-cluster sandwich",
            "year-cluster Rademacher multiplier bootstrap",
        ],
        "guardrail": (
            "No VPD/drought/soil-moisture/climate-zone/threshold/month-window scanning."
        ),
    }
    (CLIMATE_OUTPUT_DIR / "CLIMATE_MODIFIER_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pixel_ids, clim_mean, clim_count = _climate_build_climatology(cache_files)
    result = _climate_fit_modifier(
        cache_files,
        pixel_ids,
        clim_mean,
        clim_count,
    )
    _climate_plot_panel(result)

    gamma = result["parameters"].loc[
        result["parameters"]["parameter"] == "t2m_anomaly_modifier"
    ].iloc[0]
    print("\n[climate] pre-specified modifier complete")
    print(
        "  gamma per +10 ug m-3 O3 per +1 K = "
        f"{gamma['estimate_per_10ug_m3']:.8g} "
        f"[{gamma['primary_ci95_low']:.8g}, {gamma['primary_ci95_high']:.8g}], "
        f"p={gamma['primary_p']:.6g}"
    )
    print(f"  decision={result['decision']['classification']}")
    print(f"  output={CLIMATE_OUTPUT_DIR}")


# =============================================================================
# RAW DATA / v6-ALIGNED PRIMARY YEAR-SPECIFIC DML
# =============================================================================

def _clean_gpu():
    """Release Python/CuPy memory between nuisance fits when CuPy is available."""
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _parse_month(s):
    """Same month parsing rule as the v6 main program."""
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(
            s.astype(str).str.extract(r"(\d{1,2})$")[0],
            errors="coerce",
        )
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().any() or not s.between(1, 12).all():
        raise ValueError("month must be 1..12")
    return s.astype(np.int16)


def _add_month_fixed_effects(d):
    """January is reference; add February-December binary indicators."""
    m = d[MONTH].to_numpy(np.int16)
    for mm in range(2, 13):
        d[f"_month_{mm:02d}"] = (m == mm).astype(np.float32)


def _grid_info(d):
    """Same regular-grid definition as the v6 main program."""
    xs = np.sort(d[XCOL].dropna().unique())
    ys = np.sort(d[YCOL].dropna().unique())
    dxs = np.diff(xs)
    dys = np.diff(ys)
    dxs = dxs[dxs > 0]
    dys = dys[dys > 0]
    if len(dxs) == 0 or len(dys) == 0:
        raise ValueError("cannot infer positive grid spacing")
    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    if dx <= 0 or dy <= 0:
        raise ValueError("invalid grid spacing")
    return dx, dy, float(d[XCOL].min()), float(d[YCOL].min())


def _block_id(x, y, grid_tuple, factor):
    """Same spatial-block ID formula as the v6 main program."""
    dx, dy, x0, y0 = grid_tuple
    bx = np.floor((x.astype(np.float64) - x0) / (dx * factor)).astype(np.int32)
    by = np.floor((y.astype(np.float64) - y0) / (dy * factor)).astype(np.int32)
    return bx.astype(np.int64) * 10_000_000 + by.astype(np.int64)


def _pixel_id(x, y, grid_tuple):
    """Same pixel ID formula as the v6 main program."""
    dx, dy, x0, y0 = grid_tuple
    ix = np.rint((x.astype(np.float64) - x0) / dx).astype(np.int32)
    iy = np.rint((y.astype(np.float64) - y0) / dy).astype(np.int32)
    return ix.astype(np.int64) * 10_000_000 + iy.astype(np.int64)


def _format_lccs_level(v):
    """Stable label for a nominal LCCS level."""
    x = float(v)
    if np.isfinite(x) and x.is_integer():
        return str(int(x))
    return (f"{x:g}").replace("-", "m").replace(".", "p")


def _configure_lccs_encoding(d):
    """
    Configure the same deterministic nominal one-hot representation as v6.

    The smallest observed finite code is the omitted reference category only to
    avoid redundant dummy columns. No numerical ordering is imposed.
    """
    global LCCS_LEVELS, LCCS_REFERENCE

    if "lccs" not in d.columns:
        LCCS_LEVELS = tuple()
        LCCS_REFERENCE = None
        return {
            "mode": "absent",
            "levels": [],
            "reference": None,
            "n_dummy_features": 0,
        }

    vals = pd.to_numeric(d["lccs"], errors="coerce")
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
        "note": "Raw LCCS codes are not used as a continuous/ordered predictor.",
    }


def _expanded_control_names(controls):
    """Feature names in the exact order used by _make_design_matrix()."""
    controls = list(controls)
    names = [c for c in controls if c != "lccs"]
    if "lccs" in controls:
        if LCCS_REFERENCE is None or len(LCCS_LEVELS) == 0:
            raise RuntimeError(
                "LCCS requested before nominal categorical encoding was configured"
            )
        names.extend(
            f"_lccs_cat_{_format_lccs_level(lv)}"
            for lv in LCCS_LEVELS
            if float(lv) != float(LCCS_REFERENCE)
        )
    return names


def _make_design_matrix(w, controls):
    """
    Build a dense float32 nuisance-model matrix with v6 nominal LCCS encoding.

    LCCS dummy columns are generated only for the current yearly work table,
    avoiding persistent expansion of the ~83M-row master DataFrame.
    """
    controls = list(controls)
    numeric = [c for c in controls if c != "lccs"]
    feature_names = _expanded_control_names(controls)
    X = np.empty((len(w), len(feature_names)), dtype=np.float32)

    j = 0
    if numeric:
        a = w[numeric].to_numpy(np.float32)
        X[:, :len(numeric)] = a
        j = len(numeric)
        del a

    if "lccs" in controls:
        vals = pd.to_numeric(w["lccs"], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(vals).all():
            raise ValueError(
                "LCCS contains missing/non-finite values after complete-case filtering"
            )
        for lv in LCCS_LEVELS:
            if float(lv) == float(LCCS_REFERENCE):
                continue
            X[:, j] = (vals == float(lv))
            j += 1

    if j != X.shape[1]:
        raise RuntimeError("design-matrix feature count mismatch")

    return X, feature_names


def _lccs_encoding_metadata():
    return {
        "mode": "nominal_one_hot" if LCCS_REFERENCE is not None else "absent",
        "levels": [float(v) for v in LCCS_LEVELS],
        "reference": None if LCCS_REFERENCE is None else float(LCCS_REFERENCE),
        "n_dummy_features": max(0, len(LCCS_LEVELS) - 1),
        "feature_names": (
            [] if LCCS_REFERENCE is None else [
                f"_lccs_cat_{_format_lccs_level(v)}"
                for v in LCCS_LEVELS
                if float(v) != float(LCCS_REFERENCE)
            ]
        ),
        "interpretation": (
            "Nominal categorical adjustment; no numerical ordering of LCCS class codes is assumed."
        ),
    }


def _safe_r2(y_true, y_pred):
    if len(y_true) < 2 or float(np.std(y_true)) == 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


def _load_raw_data():
    """Load only v6 Model-A variables and build exactly aligned panel features."""
    header = pd.read_csv(INPUT_FILE, nrows=0)
    header_cols = set(header.columns)

    # Same robust LCCS alias handling as the v6 main program.
    lccs_source = (
        "lccs" if "lccs" in header_cols
        else ("LCCS" if "LCCS" in header_cols else None)
    )

    req = {OUTCOME, TREATMENT, XCOL, YCOL, YEAR, MONTH}
    missing_req = req - header_cols
    if missing_req:
        raise ValueError(f"Missing required columns: {sorted(missing_req)}")

    model_a_src = []
    for c in MODEL_A:
        if c == "lccs":
            if lccs_source is not None:
                model_a_src.append(lccs_source)
        elif c in header_cols:
            model_a_src.append(c)

    usecols = list(dict.fromkeys(
        [OUTCOME, TREATMENT, XCOL, YCOL, YEAR, MONTH] + model_a_src
    ))

    print(f"Reading raw data ({len(usecols)} columns)...")
    t0 = time.time()
    df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)
    print(f"Rows={len(df):,}; read time={(time.time()-t0)/60:.1f} min")

    if lccs_source == "LCCS" and "lccs" not in df.columns:
        df = df.rename(columns={"LCCS": "lccs"})

    df[YEAR] = pd.to_numeric(df[YEAR], errors="raise").astype(np.int16)
    df[MONTH] = _parse_month(df[MONTH])
    df = df.loc[df[YEAR].between(START_YEAR, END_YEAR)].copy()

    # Keep x/y in float64 until grid/block/pixel IDs are built, matching v6.
    df[XCOL] = pd.to_numeric(df[XCOL], errors="coerce").astype(np.float64)
    df[YCOL] = pd.to_numeric(df[YCOL], errors="coerce").astype(np.float64)
    for c in df.columns:
        if c not in {XCOL, YCOL, YEAR, MONTH}:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    if df[XCOL].isna().any() or df[YCOL].isna().any():
        raise ValueError("x/y contain missing coordinates; cannot construct spatial blocks")

    lccs_info = _configure_lccs_encoding(df)
    if "lccs" in df.columns:
        print(
            "LCCS encoding: nominal one-hot; "
            f"levels={len(LCCS_LEVELS)}, reference={LCCS_REFERENCE}, "
            f"dummy_features={max(0, len(LCCS_LEVELS)-1)}, "
            f"missing={lccs_info.get('missing_count', 0):,}"
        )

    # Same duplicate panel-key check as the main program.
    df["_time_index"] = (
        df[YEAR].astype(np.int32) * 12 + df[MONTH].astype(np.int32)
    )
    if df.duplicated([XCOL, YCOL, "_time_index"]).any():
        raise ValueError("duplicate x/y/month panel keys detected")

    # Replace old harmonic seasonality with 11 calendar-month indicators.
    _add_month_fixed_effects(df)

    grid_tuple = _grid_info(df)
    x64 = df[XCOL].to_numpy(np.float64)
    y64 = df[YCOL].to_numpy(np.float64)

    df["_spatial_block"] = _block_id(
        x64, y64, grid_tuple, SPATIAL_BLOCK_FACTOR
    )
    df["_pixel_id"] = _pixel_id(x64, y64, grid_tuple)

    dx, dy, x0, y0 = grid_tuple
    ix = np.rint((x64 - x0) / dx).astype(np.int32)
    iy = np.rint((y64 - y0) / dy).astype(np.int32)

    grid = {
        "x0": x0,
        "y0": y0,
        "dx": dx,
        "dy": dy,
        "nx": int(ix.max()) + 1,
        "ny": int(iy.max()) + 1,
        "xmax": float(df[XCOL].max()),
        "ymax": float(df[YCOL].max()),
    }

    # Same yearly Model-A control definition as v6.
    yearly_controls = [
        c for c in YEAR_TIME + MODEL_A
        if c in df.columns
    ]
    feature_names = _expanded_control_names(yearly_controls)

    # Main program stores x/y as float32 after IDs are built.
    df[XCOL] = df[XCOL].astype(np.float32)
    df[YCOL] = df[YCOL].astype(np.float32)

    print(f"Yearly Model-A raw controls ({len(yearly_controls)}): {yearly_controls}")
    print(f"Expanded nuisance features ({len(feature_names)}): {feature_names}")
    print(
        f"Grid dx={dx:.3f}, dy={dy:.3f}; "
        f"pixels={df['_pixel_id'].nunique():,}; "
        f"blocks={df['_spatial_block'].nunique():,}"
    )

    return df, grid, yearly_controls, feature_names


def _make_fold_map(blocks):
    """Same globally balanced spatial-fold assignment as v6 balanced_map()."""
    counts = pd.Series(blocks).value_counts(sort=False)
    items = [(int(block), int(n)) for block, n in counts.items()]

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(items)
    items.sort(key=lambda z: z[1], reverse=True)

    totals = np.zeros(N_FOLDS, dtype=np.int64)
    mapping = {}
    for block, n in items:
        f = int(np.argmin(totals))
        mapping[block] = f
        totals[f] += n

    if len(mapping) < N_FOLDS:
        raise ValueError("too few spatial blocks for requested folds")

    print(
        "Spatial fold totals: "
        + ", ".join(f"F{i+1}={int(v):,}" for i, v in enumerate(totals))
    )
    return mapping


def _make_nuisance_model(stage, year, fold):
    """Same global frozen nuisance model used by v6 primary yearly DML."""
    p = dict(HYPERPARAMS[stage])
    p.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": DEVICE,
        "max_bin": MAX_BIN,
        "n_jobs": N_JOBS,
        # v6 spatial mode: SEED + year*10 + spatial_fold_index
        "random_state": RANDOM_STATE + int(year) * 10 + int(fold),
        "verbosity": 0,
        "validate_parameters": True,
    })
    return xgb.XGBRegressor(**p)


def _fit_one_year(work, year, fold_map, controls):
    """Fit one v6-aligned year-specific spatial-blocked nuisance DML."""
    fold_series = work["_spatial_block"].map(fold_map)
    if fold_series.isna().any():
        raise RuntimeError(f"{year}: found spatial blocks missing from global fold map")
    fold = fold_series.to_numpy(np.int8)

    if np.unique(fold).size < N_FOLDS:
        raise RuntimeError(f"{year}: not all spatial folds represented")

    X, feature_names = _make_design_matrix(work, controls)
    t = work[TREATMENT].to_numpy(np.float32)
    y = work[OUTCOME].to_numpy(np.float32)

    t_hat = np.full(len(work), np.nan, np.float32)
    y_hat = np.full(len(work), np.nan, np.float32)
    diag = []

    for f in range(N_FOLDS):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        if len(te) == 0:
            raise RuntimeError(f"{year}: spatial fold {f+1} has no test observations")
        if len(tr) == 0:
            raise RuntimeError(f"{year}: spatial fold {f+1} has no training observations")

        print(
            f"  year_{year}: s{f+1} train={len(tr):,} test={len(te):,}"
        )

        for stage, target, dest in [
            ("treatment", t, t_hat),
            ("outcome", y, y_hat),
        ]:
            md = _make_nuisance_model(stage, year, f)
            md.fit(X[tr], target[tr])
            pred = md.predict(X[te]).astype(np.float32)
            dest[te] = pred

            residual = target[te].astype(np.float64) - pred.astype(np.float64)
            diag.append({
                "year": int(year),
                "stage": stage,
                "spatial_fold": int(f + 1),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "r2": _safe_r2(target[te], pred),
                "rmse": float(np.sqrt(mean_squared_error(target[te], pred))),
                "mae": float(mean_absolute_error(target[te], pred)),
                "mean_residual_bias": float(np.mean(residual)),
                "n_features": int(len(feature_names)),
            })

            del md, pred, residual
            _clean_gpu()

    if not np.isfinite(t_hat).all() or not np.isfinite(y_hat).all():
        raise RuntimeError(f"{year}: incomplete OOF nuisance predictions")

    t_res = (t - t_hat).astype(np.float32)
    y_res = (y - y_hat).astype(np.float32)

    tmp = pd.DataFrame({
        "year": np.full(len(work), year, dtype=np.int16),
        "month": work[MONTH].to_numpy(np.int16),
        "pixel_id": work["_pixel_id"].to_numpy(np.int64),
        "spatial_block": work["_spatial_block"].to_numpy(np.int64),
        "x": work[XCOL].to_numpy(np.float32),
        "y": work[YCOL].to_numpy(np.float32),
        # Keep t2m for the pre-specified climate-state modifier. t2m is already
        # part of the v6 nuisance control set; retaining it here does not change
        # the nuisance fits or the spatial-kernel estimator.
        "t2m": work["t2m"].to_numpy(np.float32),
        "num": t_res.astype(np.float64) * y_res.astype(np.float64),
        "den": t_res.astype(np.float64) ** 2,
    })

    comp = (
        tmp.groupby(
            ["year", "month", "pixel_id", "spatial_block"],
            observed=True,
            sort=False,
        )
        .agg(
            x=("x", "first"),
            y=("y", "first"),
            t2m=("t2m", "mean"),
            numerator_raw=("num", "sum"),
            denominator_raw=("den", "sum"),
            n_obs=("num", "size"),
        )
        .reset_index()
    )

    treatment_diag = {
        "r2": _safe_r2(t, t_hat),
        "rmse": float(np.sqrt(mean_squared_error(t, t_hat))),
        "mae": float(mean_absolute_error(t, t_hat)),
    }
    outcome_diag = {
        "r2": _safe_r2(y, y_hat),
        "rmse": float(np.sqrt(mean_squared_error(y, y_hat))),
        "mae": float(mean_absolute_error(y, y_hat)),
    }

    summary = {
        "year": int(year),
        "n": int(len(work)),
        "n_raw_controls": int(len(controls)),
        "n_expanded_features": int(len(feature_names)),
        "expanded_features": ";".join(feature_names),
        "seasonality": "11 calendar-month FE indicators; January reference",
        "lccs_encoding": "nominal one-hot when present",
        "treatment_oof_r2": treatment_diag["r2"],
        "treatment_oof_rmse": treatment_diag["rmse"],
        "treatment_oof_mae": treatment_diag["mae"],
        "treatment_residual_mean": float(np.mean(t_res)),
        "treatment_residual_sd": float(np.std(t_res, ddof=1)),
        "outcome_oof_r2": outcome_diag["r2"],
        "outcome_oof_rmse": outcome_diag["rmse"],
        "outcome_oof_mae": outcome_diag["mae"],
        "outcome_residual_mean": float(np.mean(y_res)),
        "outcome_residual_sd": float(np.std(y_res, ddof=1)),
    }

    del X, t, y, t_hat, y_hat, t_res, y_res, tmp
    _clean_gpu()

    return comp, pd.DataFrame(diag), summary


def _write_alignment_metadata(yearly_controls, expanded_features, grid):
    meta = {
        "script_version": SCRIPT_VERSION,
        "aligned_main_version": ALIGNED_MAIN_VERSION,
        "input": INPUT_FILE,
        "output": str(OUTPUT_DIR),
        "years": [START_YEAR, END_YEAR],
        "n_spatial_folds": N_FOLDS,
        "reference_block_factor": SPATIAL_BLOCK_FACTOR,
        "approx_reference_block_km_x": float(grid["dx"] * SPATIAL_BLOCK_FACTOR / 1000.0),
        "approx_reference_block_km_y": float(grid["dy"] * SPATIAL_BLOCK_FACTOR / 1000.0),
        "nuisance_design": (
            "year-stratified 5-fold spatial-blocked DML; same primary Model-A residualization as v6 main program"
        ),
        "seasonality": {
            "mode": "calendar_month_fixed_effects",
            "reference_month": 1,
            "dummy_columns": MONTH_FE,
            "harmonic_sin_cos_used": False,
        },
        "model_a_controls_before_lccs_expansion": list(yearly_controls),
        "expanded_nuisance_features": list(expanded_features),
        "lccs_encoding": _lccs_encoding_metadata(),
        "hyperparameters": HYPERPARAMS,
        "kernel": {
            "primary_bandwidth_km": PRIMARY_BANDWIDTH_KM,
            "sensitivity_bandwidths_km": list(SENSITIVITY_BANDWIDTHS_KM),
            "truncate_sigma": KERNEL_TRUNCATE,
        },
        "cache_compatibility": (
            "Only caches produced by this v6-aligned script are compatible. Pre-v6 caches must not be reused."
        ),
        "climate_state_modifier": {
            "enabled": RUN_CLIMATE_STATE_MODIFIER,
            "active_months": list(CLIMATE_ACTIVE_MONTHS),
            "modifier": "same-pixel same-calendar-month interannual t2m anomaly",
            "minimum_climatology_years": CLIMATE_MIN_CLIMATOLOGY_YEARS,
            "cache_retains_t2m": True,
        },
    }
    (OUTPUT_DIR / "V6_ALIGNMENT_METADATA.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _recompute_caches():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df, grid, yearly_controls, expanded_features = _load_raw_data()
    fold_map = _make_fold_map(df["_spatial_block"].to_numpy(np.int64))
    _write_alignment_metadata(yearly_controls, expanded_features, grid)

    diagnostics = []
    year_summaries = []
    t0 = time.time()

    for year in range(START_YEAR, END_YEAR + 1):
        cache_file = CACHE_DIR / f"year_{year}_month_pixel.parquet"
        diag_file = CACHE_DIR / f"year_{year}_nuisance_diagnostics.csv"
        sum_file = CACHE_DIR / f"year_{year}_nuisance_summary.json"

        if (
            RESUME
            and cache_file.exists()
            and diag_file.exists()
            and sum_file.exists()
        ):
            print(f"[resume v6 cache] {year}")
            diagnostics.append(pd.read_csv(diag_file))
            year_summaries.append(
                json.loads(sum_file.read_text(encoding="utf-8"))
            )
            continue

        cols = list(dict.fromkeys(
            [
                OUTCOME, TREATMENT, YEAR, MONTH,
                "_pixel_id", "_spatial_block",
            ] + list(yearly_controls)
        ))

        work = df.loc[df[YEAR] == year, cols].copy()
        complete_case_cols = list(dict.fromkeys(
            [OUTCOME, TREATMENT, MONTH] + list(yearly_controls)
        ))
        work = (
            work.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=complete_case_cols)
            .reset_index(drop=True)
        )

        if len(work) == 0:
            raise RuntimeError(f"{year}: no complete-case observations for Model A")

        print(
            f"[nuisance v6-aligned] {year}: "
            f"n={len(work):,}; elapsed={(time.time()-t0)/60:.1f} min"
        )

        comp, diag, summary = _fit_one_year(
            work=work,
            year=year,
            fold_map=fold_map,
            controls=yearly_controls,
        )

        comp.to_parquet(cache_file, index=False)
        diag.to_csv(diag_file, index=False)
        sum_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        diagnostics.append(diag)
        year_summaries.append(summary)

        print(
            f"  O3 OOF R2={summary['treatment_oof_r2']:.4f}; "
            f"SIF OOF R2={summary['outcome_oof_r2']:.4f}; "
            f"features={summary['n_expanded_features']}"
        )

        del work, comp, diag
        _clean_gpu()

    pd.concat(diagnostics, ignore_index=True).to_csv(
        OUTPUT_DIR / "annual_nuisance_fold_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(year_summaries).sort_values("year").to_csv(
        OUTPUT_DIR / "annual_nuisance_summary.csv",
        index=False,
    )

    del df
    _clean_gpu()
    return grid

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("MONTHLY SPATIAL KERNEL DML + CLIMATE MODIFIER — v6-ALIGNED FULL RECOMPUTATION")
    print("=" * 108)
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"Bandwidths: primary={PRIMARY_BANDWIDTH_KM:.0f} km; "
        f"sensitivity={SENSITIVITY_BANDWIDTHS_KM}"
    )
    print("=" * 108)

    grid = _recompute_caches()

    cache_files = [
        CACHE_DIR / f"year_{year}_month_pixel.parquet"
        for year in range(START_YEAR, END_YEAR + 1)
    ]

    missing = [p for p in cache_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Full recomputation did not create all cache files:\n"
            + "\n".join(str(p) for p in missing)
        )

    _kernel_stage(cache_files, OUTPUT_DIR, grid)
    _climate_state_modifier_stage(cache_files)


if __name__ == "__main__":
    main()
