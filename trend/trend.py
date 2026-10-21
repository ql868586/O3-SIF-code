import os
import warnings
import multiprocessing
import concurrent.futures

# 必须在导入 pyplot 之前设置后端，防止 Linux 终端无图形界面报错
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# 1. Core configuration
# ============================================================
csv_path = r"/root/autodl-tmp/wql/matched_data_albers_all1.csv"
output_folder = r"/root/autodl-tmp/wq/0914/month1"

shp_country_path = r"/root/autodl-tmp/map/province/World_GIS_data/country.shp"
shp_grid_path = r"/root/autodl-tmp/map/ChinaLambert/jw.shp"
shp_province_path = r"/root/autodl-tmp/map/province/sheng2022.shp"
shp_boundary_path = r"/root/autodl-tmp/map/ChinaLambert/chinaline.shp"

os.makedirs(output_folder, exist_ok=True)

ALPHA = 0.05
AUTOCORR_ALPHA = 0.05
MIN_VALID_YEAR_FRACTION = 0.70
FIG5_FDR_METHOD = "BY"
NUM_CORES = 16  # 使用 16 核并行计算

FONT_TITLE = 20
FONT_LABEL = 19
FONT_TICK = 18
FONT_TEXT = 16

COLOR_SIF = "#32CD32"
COLOR_O3 = "#1E90FF"
CMAP_SIF_MAP = "Greens"
CMAP_O3_MAP = "Blues"

# 单位修复：统一替换为 mW，year 替换为 yr
UNIT_SIF = r"mW·m$^{-2}$·sr$^{-1}$·nm$^{-1}$"
UNIT_O3 = r"μg·m$^{-3}$"
UNIT_SIF_PER_YEAR = r"mW·m$^{-2}$·sr$^{-1}$·nm$^{-1}$·yr$^{-1}$"
UNIT_O3_PER_YEAR = r"μg·m$^{-3}$·yr$^{-1}$"

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# 2. Statistical helper functions
# ============================================================
def mann_kendall_s_and_var(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return np.nan, np.nan

    s = 0.0
    for k in range(n - 1):
        s += np.sign(y[k + 1:] - y[k]).sum()

    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    return float(s), float(var_s)


def hamed_rao_modified_mk(y_vals, x_vals, alpha=0.05):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) < 3 or len(np.unique(x)) < 3:
        return {
            "n": len(y), "sen_slope": np.nan, "sen_intercept": np.nan,
            "mk_z": np.nan, "mk_p": np.nan, "lag1_acf": np.nan,
            "variance_factor": np.nan, "ols_slope": np.nan, "ols_r2": np.nan,
        }

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = len(y)

    sen = stats.theilslopes(y, x, alpha=1 - alpha)
    sen_slope = float(sen.slope)
    sen_intercept = float(np.median(y - sen_slope * x))

    try:
        ols = stats.linregress(x, y)
        ols_slope = float(ols.slope)
        ols_r2 = float(ols.rvalue ** 2)
    except Exception:
        ols_slope = np.nan
        ols_r2 = np.nan

    s, var_s = mann_kendall_s_and_var(y)
    if not np.isfinite(var_s) or var_s <= 0:
        return {
            "n": n, "sen_slope": sen_slope, "sen_intercept": sen_intercept,
            "mk_z": np.nan, "mk_p": np.nan, "lag1_acf": np.nan,
            "variance_factor": np.nan, "ols_slope": ols_slope, "ols_r2": ols_r2,
        }

    detrended = y - sen_slope * x
    ranks = stats.rankdata(detrended)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    acf_limit = zcrit / np.sqrt(n)
    weighted_sum = 0.0
    lag1_acf = np.nan

    for lag in range(1, n):
        a = ranks[:-lag]
        b = ranks[lag:]
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            rho = 0.0
        else:
            rho = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(rho):
                rho = 0.0

        if lag == 1:
            lag1_acf = rho

        if abs(rho) > acf_limit:
            weighted_sum += ((n - lag) * (n - lag - 1) * (n - lag - 2) * rho)

    variance_factor = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * weighted_sum
    if (not np.isfinite(variance_factor)) or variance_factor <= 0:
        variance_factor = 1.0

    var_s_adj = var_s * variance_factor
    if s > 0:
        z = (s - 1) / np.sqrt(var_s_adj)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s_adj)
    else:
        z = 0.0
    p = 2 * stats.norm.sf(abs(z))

    return {
        "n": n,
        "sen_slope": sen_slope,
        "sen_intercept": sen_intercept,
        "mk_z": float(z),
        "mk_p": float(p),
        "lag1_acf": float(lag1_acf) if np.isfinite(lag1_acf) else np.nan,
        "variance_factor": float(variance_factor),
        "ols_slope": ols_slope,
        "ols_r2": ols_r2,
    }


def fdr_adjust(p_values, method="BY"):
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)

    method = method.upper()
    if method == "BH":
        factor = 1.0
    elif method == "BY":
        factor = np.sum(1.0 / ranks)
    else:
        raise ValueError("method must be 'BH' or 'BY'")

    q_sorted = ranked * m * factor / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_valid = np.empty_like(q_sorted)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    return q


# ============================================================
# 3. Multiprocessing Task Function (Must be at top level)
# ============================================================
def process_pixel_task(task):
    (lon_val, lat_val), sub, min_yrs = task
    sub = sub.sort_values("year")

    valid_sif = sub.dropna(subset=["SIF", "year"])
    valid_o3 = sub.dropna(subset=["O3", "year"])

    res_s = None
    if len(valid_sif) >= min_yrs:
        res_s = hamed_rao_modified_mk(valid_sif["SIF"].values,
                                      valid_sif["year"].values, alpha=AUTOCORR_ALPHA)

    res_o = None
    if len(valid_o3) >= min_yrs:
        res_o = hamed_rao_modified_mk(valid_o3["O3"].values,
                                      valid_o3["year"].values, alpha=AUTOCORR_ALPHA)

    if res_s is not None or res_o is not None:
        return {
            "lon": lon_val,
            "lat": lat_val,
            "n_years_sif": res_s["n"] if res_s is not None else np.nan,
            "slope_sif": res_s["sen_slope"] if res_s is not None else np.nan,
            "p_sif_modified_mk": res_s["mk_p"] if res_s is not None else np.nan,
            "z_sif_modified_mk": res_s["mk_z"] if res_s is not None else np.nan,
            "lag1_sif": res_s["lag1_acf"] if res_s is not None else np.nan,
            "variance_factor_sif": res_s["variance_factor"] if res_s is not None else np.nan,
            "n_years_o3": res_o["n"] if res_o is not None else np.nan,
            "slope_o3": res_o["sen_slope"] if res_o is not None else np.nan,
            "p_o3_modified_mk": res_o["mk_p"] if res_o is not None else np.nan,
            "z_o3_modified_mk": res_o["mk_z"] if res_o is not None else np.nan,
            "lag1_o3": res_o["lag1_acf"] if res_o is not None else np.nan,
            "variance_factor_o3": res_o["variance_factor"] if res_o is not None else np.nan,
        }
    return None


# ============================================================
# 4. Map helpers
# ============================================================
def add_scale_bar_left(ax, length_km=1000, label="1000 km"):
    length_deg = 11.0
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_start = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y_start = ylim[0] + (ylim[1] - ylim[0]) * 0.10

    ax.plot([x_start, x_start + length_deg], [y_start, y_start],
            color="black", linewidth=2, zorder=200)
    tick_h = (ylim[1] - ylim[0]) * 0.015
    ax.plot([x_start, x_start], [y_start, y_start + tick_h],
            color="black", linewidth=1.5, zorder=200)
    ax.plot([x_start + length_deg, x_start + length_deg], [y_start, y_start + tick_h],
            color="black", linewidth=1.5, zorder=200)
    ax.text(x_start + length_deg / 2, y_start + tick_h * 1.5, label,
            ha="center", va="bottom", fontsize=14, zorder=200)


def plot_map_base(ax, data, val_col, cmap, vmin=None, vmax=None, norm=None, gdf_country=None, gdf_prov=None,
                  gdf_bound=None):
    if gdf_country is not None:
        gdf_country.plot(ax=ax, color="#f8f8f8", edgecolor="none", zorder=1)

    plot_data = data.dropna(subset=[val_col, "lon", "lat"])
    scatter_kwargs = {"cmap": cmap, "s": 1, "zorder": 2}
    if norm is not None:
        scatter_kwargs["norm"] = norm
    else:
        if vmin is not None:
            scatter_kwargs["vmin"] = vmin
        if vmax is not None:
            scatter_kwargs["vmax"] = vmax

    sc = ax.scatter(plot_data["lon"], plot_data["lat"], c=plot_data[val_col], **scatter_kwargs)

    if gdf_prov is not None:
        gdf_prov.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=0.2, alpha=0.5, zorder=3)
    if gdf_bound is not None:
        gdf_bound.plot(ax=ax, facecolor="none", edgecolor="black",
                       linewidth=0.8, zorder=4)

    ax.set_xlim(73, 136)
    ax.set_ylim(15, 55)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    add_scale_bar_left(ax, length_km=1000)

    ax_ins = ax.inset_axes([0.75, 0.0, 0.25, 0.28])
    ax_ins.set_xlim(105, 127)
    ax_ins.set_ylim(0, 26)
    if gdf_country is not None:
        gdf_country.plot(ax=ax_ins, color="#f8f8f8", edgecolor="none", zorder=1)

    s_mask = ((plot_data["lon"] > 105) & (plot_data["lon"] < 128) &
              (plot_data["lat"] > 0) & (plot_data["lat"] < 26))
    s_data = plot_data[s_mask]
    if len(s_data) > 0:
        ax_ins.scatter(s_data["lon"], s_data["lat"], c=s_data[val_col], **scatter_kwargs)
    if gdf_bound is not None:
        gdf_bound.plot(ax=ax_ins, facecolor="none", edgecolor="black",
                       linewidth=0.6, zorder=4)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    return sc


# ============================================================
# 5. Main Execution Block
# ============================================================
def main():
    print("正在加载数据...")
    try:
        df_full = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"数据加载失败: {exc}") from exc

    required_cols = {"x", "y", "year", "SIF", "O3"}
    missing = required_cols.difference(df_full.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {sorted(missing)}")

    crs_albers = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    crs_wgs84 = "EPSG:4326"
    transformer = Transformer.from_crs(crs_albers, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(df_full["x"].values, df_full["y"].values)
    df_full["lon"] = lon
    df_full["lat"] = lat

    def load_shp(path):
        try:
            if os.path.exists(path):
                return gpd.read_file(path).to_crs(crs_wgs84)
        except Exception as exc:
            print(f"警告：SHP加载失败 {path}: {exc}")
        return None

    gdf_country = load_shp(shp_country_path)
    gdf_prov = load_shp(shp_province_path)
    gdf_bound = load_shp(shp_boundary_path)

    # --- Prepare Fig. 4 data ---
    print("正在准备 Fig 4 数据...")
    df_full["lon_round"] = df_full["lon"].round(1)
    df_full["lat_round"] = df_full["lat"].round(1)

    spatial_stats = df_full.groupby(["lon", "lat"])[["SIF", "O3"]].agg(["mean", "std"]).reset_index()
    spatial_stats.columns = ["lon", "lat", "SIF_mean", "SIF_std", "O3_mean", "O3_std"]

    # 分别为均值和标准差计算纬向/经向剖面，修复复用bug
    lat_mean = df_full.groupby("lat_round")[["SIF", "O3"]].mean()
    lon_mean = df_full.groupby("lon_round")[["SIF", "O3"]].mean()
    lat_std = df_full.groupby("lat_round")[["SIF", "O3"]].std()
    lon_std = df_full.groupby("lon_round")[["SIF", "O3"]].std()

    # --- Prepare Fig. 5 data (Multiprocessing) ---
    print("正在准备 Fig 5：Sen slope + modified MK + spatial FDR...")
    all_years = np.sort(pd.Series(df_full["year"].dropna().unique()).astype(int).values)
    expected_years = len(all_years)
    min_valid_years = max(8, int(np.ceil(expected_years * MIN_VALID_YEAR_FRACTION)))
    print(f"  全时段年份数 = {expected_years}; Fig 5 每像元至少要求 {min_valid_years} 个有效年份")

    pixel_yearly = df_full.groupby(["lon", "lat", "year"], as_index=False)[["SIF", "O3"]].mean()
    grouped = pixel_yearly.groupby(["lon", "lat"], sort=False)

    tasks = [((lon_val, lat_val), sub, min_valid_years) for (lon_val, lat_val), sub in grouped]
    total_tasks = len(tasks)
    results = []
    step = max(1, total_tasks // 10)
    count = 0

    print(f"  开启多线程并行计算 (使用 {NUM_CORES} 个进程)...")
    # 兼容 Windows/Linux 的进程池上下文
    try:
        ctx = multiprocessing.get_context('fork')
    except ValueError:
        ctx = multiprocessing.get_context('spawn')

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES, mp_context=ctx) as executor:
        futures = {executor.submit(process_pixel_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

            count += 1
            if count % step == 0 or count == total_tasks:
                print(f"  趋势计算进度: {100 * count / total_tasks:.0f}%")

    trend_df = pd.DataFrame(results)
    for var in ["sif", "o3"]:
        p_col = f"p_{var}_modified_mk"
        trend_df[f"q_{var}_bh"] = fdr_adjust(trend_df[p_col].values, method="BH")
        trend_df[f"q_{var}_by"] = fdr_adjust(trend_df[p_col].values, method="BY")
        trend_df[f"sig_{var}_raw_p05"] = trend_df[p_col] < ALPHA
        trend_df[f"sig_{var}_bh_fdr05"] = trend_df[f"q_{var}_bh"] < ALPHA
        trend_df[f"sig_{var}_by_fdr05"] = trend_df[f"q_{var}_by"] < ALPHA

    trend_csv = os.path.join(output_folder, "Fig5_spatial_trend_ModifiedMK_FDR.csv")
    trend_df.to_csv(trend_csv, index=False)

    summary_rows = []
    for var in ["sif", "o3"]:
        p_col = f"p_{var}_modified_mk"
        valid = trend_df[p_col].notna()
        n_tests = int(valid.sum())
        raw_n = int((trend_df.loc[valid, p_col] < ALPHA).sum())
        bh_n = int((trend_df.loc[valid, f"q_{var}_bh"] < ALPHA).sum())
        by_n = int((trend_df.loc[valid, f"q_{var}_by"] < ALPHA).sum())
        summary_rows.append({
            "variable": var.upper(),
            "n_spatial_tests": n_tests,
            "raw_p_lt_0.05_n": raw_n,
            "raw_p_lt_0.05_fraction": raw_n / n_tests if n_tests else np.nan,
            "BH_FDR_q_lt_0.05_n": bh_n,
            "BH_FDR_q_lt_0.05_fraction": bh_n / n_tests if n_tests else np.nan,
            "BY_FDR_q_lt_0.05_n": by_n,
            "BY_FDR_q_lt_0.05_fraction": by_n / n_tests if n_tests else np.nan,
        })

    fdr_summary = pd.DataFrame(summary_rows)
    fdr_summary_path = os.path.join(output_folder, "Fig5_FDR_summary.csv")
    fdr_summary.to_csv(fdr_summary_path, index=False)
    print("\nFig 5 FDR summary:")
    print(fdr_summary.to_string(index=False))

    # --- Plotting ---
    print("正在生成合并总图：只包含 Fig 4 + Fig 5 ...")
    fig = plt.figure(figsize=(28, 28))
    main_gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.65, 1.0], hspace=0.20)

    fig.text(0.015, 0.982, "(a)", fontsize=24, fontweight="bold", va="top", ha="left")

    outer_gs_4 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=main_gs[0],
                                                  wspace=0.18, hspace=0.45)
    config_fig4 = [
        (0, 0, "SIF_mean", "SIF", "(a)", CMAP_SIF_MAP, COLOR_SIF, rf"SIF ({UNIT_SIF})", "mean"),
        (0, 1, "SIF_std", "SIF", "(b)", CMAP_SIF_MAP, COLOR_SIF, rf"SIF ({UNIT_SIF})", "std"),
        (1, 0, "O3_mean", "O3", "(c)", CMAP_O3_MAP, COLOR_O3, rf"O$_3$ ({UNIT_O3})", "mean"),
        (1, 1, "O3_std", "O3", "(d)", CMAP_O3_MAP, COLOR_O3, rf"O$_3$ ({UNIT_O3})", "std"),
    ]

    for r, c, col, var, panel_label, cmap, line_color, cbar_label, stat_type in config_fig4:
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs_4[r, c],
                                                    width_ratios=[1, 3], wspace=0.15)
        ax_map = fig.add_subplot(inner_gs[1])
        valid_data = spatial_stats[col].dropna()
        vmin, vmax = valid_data.min(), valid_data.max()
        sc = plot_map_base(ax_map, spatial_stats, col, cmap, vmin=vmin, vmax=vmax,
                           gdf_country=gdf_country, gdf_prov=gdf_prov, gdf_bound=gdf_bound)
        ax_map.set_title(panel_label, loc="left", fontsize=FONT_TITLE,
                         fontweight="bold", pad=6)

        cax = ax_map.inset_axes([0.1, -0.22, 0.8, 0.04])
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=FONT_TICK)
        cbar.set_label(cbar_label, fontsize=FONT_LABEL, labelpad=8)
        cbar.ax.xaxis.set_label_position("top")

        left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_gs[0], hspace=0.35)
        ax_top = fig.add_subplot(left_gs[0])
        ax_btm = fig.add_subplot(left_gs[1])

        if stat_type == "mean":
            y_lat = lat_mean[var].dropna()
            y_lon = lon_mean[var].dropna()
        else:
            y_lat = lat_std[var].dropna()
            y_lon = lon_std[var].dropna()

        common_min = min(y_lat.min(), y_lon.min()) * 0.95
        common_max = max(y_lat.max(), y_lon.max()) * 1.05

        ax_top.plot(y_lat.index, y_lat.values, color=line_color, lw=2)
        ax_top.fill_between(y_lat.index, common_min, y_lat.values,
                            color=line_color, alpha=0.2)
        ax_top.set_ylim(common_min, common_max)
        ax_top.set_xlabel("Latitude", fontsize=FONT_LABEL)
        ax_top.tick_params(labelsize=FONT_TICK)
        ax_top.grid(True, linestyle=":", alpha=0.5)

        ax_btm.plot(y_lon.index, y_lon.values, color=line_color, lw=2)
        ax_btm.fill_between(y_lon.index, common_min, y_lon.values,
                            color=line_color, alpha=0.2)
        ax_btm.set_ylim(common_min, common_max)
        ax_btm.set_xlabel("Longitude", fontsize=FONT_LABEL)
        ax_btm.tick_params(labelsize=FONT_TICK)
        ax_btm.grid(True, linestyle=":", alpha=0.5)

    fig.text(0.015, 0.395, "(b)", fontsize=24, fontweight="bold", va="top", ha="left")

    outer_gs_5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[1], wspace=0.15)

    colors_sif_trend = ["#FF8C00", "white", "#32CD32"]
    CMAP_SIF_TREND = mcolors.LinearSegmentedColormap.from_list("OrangeWhiteGreen", colors_sif_trend)
    colors_o3_trend = ["#FF8C00", "white", "#1E90FF"]
    CMAP_O3_TREND = mcolors.LinearSegmentedColormap.from_list("OrangeWhiteBlue", colors_o3_trend)

    if FIG5_FDR_METHOD.upper() == "BY":
        q_sif_col, q_o3_col, fdr_label = "q_sif_by", "q_o3_by", "BY-FDR"
    elif FIG5_FDR_METHOD.upper() == "BH":
        q_sif_col, q_o3_col, fdr_label = "q_sif_bh", "q_o3_bh", "BH-FDR"
    else:
        raise ValueError("FIG5_FDR_METHOD must be 'BY' or 'BH'")

    config_fig5 = [
        (0, "slope_sif", q_sif_col, "(a)", CMAP_SIF_TREND, -0.01, 0.01,
         rf"SIF variation ({UNIT_SIF_PER_YEAR})"),
        (1, "slope_o3", q_o3_col, "(b)", CMAP_O3_TREND, -2.0, 2.0,
         rf"Ozone MDA8 concentration variation ({UNIT_O3_PER_YEAR})"),
    ]

    for c, col, q_col, title, cmap, vmin, vmax, cbar_label in config_fig5:
        ax = fig.add_subplot(outer_gs_5[c])
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        sc = plot_map_base(ax, trend_df, col, cmap, norm=norm,
                           gdf_country=gdf_country, gdf_prov=gdf_prov, gdf_bound=gdf_bound)

        sig_points = trend_df.loc[
            trend_df[q_col].notna() & (trend_df[q_col] < ALPHA), ["lon", "lat", q_col]
        ].copy()
        if not sig_points.empty:
            sig_points["lon_grid"] = (sig_points["lon"] * 4).round() / 4
            sig_points["lat_grid"] = (sig_points["lat"] * 4).round() / 4
            sig_grid = sig_points.drop_duplicates(subset=["lon_grid", "lat_grid"])
            ax.scatter(sig_grid["lon"], sig_grid["lat"], marker=".",
                       color="black", s=2.5, alpha=0.55, zorder=2.5)

        ax.scatter([], [], marker=".", color="black", s=80, alpha=0.6,
                   label=f"{fdr_label} q < 0.05")
        ax.legend(loc="lower left", fontsize=16, framealpha=0.9)
        ax.set_title(title, loc="left", fontsize=FONT_TITLE, fontweight="bold", pad=8)

        cax = ax.inset_axes([0.1, -0.15, 0.8, 0.04])
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
        ticks = [vmin, 0, vmax]
        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels([str(t) for t in ticks])
        cbar.ax.tick_params(labelsize=FONT_TICK)
        cbar.set_label(cbar_label, fontsize=FONT_LABEL, labelpad=8)
        cbar.ax.xaxis.set_label_position("top")

    plt.subplots_adjust(top=0.965, bottom=0.055, left=0.03, right=0.99)

    combined_path = os.path.join(output_folder, "Fig4_Fig5_Combined_ab_dot_units.png")
    combined_pdf = os.path.join(output_folder, "Fig4_Fig5_Combined_ab_dot_units.pdf")
    combined_svg = os.path.join(output_folder, "Fig4_Fig5_Combined_ab_dot_units.svg")
    combined_tif = os.path.join(output_folder, "Fig4_Fig5_Combined_ab_dot_units.tif")

    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.savefig(combined_pdf, bbox_inches="tight")
    plt.savefig(combined_svg, bbox_inches="tight")
    plt.savefig(combined_tif, dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"  Combined figure saved: {combined_path}")
    print(f"  Combined figure saved: {combined_pdf}")
    print(f"  Combined figure saved: {combined_svg}")
    print(f"  Combined figure saved: {combined_tif}")
    print(f"  Fig 5 pixel statistics saved: {trend_csv}")
    print(f"  Fig 5 FDR summary saved: {fdr_summary_path}")

    # --- Reviewer-facing run note ---
    review_note = f"""
TREND ANALYSIS REVISION SUMMARY
===============================
1. Fixed the unit inconsistency: changed SIF units from W to mW to align with the main text.
2. Fixed the panel profile bug: standard deviation panels now correctly use std profiles.
3. The final publication composite includes only Fig. 4 and Fig. 5.
4. All units were converted to dot-multiplication style (with 'yr^{-1}').
5. Fig. 5 uses the same Sen + modified-MK framework at each pixel.
6. Pixel-wise spatial significance is corrected for multiple testing. The map
   uses {FIG5_FDR_METHOD.upper()} FDR q < {ALPHA:.2f}. BH and BY q-values are both exported.
7. At least {min_valid_years}/{expected_years} years are required for each pixel trend.
8. Multi-core ({NUM_CORES} workers) parallel processing was applied for trend calculations.
""".strip()

    with open(os.path.join(output_folder, "TREND_REVISION_README_Fig4_Fig5_only.txt"),
              "w", encoding="utf-8") as f:
        f.write(review_note)

    print("\n" + review_note)
    print(f"\n全部生成完成！输出目录: {output_folder}")
    print(f"最终总图: {combined_path}")


if __name__ == "__main__":
    main()