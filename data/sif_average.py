"""
OCO-2 SIF 年均 / 月均计算脚本（Python 版）
对应 R 脚本 SIF_averge.R
"""

import os
import re
import glob
import argparse
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  启用 .rio 访问器

SIF_VAR = "clear_daily_SIF"
NC_PATTERN = "OCO2.SIF.clear.inst.*.nc"


# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------
def _ensure_geo(da: xr.DataArray) -> xr.DataArray:
    """确保 DataArray 有空间维度和 CRS 信息。"""
    if da.rio.crs is None:
        lat_candidates = ["lat", "latitude", "Latitude", "Lat"]
        lon_candidates = ["lon", "longitude", "Longitude", "Lon"]
        for lat_name in lat_candidates:
            for lon_name in lon_candidates:
                if lat_name in da.dims and lon_name in da.dims:
                    da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name)
                    break
            if da.rio.x_dim is not None:
                break
        da = da.rio.write_crs("EPSG:4326")
    return da


def load_sif(nc_file: str):
    """读取 NetCDF 中的 clear_daily_SIF 波段，负值替换为 NaN。"""
    try:
        ds = xr.open_dataset(nc_file)
    except Exception as e:
        print(f"[WARN] 无法打开 {nc_file}: {e}")
        return None

    if SIF_VAR not in ds.variables:
        print(f"[WARN] {nc_file} 未找到 {SIF_VAR} 变量，跳过")
        ds.close()
        return None

    da = ds[SIF_VAR]

    # 若存在 time 维，取第一个时相（OCO2 SIF 每文件通常单时相）
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)

    # 负值置为 NaN
    da = da.where(da >= 0)

    ds.close()
    return _ensure_geo(da)


def stack_time(arrays):
    """把多个 DataArray 沿 time 维合并。"""
    # 如果已有 time 维冲突，先去掉 time 维
    arrays = [a.squeeze("time", drop=True) if "time" in a.dims else a for a in arrays]
    return xr.concat(
        arrays,
        dim=xr.DataArray(range(len(arrays)), dims="time", name="time"),
    )


def clip_to_boundary(da: xr.DataArray, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    """按边界矢量裁剪 + mask。"""
    return da.rio.clip(boundary.geometry, boundary.crs, drop=True, all_touched=True)


def load_boundary(shp_file: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_file)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


# ---------------------------------------------------------
# 年均值
# ---------------------------------------------------------
def compute_annual(nc_root_dir: str, output_dir: str, boundary: gpd.GeoDataFrame,
                   years=range(2000, 2023)):
    for year in years:
        year_dir = os.path.join(nc_root_dir, str(year))
        if not os.path.isdir(year_dir):
            print(f"[SKIP] 文件夹不存在: {year_dir}")
            continue

        nc_files = sorted(glob.glob(os.path.join(year_dir, NC_PATTERN)))
        if not nc_files:
            print(f"[SKIP] {year} 年没有匹配的 NC 文件")
            continue

        arrays = []
        for f in nc_files:
            da = load_sif(f)
            if da is not None:
                arrays.append(da)

        if not arrays:
            continue

        stack = stack_time(arrays)
        annual_mean = stack.mean(dim="time", skipna=True)
        annual_mean = _ensure_geo(annual_mean)
        annual_mean = clip_to_boundary(annual_mean, boundary)

        out_tif = os.path.join(output_dir, f"SIF_AnnualMean_China_{year}.tif")
        annual_mean.rio.to_raster(out_tif, compress="LZW")
        print(f"[OK] 年度 {year} 平均 SIF 已保存: {out_tif}")


# ---------------------------------------------------------
# 月均值
# ---------------------------------------------------------
def compute_monthly(base_dir: str, output_dir: str, boundary: gpd.GeoDataFrame):
    # 只处理名字以 4 位年份结尾的文件夹
    year_folders = [
        d for d in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(d) and re.search(r"\d{4}$", os.path.basename(d))
    ]
    year_folders.sort()

    for year_folder in year_folders:
        year = int(os.path.basename(year_folder))
        nc_files = sorted(glob.glob(os.path.join(year_folder, NC_PATTERN)))
        if not nc_files:
            continue

        monthly = {m: [] for m in range(1, 13)}

        for f in nc_files:
            # 从文件名中提取 7 位数字：YYYYDDD
            match = re.search(r"(\d{7})", os.path.basename(f))
            if not match:
                continue
            date_str = match.group(1)
            file_year = int(date_str[:4])
            doy = int(date_str[4:7])
            date = datetime(file_year, 1, 1) + timedelta(days=doy - 1)

            da = load_sif(f)
            if da is None:
                continue
            da = clip_to_boundary(da, boundary)
            monthly[date.month].append(da)

        for m in range(1, 13):
            if not monthly[m]:
                continue
            stack = stack_time(monthly[m])
            monthly_mean = stack.mean(dim="time", skipna=True)
            monthly_mean = _ensure_geo(monthly_mean)

            out_tif = os.path.join(output_dir, f"SIF_monthly_mean_{year}_{m:02d}.tif")
            monthly_mean.rio.to_raster(out_tif, compress="LZW")
            print(f"[OK] {year}-{m:02d} 月均 SIF 已保存: {out_tif}")


# ---------------------------------------------------------
# 入口
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OCO-2 SIF 年均 / 月均计算")
    parser.add_argument("--mode", choices=["annual", "monthly", "both"], default="both")
    parser.add_argument("--nc-dir", required=True,
                        help="存放所有年份文件夹的根目录（annual）或 base 目录（monthly）")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--shp", required=True, help="中国边界 Shapefile")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    boundary = load_boundary(args.shp)

    if args.mode in ("annual", "both"):
        print("=== 开始计算年均值 ===")
        compute_annual(args.nc_dir, args.out_dir, boundary)

    if args.mode in ("monthly", "both"):
        print("=== 开始计算月均值 ===")
        compute_monthly(args.nc_dir, args.out_dir, boundary)

    print("所有处理完成！")


if __name__ == "__main__":
    main()