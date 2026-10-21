"""
Python equivalent of spatiotemporal_match.R

Pipeline:
  1. Use the first SIF monthly GeoTIFF as the 0.05° spatial template.
  2. Process meteorological NC  -> per-layer TIF, aligned to SIF grid.
  3. Process ESA LCCS NC        -> 0.05° TIF using mode resampling.
  4. Process ozone NC           -> 0.05° TIF, named O3_YYYYMM.
  5. Process DEM                -> 0.05° TIF, aligned to SIF grid.
  6. For each SIF monthly file, match meteorology / LCCS / ozone / DEM
     by year-month, reproject to Albers, and export per-pixel CSV.
"""

import os
import re
import glob
import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from rasterio.enums import Resampling

warnings.filterwarnings("ignore")


# ============================================================
# 0. Paths (replaces choose.dir / choose.files)
# ============================================================
PATH_SIF     = r"/path/to/sif"
PATH_METE    = r"/path/to/mete"
PATH_LCCS    = r"/path/to/lccs"
PATH_OZONE   = r"/path/to/ozone"
CHINA_SHAPE  = r"/path/to/china.shp"          # original CRS
CHINA_ALBERS = r"/path/to/china_albers.shp"   # Albers-projected boundary
DEM_FILE     = r"/path/to/dem.tif"
OUT_DIR      = r"/path/to/output"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Utility helpers
# ============================================================
def get_folder_or_file_names(folder_path):
    """Return sub-folder names; if none, return file names."""
    subs = [d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))]
    return subs if subs else os.listdir(folder_path)


def get_crs(ds):
    """Best-effort CRS extraction from a Dataset."""
    try:
        if ds.rio.crs is not None:
            return ds.rio.crs
    except Exception:
        pass
    for v in ds.variables:
        for key in ("crs", "spatial_ref"):
            if key in ds[v].attrs:
                return ds[v].attrs[key]
    return "EPSG:4326"


def ensure_projection(obj, target_crs, resampling=Resampling.nearest):
    """Reproject DataArray / Dataset if its CRS differs from target."""
    if obj.rio.crs is None:
        obj = obj.rio.write_crs(target_crs)
    if obj.rio.crs != target_crs:
        obj = obj.rio.reproject(target_crs, resampling=resampling)
    return obj


def clip_to_extent(obj, shape_gdf):
    """Vector clip (equivalent to crop + mask in terra)."""
    return obj.rio.clip(
        shape_gdf.geometry.values,
        shape_gdf.crs,
        drop=True,
        all_touched=True,
    )


# ============================================================
# 2. Load China boundary + SIF template
# ============================================================
china_shape  = gpd.read_file(CHINA_SHAPE)
china_albers = gpd.read_file(CHINA_ALBERS)

sif_files = sorted(glob.glob(os.path.join(PATH_SIF, "SIF_monthly_mean_*.tif")))
sif_template = rxr.open_rasterio(sif_files[0], masked=True).squeeze()


# ============================================================
# 3. Meteorological NC
# ============================================================
def process_nc_file(nc_file):
    ds  = xr.open_dataset(nc_file, decode_coords="all")
    crs = get_crs(ds)
    layers = []

    for var in ds.data_vars:
        da = ds[var]
        if "time" in da.dims and da.sizes["time"] > 0:
            for t in range(da.sizes["time"]):
                single = da.isel(time=t)
                time_val = pd.to_datetime(da["time"].values[t])
                name = f"{var}{time_val.strftime('%Y-%m')}"
                single = single.rio.write_crs(crs)
                single = ensure_projection(single, china_shape.crs)
                single = clip_to_extent(single, china_shape)
                single = single.rio.reproject_match(sif_template)
                layers.append(single.rename(name))
        else:
            single = da.rio.write_crs(crs)
            single = ensure_projection(single, china_shape.crs)
            single = clip_to_extent(single, china_shape)
            single = single.rio.reproject_match(sif_template)
            layers.append(single.rename(var))

    out_ds = xr.merge(layers, compat="override")
    out_path = os.path.join(
        OUT_DIR, os.path.splitext(os.path.basename(nc_file))[0] + ".tif"
    )
    if os.path.exists(out_path):
        os.remove(out_path)
    out_ds.rio.to_raster(out_path)
    return out_ds


nc_meteo_files = sorted(glob.glob(os.path.join(PATH_METE, "*.nc")))
meteo_datasets = []
for i, f in enumerate(nc_meteo_files, 1):
    print(f"[meteo {i}/{len(nc_meteo_files)}] {os.path.basename(f)}")
    meteo_datasets.append(process_nc_file(f))
meteo_stack = xr.merge(meteo_datasets, compat="override")
print(f"meteo layers: {len(meteo_stack.data_vars)}")


# ============================================================
# 4. ESA LCCS NC (mode-resampled to 0.05°)
# ============================================================
def process_lccs_file(nc_file):
    ds = xr.open_dataset(nc_file, decode_coords="all")
    if "lccs_class" not in ds.data_vars:
        print(f"  skip (no lccs_class): {os.path.basename(nc_file)}")
        return None

    r = ds["lccs_class"]
    if "time" in r.dims and r.sizes["time"] > 0:
        r = r.isel(time=0)

    r = r.rio.write_crs(get_crs(ds))
    r = clip_to_extent(r, china_shape)
    r = ensure_projection(r, china_shape.crs, resampling=Resampling.nearest)
    r = clip_to_extent(r, china_shape)

    # Mode resampling to the 0.05° SIF grid
    r_005 = r.rio.reproject_match(sif_template, resampling=Resampling.mode)

    filename = os.path.splitext(os.path.basename(nc_file))[0]
    m = re.search(r"\d{4}", filename)
    year = m.group(0) if m else "unknown"
    out_name = f"lccs_{year}_005deg"
    r_005 = r_005.rename(out_name)

    out_path = os.path.join(OUT_DIR, out_name + ".tif")
    if os.path.exists(out_path):
        os.remove(out_path)
    r_005.rio.to_raster(out_path, dtype="uint16")
    return r_005


nc_lccs_files = sorted(glob.glob(os.path.join(PATH_LCCS, "LCCS-Map-300m*.nc")))
lccs_datasets = []
for f in nc_lccs_files:
    print(f"[LCCS] {os.path.basename(f)}")
    d = process_lccs_file(f)
    if d is not None:
        lccs_datasets.append(d)
lccs_stack = xr.merge(lccs_datasets, compat="override") if lccs_datasets else None


# ============================================================
# 5. Ozone
# ============================================================
folders = get_folder_or_file_names(PATH_OZONE)
o3_files = []
for x in folders:
    sub = os.path.join(PATH_OZONE, x)
    if os.path.isdir(sub):
        o3_files.extend(glob.glob(os.path.join(sub, "*.nc")))
if not o3_files:
    o3_files = sorted(glob.glob(os.path.join(PATH_OZONE, "*.nc")))


def process_o3_file(f):
    ds  = xr.open_dataset(f, decode_coords="all")
    var = list(ds.data_vars)[0]
    da  = ds[var]
    crs = get_crs(ds)

    def prep(single):
        single = single.rio.write_crs(crs)
        single = ensure_projection(single, china_shape.crs)
        single = clip_to_extent(single, china_shape)
        return single.rio.reproject_match(sif_template)

    if "time" in da.dims and da.sizes["time"] > 1:
        layers = []
        for t in range(da.sizes["time"]):
            time_val = pd.to_datetime(da["time"].values[t])
            single = prep(da.isel(time=t)).rename(
                f"O3_{time_val.strftime('%Y%m')}"
            )
            layers.append(single)
        out = xr.merge(layers, compat="override")
    else:
        if "time" in da.dims:
            da = da.isel(time=0)
        out = prep(da)
        m = re.search(r"\d{6}", os.path.basename(f))
        date_part = m.group(0) if m else "unknown"
        out = out.rename(f"O3_{date_part}")

    out_path = os.path.join(OUT_DIR, f"{list(out.data_vars)[0]}.tif")
    if os.path.exists(out_path):
        os.remove(out_path)
    out.rio.to_raster(out_path)
    return out


o3_datasets = []
for f in o3_files:
    print(f"[O3] {os.path.basename(f)}")
    o3_datasets.append(process_o3_file(f))
o3_stack = xr.merge(o3_datasets, compat="override") if o3_datasets else None


# ============================================================
# 6. DEM
# ============================================================
dem = rxr.open_rasterio(DEM_FILE, masked=True).squeeze()
dem = ensure_projection(dem, china_shape.crs)
dem = clip_to_extent(dem, china_shape)
dem = dem.rio.reproject_match(sif_template).rename("DEM")


# ============================================================
# 7. Monthly matching
# ============================================================
ALL_CSV = os.path.join(OUT_DIR, "matched_data_albers_all.csv")
if os.path.exists(ALL_CSV):
    os.remove(ALL_CSV)


def process_monthly_data(sif_file):
    basename = os.path.basename(sif_file)
    m = re.search(r"_(\d{4})_(\d{2})\.tif", basename)
    if not m:
        print(f"  skip: {basename}")
        return
    year_str, month_str = m.group(1), m.group(2)
    date_val  = f"{year_str}-{month_str}"
    lccs_year = year_str

    # --- SIF ---
    sif = rxr.open_rasterio(sif_file, masked=True).squeeze()
    sif = ensure_projection(sif, china_shape.crs)
    sif = clip_to_extent(sif, china_shape)
    sif = sif.rename(os.path.splitext(basename)[0])

    # --- Match meteorology by "YYYY-MM" in variable name ---
    meteo_vars = [v for v in meteo_stack.data_vars if date_val in v]
    meteo_sel  = meteo_stack[meteo_vars] if meteo_vars else None

    # --- Match LCCS by year ---
    lccs_sel = None
    if lccs_stack is not None:
        lccs_vars = [v for v in lccs_stack.data_vars if lccs_year in v]
        lccs_sel  = lccs_stack[lccs_vars] if lccs_vars else None

    # --- Match ozone by "YYYYMM" ---
    o3_sel = None
    if o3_stack is not None:
        o3_key   = f"{year_str}{month_str}"
        o3_vars  = [v for v in o3_stack.data_vars if o3_key in v]
        o3_sel   = o3_stack[o3_vars] if o3_vars else None

    # --- Merge all layers ---
    parts = [sif]
    if meteo_sel is not None:
        parts.append(meteo_sel)
    if lccs_sel is not None:
        parts.append(lccs_sel)
    if o3_sel is not None:
        parts.append(o3_sel)
    parts.append(dem)

    combined = xr.merge(parts, compat="override")

    # --- Reproject to Albers ---
    combined_albers = combined.rio.reproject(
        china_albers.crs, resampling=Resampling.nearest
    )

    # --- Convert to per-pixel wide table ---
    df = combined_albers.to_dataframe().reset_index()
    df = df.dropna(how="any")
    df["year"]  = year_str
    df["month"] = date_val

    # --- Write full-append CSV ---
    header = not os.path.exists(ALL_CSV)
    df.to_csv(ALL_CSV, mode="a", header=header, index=False)

    # --- Write per-year CSV ---
    per_year = os.path.join(OUT_DIR, f"matched_data_albers_{year_str}.csv")
    header = not os.path.exists(per_year)
    df.to_csv(per_year, mode="a", header=header, index=False)

    print(f"  {date_val}: {len(df)} rows")
    del df, combined, combined_albers
    gc.collect()


for i, sif_file in enumerate(sif_files, 1):
    print(f"[match {i}/{len(sif_files)}] {os.path.basename(sif_file)}")
    process_monthly_data(sif_file)

print("Done.")