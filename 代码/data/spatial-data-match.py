
import os
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from rasterio.enums import Resampling
from shapely.geometry import box
from exactextract import exact_extract

# ---------------------------------------------------------------------------
# 0. Working directory
# ---------------------------------------------------------------------------
os.chdir("/path/to/workdir")   # <-- change to your own path

# ---------------------------------------------------------------------------
# 1. Albers projection (Beijing54 Albers Equal Area)
# ---------------------------------------------------------------------------
albers_proj = (
    "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 "
    "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs"
)

# ---------------------------------------------------------------------------
# 2. Load China boundary
# ---------------------------------------------------------------------------
china_boundary = gpd.read_file("中国行政区划.shp")   # <-- your path

# 3. Reproject to Albers
china_albers = china_boundary.to_crs(albers_proj)

# 4. Dissolve all admin units into a single geometry (avoid grid fragmentation)
try:
    china_union = china_albers.union_all()            # geopandas >= 0.13
except AttributeError:
    china_union = china_albers.unary_union            # older geopandas

china_gdf = gpd.GeoDataFrame(geometry=[china_union], crs=albers_proj)

# ---------------------------------------------------------------------------
# 5. Build a 1 km x 1 km grid over China
# ---------------------------------------------------------------------------
def make_grid(bounds, cell_size, crs):
    """Vectorised grid builder. bounds = (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds
    nx = int(np.ceil((maxx - minx) / cell_size))
    ny = int(np.ceil((maxy - miny) / cell_size))
    xs = minx + np.arange(nx) * cell_size
    ys = miny + np.arange(ny) * cell_size
    xx, yy = np.meshgrid(xs, ys)
    xx, yy = xx.ravel(), yy.ravel()
    cells = [box(x, y, x + cell_size, y + cell_size)
             for x, y in zip(xx, yy)]
    return gpd.GeoDataFrame(geometry=cells, crs=crs)

grid = make_grid(china_gdf.total_bounds, 1000.0, albers_proj)
grid["grid_id"] = np.arange(1, len(grid) + 1)

# Clip to China (equivalent to st_intersection in the R script)
grid_china = gpd.clip(grid, china_gdf).reset_index(drop=True)
grid_china["grid_id"] = np.arange(1, len(grid_china) + 1)

# ---------------------------------------------------------------------------
# 6. Helper: reproject → clip → exact_extract
# ---------------------------------------------------------------------------
def reproject_mask_extract(raster_path, grid_gdf, stats,
                           resampling=Resampling.bilinear,
                           target_res=1000):
    """Reproject a raster to Albers, clip to China, run exact_extract."""
    da = rxr.open_rasterio(raster_path, masked=True).squeeze()

    # Reproject
    da = da.rio.reproject(albers_proj,
                          resolution=target_res,
                          resampling=resampling)

    # Clip to China boundary
    da = da.rio.clip([china_union], crs=albers_proj, drop=True)

    # Write a temporary GeoTIFF (exact_extract works best with a path)
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    try:
        da.rio.to_raster(tmp.name)
        out = exact_extract(tmp.name, grid_gdf, stats,
                            include_cols=["grid_id"])
    finally:
        os.unlink(tmp.name)

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 7. Process each year
# ---------------------------------------------------------------------------
years = [2010, 2015, 2020]
all_results = []

for year in years:
    print(f"Processing {year} ...")

    # --- Night light: sum ---
    nl_df = reproject_mask_extract(
        f"night_light_{year}.tif",
        grid_china,
        ["sum"],
        resampling=Resampling.bilinear,
    )

    # --- NPP: mean (kept the original column name npp_sum) ---
    npp_df = reproject_mask_extract(
        f"npp_{year}.tif",
        grid_china,
        ["mean"],
        resampling=Resampling.bilinear,
    )

    # --- Landuse: reclassify 190 -> 1, all else -> 0 ---
    lu_da = rxr.open_rasterio(f"landuse_{year}.tif", masked=True).squeeze()
    lu_da = lu_da.rio.reproject(albers_proj,
                                resolution=1000,
                                resampling=Resampling.nearest)
    lu_da = lu_da.rio.clip([china_union], crs=albers_proj, drop=True)

    lu_bin = (lu_da == 190).astype("float32")
    lu_bin = lu_bin.rio.write_nodata(np.nan)

    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    try:
        lu_bin.rio.to_raster(tmp.name)

        # NOTE: original R code had an operator-precedence bug
        #       sum(values == 1 * coverage_fraction) — the multiplication
        #       was applied to `1`, not the coverage fraction. Fixed here:
        def built_up_area(values, cov):
            return np.nansum((values == 1) * cov)

        def total_area(values, cov):
            return np.nansum(cov)

        built_df = pd.DataFrame(exact_extract(
            tmp.name, grid_china, [built_up_area],
            include_cols=["grid_id"]))
        total_df = pd.DataFrame(exact_extract(
            tmp.name, grid_china, [total_area],
            include_cols=["grid_id"]))
    finally:
        os.unlink(tmp.name)

    # --- Per-year result table ---
    df = pd.DataFrame({
        "grid_id":         nl_df["grid_id"],
        "night_light_sum": nl_df["sum"],
        "npp_sum":         npp_df["mean"],
        "built_up_area":   built_df["built_up_area"],
        "total_area":      total_df["total_area"],
    })
    df["built_up_ratio"] = df["built_up_area"] / df["total_area"]
    df["year"] = year

    all_results.append(
        df[["year", "grid_id", "night_light_sum",
            "npp_sum", "built_up_ratio"]]
    )

# ---------------------------------------------------------------------------
# 8. Combine all years
# ---------------------------------------------------------------------------
combined = pd.concat(all_results, ignore_index=True)

# Attach geometry for GeoPackage export
combined_gdf = grid_china.merge(combined, on="grid_id", how="right")
combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry",
                                crs=albers_proj)

# ---------------------------------------------------------------------------
# 9. Export
# ---------------------------------------------------------------------------
combined.to_csv("spatial_matched_results.csv", index=False)
combined_gdf.to_file("spatial_matched_results.gpkg", driver="GPKG")
grid_china.to_file("china_1km_grid_albers.gpkg", driver="GPKG")

print("Done.")