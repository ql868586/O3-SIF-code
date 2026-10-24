# O3-SIF Code

This repository contains the custom Python code used for data harmonization, DML (Double Machine Learning) estimation, spatial heterogeneity analysis, and robustness checks for the manuscript submitted to *Nature Communications*.

## Environment Requirements
- Python 3.12
- Required dependencies are listed in `requirements.txt`. You can install them using:
  `pip install -r requirements.txt`

## Data Sources
The raw datasets used in this study are publicly available from the following sources:

- **Surface O₃ Data**: ChinaHighO₃ dataset, available from the National Tibetan Plateau Data Center at https://data.tpdc.ac.cn/zh-hans/data/87753867-77c8-42f1-b2e6-da569679635f
- **SIF Data**: CSIF dataset, available from the National Tibetan Plateau Data Center at https://data.tpdc.ac.cn/zh-hans/data/d7cccf31-9bb5-4356-88a7-38c5458f052b/
- **Meteorological Data**: ERA5-Land monthly means, available from the Copernicus Climate Data Store at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview
- **Land Cover Data**: ESA CCI Land Cover, available from the ESA CCI Land Cover archive at https://maps.elie.ucl.ac.be/CCI/viewer/download.php

**Note on Processed Data**:The fully processed and merged intermediate dataset (the direct input for the DML models) is approximately 35.6 GB. Due to its large size, it is not hosted in this GitHub repository. It has been deposited in the Zenodo repository and is publicly available at https://doi.org/10.5281/zenodo.23520196.

## Code Structure

The repository is organized into five modules that follow the analysis workflow: data harmonization, double machine learning (DML) estimation, hyperparameter tuning, heterogeneity analysis, and trend analysis.

### Data and Preprocessing

- `data/` : Root directory for the raw and harmonized datasets (NetCDF, GeoTIFF and CSV); large data files are excluded from version control. **The fully processed intermediate dataset is available in the Zenodo repository.**
* **`data/sif_average.py`** : Computes the annual and monthly average of the OCO-2 SIF dataset (`clear_daily_SIF`). It masks out negative values (missing data), clips the raster to the China boundary (provided as a shapefile), and exports the aggregated results as GeoTIFF files.
- `data/spatiotemporal_match.py` : Aligns SIF, meteorological, LCCS, ozone and DEM layers to a common 0.05° grid by year–month and exports the analysis-ready pixel-level CSV.

### Model Training (Double Machine Learning)

- `dml/` : Double machine learning module for the main O3–SIF estimation.
- `dml/dml.py` : Core DML script estimating the O3 effect on SIF with Model A/B/C control sets, five-fold spatial and temporal cross-fitting, cluster-robust inference, wild bootstrap, and the EXP09 specification-calibration experiment.

### Hyperparameter Tuning

- `Hyperparameter/` : Nested tuning of the XGBoost nuisance models.
- `Hyperparameter/o3_sif_nested_blocked_hyperopt.py` : Performs nested, spatially blocked hyperparameter optimization for the treatment and outcome models and exports the frozen parameter set used by all DML runs.

### Heterogeneity Analysis

- `month/` : Seasonal (calendar-month) heterogeneity analysis.
- `month/o3_sif_monthly_heterogeneity_dml.py` : Month-specific DML estimation of the O3–SIF relationship, with month-by-year heatmaps, pooled forest plots and formal heterogeneity tests.
- `lccs/` : Land-cover heterogeneity analysis.
- `lccs/o3_sif_lccs_heterogeneity_dml.py` : Land-cover (LCCS)-stratified DML estimation comparing the O3–SIF relationship across vegetation classes, including threshold sensitivity and group-level forest plots.
- `spatial heterogeneity/` : Spatially explicit heterogeneity analysis.
- `spatial heterogeneity/o3_sif_spatial_heterogeneity_dml.py` : Pixel-resolved DML analysis with an SSRD radiation modifier, DuckDB-backed caching and bootstrap inference to characterize spatial heterogeneity in the O3–SIF relationship.

### Trend Analysis and Visualization

- `trend/` : Temporal trend analysis and figure generation.
- `trend/trend.py` : Computes pixel-level Sen's slope and modified Mann–Kendall trends for SIF and O3 with BH/BY FDR correction, and renders the manuscript maps (Figures 4–5).

## License
This project is licensed under the MIT License.
