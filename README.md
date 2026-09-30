# O3-SIF Code

This repository contains the custom Python code used for data harmonization, DML (Double Machine Learning) estimation, spatial heterogeneity analysis, and robustness checks for the manuscript submitted to *Nature Climate Change*.

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

**Note on Processed Data**: The fully processed and merged intermediate dataset (the direct input for the DML models) is approximately 35.6 GB. It is not hosted in this GitHub repository due to its large size. It is available from the corresponding author upon reasonable request for peer review.

## Code Structure
- `dml_reviewer_compact_v6_1_full_data.py`: Main script for DML model training and baseline analysis.
- `o3_sif_spatial_heterogeneity_plus_climate_modifier_v7_full.py`: Script for spatial heterogeneity and climate modifier analysis.
- `LCCS_DML_v3_aligned_with_main.py`: Robustness check using LCCS variables, aligned with the main analysis pipeline.
- `monthly_o3_sif_dml_v2_main_aligned.py`: Robustness check using monthly aggregated data, aligned with the main analysis pipeline.

## License
This project is licensed under the MIT License.
