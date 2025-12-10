# Biomass HTT/HTL Dataset (Version 1.0)

## 1. Overview

This dataset compiles experimental data on hydrothermal treatment and liquefaction (HTT/HTL) of lignocellulosic and lignin-rich feedstocks. It contains **3,386 experimental runs** extracted and curated from peer-reviewed literature published between **1993 and 2025**.

The primary goals are:
- Provide a unified, machine-learning-ready dataset for predicting yields and product properties
- Support hybrid ML–LCA workflows for biomass conversion systems
- Enable systematic comparison and analysis of hydrothermal processing conditions

The dataset covers:
- **Hydrothermal liquefaction (HTL)** – subcritical and supercritical
- **Hydrothermal carbonization (HTC)**
- **Solvothermal liquefaction**
- **Related thermal conversion processes**

## 2. How to cite

If you use this dataset, please cite:

> Seifallah El Fetni et al., "Biomass HTT/HTL Dataset (Version 1.0)", Zenodo, 2025, DOI: [...].

See also `CITATION.cff` in this repository for machine-readable citation metadata.

## 3. File contents

- `master_dataset.csv` – Main dataset, one row per experiment (3,386 rows)
- `metadata.json` – Machine-readable schema and dataset metadata
- `schema_description.pdf` – Human-readable description of all columns
- `CITATION.cff` – Citation metadata for tools (GitHub, Zotero, etc.)
- `LICENSE` – License information (CC BY 4.0)
- `notebooks/load_and_inspect.ipynb` – Example notebook to load and inspect the dataset
- `modules/` – Python utilities for data visualization and QA

## 4. Column groups and units

Unless otherwise noted:

- **Temperatures**: `T_reaction_C` → **°C**  
- **Times**: `t_residence_min`, `t_ramp_min` → **minutes**  
- **Pressures**: `pressure_reaction_MPa` → **MPa**  
- **Elemental composition**: `*_feed_wt_pct`, `*_biooil_wt_pct`, `*_char_wt_pct` → **wt% dry basis**  
- **Moisture**: `Moisture_min_wt_pct_ar`, `Moisture_max_wt_pct_ar` → **wt% as received**  
- **Yields**: `Yield_*_wt_pct` → **wt% of initial dry feedstock**  
- **Energy yields**: `Energy_yield_*_pct` → **% of initial feedstock energy (HHV·mass)**  
- **Carbon yields**: `Carbon_yield_*_pct` → **% of initial feedstock carbon**  
- **Heating values**: `HHV_*_MJ_per_kg` → **MJ/kg (dry basis)**  
- **Van Krevelen ratios**: `O_C_*_molar`, `H_C_*_molar` → **molar ratio (dimensionless)**  

### 4.1 Provenance and identifiers

- `paper_title` – Title of source publication  
- `DOI` – Digital Object Identifier of the source paper  
- `year` – Publication year  
- `Provenance` – Origin of the data point (table, figure, digitized, etc.)  
- `Ref` – Internal reference key  

### 4.2 Feedstock characterization

**Identity:**
- `Feedstock` – Feedstock name as reported in source  
- `Family_std` – Standardized feedstock family (Woody Biomass, Agricultural Residues, etc.)

**Ultimate analysis (wt% dry basis):**
- `C_feed_wt_pct`, `H_feed_wt_pct`, `O_feed_wt_pct`, `N_feed_wt_pct`, `S_feed_wt_pct`, `Ash_feed_wt_pct`

**Structural composition (wt% dry basis):**
- `Lignin_feed_wt_pct`, `Cellulose_feed_wt_pct`, `Hemicellulose_feed_wt_pct`, `Extractives_feed_wt_pct`

**Moisture (wt% as received):**
- `Moisture_min_wt_pct_ar`, `Moisture_max_wt_pct_ar`

**Derived properties:**
- `O_C_feed_molar`, `H_C_feed_molar` – Van Krevelen atomic ratios  
- `HHV_feed_MJ_per_kg` – Higher heating value [MJ/kg dry]  
- `LRI` – Lignin Readiness Index (composite feedstock quality metric)
- `LRI_imputed`, `LRI_imputed_source` – Imputed LRI values and their source

### 4.3 Process conditions

**Process type:**
- `process_type`, `process_subtype` – HTL, HTC, solvolysis, etc.  
- `reactor` – Reactor configuration  
- `atmosphere` – Gas atmosphere (N₂, CO₂, etc.)  
- `solvent_or_medium` – Main solvent/medium used

**Thermal conditions:**
- `T_reaction_C` – Reaction temperature [°C]  
- `t_residence_min` – Isothermal residence time [min]  
- `t_ramp_min` – Heating/ramp time to target temperature [min]  
- `heating_rate_C_per_min` – Heating rate [°C/min]

**Feed preparation:**
- `IC_feed_wt_pct_slurry` – Initial feed solids concentration in slurry [wt%]  
- `water_biomass_ratio_kg_kg` – Water:biomass mass ratio [kg/kg]

**Operating conditions:**
- `pressure_reaction_MPa` – Reaction pressure [MPa]  
- `stirring_rpm` – Stirring speed [rpm]

**Catalyst:**
- `catalyst` – Catalyst name; `"none"` used for blank (non-catalytic) runs  
- `cat_biomass_ratio_kg_kg` – Catalyst:biomass mass ratio [kg/kg]; 0.0 for blank runs

**Post-processing:**
- `yield_basis` – Original yield basis text (dry, daf, ar)  
- `separation_method` – Post-reaction separation method

### 4.4 Product yields and distributions

**Mass yields (wt% of dry feedstock):**
- `Yield_biooil_wt_pct` – Bio-oil/biocrude yield  
- `Yield_char_wt_pct` – Biochar/hydrochar yield  
- `Yield_aqueous_wt_pct` – Aqueous phase yield  
- `Yield_gas_wt_pct` – Gas yield  
- `Yield_gas_water_wt_pct` – Combined gas + water yield

**Energy recovery (% of initial feedstock energy):**
- `Energy_yield_biooil_pct` – Energy in bio-oil  
- `Energy_yield_char_pct` – Energy in char

**Carbon recovery (% of initial feedstock carbon):**
- `Carbon_yield_biooil_pct` – Carbon in bio-oil  
- `Carbon_yield_char_pct` – Carbon in char

### 4.5 Bio-oil properties

**Heating value:**
- `HHV_biooil_MJ_per_kg` – Higher heating value [MJ/kg dry]

**Elemental composition (wt% dry basis):**
- `C_biooil_wt_pct`, `H_biooil_wt_pct`, `O_biooil_wt_pct`, `N_biooil_wt_pct`, `S_biooil_wt_pct`

**Van Krevelen ratios (molar, dimensionless):**
- `O_C_biooil_molar`, `H_C_biooil_molar`

### 4.6 Char properties

**Heating value:**
- `HHV_char_MJ_per_kg` – Higher heating value [MJ/kg dry]

**Elemental composition (wt% dry basis):**
- `C_char_wt_pct`, `H_char_wt_pct`, `O_char_wt_pct`, `N_char_wt_pct`, `S_char_wt_pct`

**Van Krevelen ratios (molar, dimensionless):**
- `O_C_char_molar`, `H_C_char_molar`

### 4.7 Tracking columns (data provenance)

Located at the end of each row:

- `*_method` – Measurement or estimation method (e.g., `C_method`, `HHV_feedstock_method`)
- `*_Note` – Additional notes about the measurement  
- `*_imputed` – Boolean flags for imputed values (e.g., `Lignin_imputed`, `Ash_imputed`)
- `*_source` – Source of imputed values  
- `extra` – JSON field with additional non-standard information from source

## 5. Known limitations

- **Incomplete data**: Not all experiments report all properties. Column completeness varies (see `schema_description.pdf`).
- **Imputed values**: Where values were imputed, this is flagged by `*_imputed` and `*_method` columns.
- **Pressure reporting**: Many studies report "autogenic" without exact values; flagged in `pressure_reaction_MPa`.
- **Yield basis harmonization**: All yields normalized to dry feedstock basis where possible.
- **Critical applications**: Users should consult source papers for precise experimental details.

## 6. Data quality and validation

This dataset includes:
- Systematic quality checks for mass balance closure (C, H, O, N, S, Ash)
- Energy balance validation
- Outlier detection and flagging
- Van Krevelen diagram validation for feedstock and product compositions
- Cross-validation with literature values

Imputation methods used:
- **ML-based**: Random Forest models for Lignin, Ash, HHV (when high R² achieved)
- **Formula-based**: Dulong formula for HHV, stoichiometric calculations for atomic ratios
- **Literature**: Feedstock composition from established databases
- **Alias propagation**: Same feedstock across papers

## 7. Usage examples

### Basic loading and inspection (Python)

```python
import pandas as pd

df = pd.read_csv("master_dataset.csv")
print(f"Loaded {len(df)} experiments")
print(f"Columns: {len(df.columns)}")
print(f"\nProcess types: {df['process_type'].value_counts()}")
print(f"Feedstock families: {df['Family_std'].nunique()} unique families")
```

### Filter HTL experiments above 300°C

```python
htl_high_temp = df[
    (df['process_subtype'].str.contains('HTL', na=False)) & 
    (df['T_reaction_C'] >= 300)
]
print(f"HTL experiments ≥300°C: {len(htl_high_temp)}")
```

### Check data completeness

```python
completeness = (df.notna().sum() / len(df) * 100).sort_values(ascending=False)
print("Top 10 most complete columns:")
print(completeness.head(10))
```

See `notebooks/load_and_inspect.ipynb` for more detailed examples.

## 8. License and reuse

This dataset is released under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

You are free to:
- **Share** – copy and redistribute the material
- **Adapt** – remix, transform, and build upon the material

Under the following terms:
- **Attribution** – You must give appropriate credit, provide a link to the license, and indicate if changes were made

Full license text: https://creativecommons.org/licenses/by/4.0/

## 9. Contact and contributions

**Maintainer:** Seifallah El Fetni  
**Affiliation:** CTC gGmbH  
**Email:** []

For issues, corrections, or contributions:
- Open an issue on GitHub
- Contact the maintainer directly
- Submit a pull request with corrections

## 10. Acknowledgments

This dataset was compiled from peer-reviewed literature. We thank all original authors for their contributions to biomass conversion research.

Data extraction, curation, and validation performed using:
- Python (pandas, scikit-learn)
- SQLite for relational database management
- Custom QA pipelines for validation

## 11. Version history

- **v1.0** (2025-01-XX): Initial public release
  - 3,386 experimental runs
  - 90+ features per experiment
  - Comprehensive quality assurance and validation

## 12. Others (git)
git init
git add .
git commit -m "Initial commit"
git remote add origin url 
git push -u origin master