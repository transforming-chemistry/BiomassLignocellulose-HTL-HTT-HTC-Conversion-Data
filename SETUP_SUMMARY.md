# DB Release Setup Summary

## Directory structure 

```
DB_release/
├── master_dataset.csv          # Main dataset (3,386 rows × 90 columns)
├── metadata.json                # Machine-readable schema
├── README.md                    # Human-readable documentation
├── CITATION.cff                 # Citation metadata
├── LICENSE                      # CC BY 4.0 license
├── sync_dataset.sh              # Script to update from main DB
├── generate_metadata.py         # Script to regenerate metadata.json
├── notebooks/
│   └── load_and_inspect.ipynb   # Example analysis notebook
└── modules/
    ├── README.md                # Module documentation
    ├── van_krevelen_plotter.py  # Van Krevelen diagrams
    ├── family_normalizer.py     # Feedstock family visualization
    └── qa_envelopes.py          # QA and validation tools
```

## Automatic synchronization

The `Data_inspection.ipynb` notebook  includes automatic sync after saving:
- When `df_cleaned` is saved to `DB/Unified_HTT_Biomass_Database.csv`
- It automatically copies to `DB_release/master_dataset.csv`
- Keeps the release dataset in sync with the working dataset

## Manual synchronization
If needed, run:
```bash
cd DB_release
./sync_dataset.sh
```

## Key files explained

### README.md
Comprehensive documentation including:
- Dataset overview (3,386 runs from 1993-2025)
- Column descriptions grouped by category
- Units and measurement conventions
- Known limitations and data quality notes
- Usage examples in Python
- Citation information

### metadata.json
Machine-readable metadata with:
- Dataset statistics (rows, columns, year range)
- Column groups (provenance, feedstock, process, yields, products, tracking)
- Units and data types
- Contact information
- Keywords for discovery

### CITATION.cff
Citation metadata for:
- GitHub auto-citation
- Zotero import
- Other citation tools
Format: Citation File Format v1.2.0

### LICENSE
CC BY 4.0 (Creative Commons Attribution)
- Free to share and adapt
- Requires attribution
- Includes citation instructions

### load_and_inspect.ipynb
Example notebook demonstrating:
- Basic data loading
- Data structure inspection
- Process type and temperature analysis
- Feedstock family distribution
- Data completeness checks
- Temperature and yield distributions
- Van Krevelen diagrams
- Feedstock family circular plots
- QA envelope checks
- Filtering examples
- Numeric feature distributions

## Next steps for release

1. **Review and update contact email** in:
   - `README.md` (section 9)
   - `CITATION.cff` (authors section)
   - `metadata.json` (contact section)

2. **Update ORCID** in `CITATION.cff` if available

3. **Generate schema_description.pdf**:
   - Export `Reports/column_metadata_cleaned.csv` to formatted table
   - Add 1-2 paragraph overview
   - Export as PDF

4. **Test the notebook**:
   ```bash
   cd DB_release/notebooks
   jupyter notebook load_and_inspect.ipynb
   ```

5. ** to prepare for  upload**:
   - Create ZIP archive: `DB_release.zip`
   - Include all files and subdirectories
   - Ready for Zenodo dataset upload

6. **After  upload**:
   - Update DOI in README.md
   - Update DOI in CITATION.cff
   - Update DOI in metadata.json

## File sizes and performance

- `master_dataset.csv`: ~1.5 MB
- Total release package: ~2 MB (with modules and notebook)
- Load time: <1 second for full dataset
- Memory usage: ~50 MB loaded in pandas

## Quality metrics

- **3,386 experimental runs**
- **90 features** per experiment
- **1993-2025** publication years
- **Completeness**:
  - Core provenance: 100%
  - Process conditions: 80-95%
  - Feedstock composition: 70-85%
  - Product properties: 40-60%
  - Tracking columns: document all imputation

## Validation performed

- Mass balance closure (C, H, O, N, S, Ash)
- Energy balance validation
- Van Krevelen consistency checks
- Outlier detection and flagging
- Cross-validation with literature
- ML imputation quality checks (R² > 0.80 for deployment)

## License compliance

- All data extracted from published literature
- CC BY 4.0 allows:
  - Commercial use
  - Modification
  - Distribution
  - Private use
- Requires:
  - Attribution (citation)
  - License and copyright notice
