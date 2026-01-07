# Repository Structure

```
DB_release/
├── master_dataset.csv          # Main dataset (3,693 rows × 145 columns)
│
├── README.md                    # Main documentation
├── LICENSE                      # CC BY 4.0
├── CITATION.cff                # Citation metadata
│
├── metadata/                    # All metadata files
│   ├── README.md               # Metadata documentation
│   ├── metadata.json           # Complete schema (JSON)
│   ├── metadata.xml            # Complete schema (XML)
│   ├── metadata_radar.xml      # RADAR repository format
│   ├── column_metadata.csv     # Column documentation (tabular)
│   ├── technical_metadata.txt  # Technical specifications
│   ├── ABSTRACT.txt            # Dataset abstract
│   ├── RADAR_DESCRIPTION.txt   # Repository description
│   └── generate_metadata.py    # Metadata generator script
│
├── modules/                     # Analysis utilities
│   ├── README.md
│   ├── closure_validator.py
│   ├── family_normalizer.py
│   ├── feature_distributions.py
│   ├── qa_envelopes.py
│   ├── van_krevelen_clustering.py
│   ├── van_krevelen_plotter.py
│   └── yield_comparator.py
│
└── notebooks/                   # Example notebooks
    ├── load_and_inspect.ipynb
    └── Samples_ingestion_notebooks/
        ├── CSVs/               # Sample data extractions
        ├── PDFs/               # Source publications
        └── *.ipynb             # Data ingestion notebooks
```

## File Descriptions

### Root Level
- **master_dataset.csv**: Primary dataset file with all experimental data
- **README.md**: Main documentation with quick start and usage examples
- **LICENSE**: CC BY 4.0 license terms
- **CITATION.cff**: Machine-readable citation information

### metadata/
Contains all metadata in multiple formats for different use cases:
- JSON/XML for programmatic access
- RADAR XML for repository upload
- TXT files for human reading
- CSV for spreadsheet compatibility

### modules/
Python utilities for data analysis and quality assurance:
- Closure validation
- Van Krevelen plotting
- Feature distributions
- Yield comparisons

### notebooks/
Jupyter notebooks demonstrating:
- Data loading and inspection
- Data ingestion from literature
- Analysis examples
