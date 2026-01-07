# Metadata Files

This folder contains comprehensive metadata for the Biomass HTT/HTL Dataset.

## Files

### Generated Metadata
- **metadata.json** - Complete dataset schema with column documentation
- **metadata.xml** - XML format for interoperability
- **metadata_radar.xml** - RADAR repository-compliant format
- **column_metadata.csv** - Tabular column documentation

### Documentation
- **ABSTRACT.txt** - Dataset abstract
- **RADAR_DESCRIPTION.txt** - Structured description (Abstract, Method, Table of Content, Technical Information, Technical Remarks)
- **technical_metadata.txt** - Detailed technical specifications

### Generator
- **generate_metadata.py** - Script to regenerate metadata files

## Regenerating Metadata

From the repository root:

```bash
.venv/bin/python metadata/generate_metadata.py
```

This will update all metadata files based on the current `master_dataset.csv`.

## Metadata Contents

- Dataset dimensions (rows, columns)
- Column-level documentation (descriptions, units, data types)
- Data completeness statistics
- Column grouping by category
- Contact information
- Keywords and citations
- Year range and temporal coverage
