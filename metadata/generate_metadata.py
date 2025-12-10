#!/usr/bin/env python
"""
Generate metadata.json and synchronize with column_metadata.csv for the Biomass HTT/HTL Dataset.
Ensures unified metadata across JSON and CSV formats.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

COLUMN_GROUPS = {
    "provenance": [
        "doi", "study", "year", "author", "journal", "biomass_type",
        "source_category", "process_type", "paper_title", "DOI", "Provenance", "Ref"
    ],
    "feedstock_identity": [
        "Family_std", "Feedstock", "Biomass_class", "Pretreatment"
    ],
    "feedstock_composition": [
        "C_feed_wt_pct", "H_feed_wt_pct", "N_feed_wt_pct", "O_feed_wt_pct", "S_feed_wt_pct",
        "H_C_feed_molar", "O_C_feed_molar", "N_C_feed_molar", "S_C_feed_molar",
        "HHV_feed_MJ_per_kg", "cellulose_wt_pct", "hemicellulose_wt_pct",
        "Lignin_wt_pct", "Ash_feed_wt_pct", "Extractives_wt_pct", "Protein_wt_pct",
        "Lipids_wt_pct", "Volatiles_wt_pct", "Fixed_C_wt_pct", "LRI",
        "Lignin_feed_wt_pct", "Cellulose_feed_wt_pct", "Hemicellulose_feed_wt_pct",
        "Extractives_feed_wt_pct", "Moisture_min_wt_pct_ar", "Moisture_max_wt_pct_ar"
    ],
    "process_conditions": [
        "t_C", "t_min", "solvent", "Biomass_loading_wt_pct", "ramp_C_per_min",
        "catalyst_type", "catalyst_loading_wt_pct", "p_max_MPa", "stirring_RPM",
        "atm_reducing_gas", "H2_pressure_MPa", "process_subtype", "reactor",
        "atmosphere", "solvent_or_medium", "T_reaction_C", "t_residence_min",
        "t_ramp_min", "IC_feed_wt_pct_slurry", "pressure_reaction_MPa",
        "heating_rate_C_per_min", "stirring_rpm", "water_biomass_ratio_kg_kg",
        "catalyst", "cat_biomass_ratio_kg_kg", "yield_basis", "separation_method"
    ],
    "yields": [
        "Yield_biooil_wt_pct", "Yield_solid_wt_pct", "Yield_gas_wt_pct",
        "Yield_aq_wt_pct", "Yield_loss_wt_pct", "Yield_closure_wt_pct",
        "Yield_char_wt_pct", "Yield_aqueous_wt_pct", "Yield_gas_water_wt_pct",
        "Energy_yield_biooil_pct", "Energy_yield_char_pct"
    ],
    "biooil_properties": [
        "C_biooil_wt_pct", "H_biooil_wt_pct", "N_biooil_wt_pct", "O_biooil_wt_pct",
        "S_biooil_wt_pct", "HHV_biooil_MJ_per_kg", "H_C_biooil_molar",
        "O_C_biooil_molar", "N_C_biooil_molar", "S_C_biooil_molar",
        "Carbon_yield_biooil_pct"
    ],
    "char_properties": [
        "C_char_wt_pct", "H_char_wt_pct", "N_char_wt_pct", "O_char_wt_pct",
        "S_char_wt_pct", "HHV_char_MJ_per_kg", "H_C_char_molar", "O_C_char_molar",
        "N_C_char_molar", "S_C_char_molar", "Carbon_yield_char_pct"
    ],
    "tracking": [
        "HC_method", "t_note", "HHV_feedstock_method", "HHV_feedstock_imputed",
        "HHV_biooil_method", "HHV_biochar_method", "S_method", "S_biooil_method",
        "S_biochar_method", "Lignin_method", "Lignin_imputed", "cellulose_method",
        "cellulose_imputed", "hemicellulose_method", "hemicellulose_imputed",
        "Ash_method", "Ash_imputed", "LRI_imputed", "LRI_imputed_source",
        "extra", "C_Note", "C_method", "O/C_Note", "OC_method", "H/C_Note"
    ]
}

def get_column_group(col_name):
    """Determine which group a column belongs to"""
    for group, cols in COLUMN_GROUPS.items():
        if col_name in cols:
            return group
    return "other"

def infer_dtype(series):
    """Infer data type from pandas series dtype"""
    dtype_str = str(series.dtype)
    if 'float' in dtype_str:
        return "float"
    elif 'int' in dtype_str:
        return "integer"
    elif 'bool' in dtype_str:
        return "boolean"
    else:
        return "string"

def generate_metadata():
    """Generate metadata.json and update column_metadata.csv with synchronized content"""
    
    base_path = Path(__file__).parent.parent
    metadata_path = Path(__file__).parent
    
    df = pd.read_csv(base_path / 'master_dataset.csv')
    
    existing_metadata = pd.read_csv(metadata_path / 'column_metadata.csv')
    desc_dict = dict(zip(existing_metadata['column_name'], existing_metadata['description']))
    unit_dict = dict(zip(existing_metadata['column_name'], existing_metadata['unit']))
    
    columns = []
    csv_rows = []
    
    for col_name in df.columns:
        desc = desc_dict.get(col_name, f"Column: {col_name}")
        unit = unit_dict.get(col_name, "")
        completeness = (df[col_name].notna().sum() / len(df) * 100)
        dtype = infer_dtype(df[col_name])
        group = get_column_group(col_name)
        
        col_info = {
            "name": col_name,
            "description": desc,
            "unit": unit if pd.notna(unit) and unit else None,
            "dtype": dtype,
            "group": group,
            "completeness_pct": round(completeness, 2)
        }
        columns.append(col_info)
        
        csv_rows.append({
            "column_name": col_name,
            "description": desc,
            "unit": unit if pd.notna(unit) else "",
            "dtype": dtype,
            "group": group,
            "completeness_pct": round(completeness, 2)
        })
    
    metadata = {
        "title": "Biomass HTT/HTL Dataset",
        "version": "1.0.0",
        "description": "Unified, curated dataset of biomass hydrothermal treatment/liquefaction experiments for lignocellulosic and lignin-rich feedstocks",
        "license": "CC-BY-4.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "year_range": [int(df['year'].min()), int(df['year'].max())],
        "contact": {
            "name": "Seifallah El Fetni",
            "affiliation": "CTC gGmbH",
            "email": "your.email@institution.de"
        },
        "keywords": [
            "biomass", "hydrothermal liquefaction", "hydrothermal treatment",
            "HTL", "HTC", "bio-oil", "biochar", "lignocellulosic biomass",
            "lignin", "machine learning", "LCA"
        ],
        "column_groups": {
            group: [c["name"] for c in columns if c["group"] == group]
            for group in set(c["group"] for c in columns)
        },
        "columns": columns
    }
    
    with open(metadata_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    pd.DataFrame(csv_rows).to_csv(metadata_path / 'column_metadata.csv', index=False)
    
    print(f"✅ Generated metadata.json and updated column_metadata.csv")
    print(f"   {len(columns)} columns documented")
    print(f"   {len(df)} experimental runs")
    print(f"   Year range: {metadata['year_range']}")

if __name__ == "__main__":
    generate_metadata()
