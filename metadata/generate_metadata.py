#!/usr/bin/env python
"""Generate metadata files for Biomass HTT/HTL Dataset."""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

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
    for group, cols in COLUMN_GROUPS.items():
        if col_name in cols:
            return group
    return "other"

def infer_dtype(series):
    dtype_str = str(series.dtype)
    if 'float' in dtype_str:
        return "float"
    elif 'int' in dtype_str:
        return "integer"
    elif 'bool' in dtype_str:
        return "boolean"
    else:
        return "string"

def escape_xml_text(text):
    if text is None:
        return None
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text

def generate_radar_xml_metadata(metadata):
    description_sections = []
    
    description_sections.append("ABSTRACT:")
    description_sections.append("This dataset provides a unified, curated collection of experimental results on the hydrothermal conversion of lignocellulosic and lignin-rich biomass, including hydrothermal treatment (HTT), hydrothermal liquefaction (HTL), and related process variants. The database covers diverse biomass feedstocks, reactor configurations, solvents, catalysts, temperatures, residence times, and operating conditions, with all records normalized to consistent units and yield bases.")
    description_sections.append("")
    
    description_sections.append("METHOD:")
    description_sections.append("Data were systematically extracted from peer-reviewed literature, digitized when necessary, and harmonized to a standardized schema. All experimental records include comprehensive provenance tracking. Elemental compositions were normalized to dry or dry-ash-free basis. Missing values for key descriptors (e.g., HHV, lignocellulosic composition, elemental ratios) were imputed using Random Forest models or family-specific median values. Quality-control flags document closure checks for elemental and polymer balances. Derived descriptors include O/C and H/C molar ratios, Lignin Readiness Index (LRI), and energy/carbon recovery metrics.")
    description_sections.append("")
    
    description_sections.append("TABLE OF CONTENT:")
    description_sections.append("The dataset comprises one primary CSV file (master_dataset.csv) with 3,693 rows (experimental runs) and 145 columns organized into nine groups:")
    description_sections.append("- Provenance: source publication metadata (DOI, year, author, reference)")
    description_sections.append("- Feedstock Identity: biomass type and family classification")
    description_sections.append("- Feedstock Composition: elemental analysis (C, H, O, N, S, Ash), lignocellulosic components (Lignin, Cellulose, Hemicellulose), HHV, elemental ratios")
    description_sections.append("- Process Conditions: temperature (typically 200-400°C), residence time, pressure, solvent, catalyst, reactor type, heating rate, biomass loading")
    description_sections.append("- Yields: bio-oil, char, gas, and aqueous phase yields (wt%)")
    description_sections.append("- Bio-oil Properties: elemental composition, HHV, O/C and H/C ratios, carbon yield")
    description_sections.append("- Char Properties: elemental composition, HHV, O/C and H/C ratios, carbon yield")
    description_sections.append("- Tracking: flags and methods documenting imputation sources, measurement methods, and quality-control notes")
    description_sections.append("- Other: derived descriptors, closure flags, normalized indices")
    description_sections.append("")
    
    description_sections.append("TECHNICAL INFORMATION:")
    description_sections.append("Format: CSV (UTF-8 encoding)")
    description_sections.append("Size: 3,693 rows × 145 columns")
    description_sections.append("Temporal Coverage: 1982-2026 (40+ years of published research)")
    description_sections.append("Temperature Range: 200-400°C (typical hydrothermal conditions)")
    description_sections.append("Feedstock Families: 15+ standardized categories (e.g., Softwood, Hardwood, Agricultural Residue, Herbaceous, Kraft Lignin)")
    description_sections.append("Data Completeness: Core features (C, O, temperature, time) >98%; product yields 70-80%; detailed product composition 20-40%")
    description_sections.append("Schema: Standardized column names, units, and yield bases; metadata provided in JSON, XML, and CSV formats")
    description_sections.append("")
    
    description_sections.append("TECHNICAL REMARKS:")
    description_sections.append("- All yields normalized to dry or dry-ash-free feedstock basis as specified in yield_basis column")
    description_sections.append("- Oxygen content calculated by difference when not directly measured (documented in O_method column)")
    description_sections.append("- HHV estimated using Channiwala-Parikh correlation when not experimentally measured (flagged in HHV_feedstock_method)")
    description_sections.append("- Lignocellulosic composition imputed for ~4% of records using ML models trained on family-specific patterns")
    description_sections.append("- Closure flags (LCH_closure_flag) indicate quality of lignocellulosic balance (sum of Lignin+Cellulose+Hemicellulose)")
    description_sections.append("- Some studies report combined gas+water yields; separated when possible, otherwise flagged")
    description_sections.append("- Catalyst = 'none' explicitly indicates non-catalytic runs")
    description_sections.append("- FAIR-compliant: machine-readable metadata, standardized schemas, comprehensive provenance, CC-BY-4.0 license")
    
    radar_description = "\\n".join(description_sections)
    
    xml_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>']
    xml_lines.append('<ns2:radarDataset xmlns="http://radar-service.eu/schemas/descriptive/radar/v09/radar-elements" xmlns:ns2="http://radar-service.eu/schemas/descriptive/radar/v09/radar-dataset">')
    xml_lines.append('    <identifier identifierType="DOI">10.22000/XXXXX</identifier>')
    xml_lines.append('    <creators>')
    xml_lines.append('        <creator>')
    xml_lines.append('            <creatorName>Elfetni, Seifallah</creatorName>')
    xml_lines.append('            <givenName>Seifallah</givenName>')
    xml_lines.append('            <familyName>Elfetni</familyName>')
    xml_lines.append('            <nameIdentifier schemeURI="http://orcid.org/" nameIdentifierScheme="ORCID">https://orcid.org/0000-0003-3615-627X</nameIdentifier>')
    xml_lines.append(f'            <creatorAffiliation>{escape_xml_text(metadata["contact"]["affiliation"])}</creatorAffiliation>')
    xml_lines.append('        </creator>')
    xml_lines.append('    </creators>')
    xml_lines.append(f'    <title>{escape_xml_text(metadata["title"])}</title>')
    xml_lines.append('    <publishers>')
    xml_lines.append('        <publisher nameIdentifierScheme="Other">Elfetni, Seifallah</publisher>')
    xml_lines.append('    </publishers>')
    xml_lines.append(f'    <productionYear>{datetime.now().year}</productionYear>')
    xml_lines.append('    <subjectAreas>')
    xml_lines.append('        <subjectArea>')
    xml_lines.append('            <controlledSubjectAreaName>Chemistry</controlledSubjectAreaName>')
    xml_lines.append('        </subjectArea>')
    xml_lines.append('    </subjectAreas>')
    xml_lines.append('    <resource resourceType="Collection">Curated database from the scientific literature</resource>')
    xml_lines.append(f'    <descriptions>')
    xml_lines.append(f'        <description descriptionType="Abstract">{escape_xml_text(radar_description)}</description>')
    xml_lines.append(f'    </descriptions>')
    xml_lines.append('    <geoLocations>')
    xml_lines.append('        <geoLocation>')
    xml_lines.append('            <geoLocationCountry>GERMANY</geoLocationCountry>')
    xml_lines.append('        </geoLocation>')
    xml_lines.append('    </geoLocations>')
    xml_lines.append('    <rights>')
    xml_lines.append('        <controlledRights>CC BY 4.0 Attribution</controlledRights>')
    xml_lines.append('    </rights>')
    xml_lines.append('    <rightsHolders>')
    xml_lines.append(f'        <rightsHolder nameIdentifierScheme="Other">{escape_xml_text(metadata["contact"]["affiliation"])}</rightsHolder>')
    xml_lines.append('    </rightsHolders>')
    xml_lines.append('</ns2:radarDataset>')
    
    return '\n'.join(xml_lines)

def generate_xml_metadata(metadata):
    root = ET.Element("dataset_metadata")
    
    ET.SubElement(root, "title").text = escape_xml_text(metadata["title"])
    ET.SubElement(root, "version").text = escape_xml_text(metadata["version"])
    ET.SubElement(root, "description").text = escape_xml_text(metadata["description"])
    ET.SubElement(root, "license").text = escape_xml_text(metadata["license"])
    ET.SubElement(root, "created").text = escape_xml_text(metadata["created"])
    ET.SubElement(root, "n_rows").text = str(metadata["n_rows"])
    ET.SubElement(root, "n_columns").text = str(metadata["n_columns"])
    
    year_range_elem = ET.SubElement(root, "year_range")
    ET.SubElement(year_range_elem, "min").text = str(metadata["year_range"][0])
    ET.SubElement(year_range_elem, "max").text = str(metadata["year_range"][1])
    
    contact_elem = ET.SubElement(root, "contact")
    ET.SubElement(contact_elem, "name").text = escape_xml_text(metadata["contact"]["name"])
    ET.SubElement(contact_elem, "affiliation").text = escape_xml_text(metadata["contact"]["affiliation"])
    ET.SubElement(contact_elem, "email").text = escape_xml_text(metadata["contact"]["email"])
    
    keywords_elem = ET.SubElement(root, "keywords")
    for keyword in metadata["keywords"]:
        ET.SubElement(keywords_elem, "keyword").text = escape_xml_text(keyword)
    
    column_groups_elem = ET.SubElement(root, "column_groups")
    for group_name, columns in metadata["column_groups"].items():
        group_elem = ET.SubElement(column_groups_elem, "group", name=escape_xml_text(group_name))
        for col in columns:
            ET.SubElement(group_elem, "column").text = escape_xml_text(col)
    
    columns_elem = ET.SubElement(root, "columns")
    for col in metadata["columns"]:
        col_elem = ET.SubElement(columns_elem, "column")
        ET.SubElement(col_elem, "name").text = escape_xml_text(col["name"])
        
        desc_elem = ET.SubElement(col_elem, "description")
        if col["description"] is not None:
            desc_elem.text = escape_xml_text(col["description"])
        else:
            desc_elem.set("null", "true")
        
        unit_elem = ET.SubElement(col_elem, "unit")
        if col["unit"] is not None:
            unit_elem.text = escape_xml_text(col["unit"])
        else:
            unit_elem.set("null", "true")
        
        ET.SubElement(col_elem, "dtype").text = escape_xml_text(col["dtype"])
        ET.SubElement(col_elem, "group").text = escape_xml_text(col["group"])
        ET.SubElement(col_elem, "completeness_pct").text = str(col["completeness_pct"])
    
    ET.indent(root, space="  ", level=0)
    xml_str = ET.tostring(root, encoding='unicode', xml_declaration=False)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

def generate_metadata():
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
        
        desc = desc if pd.notna(desc) else None
        unit = unit if pd.notna(unit) and unit else None
        
        col_info = {
            "name": col_name,
            "description": desc,
            "unit": unit,
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
        "description": "This dataset provides a unified, curated collection of experimental results on the hydrothermal conversion of lignocellulosic and lignin-rich biomass, including hydrothermal treatment (HTT), hydrothermal liquefaction (HTL), and related process variants. It comprises 3,693 experimental runs extracted from peer-reviewed literature spanning over 40 years (1982-2026), encompassing 145 features including feedstock composition, reaction conditions, product yields, and energy/carbon recovery metrics. The database covers diverse biomass feedstocks, reactor configurations, solvents, catalysts, temperatures (typically 200-400°C), residence times, and operating conditions, with all records normalized to consistent units and yield bases. The dataset includes comprehensive provenance tracking, quality-control flags, and documentation of imputed values. Beyond raw experimental data, it provides derived descriptors such as elemental ratios (O/C, H/C), lignocellulosic composition fractions, energy yields, carbon recoveries, and composite indicators (e.g., Lignin Readiness Index). With standardized schemas and column-level metadata, the dataset is designed to support machine-learning model development, comparative process analysis, life-cycle assessment (LCA) studies, and serves as a FAIR-compliant resource for the biomass conversion, sustainable chemistry, and digital chemistry communities.",
        "license": "CC-BY-4.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "year_range": [int(df['year'].min()) if pd.notna(df['year'].min()) else None, 
                       int(df['year'].max()) if pd.notna(df['year'].max()) else None],
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
    
    xml_content = generate_xml_metadata(metadata)
    with open(metadata_path / 'metadata.xml', 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    radar_xml_content = generate_radar_xml_metadata(metadata)
    with open(metadata_path / 'metadata_radar.xml', 'w', encoding='utf-8') as f:
        f.write(radar_xml_content)
 
    pd.DataFrame(csv_rows).to_csv(metadata_path / 'column_metadata.csv', index=False)
    
    print(f"✅ Generated metadata files in {metadata_path.name}/")
    print(f"   • metadata.json")
    print(f"   • metadata.xml")
    print(f"   • metadata_radar.xml")
    print(f"   • column_metadata.csv")
    print(f"\n   Dataset: {len(df)} rows × {len(columns)} columns")
    print(f"   Years: {metadata['year_range'][0]}-{metadata['year_range'][1]}")

if __name__ == "__main__":
    generate_metadata()
