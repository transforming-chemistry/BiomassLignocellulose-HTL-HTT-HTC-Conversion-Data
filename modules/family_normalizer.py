import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple

# -----------------------
# Canonical family labels
# -----------------------
FAMILIES_STD = [
    "Woody Biomass / Hardwood",
    "Woody Biomass / Softwood",
    "Woody Biomass / Unspecified",
    "Herbaceous Biomass",
    "Agricultural Residues",
    "Aquatic Biomass",
    "Lignin-rich Streams (Technical)",
    "Mixed Biomass",
    "Model Compound",
    "Waste Biomass / Sludge",
    "Animal Manure",
    "Waste Wood / Construction",
    "Plastics / Polymer Blends",
    "Unknown",
]

# ----------------------------------------------------
# Helper: normalize strings for robust matching
# ----------------------------------------------------
def _norm_key(s: Any) -> str:
    s = str(s or "")
    s = s.strip().lower()
    s = re.sub(r"[^\x00-\x7F]+", "", s)       # strip non-ascii (™, ®, etc.)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s*\+\s*", " + ", s)
    return s

def _has_any(text: str, keys: List[str]) -> bool:
    return any(k in text for k in keys)

def _is_polymer_mix(feed: str) -> bool:
    return _has_any(feed, [" ldpe", " hdpe", " pe ", " pp", " ps", " pet", "polyethy", "polyprop", "polysty", "polyethylene", "polypropylene"])

# ----------------------------------------------------
# Exact/alias maps - aggressively expanded for your DB
# Keys normalized via _norm_key
# ----------------------------------------------------
_EXACT_MAP_RAW = {
    # Woody hardwoods / softwoods
    "bamboo": "Woody Biomass / Hardwood",
    "oak wood": "Woody Biomass / Hardwood",
    "mongolian oak": "Woody Biomass / Hardwood",
    "beech wood": "Woody Biomass / Hardwood",
    "beechwood": "Woody Biomass / Hardwood",
    "birch": "Woody Biomass / Hardwood",
    "birch wood powder (<20 mesh)": "Woody Biomass / Hardwood",
    "birch wood sawdust": "Woody Biomass / Hardwood",
    "birchwood sawdust": "Woody Biomass / Hardwood",
    "poplar": "Woody Biomass / Hardwood",
    "poplar wood": "Woody Biomass / Hardwood",
    "populus tomentosa": "Woody Biomass / Hardwood",
    "willow": "Woody Biomass / Hardwood",
    "willow wood": "Woody Biomass / Hardwood",
    "paulownia wood": "Woody Biomass / Hardwood",
    "eucalyptus": "Woody Biomass / Hardwood",
    "eucalyptus wood": "Woody Biomass / Hardwood",
    "eucalyptus bark": "Woody Biomass / Hardwood",
    "eucalyptus (eu)": "Woody Biomass / Hardwood",
    "eucalyptus tereticornis branch (40–60 mesh)": "Woody Biomass / Hardwood",
    "camellia oleifera abel (coa)": "Woody Biomass / Hardwood",
    "camellia oleifera": "Woody Biomass / Hardwood",

    "spruce": "Woody Biomass / Softwood",
    "spruce wood": "Woody Biomass / Softwood",
    "norway spruce": "Woody Biomass / Softwood",
    "white spruce bark": "Woody Biomass / Softwood",
    "pinewood": "Woody Biomass / Softwood",
    "pine wood": "Woody Biomass / Softwood",
    "loblolly pine": "Woody Biomass / Softwood",
    "loblolly pine (lp)": "Woody Biomass / Softwood",
    "slash pine": "Woody Biomass / Softwood",
    "ponderosa pinewood": "Woody Biomass / Softwood",
    "pine flakes (biller 2018)": "Woody Biomass / Softwood",
    "pinyon/juniper": "Woody Biomass / Softwood",
    "pinus massoniana": "Woody Biomass / Softwood",
    "cunninghamia lanceolata": "Woody Biomass / Softwood",
    "chinese fir": "Woody Biomass / Softwood",
    "white pine bark": "Woody Biomass / Softwood",
    "larch": "Woody Biomass / Softwood",
    "scotch pine wood": "Woody Biomass / Softwood",
    "pipeline-transported softwood": "Woody Biomass / Softwood",

    # Herbaceous & agricultural residues
    "miscanthus": "Herbaceous Biomass",
    "switchgrass (biller 2018)": "Herbaceous Biomass",
    "maize silage": "Herbaceous Biomass",
    "natural hay": "Herbaceous Biomass",

    "wheat straw": "Agricultural Residues",
    "wheat straw (ws)": "Agricultural Residues",
    "corn stover": "Agricultural Residues",
    "cornstalk": "Agricultural Residues",
    "corn stalk": "Agricultural Residues",
    "corn straw": "Agricultural Residues",
    "corn straw (cs)": "Agricultural Residues",
    "corn straw (zhang 2020)": "Agricultural Residues",
    "corn straw (ref 73)": "Agricultural Residues",
    "soybean straw": "Agricultural Residues",
    "soybean straw (ref 73)": "Agricultural Residues",
    "peanut straw": "Agricultural Residues",
    "rice straw": "Agricultural Residues",
    "rice straw (hunan)": "Agricultural Residues",
    "rice straw (ref 73)": "Agricultural Residues",
    "barley straw": "Agricultural Residues",
    "canola straw": "Agricultural Residues",
    "flaxseed straw": "Agricultural Residues",
    "sugarcane bagasse": "Agricultural Residues",
    "sugarcane bagasse (bca)": "Agricultural Residues",
    "sugarcane ghr": "Agricultural Residues",
    "palm oil empty fruit bunch": "Agricultural Residues",
    "oil palm empty fruit bunch (efb)": "Agricultural Residues",
    "palm empty fruit bunches": "Agricultural Residues",
    "oil palm pressed fiber (oppf)": "Agricultural Residues",
    "palm mesocarp fiber (pmf)": "Agricultural Residues",
    "palm kernel shell (pks)": "Agricultural Residues",
    "coconut fiber": "Agricultural Residues",
    "coconut shell": "Agricultural Residues",
    "coconut husk": "Agricultural Residues",
    "walnut shell": "Agricultural Residues",
    "walnut shells": "Agricultural Residues",
    "corn cobs": "Agricultural Residues",
    "corn cob residues": "Agricultural Residues",
    "sunflower stalk": "Agricultural Residues",
    "grape marc": "Agricultural Residues",
    "blackcurrant pomace": "Agricultural Residues",
    "spent coffee grounds": "Agricultural Residues",
    "draff (brewers spent grains)": "Agricultural Residues",
    "distillers grains": "Agricultural Residues",
    "litsea cubeba seed": "Agricultural Residues",
    "onopordum heteracanthum stalks": "Agricultural Residues",
    "tobacco stalk": "Agricultural Residues",
    "tobacco processing waste": "Agricultural Residues",
    "bagasse pith": "Agricultural Residues",
    "rice husk": "Agricultural Residues",
    "rice hulls": "Agricultural Residues",
    "rice husk bran": "Agricultural Residues",
    "banana stem": "Agricultural Residues",
    "peanut straw (ps)": "Agricultural Residues",
    "soybean straw (ss)": "Agricultural Residues",
    "olive oil residues (oor), izmir, turkey": "Agricultural Residues",
    "olive oil residues": "Agricultural Residues",

    # Aquatic
    "water hyacinth (eichhornia crassipes)": "Aquatic Biomass",
    "algae": "Aquatic Biomass",
    "macroalgae": "Aquatic Biomass",
    "microalgae": "Aquatic Biomass",

    # Lignin-rich / technical streams
    "lignin-rich stream (lrs, from cellulosic ethanol)": "Lignin-rich Streams (Technical)",
    "ethanol lignin (ol)": "Lignin-rich Streams (Technical)",
    "ethanol lignin": "Lignin-rich Streams (Technical)",
    "pyroligneous acid (pa)": "Lignin-rich Streams (Technical)",
    "pyroligneous acid": "Lignin-rich Streams (Technical)",
    "lignin-rich stream (poplar, lignocellulosic ethanol biorefinery)": "Lignin-rich Streams (Technical)",
    "lignin-rich stream (poplar, ethanol biorefinery, northern italy)": "Lignin-rich Streams (Technical)",
    "enzymatic hydrolysis lignin": "Lignin-rich Streams (Technical)",
    "hydrolysis lignin (cellunolix)": "Lignin-rich Streams (Technical)",
    "alkali lignin": "Lignin-rich Streams (Technical)",
    "soda lignin": "Lignin-rich Streams (Technical)",
    "organosolv lignin (ol)": "Lignin-rich Streams (Technical)",
    "lignosulfonate lignin (ls)": "Lignin-rich Streams (Technical)",
    "alkaline lignin (al)": "Lignin-rich Streams (Technical)",
    "dealkaline lignin (dal)": "Lignin-rich Streams (Technical)",
    "kraft lignin (commercial)": "Lignin-rich Streams (Technical)",
    "kraft lignin (wheat straw)": "Lignin-rich Streams (Technical)",
    "industrial kraft lignin (ikl)": "Lignin-rich Streams (Technical)",
    "lignoboost kraft lignin (softwood)": "Lignin-rich Streams (Technical)",
    "lignoboost kraft lignin (softwood)": "Lignin-rich Streams (Technical)",
    "kraft lignin": "Lignin-rich Streams (Technical)",
    "black liquor": "Lignin-rich Streams (Technical)",
    "lignin-like product (acid-precipitated from bl)": "Lignin-rich Streams (Technical)",

    # Waste, sludge, manure, construction wood, RDF
    "waste activated sludge": "Waste Biomass / Sludge",
    "sewage sludge": "Waste Biomass / Sludge",
    "sludge": "Waste Biomass / Sludge",
    "was": "Waste Biomass / Sludge",
    "manure": "Animal Manure",
    "manure/lp (50:50)": "Mixed Biomass",
    "sludge/lp (50:50)": "Mixed Biomass",
    "brdf (biffa rdf)": "Waste Wood / Construction",
    "floc residue": "Waste Wood / Construction",
    "waste furniture sawdust": "Waste Wood / Construction",
    "newspaper": "Waste Wood / Construction",
    "construction wood (unw)": "Waste Wood / Construction",
    "construction wood (nhzw)": "Waste Wood / Construction",
    "construction wood (hzw)": "Waste Wood / Construction",
    "construction wood (mxw)": "Waste Wood / Construction",
    "charcoal from untreated softwood mix": "Waste Wood / Construction",
    "charcoal from ccb-treated softwood mix": "Waste Wood / Construction",

    # Model compounds
    "cellulose": "Model Compound",
    "holocellulose from wood fibers": "Model Compound",
    "wood (non-pretreated, powder mix)": "Woody Biomass / Unspecified",
    "wood (non-pretreated, powder)": "Woody Biomass / Unspecified",
    "wood (pretreated chips + raw powder)": "Woody Biomass / Unspecified",
    "wood (alkali pretreated chips)": "Woody Biomass / Unspecified",

    # Plastics / blends
    "cgt/ldpe (2/1)": "Plastics / Polymer Blends",
    "cgt/ldpe (1/2)": "Plastics / Polymer Blends",


    # Short-hands / typos / specific residues
    "lp": "Woody Biomass / Softwood",                              # lab shorthand for Loblolly Pine
    "miscanthus (biller 2018": "Herbaceous Biomass",               # missing right-parenthesis in source table
    "switchgrass (biller 2018": "Herbaceous Biomass",              # same pattern as above
    "tahoe mix": "Mixed Biomass",                                  # site-specific mixed forest fuel
    "orange peel (powder, <=500 m)": "Agricultural Residues",      # normalize the µm variant to <=500 m (ascii)
    "orange peel (powder, ≤500 µm)": "Agricultural Residues",      # keep the exact unicode version too
    "spent mushroom substrate (pleurotus ostreatus)": "Agricultural Residues",
    "hazelnut shell": "Agricultural Residues",

    # Safety for literal "None" strings appearing in Feedstock
    "none": "Unknown",

    # Fossil fuels / non-biomass materials (not suitable for HTT biomass classification)
    "kukersite oil shale (kos)": "Unknown",
    "kukersite oil shale": "Unknown",
    "lignite": "Unknown",
    "oil shale": "Unknown",

    # Additional mappings for unmapped feedstocks
    "acacia mangium": "Woody Biomass / Hardwood",                # Acacia species - tropical hardwood
    "kenaf": "Herbaceous Biomass",                               # Hibiscus cannabinus - herbaceous fiber crop
    "korean native kenaf (knk)": "Herbaceous Biomass",          # Korean variety of kenaf
    "metroxylon sp. stem": "Woody Biomass / Unspecified",       # Sago palm stem - woody but not typical hardwood/softwood
    "oil-palm empty fruit bunch": "Agricultural Residues",      # Palm oil processing residue
    "oil-palm shell": "Agricultural Residues",                  # Palm kernel shell residue
    "rubber tree": "Woody Biomass / Hardwood",                  # Hevea brasiliensis - tropical hardwood

}

# add this to _EXACT_MAP_RAW  (just below where you define it)

_EXACT_MAP_RAW.update({
    # 1) obvious woods from Japanese / SE Asian lists
    "ailanthus": "Woody Biomass / Hardwood",                     # tree, broadleaf
    "japanese cedar (sugi)": "Woody Biomass / Softwood",         # Cryptomeria japonica
    "japanese hemlock (tsuga)": "Woody Biomass / Softwood",
    "kamerere (nanyo-zai)": "Woody Biomass / Hardwood",
    "kapur (nanyo-zai)": "Woody Biomass / Hardwood",
    "red lauan (nanyo-zai)": "Woody Biomass / Hardwood",

    # 2) short crop / aquatic / ag wastes
    "duckweed (lemna sp.)": "Aquatic Biomass",
    "kenaf (stem)": "Herbaceous Biomass",
    "tea waste": "Agricultural Residues",
    "scg": "Agricultural Residues",                              # SCG → Spent coffee grounds
    "smc": "Agricultural Residues",                              # SMC → spent mushroom compost/substrate
    "cs": "Agricultural Residues",                               # "Cotton stalk (CS)"
    "pf": "Agricultural Residues",                               # palm fiber / plant fiber in your context

    # 3) odd but wood-ish / construction-ish
    "wpb": "Waste Wood / Construction",                          # waste particle/ply/pressed board
    "aw": "Aquatic Biomass",                                     # common HTL shorthand: algal waste

    # 4) literal None
    "none": "Unknown",
})
_EXACT_MAP_RAW["syrian mesquite (prosopis farcta)"] = "Woody Biomass / Hardwood"

EXACT_MAP_LC = { _norm_key(k): v for k, v in _EXACT_MAP_RAW.items() }


# ----------------------------------------------------
# Paper-provided Family aliases (already in df['Family'])
# ----------------------------------------------------
FAMILY_TEXT_MAP = {
    "woody biomass/softwood": "Woody Biomass / Softwood",
    "woody biomass / softwood": "Woody Biomass / Softwood",
    "woody biomass/ hardwood": "Woody Biomass / Hardwood",
    "woody biomass / hardwood": "Woody Biomass / Hardwood",
    "woody biomass": "Woody Biomass / Unspecified",
    "lignocellulosic (wood)": "Woody Biomass / Unspecified",
    "lignocellulosic (hardwood)": "Woody Biomass / Hardwood",
    "lignocellulosic / softwood": "Woody Biomass / Softwood",
    "woody biomass / mixed": "Mixed Biomass",
    "woody biomass / mixed (hardwood+softwood)": "Mixed Biomass",

    "herbaceous biomass": "Herbaceous Biomass",
    "herbaceous / agricultural residues": "Agricultural Residues",
    "ag residues / herbaceous": "Agricultural Residues",
    "ag residues / straw": "Agricultural Residues",
    "grasses / forage": "Herbaceous Biomass",
    "herbaceous biomass (sorghum)": "Herbaceous Biomass",
    "herbaceous lignocellulosic biomass": "Herbaceous Biomass",

    "aquatic": "Aquatic Biomass",
    "aquatic lignocellulosic biomass": "Aquatic Biomass",

    "lignin / industrial": "Lignin-rich Streams (Technical)",
    "technical lignin": "Lignin-rich Streams (Technical)",
    "technical lignin / kraft": "Lignin-rich Streams (Technical)",
    "lignin-rich streams": "Lignin-rich Streams (Technical)",
    "lignin-rich stream / hardwood-derived": "Lignin-rich Streams (Technical)",
    "derived or processed": "Lignin-rich Streams (Technical)",

    "mixed biomass (ss/ws)": "Mixed Biomass",
    "mixed biomass (cm/ws)": "Mixed Biomass",
    "mixed biomass (ss/cm)": "Mixed Biomass",
    "mixed biomass / herbaceous+woody": "Mixed Biomass",

    "waste wood / construction": "Waste Wood / Construction",
    "sludge": "Waste Biomass / Sludge",
    "poaceae": "Herbaceous Biomass",
    "pinaceae": "Woody Biomass / Softwood",
    "fagaceae": "Woody Biomass / Hardwood",
}

FAMILY_TEXT_MAP_LC = { _norm_key(k): v for k, v in FAMILY_TEXT_MAP.items() }

# ----------------------------------------------------
# Botanical family → group (when Family holds e.g., "Poaceae")
# ----------------------------------------------------
BOTANICAL_FAMILY_TO_GROUP = {
    "pinaceae": "Woody Biomass / Softwood",
    "cupressaceae": "Woody Biomass / Softwood",
    "taxaceae": "Woody Biomass / Softwood",
    "fagaceae": "Woody Biomass / Hardwood",
    "betulaceae": "Woody Biomass / Hardwood",
    "salicaceae": "Woody Biomass / Hardwood",
    "myrtaceae": "Woody Biomass / Hardwood",
    "oleaceae": "Woody Biomass / Hardwood",
    "lauraceae": "Woody Biomass / Hardwood",
    "poaceae": "Herbaceous Biomass",
}

# ----------------------------------------------------
# Core classifier
# ----------------------------------------------------
def normalize_family(row: pd.Series) -> str:
    feed_raw = row.get("Feedstock")
    fam_raw  = row.get("Family")

    # 0) Handle None/empty/NaN feedstocks early
    if feed_raw is None or str(feed_raw).strip().lower() in {"", "none", "nan"}:
        # fall back to Family text if it's informative; else Unknown
        if fam_raw:
            fam = _norm_key(fam_raw)
            if fam in FAMILY_TEXT_MAP_LC:
                return FAMILY_TEXT_MAP_LC[fam]
            if fam in BOTANICAL_FAMILY_TO_GROUP:
                return BOTANICAL_FAMILY_TO_GROUP[fam]
        return "Unknown"

    feed = _norm_key(feed_raw)
    fam  = _norm_key(fam_raw)

    # 1) Exact by feed name
    if feed in EXACT_MAP_LC:
        return EXACT_MAP_LC[feed]

    # 2) Model compounds / polymers
    if _has_any(feed, ["cellulose", "holocellulose"]):
        return "Model Compound"
    if _is_polymer_mix(feed) or _has_any(feed, ["ldpe", "polymer", "plastic"]):
        return "Plastics / Polymer Blends"

    # 3) Sludge / manure
    if _has_any(feed, ["sewage sludge", "waste activated sludge", " was ", "(was", " sludge"]):
        return "Waste Biomass / Sludge"
    if _has_any(feed, ["manure"]):
        return "Animal Manure"

    # 4) Mixtures (shorthands like “A/B”, “A+B”, ratios)
    if re.search(r"\b\d+\s*:\s*\d+\b", feed) or _has_any(feed, [" / ", " + ", " blend ", " mixture ", "mix)"]):
        return "Mixed Biomass"

    # 5) Lignin / technical streams by feed text
    if _has_any(feed, [
        "lignin-rich stream", "hydrolysis lignin", "enzymatic hydrolysis lignin",
        "alkali lignin", "soda lignin", "organosolv lignin", "lignosulfonate",
        "kraft lignin", "black liquor", "lignoboost", "ethanol lignin",
        "pyroligneous acid"
    ]):
        return "Lignin-rich Streams (Technical)"

    # 6) Aquatic by feed text
    if _has_any(feed, ["water hyacinth", "eichhornia", "algae", "macroalgae", "microalgae", "seaweed", "kelp"]):
        return "Aquatic Biomass"

    # 7) Agricultural residue by feed text
    if _has_any(feed, [
        "bagasse","stover","stalk","straw","husk","hull","cob","grain","silage",
        "marc","pomace","grounds","coffee","oppf","efb","mesocarp","kernel shell",
        "spent brewery","distillers","draff","seed coat","seed","olive oil residues"
    ]):
        return "Agricultural Residues"

    # 8) Woody species by feed text
    if _has_any(feed, ["spruce","pine","fir","juniper","cunninghamia","pinus","masson","larch"]):
        return "Woody Biomass / Softwood"
    if _has_any(feed, ["oak","beech","birch","eucalyptus","poplar","populus","willow","bamboo","fraxinus","ash","paulownia","camellia oleifera"]):
        return "Woody Biomass / Hardwood"
    if "wood" in feed:
        return "Woody Biomass / Unspecified"

    # 9) Family text overrides (paper-provided)
    if fam in FAMILY_TEXT_MAP_LC:
        return FAMILY_TEXT_MAP_LC[fam]

    # 10) Botanical family names in Family column
    if fam in BOTANICAL_FAMILY_TO_GROUP:
        return BOTANICAL_FAMILY_TO_GROUP[fam]

    # 11) Heuristics on Family text
    if _has_any(fam, ["model", "pure substrate", "surrogate"]):
        return "Model Compound"
    if _has_any(fam, ["lignin", "derived", "processed", "black liquor", "kraft", "organosolv", "soda lignin"]):
        return "Lignin-rich Streams (Technical)"
    if _has_any(fam, ["aquatic", "algae", "seaweed", "macroalgae", "microalgae", "hyacinth"]):
        return "Aquatic Biomass"
    if _has_any(fam, ["herbaceous", "grass", "forage", "poaceae"]):
        return "Herbaceous Biomass"
    if _has_any(fam, ["agricultur", "fruit", "nut", "shell", "straw", "stalk", "cob", "bagasse", "residue", "grain", "silage", "husk", "hull", "marc", "pomace", "grounds", "coffee"]):
        return "Agricultural Residues"
    if _has_any(fam, ["softwood", "spruce", "pine", "conifer", "cunninghamia", "pinus", "masson"]):
        return "Woody Biomass / Softwood"
    if _has_any(fam, ["hardwood", "oak", "beech", "birch", "eucalyptus", "poplar", "willow", "bamboo", "lauraceae", "fraxinus", "ash"]):
        return "Woody Biomass / Hardwood"
    if "wood" in fam or "woody" in fam:
        return "Woody Biomass / Unspecified"

    # 12) Waste wood / construction
    if _has_any(feed, ["newspaper", "waste furniture", "construction wood", "rdf", "floc residue", "charcoal from"]):
        return "Waste Wood / Construction"

    return "Unknown"

# ----------------------------------------------------
# Public API
# ----------------------------------------------------
def apply_family_normalization(
    df: pd.DataFrame,
    feedstock_col: str = "Feedstock",
    family_col: str = "Family",
    create_std_col: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    df2 = df.copy()
    if create_std_col and "Family_std" not in df2.columns:
        df2["Family_std"] = np.nan
    df2["Family_std"] = df2.apply(normalize_family, axis=1)
    # Replace the working Family with standardized
    df2[family_col] = df2["Family_std"]

    # Audit unknown
    unknowns = (
        df2.loc[df2[family_col] == "Unknown", feedstock_col]
        .astype(str)
        .dropna()
        .drop_duplicates()
        .sort_values(key=lambda s: s.str.lower())
        .tolist()
    )
    if verbose:
        total = df2[feedstock_col].nunique()
        mapped = total - len(unknowns)
        rate = (mapped / total * 100) if total else 0
        print("=== FEEDSTOCK FAMILY NORMALIZATION RESULTS ===")
        print(f"Total unique feedstocks: {total}")
        print(f"Successfully mapped:    {mapped} ({rate:.1f}%)")
        print(f"Still unmapped:         {len(unknowns)} ({100-rate:.1f}%)")
        if unknowns:
            print("\nStill unmapped feedstocks (sample up to 100):")
            for name in unknowns[:100]:
                print(" •", name)
            if len(unknowns) > 100:
                print(f"... and {len(unknowns) - 100} more")
    return df2

def get_family_statistics(
    df: pd.DataFrame,
    family_col: str = "Family",
    feedstock_col: str = "Feedstock"
) -> Dict[str, Any]:
    fam_counts = df[family_col].value_counts(dropna=False)
    feedstock_counts = df.groupby(family_col)[feedstock_col].nunique().sort_values(ascending=False)
    total_rows = len(df)
    unknown_rows = int((df[family_col] == "Unknown").sum())
    mapped_rows = total_rows - unknown_rows
    coverage_rate = (mapped_rows / total_rows * 100) if total_rows else 0
    return {
        "total_rows": total_rows,
        "total_families": fam_counts.size,
        "total_feedstocks": df[feedstock_col].nunique(),
        "mapped_rows": mapped_rows,
        "unknown_rows": unknown_rows,
        "coverage_rate": coverage_rate,
        "family_counts": fam_counts,
        "feedstock_counts": feedstock_counts,
        "family_distribution": fam_counts.to_dict(),
        "most_common_family": fam_counts.index[0] if fam_counts.size else None,
        "least_common_families": fam_counts[fam_counts == 1].index.tolist()
    }

def complete_family_normalization_pipeline(
    df: pd.DataFrame,
    feedstock_col: str = "Feedstock",
    family_col: str = "Family",
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if verbose:
        print("=== FEEDSTOCK FAMILY NORMALIZATION PIPELINE ===")
        print(f"Input: {len(df)} rows, {df[feedstock_col].nunique()} unique feedstocks")

    df_norm = apply_family_normalization(
        df, feedstock_col=feedstock_col, family_col=family_col, create_std_col=True, verbose=verbose
    )
    stats = get_family_statistics(df_norm, family_col=family_col, feedstock_col=feedstock_col)

    if verbose:
        print("\n=== CLASSIFICATION SUMMARY (top 10) ===")
        head = stats["family_counts"].head(10)
        for fam, count in head.items():
            pct = count / stats["total_rows"] * 100 if stats["total_rows"] else 0
            print(f"  {fam}: {count} rows ({pct:.1f}%)")
        if stats["family_counts"].size > 10:
            print(f"  ... and {stats['family_counts'].size - 10} more families")

    return df_norm, stats
# modules/family_normalizer.py (or wherever you keep QA/plots)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Sequence

def plot_family_circular_distribution(
    df: pd.DataFrame,
    family_col: str = "Family",
    min_share_to_label: float = 0.03,   # show text labels only if >= 3%
    min_share_to_keep: float = 0.015,   # aggregate slices <1.5% into "Other"
    top_n: Optional[int] = None,        # alternatively keep top N families, rest -> "Other"
    figsize: Tuple[float, float] = (7, 6),
    title: Optional[str] = "Feedstock Family Distribution",
    show: bool = True,
    return_table: bool = True,
    savepath: Optional[str] = None,     # e.g., "family_distribution.png"
) -> Optional[Dict[str, Any]]:
    """
    Draw a donut chart of Family distribution (percent of rows).
    Assumes 'Family' has already been normalized (e.g., via complete_family_normalization_pipeline).
    """
    # --- counts & shares
    counts = df[family_col].fillna("Unknown").value_counts()
    total = counts.sum()
    if total == 0:
        print("No rows to plot.")
        return None

    shares = (counts / total).sort_values(ascending=False)

    # --- keep top or threshold; fold small slices to "Other"
    if top_n is not None and top_n > 0 and len(shares) > top_n:
        kept = shares.head(top_n)
        other_share = shares.iloc[top_n:].sum()
        if other_share > 0:
            kept.loc["Other"] = other_share
        shares_plot = kept
    else:
        big = shares[shares >= min_share_to_keep]
        other_share = shares[shares < min_share_to_keep].sum()
        if other_share > 0:
            big.loc["Other"] = other_share
        shares_plot = big

    labels = shares_plot.index.tolist()
    sizes  = shares_plot.values

    # --- figure
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts = ax.pie(
        sizes,
        startangle=90,
        wedgeprops=dict(width=0.38, edgecolor="white"),  # donut look
        labels=None,                                     # we'll place labels manually
    )
    ax.axis("equal")

    # center hole text
    ax.text(0, 0, f"n={total}", ha="center", va="center", fontsize=11, alpha=0.8)

    # label only meaningful slices
    for w, lab, share in zip(wedges, labels, sizes):
        pct = share * 100
        if pct / 100.0 >= min_share_to_label:
            ang = (w.theta2 + w.theta1) / 2.0
            x = 0.72 * np.cos(np.deg2rad(ang))
            y = 0.72 * np.sin(np.deg2rad(ang))
            ax.text(x, y, f"{lab}\n{pct:.1f}%", ha="center", va="center", fontsize=9)

    # title + legend (optional)
    if title:
        ax.set_title(title, pad=12)

    # lightweight legend (sorted)
    legend_labels = [f"{lab} — {share*100:.1f}%" for lab, share in zip(labels, sizes)]
    ax.legend(
        wedges, legend_labels,
        title="Families", loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=9
    )

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_table:
        table = (shares_plot * 100).round(2).rename("percent").reset_index().rename(columns={"index": "Family"})
        return {"figure": fig, "table": table, "total_rows": int(total)}
    return None
