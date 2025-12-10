from __future__ import annotations
"""
qa_envelopes.py — Envelope-aware QA utilities for HTT/HTL/HTC datasets.

Public API:
- plot_energy_carbon_envelopes(df): histograms for E_B, E_H, C_B, C_H with logical borders
- plot_yield_envelopes(df): histograms for B_Y, C_Y, A_Y, G_Y with logical borders
- run_basic_qc(df): quick consistency checks and small peeks
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Dict, Any, Tuple, Optional

def _norm_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()

def _safe_json_load(maybe_json):
    try:
        if pd.isna(maybe_json):
            return {}
    except Exception:
        pass
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, (str, bytes)):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.copy()
    for c in cols:
        if c not in g.columns:
            g[c] = np.nan
    return g

def _first_present(row: pd.Series, keys: list[str], default=None):
    for k in keys:
        if k in row.index:
            v = row.get(k, default)
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            return v
    return default

def _proc_from_row(row: pd.Series) -> str | None:
    v = row.get("Process_type", None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = row.get("process_type", None)
    return v



def _proc_default_envelopes(proc: str | None) -> dict:
    p = _norm_text(proc)
    if ("hydrothermal liquefaction" in p) or (p == "htl") or ("liquefaction" in p):
        return {"E_B_range": (0.0, 25.0), "E_H_range": (0.0, 8.0),
                "C_B_range": (0.0, 0.55), "C_H_range": (0.0, 0.25)}
    if ("hydrothermal carbon" in p) or (p == "htc") or ("carbonization" in p):
        return {"E_B_range": (0.0, 3.0), "E_H_range": (0.0, 22.0),
                "C_B_range": (0.0, 0.10), "C_H_range": (0.0, 0.65)}
    if "pyrolysis" in p:
        return {"E_B_range": (0.0, 22.0), "E_H_range": (0.0, 18.0),
                "C_B_range": (0.0, 0.60), "C_H_range": (0.0, 0.60)}
    return {"E_B_range": (0.0, 25.0), "E_H_range": (0.0, 22.0),
            "C_B_range": (0.0, 0.60), "C_H_range": (0.0, 0.60)}

def _row_caps_from_values(r: pd.Series, eps: float = 0.05) -> dict:
    BY      = pd.to_numeric(r.get("B_Y"), errors="coerce")
    CY      = pd.to_numeric(r.get("C_Y"), errors="coerce")
    HHV_bo  = pd.to_numeric(r.get("HHV_biooil"), errors="coerce")
    HHV_ch  = pd.to_numeric(r.get("HHV_biochar"), errors="coerce")
    C_bo    = pd.to_numeric(r.get("C_biooil"), errors="coerce")
    C_ch    = pd.to_numeric(r.get("C_biochar"), errors="coerce")

    caps = {}
    if pd.notna(BY) and pd.notna(HHV_bo):
        caps["E_B_range"] = (0.0, max(0.0, (BY/100.0) * HHV_bo * (1.0 + eps)))
    if pd.notna(CY) and pd.notna(HHV_ch):
        caps["E_H_range"] = (0.0, max(0.0, (CY/100.0) * HHV_ch * (1.0 + eps)))
    if pd.notna(BY) and pd.notna(C_bo):
        caps["C_B_range"] = (0.0, max(0.0, (BY/100.0) * (C_bo/100.0) * (1.0 + eps)))
    if pd.notna(CY) and pd.notna(C_ch):
        caps["C_H_range"] = (0.0, max(0.0, (CY/100.0) * (C_ch/100.0) * (1.0 + eps)))
    return caps

def _resolve_envelope(row: pd.Series) -> dict:
    ex = _safe_json_load(row.get("extra"))
    explicit = (ex.get("QA") or {}).get("envelope") or {}
    proc = _proc_from_row(row)

    if explicit:
        defaults = _proc_default_envelopes(proc)
        return {
            "E_B_range": tuple(explicit.get("E_B_range", defaults["E_B_range"])),
            "E_H_range": tuple(explicit.get("E_H_range", defaults["E_H_range"])),
            "C_B_range": tuple(explicit.get("C_B_range", defaults["C_B_range"])),
            "C_H_range": tuple(explicit.get("C_H_range", defaults["C_H_range"])),
        }
    caps = _row_caps_from_values(row, eps=0.05)
    if caps:
        defaults = _proc_default_envelopes(proc)
        return {
            "E_B_range": tuple(caps.get("E_B_range", defaults["E_B_range"])),
            "E_H_range": tuple(caps.get("E_H_range", defaults["E_H_range"])),
            "C_B_range": tuple(caps.get("C_B_range", defaults["C_B_range"])),
            "C_H_range": tuple(caps.get("C_H_range", defaults["C_H_range"])),
        }
    return _proc_default_envelopes(proc)

def _global_envelope_bounds(df: pd.DataFrame, range_key: str) -> tuple[float, float]:
    los, his = [], []
    for _, r in df.iterrows():
        env = _resolve_envelope(r)
        lo, hi = env[range_key]
        los.append(lo); his.append(hi)
    return (np.nanmin(los), np.nanmax(his))

def _plot_distribution(df: pd.DataFrame, value_col: str, range_key: str, xlabel: str, title: str):
    vals = pd.to_numeric(df.get(value_col), errors="coerce").dropna().values
    if vals.size == 0:
        print(f"No data to plot for {value_col}."); return
    lo, hi = _global_envelope_bounds(df, range_key)
    plt.figure()
    plt.hist(vals, bins=40)
    plt.axvline(lo, linestyle="--", linewidth=2, label="Envelope low")
    plt.axvline(hi, linestyle="--", linewidth=2, label="Envelope high")
    plt.xlabel(xlabel)
    plt.title(title + f"\nGlobal envelope: [{lo:.3g}, {hi:.3g}]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_carbon_envelopes(df: pd.DataFrame) -> None:
    cols_needed = ["extra","Process_type","process_type","E_B","E_H","C_B","C_H",
                   "B_Y","C_Y","HHV_biooil","HHV_biochar","C_biooil","C_biochar"]
    g = _ensure_cols(df, cols_needed)
    _plot_distribution(g, "E_B", "E_B_range", "E_B (MJ/kg dry biomass)",
                       "Distribution of E_B with logical border")
    _plot_distribution(g, "E_H", "E_H_range", "E_H (MJ/kg dry biomass)",
                       "Distribution of E_H with logical border")
    _plot_distribution(g, "C_B", "C_B_range", "C_B (kg C / kg dry biomass)",
                       "Distribution of C_B with logical border")
    _plot_distribution(g, "C_H", "C_H_range", "C_H (kg C / kg dry biomass)",
                       "Distribution of C_H with logical border")



def _proc_default_yield_envelopes(proc: str | None) -> dict:
    p = _norm_text(proc)
    if ("hydrothermal liq" in p) or ("liquefaction" in p) or (p == "htl"):
        return {"B_Y_range": (0.0, 65.0), "C_Y_range": (0.0, 35.0),
                "A_Y_range": (0.0, 80.0), "G_Y_range": (0.0, 30.0)}
    if ("hydrothermal carbon" in p) or ("carbonization" in p) or (p == "htc"):
        return {"B_Y_range": (0.0, 15.0), "C_Y_range": (20.0, 90.0),
                "A_Y_range": (0.0, 60.0), "G_Y_range": (0.0, 30.0)}
    if "pyrolysis" in p:
        return {"B_Y_range": (0.0, 75.0), "C_Y_range": (0.0, 50.0),
                "A_Y_range": (0.0, 15.0), "G_Y_range": (0.0, 40.0)}
    return {"B_Y_range": (0.0, 80.0), "C_Y_range": (0.0, 90.0),
            "A_Y_range": (0.0, 80.0), "G_Y_range": (0.0, 50.0)}

def _row_caps_from_yields(r: pd.Series, eps: float = 0.02) -> dict:
    BY = pd.to_numeric(r.get("B_Y"), errors="coerce")
    CY = pd.to_numeric(r.get("C_Y"), errors="coerce")
    AY = pd.to_numeric(r.get("A_Y"), errors="coerce")
    GY = pd.to_numeric(r.get("G_Y"), errors="coerce")

    def cap_from_others(others):
        if all(pd.notna(x) for x in others):
            rem = max(0.0, 100.0 - float(np.nansum(others)))
            return (0.0, rem * (1.0 + eps))
        return None

    caps = {}
    c = cap_from_others([CY, AY, GY])
    if c: caps["B_Y_range"] = c
    c = cap_from_others([BY, AY, GY])
    if c: caps["C_Y_range"] = c
    c = cap_from_others([BY, CY, GY])
    if c: caps["A_Y_range"] = c
    c = cap_from_others([BY, CY, AY])
    if c: caps["G_Y_range"] = c
    return caps

def _resolve_yield_envelope(row: pd.Series) -> dict:
    ex = _safe_json_load(row.get("extra"))
    explicit = (ex.get("QA") or {}).get("envelope") or {}
    defaults = _proc_default_yield_envelopes(_proc_from_row(row))
    if explicit:
        return {
            "B_Y_range": tuple(explicit.get("B_Y_range", defaults["B_Y_range"])),
            "C_Y_range": tuple(explicit.get("C_Y_range", defaults["C_Y_range"])),
            "A_Y_range": tuple(explicit.get("A_Y_range", defaults["A_Y_range"])),
            "G_Y_range": tuple(explicit.get("G_Y_range", defaults["G_Y_range"])),
        }
    caps = _row_caps_from_yields(row, eps=0.02)
    return {
        "B_Y_range": tuple(caps.get("B_Y_range", defaults["B_Y_range"])),
        "C_Y_range": tuple(caps.get("C_Y_range", defaults["C_Y_range"])),
        "A_Y_range": tuple(caps.get("A_Y_range", defaults["A_Y_range"])),
        "G_Y_range": tuple(caps.get("G_Y_range", defaults["G_Y_range"])),
    }

def _global_yield_bounds(df: pd.DataFrame, range_key: str) -> tuple[float, float]:
    los, his = [], []
    for _, r in df.iterrows():
        env = _resolve_yield_envelope(r)
        lo, hi = env[range_key]
        los.append(lo); his.append(hi)
    return (np.nanmin(los), np.nanmax(his))

def _plot_yield_distribution(df: pd.DataFrame, yield_col: str, range_key: str, title: str):
    vals = pd.to_numeric(df.get(yield_col), errors="coerce").dropna().values
    if vals.size == 0:
        print(f"No data to plot for {yield_col}."); return
    lo, hi = _global_yield_bounds(df, range_key)
    plt.figure()
    plt.hist(vals, bins=40)
    plt.axvline(lo, linestyle="--", linewidth=2, label="Envelope low")
    plt.axvline(hi, linestyle="--", linewidth=2, label="Envelope high")
    plt.xlabel(f"{yield_col} (wt% of dry feedstock)")
    plt.title(f"Distribution of {yield_col} with logical border\nGlobal envelope: [{lo:.3g}, {hi:.3g}]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_yield_envelopes(df: pd.DataFrame) -> None:
    cols_needed = ["extra","Process_type","process_type","B_Y","C_Y","A_Y","G_Y"]
    g = _ensure_cols(df, cols_needed)

    mask_all4 = g[["B_Y","C_Y","A_Y","G_Y"]].notna().all(axis=1)
    if mask_all4.any():
        mb = g.loc[mask_all4, ["B_Y","C_Y","A_Y","G_Y"]].sum(axis=1)
        n_bad = ((mb < 98.0) | (mb > 102.0)).sum()
        print(f"Mass-balance check (B_Y+C_Y+A_Y+G_Y ≈ 100): {n_bad} violations out of {mask_all4.sum()} rows")

    _plot_yield_distribution(g, "B_Y", "B_Y_range", "Biocrude / Liquid yield (B_Y)")
    _plot_yield_distribution(g, "C_Y", "C_Y_range", "Char / Hydrochar yield (C_Y)")
    _plot_yield_distribution(g, "A_Y", "A_Y_range", "Aqueous-organics yield (A_Y)")
    _plot_yield_distribution(g, "G_Y", "G_Y_range", "Gas yield (G_Y)")



def run_basic_qc(df: pd.DataFrame) -> None:
    need = [
        "process_type","Process_type",
        "E_H","E_B","B_Y","C_Y","C_B","C_H",
        "HHV_input","H/C","O/C",
        "HHV_biooil","HHV_biochar","E_B","E_H","C_B","C_H",
        "T","t","Temp_C","t_min","Ref","Source_Figure"
    ]
    g = _ensure_cols(df, need)

    if "T" not in g or g["T"].isna().all():
        g["T"] = g["Temp_C"]
    if "t" not in g or g["t"].isna().all():
        g["t"] = g["t_min"]

    g["proc"] = g.get("Process_type").astype(str).str.lower()
    null_mask = g["proc"].isin(["nan", "none"]) | g["proc"].isna()
    g.loc[null_mask, "proc"] = g.get("process_type").astype(str).str.lower()

    to_num = lambda s: pd.to_numeric(s, errors="coerce")

    EH_hi_default = 25.0
    EB_hi_default = 25.0

    bad_EH = g[to_num(g["E_H"]) > EH_hi_default]
    bad_EB = g[to_num(g["E_B"]) > EB_hi_default]

    print(f"E_H > {EH_hi_default} MJ/kg: {len(bad_EH)} rows")
    print(f"E_B > {EB_hi_default} MJ/kg: {len(bad_EB)} rows")

    bad_BY = g[(to_num(g["B_Y"]) < 0) | (to_num(g["B_Y"]) > 100)]
    bad_CY = g[(to_num(g["C_Y"]) < 0) | (to_num(g["C_Y"]) > 100)]
    print(f"B_Y outside [0,100]: {len(bad_BY)} rows")
    print(f"C_Y outside [0,100]: {len(bad_CY)} rows")

    bad_CB = g[to_num(g["C_B"]) > 0.60]
    bad_CH = g[to_num(g["C_H"]) > 0.65]
    print(f"C_B > 0.60 (fraction): {len(bad_CB)} rows")
    print(f"C_H > 0.65 (fraction): {len(bad_CH)} rows")

    bad_HHVf = g[to_num(g["HHV_input"]) < 12]
    bad_HC   = g[(to_num(g["H/C"]) < 0.8) | (to_num(g["H/C"]) > 1.9)]
    bad_OC   = g[(to_num(g["O/C"]) < 0.35) | (to_num(g["O/C"]) > 0.95)]
    print(f"HHV_input < 12 MJ/kg: {len(bad_HHVf)} rows")
    print(f"H/C outside ~[0.8,1.9]: {len(bad_HC)} rows")
    print(f"O/C outside ~[0.35,0.95]: {len(bad_OC)} rows")

    def _peek(df_, cols, n=5, tag=""):
        if len(df_) == 0:
            return
        cols = [c for c in cols if c in g.columns]
        print(f"\n-- {tag} (showing {min(n,len(df_))} of {len(df_)}) --")
        print(df_[cols].head(n))

    show_cols = ["Feedstock","proc","T","t","B_Y","C_Y","A_Y","G_Y",
                 "HHV_biooil","HHV_biochar","E_B","E_H","C_B","C_H",
                 "HHV_input","O/C","H/C","Ref","Source_Figure"]

    _peek(bad_EH, show_cols, tag=f"E_H > {EH_hi_default}")
    _peek(bad_EB, show_cols, tag=f"E_B > {EB_hi_default}")
    _peek(bad_BY, show_cols, tag="B_Y out of [0,100]")
    _peek(bad_CB, show_cols, tag="C_B > 0.60")
    _peek(bad_HHVf, show_cols, tag="HHV_input < 12")



def build_df_for_qa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the new standardized df and return a df_derived with the
    column names that qa_envelopes expects (B_Y, C_Y, E_B, E_H, HHV_input, etc.).
    """
    df_derived = df.copy()

    rename_map = {
        "C_feed_wt_pct": "C",
        "H_feed_wt_pct": "H",
        "O_feed_wt_pct": "O",
        "N_feed_wt_pct": "N",
        "S_feed_wt_pct": "S",
        "Ash_feed_wt_pct": "Ash",
        "Lignin_feed_wt_pct": "Lignin",
        "Cellulose_feed_wt_pct": "cellulose",
        "Hemicellulose_feed_wt_pct": "hemicellulose",
        "Extractives_feed_wt_pct": "extractives",

        "O_C_feed_molar": "O/C",
        "H_C_feed_molar": "H/C",


        "HHV_feed_MJ_per_kg": "HHV_input",


        "T_reaction_C": "T",
        "t_residence_min": "t",
        "t_ramp_min": "t_ramp",
        "IC_feed_wt_pct_slurry": "IC",
        "pressure_reaction_MPa": "pressure",
        "heating_rate_C_per_min": "heating_rate",
        "stirring_rpm": "stirring_velocity",
        "water_biomass_ratio_kg_kg": "water_biomass_ratio",
        "cat_biomass_ratio_kg_kg": "catalyst_biomass_ratio",

        "Yield_biooil_wt_pct": "B_Y",
        "Yield_char_wt_pct": "C_Y",
        "Yield_aqueous_wt_pct": "A_Y",
        "Yield_gas_wt_pct": "G_Y",
        "Yield_gas_water_wt_pct": "GW_Y",


        "Energy_yield_biooil_pct": "E_B",
        "Energy_yield_char_pct": "E_H",

        "Carbon_yield_biooil_pct": "C_B",
        "Carbon_yield_char_pct": "C_H",


        "HHV_biooil_MJ_per_kg": "HHV_biooil",
        "HHV_char_MJ_per_kg": "HHV_biochar",


        "C_char_wt_pct": "C_biochar",
        "H_char_wt_pct": "H_biochar",
        "O_char_wt_pct": "O_biochar",
        "N_char_wt_pct": "N_biochar",
        "S_char_wt_pct": "S_biochar",
        "O_C_char_molar": "O/C_biochar",
        "H_C_char_molar": "H/C_biochar",

        "O_C_biooil_molar": "O/C_biooil",
        "H_C_biooil_molar": "H/C_biooil",


        "Moisture_min_wt_pct_ar": "Moisture_min",
        "Moisture_max_wt_pct_ar": "Moisture_max",

        "VK_cluster": "VK_cluster_k2",
    }

    df_derived = df_derived.rename(columns=rename_map)


    if "process_type" in df_derived.columns and "Process_type" not in df_derived.columns:
        df_derived["Process_type"] = df_derived["process_type"]


    numeric_cols_for_qa = [
        "C", "H", "O", "N", "S", "Ash",
        "Lignin", "cellulose", "hemicellulose", "extractives",
        "O/C", "H/C",
        "HHV_input",
        "B_Y", "C_Y", "A_Y", "G_Y", "GW_Y",
        "E_B", "E_H",
        "C_B", "C_H",
        "HHV_biooil", "HHV_biochar",
        "T", "t", "t_ramp", "IC",
        "pressure", "heating_rate", "stirring_velocity", "water_biomass_ratio",
    ]

    for col in numeric_cols_for_qa:
        if col in df_derived.columns:
            df_derived[col] = pd.to_numeric(df_derived[col], errors="coerce")

    return df_derived


def drop_by_yield_envelope(df: pd.DataFrame,
                           col: str = "B_Y",
                           override_range: tuple[float, float] | None = None,
                           keep_na: bool = True,
                           inplace: bool = False) -> pd.DataFrame:
    """
    Drop rows where `col` is outside its logical envelope.
    - By default uses row-adaptive envelopes (from _resolve_yield_envelope).
    - If override_range=(lo, hi) is given, uses that constant range instead.
    - keep_na=True keeps rows where `col` is NaN.
    Returns a filtered DataFrame (or modifies in place if inplace=True).
    """
    g = df if inplace else df.copy()

    needed = ["extra", "Process_type", "process_type", "B_Y", "C_Y", "A_Y", "G_Y"]
    g = _ensure_cols(g, needed)

    if override_range is not None:
        lo, hi = override_range
        vals = pd.to_numeric(g[col], errors="coerce")
        keep_mask = vals.isna() if keep_na else vals.notna()
        keep_mask |= (vals >= lo) & (vals <= hi)
    else:
        def _keep_row(row: pd.Series) -> bool:
            v = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(v):
                return keep_na
            env = _resolve_yield_envelope(row)
            lo, hi = env[f"{col}_range"]
            return (v >= lo) and (v <= hi)
        keep_mask = g.apply(_keep_row, axis=1)

    dropped = int((~keep_mask).sum())
    kept = int(keep_mask.sum())
    total = int(len(g))
    print(f"🧹 Dropping {dropped} outlier rows for {col} (kept {kept}/{total}).")

    if inplace:
        g.drop(index=g.index[~keep_mask], inplace=True)
        g.reset_index(drop=True, inplace=True)
        return g
    else:
        return g.loc[keep_mask].reset_index(drop=True)


def homogenize_ic_to_percent(
    df: pd.DataFrame,
    bounds: tuple[float, float] = (0.5, 40.0),
    prefer_wbr: bool = True,
    diff_tol_pp: float = 1.0,
    inplace: bool = False,
    dedupe_by_paper: bool = True,
    paper_key: str | None = "paper_title",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize IC to percent (0..100) and return an audit.
      - IC in (0,1] => fraction -> %; 1< IC <=100 => already %.
      - If Water_Biomass_ratio present: IC_wbr = 100/(1+W/B).
      - Prefer W/B when |IC% - IC_wbr%| > diff_tol_pp (if prefer_wbr).
      - Flag out-of-bounds vs 'bounds'.
    Returns (df_out, audit_df). audit_df is one row per paper when dedupe_by_paper=True.
    """
    g = df if inplace else df.copy()

    if paper_key is None or paper_key not in g.columns:
        for cand in ("paper_title", "DOI", "Ref"):
            if cand in g.columns:
                paper_key = cand
                break
        else:
            paper_key = None  # no grouping key; we'll still work

    wbr_col = None
    for cand in ["water_biomass_ratio","Water_Biomass_ratio","W_B_ratio","WBR","Water/Biomass","water:biomass"]:
        if cand in g.columns:
            wbr_col = cand
            break

    IC_raw = pd.to_numeric(g.get("IC"), errors="coerce")
    IC_norm = IC_raw.copy()

    frac_mask = (IC_raw > 0) & (IC_raw <= 1.0)
    IC_norm[frac_mask] = 100.0 * IC_raw[frac_mask]

    ic_from_wbr = pd.Series(np.nan, index=g.index)
    if wbr_col is not None:
        WBr = pd.to_numeric(g[wbr_col], errors="coerce")
        ic_from_wbr = 100.0 / (1.0 + WBr)
        ic_from_wbr = ic_from_wbr.where(np.isfinite(ic_from_wbr))

    both = IC_norm.notna() & ic_from_wbr.notna()
    big_diff = both & (IC_norm.sub(ic_from_wbr).abs() > diff_tol_pp)

    if prefer_wbr and wbr_col is not None:
        IC_norm.loc[big_diff] = ic_from_wbr.loc[big_diff]

    fill_from_wbr = IC_norm.isna() & ic_from_wbr.notna()
    IC_norm.loc[fill_from_wbr] = ic_from_wbr.loc[fill_from_wbr]

    lo, hi = bounds
    out_of_bounds = IC_norm.notna() & ((IC_norm < lo) | (IC_norm > hi))

    g["IC"] = IC_norm

    report = pd.DataFrame({
        paper_key if paper_key else "paper_key": g.get(paper_key, pd.Series(index=g.index, dtype=object)),
        "IC_raw": IC_raw,
        "IC_norm_pct": IC_norm,
        "IC_from_wbr_pct": ic_from_wbr if wbr_col is not None else pd.Series(np.nan, index=g.index),
        "used_wbr": ((prefer_wbr & big_diff) | fill_from_wbr),
        "flag_disagree_with_wbr": big_diff if wbr_col is not None else pd.Series(False, index=g.index),
        "flag_out_of_bounds": out_of_bounds,
    })

    try:
        n_total = len(g)
        n_wbr = int(ic_from_wbr.notna().sum()) if wbr_col is not None else 0
        n_fill = int(fill_from_wbr.sum())
        n_diff = int(big_diff.sum()) if wbr_col is not None else 0
        n_oob  = int(out_of_bounds.sum())
        print(f"IC homogenization: rows={n_total}, with W/B={n_wbr}, filled_from_W/B={n_fill}, "
              f"disagreed>{diff_tol_pp}pp={n_diff}, out_of_bounds[{lo},{hi}]={n_oob}")
    except Exception:
        pass

    if dedupe_by_paper and paper_key:
        key = report[paper_key]
        grp = pd.DataFrame({
            "rows_total": key.groupby(key).size()
        })
        grp["rows_with_IC_raw"]   = IC_raw.notna().groupby(key).sum().astype(int)
        grp["rows_with_WB"]       = (ic_from_wbr.notna()).groupby(key).sum().astype(int) if wbr_col else 0
        grp["rows_used_wbr"]      = report.groupby(key)["used_wbr"].sum().astype(int)
        grp["rows_disagree_wbr"]  = report.groupby(key)["flag_disagree_with_wbr"].sum().astype(int)
        grp["rows_out_of_bounds"] = report.groupby(key)["flag_out_of_bounds"].sum().astype(int)
        grp["IC_min"]             = IC_norm.groupby(key).min()
        grp["IC_max"]             = IC_norm.groupby(key).max()
        grp["IC_median"]          = IC_norm.groupby(key).median()
        if wbr_col:
            grp["IC_wbr_min"]     = ic_from_wbr.groupby(key).min()
            grp["IC_wbr_max"]     = ic_from_wbr.groupby(key).max()

        audit = grp.reset_index().rename(columns={paper_key: "paper_title"})
    else:
        audit = report.loc[
            report["used_wbr"] | report["flag_disagree_with_wbr"] | report["flag_out_of_bounds"]
        ].reset_index(drop=True)

    return (g if inplace else g, audit)

def fill_missing_energy_carbon(
    df: pd.DataFrame,
    targets: tuple = ("E_B", "C_B"),          # choose from {"E_B","C_B","E_H","C_H"}
    eps_cap: float = 0.02,
    bounds: dict | None = None,
    recompute_existing: bool = False,
    dedupe_by_paper: bool = True,
    paper_key: str | None = "paper_title",
    inplace: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill E_B/C_B (and/or E_H/C_H) from yields & product properties.
    Returns (df_out, audit_df_dedup).
    """
    g = df if inplace else df.copy()

    if paper_key is None or paper_key not in g.columns:
        for cand in ("paper_title", "DOI", "Ref"):
            if cand in g.columns:
                paper_key = cand
                break
        else:
            paper_key = None

    BY = pd.to_numeric(g.get("B_Y"), errors="coerce")
    CY = pd.to_numeric(g.get("C_Y"), errors="coerce")
    HHV_bo = pd.to_numeric(g.get("HHV_biooil"), errors="coerce")
    HHV_ch = pd.to_numeric(g.get("HHV_biochar"), errors="coerce")
    C_bo = pd.to_numeric(g.get("C_biooil"), errors="coerce")
    C_ch = pd.to_numeric(g.get("C_biochar"), errors="coerce")

    EB0 = pd.to_numeric(g.get("E_B"), errors="coerce") if "E_B" in g.columns else pd.Series(np.nan, index=g.index)
    EH0 = pd.to_numeric(g.get("E_H"), errors="coerce") if "E_H" in g.columns else pd.Series(np.nan, index=g.index)
    CB0 = pd.to_numeric(g.get("C_B"), errors="coerce") if "C_B" in g.columns else pd.Series(np.nan, index=g.index)
    CH0 = pd.to_numeric(g.get("C_H"), errors="coerce") if "C_H" in g.columns else pd.Series(np.nan, index=g.index)

    default_bounds = {
        "E_B": (0.0, 25.0), "E_H": (0.0, 22.0),
        "C_B": (0.0, 0.60), "C_H": (0.0, 0.60),
    }
    B = default_bounds if bounds is None else {**default_bounds, **bounds}

    EB_cand = (BY/100.0) * HHV_bo
    CB_cand = (BY/100.0) * (C_bo/100.0)
    EH_cand = (CY/100.0) * HHV_ch
    CH_cand = (CY/100.0) * (C_ch/100.0)

    EB_cap = (BY/100.0) * HHV_bo * (1.0 + eps_cap)
    CB_cap = (BY/100.0) * (C_bo/100.0) * (1.0 + eps_cap)
    EH_cap = (CY/100.0) * HHV_ch * (1.0 + eps_cap)
    CH_cap = (CY/100.0) * (C_ch/100.0) * (1.0 + eps_cap)

    def _fill_metric(name, base0, cand, cap, lohi):
        lo, hi = lohi
        have_inputs = cand.notna() & cap.notna()
        missing_or_recompute = base0.isna() if not recompute_existing else have_inputs
        opportun = have_inputs & missing_or_recompute

        ok_cap = cand <= cap
        ok_bounds = cand.between(lo, hi)
        accept = opportun & ok_cap & ok_bounds

        base = base0.copy()
        base.loc[accept] = cand.loc[accept]
        if name in g.columns:
            g[name] = base
        else:
            g[name] = base

        return {
            "metric": name,
            "opportunities": int(opportun.sum()),
            "filled": int(accept.sum()),
            "rejected_cap": int((opportun & ~ok_cap).sum()),
            "rejected_bounds": int((opportun & ok_cap & ~ok_bounds).sum()),
            "existing_non_nan": int(base0.notna().sum()),
            "cand_min": float(cand.loc[opportun].min()) if opportun.any() else np.nan,
            "cand_max": float(cand.loc[opportun].max()) if opportun.any() else np.nan,
        }, accept

    audits = []
    accepts = {}

    if "E_B" in targets:
        a, m = _fill_metric("E_B", EB0, EB_cand, EB_cap, B["E_B"]); audits.append(a); accepts["E_B"] = m
    if "C_B" in targets:
        a, m = _fill_metric("C_B", CB0, CB_cand, CB_cap, B["C_B"]); audits.append(a); accepts["C_B"] = m
    if "E_H" in targets:
        a, m = _fill_metric("E_H", EH0, EH_cand, EH_cap, B["E_H"]); audits.append(a); accepts["E_H"] = m
    if "C_H" in targets:
        a, m = _fill_metric("C_H", CH0, CH_cand, CH_cap, B["C_H"]); audits.append(a); accepts["C_H"] = m

    audit_df = pd.DataFrame.from_records(audits)

    if paper_key:
        rows = []
        for met, mask in accepts.items():
            if mask.any():
                tmp = pd.DataFrame({
                    "paper_title": g[paper_key],
                    "metric": met,
                    "filled": mask.astype(int)
                })
                rows.append(tmp)
        if rows:
            long = pd.concat(rows, axis=0, ignore_index=True)
            per_paper = (long.groupby(["paper_title","metric"], dropna=False)["filled"]
                              .sum().reset_index()
                              .pivot(index="paper_title", columns="metric", values="filled")
                              .fillna(0).astype(int).reset_index())
        else:
            per_paper = pd.DataFrame(columns=["paper_title"])
    else:
        per_paper = pd.DataFrame(columns=["paper_title"])

    try:
        totals = ", ".join([f"{r['metric']}: filled {r['filled']}/{r['opportunities']}" for _, r in audit_df.iterrows()])
        print(f"Energy/Carbon fill — {totals}")
    except Exception:
        pass

    return (g if inplace else g, per_paper)

def drop_reaction_time_outliers(
    df: pd.DataFrame,
    strategy: str = "combined",         # "rule" | "mad" | "iqr" | "combined"
    keep_na: bool = True,
    min_valid_t: float = 0.5,
    mad_k: float = 3.5,
    iqr_k: float = 1.5,
    dedupe_by_paper: bool = True,
    paper_key: str | None = "paper_title",
    inplace: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows with non-logical reaction times:
      - 'rule': per-subtype envelope for t (min..max)
      - 'mad':  robust per-group (process_subtype) MAD outlier removal
      - 'iqr':  per-group Tukey IQR outlier removal
      - 'combined': apply both rule AND (mad/iqr), i.e., drop if violates either
    Returns (df_out, audit_df_per_paper).
    """
    g = df if inplace else df.copy()

    g = _ensure_cols(g, ["t", "process_subtype", "paper_title"])

    t = pd.to_numeric(g["t"], errors="coerce")

    t = t.where(~(t.notna() & (t < min_valid_t)), np.nan)
    g["t"] = t

    ENV_T = {
        "HTL": (1.0, 120.0),
        "HTC": (10.0, 1440.0),
        "HTG": (1.0, 120.0),
        "APR": (10.0, 480.0),
        "HT?": (1.0, 1440.0),
    }
    def _env_for(st: str):
        s = _norm_text(st).upper()
        if "HTL" in s: return ENV_T["HTL"]
        if "HTC" in s: return ENV_T["HTC"]
        if "HTG" in s: return ENV_T["HTG"]
        if "APR" in s: return ENV_T["APR"]
        return ENV_T["HT?"]

    subtype = g["process_subtype"].astype(str)
    bounds_lo = subtype.map(lambda s: _env_for(s)[0])
    bounds_hi = subtype.map(lambda s: _env_for(s)[1])

    if keep_na:
        ok_rule = t.isna() | ((t >= bounds_lo) & (t <= bounds_hi))
    else:
        ok_rule = t.notna() & (t >= bounds_lo) & (t <= bounds_hi)

    grpkey = g["process_subtype"].fillna("HT?")
    ok_outlier = pd.Series(True, index=g.index)

    def _mad_mask(x: pd.Series, k: float) -> pd.Series:
        x = x.dropna()
        if x.size < 8:
            return pd.Series(True, index=x.index)
        med = x.median()
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return pd.Series(True, index=x.index)
        z = 0.6745 * (x - med) / mad
        return z.abs() <= k

    def _iqr_mask(x: pd.Series, k: float) -> pd.Series:
        x = x.dropna()
        if x.size < 8:
            return pd.Series(True, index=x.index)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = (q1 - k*iqr, q3 + k*iqr)
        return (x >= lo) & (x <= hi)

    for name, idx in grpkey.groupby(grpkey).groups.items():
        tx = t.loc[idx]
        if strategy in ("mad", "combined"):
            ok_sub = _mad_mask(tx, mad_k)
        elif strategy == "iqr":
            ok_sub = _iqr_mask(tx, iqr_k)
        else:  # "rule"
            ok_sub = pd.Series(True, index=tx.index)
        ok_outlier.loc[ok_sub.index] = ok_outlier.loc[ok_sub.index] & ok_sub

    if strategy == "rule":
        keep_mask = ok_rule
    elif strategy in ("mad", "iqr"):
        keep_mask = ok_outlier if keep_na else (t.notna() & ok_outlier)
    else:  # "combined"
        keep_mask = ok_rule & ok_outlier

    dropped_idx = g.index[~keep_mask]
    dropped = g.loc[dropped_idx, ["paper_title", "process_subtype", "t"]].copy()

    dropped["bad_rule"] = ~ok_rule.loc[dropped_idx]
    dropped["bad_outlier"] = ~ok_outlier.loc[dropped_idx]

    if (paper_key is None) or (paper_key not in g.columns):
        paper_key = "paper_title"

    if not dropped.empty and dedupe_by_paper:
        audit = (dropped
                 .groupby([paper_key, "process_subtype"], dropna=False)
                 .agg(
                     rows_dropped=("t","size"),
                     bad_rule=("bad_rule","sum"),
                     bad_outlier=("bad_outlier","sum"),
                     t_min=("t","min"),
                     t_max=("t","max")
                 )
                 .reset_index()
                 .sort_values(["bad_rule","bad_outlier","rows_dropped"], ascending=False))
    else:
        audit = dropped.rename(columns={paper_key: "paper_title"})

    g = g.loc[keep_mask].reset_index(drop=True)

    try:
        print(f"🧹 drop_reaction_time_outliers: dropped {len(dropped_idx)} / {len(df)} rows "
              f"({strategy}, keep_na={keep_na})")
    except Exception:
        pass

    return (g if inplace else g, audit)

def relabel_low_T_htl(df, T_col="T", subtype_col="process_subtype",
                      threshold=170.0, label="HTC"):
    """
    Relabel HTL rows with T < threshold to HTC (not 'HTC?' which is deprecated).
    Low-temperature (<170°C) processes labeled as HTL are typically HTC.
    """
    t = pd.to_numeric(df.get(T_col), errors="coerce")
    is_htl = df.get(subtype_col, "").astype(str).str.upper().str.contains("HTL", na=False)
    mask = is_htl & t.notna() & (t < threshold)
    print(f"Relabeling {int(mask.sum())} HTL rows with {T_col} < {threshold}°C → {label}")
    df.loc[mask, subtype_col] = label
    return df

def audit_htl_sanity_and_catalyst(
    df,
    base_temp_range=(180.0, 380.0),
    scw_upper=420.0,
    temp_soft_tol=10.0,
    allowed_gases=("n2","nitrogen","co2","argon","ar"),
    disallowed_gases=("h2","hydrogen","o2","oxygen","air"),
    medium_ok=(
        "water","h2o","aqueous","subcritical water","near-critical water",
        "supercritical water","scw","water-ethanol","ethanol/water","water/ethanol",
        "ethanol","methanol","isopropanol","propanol","butanol","acetone",
        "ethylene glycol","glycerol","phenol","binary","mixture","co-solvent","cosolvent",
        "dioxane","1,4-dioxane","thf","tetrahydrofuran","tetralin","cyclohexane",
        "benzyl alcohol","decane","n-hexane"
    ),
    treat_missing_as_ok=True,
    strict_gas=False,
    strict_medium=False,
    inplace=False,
    dedupe_by_paper=True,
):
    import pandas as pd, numpy as np

    def _tx(x):
        try:
            if pd.isna(x): return ""
        except Exception:
            pass
        return str(x).strip().lower() if x is not None else ""

    if "process_subtype" not in df.columns:
        print("No 'process_subtype' column; nothing to audit for HTL.")
        return df, pd.DataFrame()

    g = df if inplace else df.copy()
    is_htl = g["process_subtype"].astype(str).str.upper().str.contains("HTL")
    h = g.loc[is_htl].copy()
    if h.empty:
        print("No HTL rows found.")
        return g, pd.DataFrame()

    def _bag(row):
        return " ".join([
            _tx(row.get("paper_title") or row.get("title")),
            _tx(row.get("solvent_or_medium") or row.get("Solvent_or_medium")),
            _tx(row.get("Provenance")), _tx(row.get("notes")), _tx(row.get("extra"))
        ])

    bag_tokens = h.apply(_bag, axis=1)
    scw_mask = bag_tokens.str.contains("supercritical|near-critical|scw", regex=True) | (
        pd.to_numeric(h.get("T"), errors="coerce") >= 390
    )

    T = pd.to_numeric(h.get("T"), errors="coerce")
    lo, hi = base_temp_range
    hi_dynamic = np.where(scw_mask, scw_upper, hi)
    ok_T_hard = T.isna() | ((T >= lo) & (T <= hi_dynamic))
    ok_T_soft = T.isna() | ((T >= lo - temp_soft_tol) & (T <= hi_dynamic + temp_soft_tol))

    bad_T      = ~ok_T_hard
    near_T     = ok_T_soft & ~ok_T_hard
    bad_T_low  = bad_T & (T.notna() & (T < lo))
    bad_T_high = bad_T & (T.notna() & (T > hi_dynamic))

    med = (h.get("solvent_or_medium") if "solvent_or_medium" in h.columns else
           h.get("Solvent_or_medium"))
    med_t = med.apply(_tx) if med is not None else pd.Series("", index=h.index)



    def _medium_ok(s):
        if not s:
            return treat_missing_as_ok or (not strict_medium)
        return any(k in s for k in medium_ok)

    ok_medium = med_t.apply(_medium_ok)
    bad_medium = ~ok_medium

    def _gasbag(row):
        return " ".join([
            _tx(row.get("gas")), _tx(row.get("gas_type")),
            _tx(row.get("Provenance")), _tx(row.get("reactor")),
            _tx(row.get("notes")), _tx(row.get("extra"))
        ])

    if "n2_flow" in h.columns:
        n2 = pd.to_numeric(h["n2_flow"], errors="coerce")
        explicit_inert = n2.notna() & (n2 > 0)
    else:
        explicit_inert = pd.Series(False, index=h.index)

    gas_ok = []
    for i, row in h.iterrows():
        if explicit_inert.loc[i]:
            gas_ok.append(True); continue
        bag = _gasbag(row)
        if not bag:
            gas_ok.append(treat_missing_as_ok or (not strict_gas)); continue
        if any(tok in bag for tok in disallowed_gases):
            gas_ok.append(False); continue
        if any(tok in bag for tok in allowed_gases):
            gas_ok.append(True); continue
        if any(kw in bag for kw in ["sealed","autogenous","no purge","without gas","no gas","batch reactor"]):
            gas_ok.append(True); continue
        gas_ok.append(treat_missing_as_ok or (not strict_gas))

    ok_gas = pd.Series(gas_ok, index=h.index)
    bad_gas = ~ok_gas

    cat = h.get("catalyst")
    def categorize_catalyst(catalyst_text: object) -> str:
        t = _tx(catalyst_text)
        if t in {"", "none", "no catalyst", "nc"}: return "none"
        acid_kw = {"acid","h2so4","hno3","hcl","h3po4","formic","acetic","p-toluenesulfonic","pts","sulfonic",
                   "zeolite","hzsm-5","h-beta","hbeta","usy","us*y","mordenite"}
        base_kw = {"base","alkaline","naoh","koh","nh3","ammonia","k2co3","na2co3","cao","ca(oh)2","mgo"}
        metal_kw = {" ni "," pd "," pt "," ru "," fe "," co "," cu "," zn "," mo "," w "," sn "}
        oxide_kw = {"al2o3","sio2","tio2","ceo2","zro2","nio","cuo","mno2","zno","oxide"}
        sulf_kw = {"mos2","cos","nis","fes","sulfide","sulfidized"}
        tt = " " + t + " "
        tags = []
        if any(k in t for k in acid_kw): tags.append("acid")
        if any(k in t for k in base_kw): tags.append("alkaline")
        if any(k in tt for k in metal_kw): tags.append("metal")
        if any(k in t for k in oxide_kw): tags.append("metal oxide")
        if any(k in t for k in sulf_kw): tags.append("metal sulfide")
        if not tags: return "unknown"
        if ("metal" in tags) and (("acid" in tags) or ("alkaline" in tags)): return "bifunctional (metal + acid/base)"
        if ("metal oxide" in tags) and ("acid" in tags): return "bifunctional (acidic oxide)"
        if "metal sulfide" in tags: return "metal sulfide"
        if "metal oxide" in tags: return "metal oxide"
        if "metal" in tags: return "metal"
        if "alkaline" in tags: return "alkaline"
        if "acid" in tags: return "acid"
        return tags[0]

    cat_cat = cat.apply(categorize_catalyst) if cat is not None else pd.Series("unknown", index=h.index)
    if inplace:
        g.loc[h.index, "catalyst_category"] = cat_cat.values

    issues = pd.DataFrame({
        "paper_title": h.get("paper_title", pd.Series(index=h.index, dtype=object)),
        "process_subtype": h["process_subtype"],
        "T": T,
        "solvent_or_medium": med,
        "gas_ok": ok_gas,
        "catalyst": cat,
        "catalyst_category": cat_cat,
        "bad_T": bad_T.fillna(False),
        "bad_T_low": bad_T_low.fillna(False),
        "bad_T_high": bad_T_high.fillna(False),
        "near_T": near_T.fillna(False),
        "bad_medium": bad_medium.fillna(False),
        "bad_gas": bad_gas.fillna(False),
        "scw_detected": scw_mask
    }, index=h.index)

    hard = ((issues["bad_T"] & ~issues["near_T"])
        | issues["bad_medium"]
        | issues["bad_gas"])
    issues_hard = issues.loc[hard]

    if dedupe_by_paper and not issues_hard.empty:
        audit = (issues_hard
                 .groupby(["paper_title"], dropna=False)
                 .agg(rows_flagged=("T","size"),
                      bad_T=("bad_T","sum"),
                      bad_T_low=("bad_T_low","sum"),
                      bad_T_high=("bad_T_high","sum"),
                      bad_medium=("bad_medium","sum"),
                      bad_gas=("bad_gas","sum"),
                      T_min=("T","min"), T_max=("T","max"))
                 .reset_index()
                 .sort_values(["bad_T_high","bad_T_low","bad_medium","bad_gas","rows_flagged"], ascending=False))
    else:
        audit = issues_hard.reset_index(drop=True)

    try:
        n_htl = len(h); n_bad = len(issues_hard)
        print(f"HTL sanity check: {n_htl} rows | base T [{lo},{hi}]°C (+/-{temp_soft_tol}), SCW≤{scw_upper}°C; "
              f"lenient medium/gas={'yes' if (not strict_medium and not strict_gas) else 'no'}")
        if n_bad:
            print(f"⚠️  Hard violations: {n_bad} rows across {audit['paper_title'].nunique()} papers.")
        else:
            print("✅ No hard violations; only near-misses (see 'near_T').")
    except Exception:
        pass

    return g, audit



import re
import json


_MAP_SINGLE = {
    "none":"none", "non-catalytic":"none", "no catalyst":"none", "h2o":"none",
    "h2so4":"h2so4","hcl":"hcl","hno3":"hno3","ch3cooh":"acetic_acid","formic acid":"formic_acid",
    "naoh":"naoh","k2co3":"k2co3","koh":"koh","na2co3":"na2co3","ca(oh)2":"caoh2","t-buok":"tbuok","nh4oh":"nh4oh",
    "fe":"fe","ni":"ni","co":"co","ru":"ru","pd":"pd","iron powder":"fe","fe (metal powder)":"fe","ni (metal powder)":"ni",
    "nio":"nio","fe2o3":"fe2o3","fe3o4":"fe3o4","zno":"zno","mgo":"mgo","al2o3":"al2o3","tio2":"tio2","ceo2":"ceo2","sio2":"sio2","zro2":"zro2",
    "fes":"fes","feso4":"feso4","zncl2":"zncl2","k3po4":"k3po4",
    "hzsm-5":"hzsm5","zeolite":"zeolite","sba-15":"sba15",
    "ru/c":"ru/c","pd/c":"pd/c","ni/ac":"ni/c","co/ac":"co/c","ni/sba-15":"ni/sba15","al/sba-15":"al/sba15","ni/al2o3":"ni/al2o3",
    "kf/al2o3":"kf/al2o3","colemanite":"colemanite",
}

_CLASS_RULES = {
    "h2so4":"acid","hcl":"acid","hno3":"acid","acetic_acid":"acid","formic_acid":"acid",
    "naoh":"alkaline","koh":"alkaline","k2co3":"alkaline","na2co3":"alkaline","caoh2":"alkaline","tbuok":"alkaline","nh4oh":"alkaline","kf/al2o3":"alkaline",
    "fe":"metal","ni":"metal","co":"metal","ru":"metal","pd":"metal",
    "nio":"oxide","fe2o3":"oxide","fe3o4":"oxide","zno":"oxide","mgo":"oxide","al2o3":"oxide","tio2":"oxide","ceo2":"oxide","sio2":"oxide","zro2":"oxide",
    "fes":"sulfide","feso4":"salt","zncl2":"salt","k3po4":"salt",
    "hzsm5":"zeolite","zeolite":"zeolite","sba15":"zeolite",
    "ru/c":"metal","pd/c":"metal","ni/al2o3":"metal","ni/sba15":"metal","al/sba15":"oxide",
    "colemanite":"other","none":"none",
}

_MAP_SINGLE.update({
    "none(de-ashedfeed)":"none",
    "fecatalytic":"fe",
    "fe(metalpowder)":"fe",
    "ironpowder":"fe",
    "ni(metalpowder)":"ni",
    "ni(powders)":"ni",
    "k2co3(%)":"k2co3",
    "koh(%)":"koh",
    "%fes":"fes",
    "formicacid":"formic_acid",
    "ac":"c",
    "ni2p":"ni2p",
    "ni/si-al":"ni/si-al",
})

_CLASS_RULES.update({
    "kf":"alkaline",          # for 'kf/al2o3' split: active 'kf' → alkaline
    "al":"oxide",             # 'al/sba15' → treat Al as oxide-like active site
    "c":"support",            # AC/carbon-only → support (not 'other')
    "ni2p":"phosphide",
    "colemanite":"salt",      # mineral → salt-like
})


_MAP_SINGLE.update({
    "bht": "bht",
})

_CLASS_RULES.update({
    "bht": "additive",   # antioxidant; not a catalyst
})

_SUPPORT_TOKENS = {"al2o3","sio2","tio2","ceo2","zro2","c","sba15"}
_SUPPORT_TOKENS.update({"c","sba15","si-al"})

def _tx(x) -> str:
    try:
        from math import isnan
        if x is None: return ""
        if isinstance(x, float) and isnan(x): return ""
    except Exception:
        pass
    s = str(x).strip().lower()
    s = s.replace("–","-").replace("—","-")
    s = re.sub(r"\s+"," ", s)
    return s

def _strip_loading(s: str):
    """
    Remove load/concentration annotations like:
      '5 wt%', '(5 wt%)', '0.16%', '1N', '0.94 M', '10 ppm'.
    (We also clean glued patterns later inside _normalize_component.)
    """
    import re
    s = re.sub(r"[–—]", "-", s)

    patterns = [
        r"\(\s*\d+(?:\.\d+)?\s*(?:wt\.?%?|%|w\/w|v\/v|mol\.?%?|m|n|M|N|g|mg|ppm)\s*\)",  # (5 wt%), (0.5N)
        r"\b\d+(?:\.\d+)?\s*(?:wt\.?%?|%|w\/w|v\/v|mol\.?%?|m|n|M|N|g|mg|ppm)\b",        # 5 wt%, 0.16%, 1N, 0.94 M
    ]
    changed = True
    while changed:
        s_old = s
        for pat in patterns:
            s = re.sub(pat, " ", s, flags=re.I)
        changed = (s != s_old)

    s = re.sub(r"\s*\+\s*", " + ", s)   # tidy ' + '
    s = re.sub(r"\s+", " ", s).strip(" ,;+-")
    return s, None



def _split_components(s: str):
    parts = re.split(r"\s*\+\s*|\s*,\s*", s)
    return [p.strip() for p in parts if p.strip()]

def _normalize_component(token: str):
    import re
    tok = token.strip().lower()
    tok = tok.replace("–","-").replace("—","-")

    tok = re.sub(r"\b\d+(?:\.\d+)?\s*(?:wt\.?%?|%|m|n|M|N)\b", "", tok, flags=re.I)

    tok = tok.replace("hzsm-5","hzsm5").replace("sba-15","sba15")
    tok = tok.replace("/ac","/c")   # only treat '/AC' as support (won't affect 'acetic')

    tok = tok.replace(" ", "")
    tok = re.sub(r"\d+(?:\.\d+)?(?:wt%|wt|%|m|n|M|N)$", "", tok, flags=re.I)
    tok = re.sub(r"\d+(?:\.\d+)?%", "", tok)

    if tok.startswith("tio2("): tok = "tio2"
    tok = re.sub(r"ceo2\(.+?\)", "ceo2", tok)

    if tok in _MAP_SINGLE:
        return _MAP_SINGLE[tok]
    return tok



def _component_active_support(tok: str):
    return tok.split("/",1) if "/" in tok else (tok, None)

def _class_for_token(tok: str):
    import re
    metals = {"ni","fe","co","ru","pd","pt","cu","zn","mo","w","sn"}
    if "-" in tok and any(p in metals for p in tok.split("-")):
        return "metal"
    if tok in _CLASS_RULES: return _CLASS_RULES[tok]
    if re.search(r"(oh|co3|po4|ok)$", tok): return "alkaline"
    if re.search(r"(o|o2|o3|o4)$", tok) and tok not in {"co","po","kpo"}: return "oxide"
    if tok in {"none","non-catalytic"}: return "none"
    return "other"


def _parse_catalyst_cell(raw) -> dict:
    if raw is None or str(raw).strip() == "" or (isinstance(raw,float) and pd.isna(raw)):
        return {"norm": np.nan, "load": None, "comps": [], "actives": [], "supports": [], "classes": set(), "primary": "none"}
    s = _tx(raw)
    s, load = _strip_loading(s)
    import re as _re
    if _re.search(r"\b(no catalyst|non[- ]?catalytic|none)\b", s):
        return {"norm": "none", "load": load, "comps": ["none"],
                "actives": [], "supports": [], "classes": {"none"}, "primary":"none"}
    if s in {"","-","none","non-catalytic","no catalyst","h2o"}:
        return {"norm": "none", "load": load, "comps": ["none"], "actives": [], "supports": [], "classes": {"none"}, "primary":"none"}

    comps, acts, sups, classes = [], [], [], set()
    for part in _split_components(s):
        tok = _normalize_component(part)
        a, sup = _component_active_support(tok)
        comps.append(tok)
        acts.append(a)
        if sup: sups.append(sup)
        classes.add(_class_for_token(a))
        if sup:
            cls_s = _class_for_token(sup)
            if cls_s != "other": classes.add(cls_s)

    priority = ["phosphide","metal","alkaline","acid","oxide","sulfide","zeolite",
                "support","none","salt","additive","other"]
    primary = next((c for c in priority if c in classes), "other")

    return {"norm":" + ".join(comps), "load":load, "comps":comps, "actives":acts, "supports":sups, "classes":classes, "primary":primary}

def categorize_catalyst(x) -> str:
    return _parse_catalyst_cell(x)["primary"]

def enrich_catalyst_features(df: pd.DataFrame,
                             include_onehots: bool = True,
                             include_active_support: bool = False,
                             inplace: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 1: normalize names + primary class
    Phase 2: (optional) multi-hot class one-hots; optionally active metal & support one-hots
    Returns (df_out, summary_df)
    """
    g = df if inplace else df.copy()
    col = "catalyst" if "catalyst" in g.columns else None
    if col is None:
        return (g, pd.DataFrame())

    parsed = g[col].apply(_parse_catalyst_cell)

    g["catalyst_norm"]   = parsed.apply(lambda d: d["norm"])
    g["catalyst_load"]   = parsed.apply(lambda d: d["load"])
    g["cat_primary"]     = parsed.apply(lambda d: d["primary"])
    g["cat_classes_json"]= parsed.apply(lambda d: json.dumps(sorted(list(d["classes"]))))

    added = []

    if include_onehots:
        all_classes = sorted({c for d in parsed for c in d["classes"]})
        for cls in all_classes:
            cname = f"Cat_{cls}"
            g[cname] = parsed.apply(lambda d, c=cls: int(c in d["classes"]))
            added.append(cname)

    if include_active_support:
        actives_flat  = sorted({a for d in parsed for a in d["actives"] if a and a not in _SUPPORT_TOKENS})
        supports_flat = sorted({s for d in parsed for s in d["supports"] if s})
        for a in actives_flat:
            cname = f"CatActive_{a.upper()}"
            g[cname] = parsed.apply(lambda d, aa=a: int(aa in d["actives"]))
            added.append(cname)
        for s in supports_flat:
            cname = f"CatSup_{s.upper()}"
            g[cname] = parsed.apply(lambda d, ss=s: int(ss in d["supports"]))
            added.append(cname)

    summary = (g.groupby("cat_primary", dropna=False)
                 .size().reset_index(name="rows")
                 .sort_values("rows", ascending=False))

    if "catalyst_category" not in g.columns:
        g["catalyst_category"] = g["cat_primary"]

    return (g if inplace else g, summary)


def audit_catalyst_others(df, topn=100):
    """
    Show which raw strings and normalized tokens ended up in the 'other' bucket.
    Returns (raw_level_df, token_level_df).
    """
    import pandas as pd, json

    if "catalyst" not in df.columns:
        return (pd.DataFrame(), pd.DataFrame())

    parsed = df["catalyst"].apply(_parse_catalyst_cell)

    g = pd.DataFrame({
        "raw": df["catalyst"],
        "norm": parsed.apply(lambda d: d["norm"]),
        "classes": parsed.apply(lambda d: sorted(list(d["classes"]))),
        "primary": parsed.apply(lambda d: d["primary"])
    })
    raw_level = (g[g["primary"]=="other"]
                 .value_counts(["raw","norm"])
                 .reset_index(name="rows")
                 .sort_values("rows", ascending=False).head(topn))

    def toks(d): return d.get("comps", [])
    tokens = []
    for d in parsed:
        for t in d.get("comps", []):
            cls = _class_for_token(t.split("/",1)[0])  # check active token
            if cls == "other":
                tokens.append(t)
    token_level = (pd.Series(tokens, name="token")
                   .value_counts()
                   .rename_axis("token")
                   .reset_index(name="rows")
                   .head(topn))
    return raw_level, token_level





def audit_fix_lignin(
    df: pd.DataFrame,
    family_col: str = "Family_std",   # fallback to "Family" if missing
    lignin_col: str = "Lignin",
    comp_cols: tuple = ("cellulose","hemicellulose","Extractives_pct","Ash"),
    convert_fraction_to_percent: bool = True,
    on_hard_violation: str = "nan",   # "nan" | "clip" | "drop"
    use_family_envelopes: bool = True,
    mad_outliers: bool = True,
    mad_k: float = 4.0,
    dedupe_by_paper: bool = True,
    inplace: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforce logical Lignin values:
      - Detects & converts fractions→percent (row-wise) for composition fields.
      - Hard bounds [0,100].
      - Soft family envelopes (only flags, does not alter unless on_hard_violation triggers).
      - Optional MAD outlier flags per family.
    Returns (df_out, audit_df_per_paper).
    """
    import numpy as np, pandas as pd

    g = df if inplace else df.copy()

    fam_col = family_col if family_col in g.columns else ("Family" if "Family" in g.columns else None)
    for c in [lignin_col, fam_col, "paper_title"]:
        if c and c not in g.columns:
            g[c] = np.nan

    L = pd.to_numeric(g.get(lignin_col), errors="coerce")
    comps = {c: pd.to_numeric(g.get(c), errors="coerce") for c in comp_cols if c in g.columns}

    def _row_fraction_like(idx):
        vals = []
        if pd.notna(L.iloc[idx]) and (0 < L.iloc[idx] <= 1.0):
            vals.append(True)
        for c, s in comps.items():
            v = s.iloc[idx]
            if pd.notna(v) and (0 <= v <= 1.0):
                vals.append(True)
        return (sum(vals) >= 3)

    converted_rows = []
    if convert_fraction_to_percent:
        for i in g.index:
            try:
                if _row_fraction_like(g.index.get_loc(i)):
                    if pd.notna(L.loc[i]): L.loc[i] *= 100.0
                    for c, s in comps.items():
                        if pd.notna(s.loc[i]):
                            s.loc[i] = s.loc[i] * 100.0
                    converted_rows.append(i)
            except Exception:
                pass
        g[lignin_col] = L
        for c, s in comps.items():
            g[c] = s

    hard_low = L.notna() & (L < 0)
    hard_high = L.notna() & (L > 100)

    if on_hard_violation == "clip":
        L = L.clip(lower=0, upper=100)
    elif on_hard_violation == "nan":
        L = L.where(~(hard_low | hard_high), np.nan)
    elif on_hard_violation == "drop":
        pass
    else:
        L = L.where(~(hard_low | hard_high), np.nan)

    g[lignin_col] = L

    FAMILY_ENV = {
        "Woody Biomass":        (18, 35),
        "Herbaceous Biomass":   (10, 25),
        "Agricultural Residues":( 8, 24),
        "Energy Crops":         (12, 28),
        "Algae":                ( 0, 15),
        "Food Waste":           ( 0, 20),
        "Sewage Sludge":        ( 0, 20),
        "Manure":               ( 0, 20),
        "Lignin-Rich":          (40,100),
        "Technical Lignin":     (60,100),
        "__fallback__":         ( 0, 50),
    }

    fam_series = g.get(fam_col) if fam_col else pd.Series(index=g.index, dtype=object)
    fam_txt = fam_series.astype(str)

    lo_soft = []; hi_soft = []
    for i, fam in fam_txt.items():
        rng = FAMILY_ENV.get(fam, FAMILY_ENV["__fallback__"])
        lo_soft.append(rng[0]); hi_soft.append(rng[1])
    lo_soft = pd.Series(lo_soft, index=g.index, dtype=float)
    hi_soft = pd.Series(hi_soft, index=g.index, dtype=float)

    soft_low = use_family_envelopes and L.notna() & (L < lo_soft)
    soft_high = use_family_envelopes and L.notna() & (L > hi_soft)

    outlier = pd.Series(False, index=g.index)
    if mad_outliers and fam_col:
        for fam, idx in g.groupby(fam_col).groups.items():
            series = pd.to_numeric(g.loc[idx, lignin_col], errors="coerce").dropna()
            if series.size >= 15:
                med = series.median()
                mad = np.median(np.abs(series - med))
                if mad > 0:
                    z = 0.6745 * (series - med) / mad
                    outlier.loc[series.index] = z.abs() > mad_k

    drop_mask = pd.Series(False, index=g.index)
    if on_hard_violation == "drop":
        drop_mask = hard_low | hard_high

    kept = g.loc[~drop_mask].copy()
    dropped = g.loc[drop_mask, ["paper_title", lignin_col, fam_col]].copy()

    issues = pd.DataFrame({
        "paper_title": kept.get("paper_title", pd.Series(index=kept.index, dtype=object)),
        "family": kept.get(fam_col, pd.Series(index=kept.index, dtype=object)),
        "Lignin": kept[lignin_col],
        "converted_fraction": kept.index.isin(converted_rows),
        "hard_low": hard_low.loc[kept.index].fillna(False),
        "hard_high": hard_high.loc[kept.index].fillna(False),
        "soft_low": pd.Series(soft_low, index=kept.index).fillna(False).astype(bool),
        "soft_high": pd.Series(soft_high, index=kept.index).fillna(False).astype(bool),
        "mad_outlier": outlier.loc[kept.index].fillna(False),
        "lo_soft": lo_soft.loc[kept.index],
        "hi_soft": hi_soft.loc[kept.index],
    })

    any_flag = (issues[["converted_fraction","hard_low","hard_high","soft_low","soft_high","mad_outlier"]].any(axis=1))
    flagged = issues.loc[any_flag]

    if dedupe_by_paper and not flagged.empty:
        audit = (flagged
                 .groupby(["paper_title"], dropna=False)
                 .agg(
                     rows_flagged=("Lignin","size"),
                     converted=("converted_fraction","sum"),
                     hard_low=("hard_low","sum"),
                     hard_high=("hard_high","sum"),
                     soft_low=("soft_low","sum"),
                     soft_high=("soft_high","sum"),
                     mad_outlier=("mad_outlier","sum"),
                     L_min=("Lignin","min"),
                     L_max=("Lignin","max"),
                 )
                 .reset_index()
                 .sort_values(["hard_high","hard_low","soft_high","soft_low","mad_outlier","converted","rows_flagged"], ascending=False))
    else:
        audit = flagged.reset_index(drop=True)

    try:
        n_total = int(g[lignin_col].notna().sum())
        print(f"Lignin audit: n={n_total} | converted rows={len(converted_rows)} | "
              f"hard<0={int(hard_low.sum())} hard>100={int(hard_high.sum())} | "
              f"soft_low={int(pd.Series(soft_low, index=g.index).fillna(False).sum())} "
              f"soft_high={int(pd.Series(soft_high, index=g.index).fillna(False).sum())} "
              f"| MAD outliers={int(outlier.sum())} | dropped={int(drop_mask.sum())}")
    except Exception:
        pass

    return (g.loc[~drop_mask].reset_index(drop=True) if not inplace else g, audit)

def audit_fix_lignin_by_tier(
    df,
    tier_col: str = "Tier",
    lignin_col: str = "Lignin",
    paper_col: str = "paper_title",
    on_hard_violation: str = "nan",   # "nan" | "clip" | "drop" | "ignore"
    convert_fraction_to_percent: bool = False,
    inplace: bool = False,
    dedupe_by_paper: bool = True
):
    """
    Minimal lignin sanity using ONLY Tier:
      - Hard bounds: Lignin must be in [0,100].
      - Soft envelopes:
          * 'lignin-rich' / 'technical' tiers → [40, 100] (soft)
          * all other tiers → [0, 45] (soft)
      - No family retagging, no new taxonomy — just flags and optional hard fix.

    Returns (df_out, audit_df)
    """
    import numpy as np, pandas as pd

    g = df if inplace else df.copy()
    if lignin_col not in g.columns:
        g[lignin_col] = np.nan
    if tier_col not in g.columns:
        g[tier_col] = np.nan
    if paper_col not in g.columns:
        g[paper_col] = np.nan

    L = pd.to_numeric(g[lignin_col], errors="coerce")

    if convert_fraction_to_percent:
        frac_mask = L.between(0, 1, inclusive="both")
        g.loc[frac_mask, lignin_col] = L.loc[frac_mask] * 100.0
        L = pd.to_numeric(g[lignin_col], errors="coerce")

    hard_low  = L.notna() & (L < 0)
    hard_high = L.notna() & (L > 100)

    if on_hard_violation == "clip":
        L = L.clip(0, 100)
        g[lignin_col] = L
        drop_mask = pd.Series(False, index=g.index)
    elif on_hard_violation == "nan":
        L = L.where(~(hard_low | hard_high), np.nan)
        g[lignin_col] = L
        drop_mask = pd.Series(False, index=g.index)
    elif on_hard_violation == "drop":
        drop_mask = (hard_low | hard_high)
    else:  # "ignore"
        drop_mask = pd.Series(False, index=g.index)

    t = g[tier_col].astype(str).str.lower()
    is_rich = (
        t.str.contains("lignin-rich|lignin rich|technical|kraft|organosolv|soda")
    )

    lo_soft = np.where(is_rich, 40.0, 0.0)
    hi_soft = np.where(is_rich, 100.0, 45.0)

    soft_low  = L.notna() & (L < lo_soft)
    soft_high = L.notna() & (L > hi_soft)

    kept = g.loc[~drop_mask].copy()

    issues = pd.DataFrame({
        paper_col: kept[paper_col],
        "Tier": kept[tier_col],
        "Lignin": kept[lignin_col],
        "hard_low":  hard_low.loc[kept.index].fillna(False),
        "hard_high": hard_high.loc[kept.index].fillna(False),
        "soft_low":  pd.Series(soft_low,  index=g.index).loc[kept.index].fillna(False),
        "soft_high": pd.Series(soft_high, index=g.index).loc[kept.index].fillna(False),
        "lo_soft":   pd.Series(lo_soft,   index=g.index).loc[kept.index],
        "hi_soft":   pd.Series(hi_soft,   index=g.index).loc[kept.index],
    })

    flagged = issues[issues[["hard_low","hard_high","soft_low","soft_high"]].any(axis=1)]

    if dedupe_by_paper and not flagged.empty:
        audit = (flagged
                 .groupby(paper_col, dropna=False)
                 .agg(rows_flagged=("Lignin","size"),
                      hard_low=("hard_low","sum"),
                      hard_high=("hard_high","sum"),
                      soft_low=("soft_low","sum"),
                      soft_high=("soft_high","sum"),
                      L_min=("Lignin","min"),
                      L_max=("Lignin","max"))
                 .reset_index()
                 .sort_values(["hard_high","hard_low","soft_high","soft_low","rows_flagged"], ascending=False))
    else:
        audit = flagged.reset_index(drop=True)

    try:
        print(f"Lignin-by-Tier audit: n={int(L.notna().sum())} | "
              f"hard<0={int(hard_low.sum())} hard>100={int(hard_high.sum())} | "
              f"soft_low={int(soft_low.sum())} soft_high={int(soft_high.sum())} | "
              f"dropped={int(drop_mask.sum())}")
    except Exception:
        pass

    return (kept if not inplace else g, audit)



def _auto_scale_to_percent(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, float]:
    """
    If values look like fractions (median between 0 and ~1.5), scale by 100.
    Returns (df_scaled, scale_factor).
    """
    sub = df.loc[:, cols].astype(float)
    med = np.nanmedian(sub.values)
    if np.isnan(med):
        return sub, 1.0
    if med <= 1.5:
        return sub * 100.0, 100.0
    return sub, 1.0

def get_yield_closure_stats(
    df: pd.DataFrame,
    cols: Sequence[str] = ("B_Y", "C_Y", "A_Y", "G_Y"),
    target: float = 100.0,
    tolerance: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute B_Y + C_Y + A_Y + G_Y closure (in %) and return quality-control stats.
    Only rows with all four components present are included.
    """
    have_all = df[list(cols)].notna().all(axis=1)
    if not have_all.any():
        return {
            "n_valid": 0,
            "message": "No rows have all yield components present.",
        }

    sub, scale = _auto_scale_to_percent(df.loc[have_all, list(cols)], cols)
    closure = sub.sum(axis=1)
    within = (closure >= (target - tolerance)) & (closure <= (target + tolerance))

    stats = {
        "scale_factor": scale,
        "n_valid": int(closure.shape[0]),
        "n_within_tol": int(within.sum()),
        "share_within_tol": float(within.mean()) if closure.shape[0] else np.nan,
        "target": target,
        "tolerance": tolerance,
        "min": float(np.nanmin(closure)),
        "p5": float(np.nanpercentile(closure, 5)),
        "median": float(np.nanmedian(closure)),
        "mean": float(np.nanmean(closure)),
        "p95": float(np.nanpercentile(closure, 95)),
        "max": float(np.nanmax(closure)),
        "n_low_outliers": int((closure < (target - tolerance)).sum()),
        "n_high_outliers": int((closure > (target + tolerance)).sum()),
    }
    stats["closure_series_pct"] = closure
    stats["valid_mask"] = have_all
    return stats

def plot_yield_closure_distribution(
    df: pd.DataFrame,
    cols: Sequence[str] = ("B_Y", "C_Y", "A_Y", "G_Y"),
    target: float = 100.0,
    tolerance: float = 5.0,
    bins: int = 40,
    figsize: Tuple[float, float] = (7.0, 4.0),
    show: bool = True,
    return_stats: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Plot histogram of closure = sum(B_Y, C_Y, A_Y, G_Y) in percent.
    Auto-detects fraction vs percent inputs.
    """
    stats = get_yield_closure_stats(df, cols=cols, target=target, tolerance=tolerance)
    if stats.get("n_valid", 0) == 0:
        print("No valid rows to plot (need all four yields present).")
        return stats if return_stats else None

    s = stats["closure_series_pct"]  # already in %
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(s.values, bins=bins)

    ax.axvline(target, linestyle="--")
    ax.axvline(target - tolerance, linestyle=":")
    ax.axvline(target + tolerance, linestyle=":")

    ax.set_title("Closure Distribution: B_Y + C_Y + A_Y + G_Y")
    ax.set_xlabel("Closure (%)")
    ax.set_ylabel("Count")

    footer = (
        f"n={stats['n_valid']}, within ±{tolerance}%: {stats['n_within_tol']} "
        f"({stats['share_within_tol']*100:.1f}%)\n"
        f"median={stats['median']:.2f}%, mean={stats['mean']:.2f}% "
        f"[p5={stats['p5']:.2f}%, p95={stats['p95']:.2f}%]"
    )
    ax.text(
        0.01, -0.25, footer,
        transform=ax.transAxes,
        ha="left", va="top"
    )
    plt.tight_layout()
    if show:
        plt.show()

    return stats if return_stats else None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Any, Optional, Dict

def _closure_series(df: pd.DataFrame, cols: Sequence[str] = ("B_Y","C_Y","A_Y","G_Y")):
    """
    Return closure (%) for rows where ALL specified yields are present.
    Auto-detects fraction vs percent on the valid subset.
    """
    y = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    valid = y.notna().all(axis=1)
    yv = y.loc[valid]
    if yv.empty:
        return pd.Series(dtype=float), 1.0, valid

    s = yv.sum(axis=1, skipna=False)
    med = s.median(skipna=True)
    if med <= 2.0:
        s = s * 100.0
        scale = 100.0
    else:
        scale = 1.0
    return s, scale, valid

def _closure_stats(s: pd.Series, target=100.0, tol=5.0) -> Dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n_valid": 0, "share_within_tol": np.nan, "mean": np.nan, "median": np.nan,
                "min": np.nan, "p5": np.nan, "p95": np.nan, "max": np.nan,
                "n_low_outliers": 0, "n_high_outliers": 0}
    within = s.between(target - tol, target + tol)
    return {
        "n_valid": int(s.size),
        "share_within_tol": float(within.mean()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "p5": float(np.percentile(s, 5)),
        "p95": float(np.percentile(s, 95)),
        "max": float(s.max()),
        "n_low_outliers": int((s < target - tol).sum()),
        "n_high_outliers": int((s > target + tol).sum()),
    }

def compare_yield_closures(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Reference",
    label_b: str = "Your DB",
    cols=("B_Y","C_Y","A_Y","G_Y"),
    bins=40,
    tol=5.0,
    target=100.0,
    show_table_on_plot=True,
    figsize=(9, 4.8),
    alpha=0.55,
):
    remap_columns = {'oil_yield': 'B_Y', 'char_yield': 'C_Y', 'aqueous_yield': 'A_Y', 'gas_yield': 'G_Y'}
    df_a = df_a.rename(columns=remap_columns)

    sa, scale_a, valid_a = _closure_series(df_a, cols)
    sb, scale_b, valid_b = _closure_series(df_b, cols)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(sa.values, bins=bins, alpha=alpha, label=f"{label_a} (n={sa.size})")
    ax.hist(sb.values, bins=bins, alpha=alpha, label=f"{label_b} (n={sb.size})")
    ax.axvline(target - tol, ls="--", lw=1.25)
    ax.axvline(target + tol, ls="--", lw=1.25)
    ax.set_title("Closure Distribution: B_Y + C_Y + A_Y + G_Y")
    ax.set_xlabel("Closure (%)")
    ax.set_ylabel("Count")
    ax.legend()

    stats_a = _closure_stats(sa, target=target, tol=tol)
    stats_b = _closure_stats(sb, target=target, tol=tol)

    ks = {}
    try:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(sa.values, sb.values, alternative="two-sided", mode="auto")
        ks = {"ks_stat": float(ks_stat), "ks_pvalue": float(ks_p)}
    except Exception:
        ks = {"ks_stat": None, "ks_pvalue": None}

    tbl = pd.DataFrame({
        "dataset": [label_a, label_b],
        "n_valid": [stats_a["n_valid"], stats_b["n_valid"]],
        "mean_%": [stats_a["mean"], stats_b["mean"]],
        "median_%": [stats_a["median"], stats_b["median"]],
        "p5_%": [stats_a["p5"], stats_b["p5"]],
        "p95_%": [stats_a["p95"], stats_b["p95"]],
        f"share_within_±{tol:.0f}%": [
            round(stats_a["share_within_tol"] * 100, 2) if stats_a["n_valid"] else np.nan,
            round(stats_b["share_within_tol"] * 100, 2) if stats_b["n_valid"] else np.nan,
        ],
        "low_outliers": [stats_a["n_low_outliers"], stats_b["n_low_outliers"]],
        "high_outliers": [stats_a["n_high_outliers"], stats_b["n_high_outliers"]],
    })

    print("\n=== Yield Closure Comparison ===")
    print(tbl.to_string(index=False))
    if ks["ks_stat"] is not None:
        print(f"\nKS two-sample test: D = {ks['ks_stat']:.3f}, p = {ks['ks_pvalue']:.3e}")

    if show_table_on_plot:
        txt = (
            f"{label_a}: n={stats_a['n_valid']}, μ={stats_a['mean']:.2f}%, "
            f"med={stats_a['median']:.2f}%, within ±{tol}%={stats_a['share_within_tol']*100:.1f}%\n"
            f"{label_b}: n={stats_b['n_valid']}, μ={stats_b['mean']:.2f}%, "
            f"med={stats_b['median']:.2f}%, within ±{tol}%={stats_b['share_within_tol']*100:.1f}%"
        )
        ax.text(0.01, 0.5, txt, transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9))

    plt.tight_layout()
    plt.show()

    return {
        "stats": {label_a: stats_a, label_b: stats_b},
        "ks_test": ks,
        "scale_factors": {label_a: scale_a, label_b: scale_b},
        "valid_masks": {label_a: valid_a, label_b: valid_b},
        "table": tbl,
    }

YCOLS_DEFAULT = ("B_Y", "C_Y", "A_Y", "G_Y")

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _coerce_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(_as_float)
        else:
            out[c] = np.nan
    return out

def _detect_percent_scale(s: pd.Series) -> float:
    """Heuristic: if median sum <= 2.0, assume inputs are fractions and multiply by 100."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 1.0
    return 100.0 if np.nanmedian(s) <= 2.0 else 1.0

def compute_missing_yield_by_difference(
    df: pd.DataFrame,
    cols: Sequence[str] = YCOLS_DEFAULT,
    target: float = 100.0,
    residual_tolerance: float = 5.0,
    enforce_bounds: Tuple[float, float] = (0.0, 100.0),
    plausible_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    basis_note_col: Optional[str] = "Provenance",  # put any basis notes there if you track them
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fill exactly one missing yield among B_Y, C_Y, A_Y, G_Y by difference to 'target'.

    Rules & guards:
    - Only rows with exactly ONE NaN among the four are eligible.
    - Known three must sum to ≤ target + residual_tolerance (otherwise reject).
    - Computed value must lie within 'enforce_bounds'.
    - If 'plausible_ranges' is provided (e.g., {'A_Y':(2,60), 'G_Y':(0,25)}), reject if outside.
    - Adds audit columns:
        'Yield_fill_method' ('by_difference' or NaN),
        'Yield_fill_target',
        'Yield_fill_residual_before',
        'Yield_fill_rejected_reason' (NaN if accepted)
    """
    df = _coerce_cols(df, cols).copy()

    for col in ["Yield_fill_method", "Yield_fill_target", "Yield_fill_residual_before", "Yield_fill_rejected_reason"]:
        if col not in df.columns:
            df[col] = np.nan

    sums_all4 = df[cols].dropna().sum(axis=1)
    scale = _detect_percent_scale(sums_all4)
    if scale == 100.0:
        df[cols] = df[cols] * 100.0
        if verbose:
            print("Detected fractional inputs; converted to percent (×100).")

    lower, upper = enforce_bounds
    fills_ok = 0
    fills_rej = 0

    arr = df[cols].to_numpy()
    mask_counts_na = np.isnan(arr).sum(axis=1)
    eligible = (mask_counts_na == 1)

    for idx in np.where(eligible)[0]:
        row_vals = arr[idx, :].copy()
        known_sum = np.nansum(row_vals)
        residual = target - known_sum

        miss_i = int(np.where(np.isnan(row_vals))[0][0])
        miss_col = cols[miss_i]

        if known_sum > (target + residual_tolerance):
            df.at[df.index[idx], "Yield_fill_rejected_reason"] = f"sum_known>{target}+tol ({known_sum:.2f}%)"
            fills_rej += 1
            continue

        proposed = residual

        if not (lower <= proposed <= upper):
            df.at[df.index[idx], "Yield_fill_rejected_reason"] = f"out_of_bounds({proposed:.2f}%)"
            fills_rej += 1
            continue

        if plausible_ranges and miss_col in plausible_ranges:
            lo, hi = plausible_ranges[miss_col]
            if not (lo <= proposed <= hi):
                df.at[df.index[idx], "Yield_fill_rejected_reason"] = f"{miss_col}_implausible({proposed:.2f}%, allowed {lo}-{hi}%)"
                fills_rej += 1
                continue

        df.at[df.index[idx], miss_col] = round(proposed, 3)
        df.at[df.index[idx], "Yield_fill_method"] = "by_difference"
        df.at[df.index[idx], "Yield_fill_target"] = target
        df.at[df.index[idx], "Yield_fill_residual_before"] = round(residual, 3)
        fills_ok += 1

    total_elig = int(eligible.sum())
    if verbose:
        print(f"[Yield completion] Eligible rows (exactly one missing): {total_elig}")
        print(f"[Yield completion] Filled: {fills_ok}, Rejected: {fills_rej}")

    closure = df[list(cols)].sum(axis=1)
    stats = {
        "eligible": total_elig,
        "filled": fills_ok,
        "rejected": fills_rej,
        "closure_after_mean": float(pd.to_numeric(closure, errors="coerce").dropna().mean()),
        "closure_after_median": float(pd.to_numeric(closure, errors="coerce").dropna().median()),
    }
    return df, stats

def fill_missing_yields_conservative(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    plausible = {
        "B_Y": (0, 70),    # biocrude often 5–60%
        "C_Y": (0, 60),    # char often 0–40%
        "A_Y": (2, 70),    # aqueous can be large but rarely <2% on dry basis
        "G_Y": (0, 40),    # gas typically 0–25%
    }
    return compute_missing_yield_by_difference(
        df,
        cols=YCOLS_DEFAULT,
        target=100.0,
        residual_tolerance=5.0,
        enforce_bounds=(0.0, 100.0),
        plausible_ranges=plausible,
        verbose=verbose
    )




def list_by_gt100(df: pd.DataFrame, ycol: str = "B_Y") -> pd.DataFrame:
    s = pd.to_numeric(df[ycol], errors="coerce")
    mask = s > 100.0 + 1e-9
    out = (
        df.loc[mask, ["paper_title", "DOI", "year", "Feedstock", "Family", ycol]]
          .sort_values(ycol, ascending=False)
          .reset_index(drop=True)
    )
    print(f"Rows with {ycol} > 100%: {len(out)}")
    return out



def drop_by_gt100(df: pd.DataFrame, ycol: str = "B_Y", confirm: bool = False) -> pd.DataFrame:
    s = pd.to_numeric(df[ycol], errors="coerce")
    mask = s > 100.0 + 1e-9
    n = int(mask.sum())
    if not confirm:
        print(f"[DRY RUN] Would drop {n} row(s) where {ycol} > 100%. Pass confirm=True to apply.")
        return df
    print(f"Dropping {n} row(s) where {ycol} > 100%.")
    return df.loc[~mask].copy()


def flag_by_gt100(df: pd.DataFrame, ycol: str = "B_Y", flag_col: str = "QA_flag") -> pd.DataFrame:
    """Flag rows where ycol > 100 (percent basis) in a string column `flag_col`."""
    out = df.copy()
    s = pd.to_numeric(out.get(ycol), errors="coerce")
    mask = s > 100.0 + 1e-9

    if flag_col not in out.columns:
        out[flag_col] = ""

    base = out[flag_col].astype("string").fillna("")
    suffix = f"{ycol}>100%"

    new_vals = np.where(base.loc[mask].str.len() > 0,
                        base.loc[mask] + "; " + suffix,
                        suffix)

    out.loc[mask, flag_col] = new_vals
    print(f"Flagged {int(mask.sum())} row(s) in '{flag_col}'.")
    return out



