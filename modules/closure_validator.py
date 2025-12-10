"""
Validation and visualization of mass balance closures for feedstock composition.

Checks polymer closure (Lignin + Cellulose + Hemicellulose) and 
elemental closure (C + H + O + N + S + Ash) for data quality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def validate_closures(df, 
                     upper_tol_LCH_soft=105.0,
                     upper_tol_LCH_hard=110.0,
                     lower_tol_LCH=-1e-6,
                     show_plots=True,
                     show_stats=True):
    """
    Validate and flag polymer and elemental closures in feedstock data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with composition columns
    upper_tol_LCH_soft : float
        Soft upper tolerance for polymer sum (default 105%)
    upper_tol_LCH_hard : float
        Hard upper tolerance for polymer sum (default 110%)
    lower_tol_LCH : float
        Lower tolerance for polymer sum (default -1e-6)
    show_plots : bool
        Generate closure distribution plots
    show_stats : bool
        Print closure statistics
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added closure columns and flags
    """
    
    df = df.copy()
    
    triplet = ["Lignin_feed_wt_pct", "Hemicellulose_feed_wt_pct", "Cellulose_feed_wt_pct"]
    elem_cols = ["C_feed_wt_pct", "H_feed_wt_pct", "O_feed_wt_pct", "N_feed_wt_pct", "S_feed_wt_pct", "Ash_feed_wt_pct"]
    
   
    for col in triplet + elem_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df")
    
  
    df["sum_LCH"] = df[triplet].sum(axis=1, min_count=1)
    df["sum_CHONSAsh"] = df[elem_cols].sum(axis=1, min_count=1)
    df["sum_CHONS"] = df[["C_feed_wt_pct", "H_feed_wt_pct", "O_feed_wt_pct", "N_feed_wt_pct", "S_feed_wt_pct"]].sum(axis=1, min_count=1)
    

    mask_LCH_soft = df["sum_LCH"] > upper_tol_LCH_soft
    mask_LCH_hard = df["sum_LCH"] > upper_tol_LCH_hard
    mask_LCH_low = df["sum_LCH"] < lower_tol_LCH
    
    df["LCH_flag"] = "OK"
    df.loc[mask_LCH_soft & ~mask_LCH_hard, "LCH_flag"] = "LCH>105"
    df.loc[mask_LCH_hard, "LCH_flag"] = "LCH>110"
    df.loc[mask_LCH_low, "LCH_flag"] = "LCH<0"
    
 
    if show_stats:
        print("=== Polymer closure (Lignin + Cellulose + Hemicellulose) ===")
        print(df["sum_LCH"].describe())
        print(f"\nFlag counts:")
        print(df["LCH_flag"].value_counts())
        
        print(f"\nTop 20 rows with highest sum_LCH:")
        cols_view = ["DOI", "Feedstock", "Lignin_feed_wt_pct", "Cellulose_feed_wt_pct", "Hemicellulose_feed_wt_pct", "sum_LCH", "LCH_flag"]
        if "Tier" in df.columns:
            cols_view.insert(2, "Tier")
        print(df.sort_values("sum_LCH", ascending=False)[cols_view].head(20).to_string(index=False))
 
    if show_plots:
        _plot_closures(df)
    
    return df


def _plot_closures(df):
    """Generate closure distribution plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Polymer 
    ax = axes[0]
    vals_LCH = df["sum_LCH"].dropna()
    ax.hist(vals_LCH, bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(100.0, color='red', linestyle='--', linewidth=2, label='100%')
    ax.set_xlabel("Lignin + Cellulose + Hemicellulose [wt%]", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Polymer Closure (L+C+H)", fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Elemental 
    ax = axes[1]
    vals_elem = df["sum_CHONSAsh"].dropna()
    ax.hist(vals_elem, bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(100.0, color='red', linestyle='--', linewidth=2, label='100%')
    ax.set_xlabel("C + H + O + N + S + Ash [wt%]", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Elemental Closure (CHONS+Ash)", fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["sum_CHONSAsh"], df["sum_LCH"], alpha=0.4, s=10)
    ax.axvline(100.0, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(100.0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Elemental Closure (C+H+O+N+S+Ash) [wt%]", fontsize=11)
    ax.set_ylabel("Polymer Closure (L+C+H) [wt%]", fontsize=11)
    ax.set_title("Polymer vs Elemental Closure", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_closure_summary(df):
    """
    Get summary statistics for closure validation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with closure columns (must run validate_closures first)
    
    Returns
    -------
    dict
        Summary statistics
    """
    
    if "sum_LCH" not in df.columns:
        raise ValueError("Run validate_closures() first to compute closure sums")
    
    summary = {
        "n_rows": len(df),
        "LCH_mean": df["sum_LCH"].mean(),
        "LCH_median": df["sum_LCH"].median(),
        "LCH_std": df["sum_LCH"].std(),
        "LCH_flags": df["LCH_flag"].value_counts().to_dict(),
        "elemental_mean": df["sum_CHONSAsh"].mean(),
        "elemental_median": df["sum_CHONSAsh"].median(),
        "elemental_std": df["sum_CHONSAsh"].std(),
    }
    
    return summary
