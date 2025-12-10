# feature_distributions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple

def _coerce_num(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=float)
    return pd.to_numeric(s, errors="coerce")

def _available(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def _default_hard_bounds() -> Dict[str, Tuple[float, float]]:
    b = {
        # feed chemistry (wt%)
        "C_feed_wt_pct": (0.0, 100.0),
        "H_feed_wt_pct": (0.0, 100.0),
        "O_feed_wt_pct": (0.0, 100.0),
        "N_feed_wt_pct": (0.0, 100.0),
        "S_feed_wt_pct": (0.0, 100.0),
        "Ash_feed_wt_pct": (0.0, 100.0),
        "Lignin_feed_wt_pct": (0.0, 100.0),
        "Cellulose_feed_wt_pct": (0.0, 100.0),
        "Hemicellulose_feed_wt_pct": (0.0, 100.0),
        # process
        "T_reaction_C": (0.0, 1000.0),
        "t_residence_min": (0.0, 1e5),
        "IC_feed_wt_pct_slurry": (0.0, 100.0),
        "pressure_reaction_MPa": (0.0, 300.0),
        "cat_biomass_ratio_kg_kg": (0.0, 100.0),
    }
    return b

def _logical_borders_1d(x: pd.Series, p_lo: float, p_hi: float, hard: Tuple[float, float]) -> Tuple[float, float]:
    x = _coerce_num(x).dropna()
    if len(x) == 0:
        return np.nan, np.nan
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    lo = max(lo, hard[0])
    hi = min(hi, hard[1])
    if lo > hi:
        lo, hi = sorted([lo, hi])
    return float(lo), float(hi)

def calculate_logical_borders(
    df: pd.DataFrame,
    feature: str,
    process_col: str = "process_subtype",
    percentile_low: float = 2.5,
    percentile_high: float = 97.5,
    hard_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Tuple[float, float]]:
    hb = _default_hard_bounds() if hard_bounds is None else hard_bounds
    hb_feat = hb.get(feature, (-np.inf, np.inf))
    borders = {}
    if process_col not in df.columns:
        x = _coerce_num(df.get(feature)).dropna()
        if len(x):
            borders["_all_"] = _logical_borders_1d(x, percentile_low, percentile_high, hb_feat)
        return borders
    for proc in sorted(df[process_col].dropna().unique()):
        x = df.loc[df[process_col] == proc, feature]
        lo, hi = _logical_borders_1d(x, percentile_low, percentile_high, hb_feat)
        if not np.isnan(lo) and not np.isnan(hi):
            borders[proc] = (lo, hi)
    return borders

def plot_feature_distribution_with_borders(
    df: pd.DataFrame,
    feature: str,
    process_col: str = "process_subtype",
    bins: int = 40,
    figsize: Tuple[float, float] = (10, 5),
    percentile_low: float = 2.5,
    percentile_high: float = 97.5,
    hard_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    show_borders: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    data = _coerce_num(df.get(feature)).dropna()
    if len(data) == 0:
        ax.text(0.5, 0.5, f"No numeric data for {feature}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)
    if show_borders:
        borders = calculate_logical_borders(
            df, feature, process_col, percentile_low, percentile_high, hard_bounds
        )
        styles = ["--", ":", "-.", (0, (3, 2, 1, 2))]
        for k, (proc, (lo, hi)) in enumerate(borders.items()):
            ls = styles[k % len(styles)]
            ax.axvline(lo, linestyle=ls, linewidth=1.5)
            ax.axvline(hi, linestyle=ls, linewidth=1.5, label=f"{proc}: [{lo:.2f}, {hi:.2f}]")

    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.set_title(f"{feature} — distribution with logical borders")
    if show_borders:
        ax.legend(fontsize=8, frameon=False)

    gmin, gmax = data.min(), data.max()
    ax.text(0.02, 0.98, f"Global: [{gmin:.2f}, {gmax:.2f}]", transform=ax.transAxes, fontsize=9, va="top")
    ax.grid(axis="y", alpha=0.3)
    return ax

def plot_all_feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    process_col: str = "process_subtype",
    bins: int = 40,
    ncols: int = 3,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    percentile_low: float = 2.5,
    percentile_high: float = 97.5,
    hard_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    show_borders: bool = True
) -> plt.Figure:
    if features is None:
        features = [
            "C_feed_wt_pct","H_feed_wt_pct","O_feed_wt_pct","N_feed_wt_pct","S_feed_wt_pct","Ash_feed_wt_pct",
            "Lignin_feed_wt_pct","Cellulose_feed_wt_pct","Hemicellulose_feed_wt_pct",
            "T_reaction_C","t_residence_min","IC_feed_wt_pct_slurry","pressure_reaction_MPa","cat_biomass_ratio_kg_kg"
        ]
    feats = _available(df, features)
    nrows = (len(feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])
    for i, f in enumerate(feats):
        plot_feature_distribution_with_borders(
            df, f, process_col=process_col, bins=bins,
            percentile_low=percentile_low, percentile_high=percentile_high,
            hard_bounds=hard_bounds, show_borders=show_borders, ax=axes[i]
        )
    for j in range(len(feats), len(axes)):
        axes[j].set_axis_off()
    plt.tight_layout()
    return fig

def plot_feature_distributions_by_process(
    df: pd.DataFrame,
    feature: str,
    process_col: str = "process_subtype",
    bins: int = 30,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    procs = sorted(df[process_col].dropna().unique()) if process_col in df.columns else ["_all_"]
    ncols = 2
    nrows = (len(procs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])
    for i, p in enumerate(procs):
        if p == "_all_":
            data = _coerce_num(df.get(feature)).dropna()
            title = "_all_"
        else:
            data = _coerce_num(df.loc[df[process_col] == p, feature]).dropna()
            title = p
        if len(data):
            axes[i].hist(data, bins=bins, edgecolor="black", alpha=0.7)
            mean_v, med_v = data.mean(), data.median()
            axes[i].axvline(mean_v, linestyle="--", linewidth=1.5, label=f"Mean {mean_v:.2f}")
            axes[i].axvline(med_v, linestyle=":", linewidth=1.5, label=f"Median {med_v:.2f}")
            axes[i].legend(fontsize=8, frameon=False)
        else:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_axis_off()
            continue
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"{title} (n={len(data)})")
        axes[i].grid(axis="y", alpha=0.3)
    for j in range(len(procs), len(axes)):
        axes[j].set_axis_off()
    plt.tight_layout()
    return fig

def plot_features_serial(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    process_col: str = "process_subtype",
    bins: int = 40,
    figsize: Tuple[float, float] = (12, 5),
    percentile_low: float = 2.5,
    percentile_high: float = 97.5,
    hard_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    show_borders: bool = True
) -> None:
    if features is None:
        features = [
            "C_feed_wt_pct","H_feed_wt_pct","O_feed_wt_pct","N_feed_wt_pct","S_feed_wt_pct","Ash_feed_wt_pct",
            "Lignin_feed_wt_pct","Cellulose_feed_wt_pct","Hemicellulose_feed_wt_pct",
            "T_reaction_C","t_residence_min","IC_feed_wt_pct_slurry","pressure_reaction_MPa","cat_biomass_ratio_kg_kg"
        ]
    feats = _available(df, features)
    print(f"Serial plots: {len(feats)} features")
    if process_col in df.columns:
        print(f"Processes: {sorted(df[process_col].dropna().unique())}")
    for f in feats:
        data = _coerce_num(df.get(f)).dropna()
        if not len(data):
            print(f"Skipping {f}: no numeric data")
            continue
        _, ax = plt.subplots(figsize=figsize)
        plot_feature_distribution_with_borders(
            df, f, process_col=process_col, bins=bins,
            percentile_low=percentile_low, percentile_high=percentile_high,
            hard_bounds=hard_bounds, show_borders=show_borders, ax=ax
        )
        plt.show()
        print(f"{f}: n={len(data)} range=[{data.min():.4g}, {data.max():.4g}]")

def run_feature_distribution_analysis(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    process_col: str = "process_subtype",
    show_combined: bool = True,
    show_individual: bool = False,
    show_serial: bool = False,
    bins: int = 40,
    percentile_low: float = 2.5,
    percentile_high: float = 97.5,
    hard_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> None:
    if features is None:
        features = [
            "C_feed_wt_pct","H_feed_wt_pct","O_feed_wt_pct","N_feed_wt_pct","S_feed_wt_pct","Ash_feed_wt_pct",
            "Lignin_feed_wt_pct","Cellulose_feed_wt_pct","Hemicellulose_feed_wt_pct",
            "T_reaction_C","t_residence_min","IC_feed_wt_pct_slurry","pressure_reaction_MPa","cat_biomass_ratio_kg_kg"
        ]
    feats = _available(df, features)
    print(f"Features: {len(feats)} | Total rows: {len(df)}")
    if process_col in df.columns:
        print(f"Processes: {sorted(df[process_col].dropna().unique())}")
    if show_serial:
        plot_features_serial(
            df, feats, process_col=process_col, bins=bins,
            percentile_low=percentile_low, percentile_high=percentile_high,
            hard_bounds=hard_bounds, show_borders=True
        )
    elif show_combined:
        fig = plot_all_feature_distributions(
            df, feats, process_col=process_col, bins=bins,
            percentile_low=percentile_low, percentile_high=percentile_high,
            hard_bounds=hard_bounds, show_borders=True
        )
        plt.show()
    if show_individual:
        for f in feats:
            fig = plot_feature_distributions_by_process(df, f, process_col=process_col)
            plt.show()
