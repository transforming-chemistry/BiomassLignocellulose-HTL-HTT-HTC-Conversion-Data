"""
Yield Closure Comparison Module

Compare mass yield closure between two datasets to assess data quality
and completeness of yield reporting (bio-oil, char, aqueous, gas).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_yield_closures(
    df_a,
    df_b,
    label_a="Dataset A",
    label_b="Dataset B",
    cols=("Yield_biooil_wt_pct", "Yield_char_wt_pct", 
          "Yield_aqueous_wt_pct", "Yield_gas_wt_pct"),
    alt_cols=("Yield_biooil_wt_pct", "Yield_char_wt_pct", 
              "Yield_gas_water_wt_pct"),
    bins=36,
    tol=5.0,
    figsize=(14, 5)
):
    """
    Compare yield closure distributions between two datasets.
    
    Mass closure is computed as the sum of all product yields. Ideally
    should be close to 100%. This function visualizes the distribution
    of closure values and identifies outliers.
    
    Parameters
    ----------
    df_a : pd.DataFrame
        First dataset for comparison
    df_b : pd.DataFrame
        Second dataset for comparison
    label_a : str
        Label for first dataset
    label_b : str
        Label for second dataset
    cols : tuple of str
        Primary column names for yields (bio-oil, char, aqueous, gas)
    alt_cols : tuple of str
        Alternative column names if primary not available
        (bio-oil, char, combined gas+water)
    bins : int
        Number of bins for histogram
    tol : float
        Tolerance in % for flagging closure outliers (e.g., 5.0 means
        samples with closure < 95% or > 105% are flagged)
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns
    -------
    dict
        Results containing:
        - closure_a: Series with closure values for dataset A
        - closure_b: Series with closure values for dataset B
        - stats_a: dict with statistics for dataset A
        - stats_b: dict with statistics for dataset B
        - outliers_a: indices of outliers in dataset A
        - outliers_b: indices of outliers in dataset B
    """
    
    def compute_closure(df, cols_primary, cols_alt):
        """Compute mass closure, trying primary columns first, then alternatives."""
        
        available_primary = [c for c in cols_primary if c in df.columns]
        available_alt = [c for c in cols_alt if c in df.columns]
        
        if len(available_primary) >= 3:
            closure = df[available_primary].sum(axis=1)
            used_cols = available_primary
        elif len(available_alt) >= 2:
            closure = df[available_alt].sum(axis=1)
            used_cols = available_alt
        else:
            closure = pd.Series(np.nan, index=df.index)
            used_cols = []
        
        return closure.dropna(), used_cols
    
    closure_a, cols_used_a = compute_closure(df_a, cols, alt_cols)
    closure_b, cols_used_b = compute_closure(df_b, cols, alt_cols)
    
    if len(closure_a) == 0:
        raise ValueError(f"{label_a}: No valid yield data found. "
                        f"Required columns: {cols} or {alt_cols}")
    if len(closure_b) == 0:
        raise ValueError(f"{label_b}: No valid yield data found. "
                        f"Required columns: {cols} or {alt_cols}")
    
    stats_a = {
        'mean': closure_a.mean(),
        'median': closure_a.median(),
        'std': closure_a.std(),
        'min': closure_a.min(),
        'max': closure_a.max(),
        'n_samples': len(closure_a),
        'columns_used': cols_used_a
    }
    
    stats_b = {
        'mean': closure_b.mean(),
        'median': closure_b.median(),
        'std': closure_b.std(),
        'min': closure_b.min(),
        'max': closure_b.max(),
        'n_samples': len(closure_b),
        'columns_used': cols_used_b
    }
    
    outliers_a = closure_a[(closure_a < 100 - tol) | (closure_a > 100 + tol)].index
    outliers_b = closure_b[(closure_b < 100 - tol) | (closure_b > 100 + tol)].index
    
    print("="*70)
    print("YIELD CLOSURE COMPARISON")
    print("="*70)
    print(f"\n{label_a}:")
    print(f"  Samples:        {stats_a['n_samples']:,}")
    print(f"  Columns used:   {', '.join(stats_a['columns_used'])}")
    print(f"  Mean closure:   {stats_a['mean']:.1f}%")
    print(f"  Median closure: {stats_a['median']:.1f}%")
    print(f"  Std dev:        {stats_a['std']:.1f}%")
    print(f"  Range:          {stats_a['min']:.1f}% – {stats_a['max']:.1f}%")
    print(f"  Outliers:       {len(outliers_a):,} ({100*len(outliers_a)/len(closure_a):.1f}%)")
    
    print(f"\n{label_b}:")
    print(f"  Samples:        {stats_b['n_samples']:,}")
    print(f"  Columns used:   {', '.join(stats_b['columns_used'])}")
    print(f"  Mean closure:   {stats_b['mean']:.1f}%")
    print(f"  Median closure: {stats_b['median']:.1f}%")
    print(f"  Std dev:        {stats_b['std']:.1f}%")
    print(f"  Range:          {stats_b['min']:.1f}% – {stats_b['max']:.1f}%")
    print(f"  Outliers:       {len(outliers_b):,} ({100*len(outliers_b)/len(closure_b):.1f}%)")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].hist(closure_a, bins=bins, alpha=0.7, label=label_a, 
                 edgecolor='black', color='cornflowerblue')
    axes[0].axvline(100, color='red', linestyle='--', linewidth=2, 
                   label='Perfect closure (100%)')
    axes[0].axvline(stats_a['mean'], color='blue', linestyle='-', 
                   linewidth=2, label=f"Mean: {stats_a['mean']:.1f}%")
    axes[0].axvspan(100-tol, 100+tol, alpha=0.2, color='green', 
                   label=f'Acceptable range (±{tol}%)')
    axes[0].set_xlabel('Mass Closure (%)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'{label_a}\n(n={len(closure_a):,})', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].hist(closure_b, bins=bins, alpha=0.7, label=label_b, 
                 edgecolor='black', color='lightcoral')
    axes[1].axvline(100, color='red', linestyle='--', linewidth=2, 
                   label='Perfect closure (100%)')
    axes[1].axvline(stats_b['mean'], color='darkred', linestyle='-', 
                   linewidth=2, label=f"Mean: {stats_b['mean']:.1f}%")
    axes[1].axvspan(100-tol, 100+tol, alpha=0.2, color='green', 
                   label=f'Acceptable range (±{tol}%)')
    axes[1].set_xlabel('Mass Closure (%)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'{label_b}\n(n={len(closure_b):,})', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'closure_a': closure_a,
        'closure_b': closure_b,
        'stats_a': stats_a,
        'stats_b': stats_b,
        'outliers_a': outliers_a,
        'outliers_b': outliers_b
    }
