import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Any

# Default family markers for consistent visualization
DEFAULT_FAMILY_MARKERS = {
    "Woody Biomass / Hardwood": "o",      # circle
    "Woody Biomass / Softwood": "s",      # square
    "Woody Biomass / Unspecified": "D",   # diamond
    "Herbaceous Biomass": "^",            # triangle up
    "Agricultural Residues": "v",         # triangle down
    "Aquatic Biomass": "P",               # plus (filled)
    "Lignin-rich Streams": "X",           # X
    "Model Compound": "*",                # star
    "Mixed Biomass": "p",                 # pentagon
    "Waste Biomass / Sludge": "h",        # hexagon
    "Animal Manure": "<",                 # triangle left
    "Waste Wood / Construction": ">",     # triangle right
    "Unknown": "+"                        # plus marker for unknown
}

def plot_van_krevelen_diagram(
    df: pd.DataFrame,
    family_col: str = "Family",
    oc_col: str = "O/C",
    hc_col: str = "H/C",
    figsize: Tuple[int, int] = (12, 8),
    family_markers: Optional[Dict[str, str]] = None,
    color_palette: str = "Set2",
    title: Optional[str] = None,
    show_reference_lines: bool = True,
    show_lignin_zone: bool = True,
    show_marker_legend: bool = True,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a Van Krevelen diagram with family-specific markers and colors.
    
    Args:
        df: DataFrame containing the data
        family_col: Column name for feedstock families
        oc_col: Column name for O/C ratios
        hc_col: Column name for H/C ratios
        figsize: Figure size (width, height)
        family_markers: Custom marker mapping for families
        color_palette: Seaborn color palette name
        title: Custom title for the plot
        show_reference_lines: Whether to show reference lines
        show_lignin_zone: Whether to show lignin zone rectangle
        show_marker_legend: Whether to print marker legend
        legend_kwargs: Additional legend parameters
        **kwargs: Additional parameters for scatter plots
        
    Returns:
        matplotlib Figure object
    """
    # Set up style
    sns.set(style="whitegrid")
    
    # Filter data
    plot_df = df.dropna(subset=[oc_col, hc_col]).copy()
    
    if len(plot_df) == 0:
        raise ValueError(f"No valid data found for columns {oc_col} and {hc_col}")
    
    # Use default markers if not provided
    if family_markers is None:
        family_markers = DEFAULT_FAMILY_MARKERS.copy()
    
    # Get unique families and colors
    families = plot_df[family_col].unique()
    colors = sns.color_palette(color_palette, n_colors=len(families))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default scatter plot parameters
    scatter_defaults = {
        's': 100,
        'edgecolor': 'black',
        'linewidth': 0.5,
        'alpha': 0.7
    }
    scatter_defaults.update(kwargs)
    
    # Create scatter plots for each family
    for i, family in enumerate(families):
        family_data = plot_df[plot_df[family_col] == family]
        marker = family_markers.get(family, "o")  # default to circle if not found
        
        # Prepare scatter parameters (avoid edgecolor for unfilled markers)
        scatter_params = scatter_defaults.copy()
        if marker in ['+', 'x', '|', '_', 'X', 'P', 1, 2, 3, 4]:  # unfilled markers
            scatter_params.pop('edgecolor', None)
            scatter_params.pop('linewidth', None)
        
        ax.scatter(
            family_data[oc_col], 
            family_data[hc_col],
            c=[colors[i]], 
            marker=marker,
            label=family,
            **scatter_params
        )
    
    # Add reference lines if requested
    if show_reference_lines:
        ax.axvline(0.5, color='cyan', linestyle='--', linewidth=1, alpha=0.8, label="O/C = 0.5")
        ax.axvline(0.65, color='magenta', linestyle=':', linewidth=1, alpha=0.8, label="O/C = 0.65")
        ax.axhline(1.3, color='cyan', linestyle='--', linewidth=1, alpha=0.8, label="H/C = 1.3")
        ax.axhline(1.6, color='magenta', linestyle=':', linewidth=1, alpha=0.8, label="H/C = 1.6")
    
    # Add lignin zone if requested
    if show_lignin_zone:
        lignin_rect = plt.Rectangle((0.3, 1.0), 0.2, 0.3, color='lightblue', 
                                  alpha=0.2, label="Typical Lignin Zone")
        ax.add_patch(lignin_rect)
    
    # Set title
    if title is None:
        title = "Van Krevelen Diagram – Feedstock classification by family"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set axis labels
    ax.set_xlabel(f"{oc_col} (atomic ratio)", fontsize=12)
    ax.set_ylabel(f"{hc_col} (atomic ratio)", fontsize=12)
    
    # Configure legend
    legend_defaults = {
        'bbox_to_anchor': (1.05, 1),
        'loc': 'upper left',
        'ncol': 1,
        'fontsize': 10,
        'title': "Feedstock Family",
        'frameon': True,
        'fancybox': True,
        'shadow': True
    }
    if legend_kwargs:
        legend_defaults.update(legend_kwargs)
    
    ax.legend(**legend_defaults)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Print marker legend if requested
    if show_marker_legend:
        print("=== MARKER LEGEND FOR FAMILIES ===")
        for family, marker in family_markers.items():
            if family in families:
                print(f"{marker:2s} = {family}")

    
    return fig

def plot_van_krevelen_simple(
    df: pd.DataFrame,
    family_col: str = "Family",
    oc_col: str = "O/C", 
    hc_col: str = "H/C",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a simplified Van Krevelen diagram with minimal styling.
    
    Args:
        df: DataFrame containing the data
        family_col: Column name for feedstock families
        oc_col: Column name for O/C ratios
        hc_col: Column name for H/C ratios
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    plot_df = df.dropna(subset=[oc_col, hc_col]).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Simple scatter plot colored by family
    families = plot_df[family_col].unique()
    colors = sns.color_palette("tab10", n_colors=len(families))
    
    for i, family in enumerate(families):
        family_data = plot_df[plot_df[family_col] == family]
        ax.scatter(family_data[oc_col], family_data[hc_col], 
                  c=[colors[i]], label=family, alpha=0.7, s=50)
    
    ax.set_xlabel(f"{oc_col} (atomic ratio)")
    ax.set_ylabel(f"{hc_col} (atomic ratio)")
    ax.set_title("Van Krevelen Diagram")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def get_family_marker_summary(df: pd.DataFrame, family_col: str = "Family") -> Dict[str, str]:
    """
    Get a summary of which families are present in the data and their assigned markers.
    
    Args:
        df: DataFrame containing the data
        family_col: Column name for feedstock families
        
    Returns:
        Dictionary mapping family names to marker symbols
    """
    families = df[family_col].dropna().unique()
    return {family: DEFAULT_FAMILY_MARKERS.get(family, "o") for family in families}

def add_custom_family_markers(custom_markers: Dict[str, str]) -> Dict[str, str]:
    """
    Add or update family markers with custom definitions.
    
    Args:
        custom_markers: Dictionary of family -> marker symbol mappings
        
    Returns:
        Updated marker dictionary
    """
    updated_markers = DEFAULT_FAMILY_MARKERS.copy()
    updated_markers.update(custom_markers)
    return updated_markers