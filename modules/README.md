# Visualization and QA Modules

This directory contains Python modules for data visualization and quality assurance.

## Modules

### van_krevelen_plotter.py
Plot Van Krevelen diagrams (H/C vs O/C) for feedstock and product characterization.

**Key function:**
```python
from van_krevelen_plotter import plot_van_krevelen_diagram

fig = plot_van_krevelen_diagram(
    df=df, 
    family_col='Family_std', 
    oc_col='O_C_feed_molar', 
    hc_col='H_C_feed_molar',
    figsize=(12, 8), 
    show_reference_lines=True, 
    show_lignin_zone=True
)
```

### family_normalizer.py
Visualize feedstock family distributions with circular/pie charts.

**Key function:**
```python
from family_normalizer import plot_family_circular_distribution

res = plot_family_circular_distribution(
    df,
    family_col='Family_std',
    min_share_to_label=0.03,
    top_n=10,
    title='Feedstock Families'
)
```

### qa_envelopes.py
Quality assurance checks for yields, energy balances, and carbon balances.

**Key functions:**
```python
import qa_envelopes

qa_envelopes.run_basic_qc(df)
qa_envelopes.plot_energy_carbon_envelopes(df)
qa_envelopes.plot_yield_envelopes(df)
```

## Usage

See `notebooks/load_and_inspect.ipynb` for complete examples.
