"""
Van Krevelen Clustering Analysis Module

Provides K-means clustering analysis on Van Krevelen (H/C vs O/C) space
with automatic optimal cluster selection using silhouette score.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def complete_van_krevelen_clustering_analysis(
    df,
    oc_col="O_C_feed_molar",
    hc_col="H_C_feed_molar",
    family_col="Family_std",
    k_min=2,
    k_max=10,
    random_state=42,
    show_plots=True,
    show_tables=True,
    verbose=True
):
    """
    Complete Van Krevelen clustering analysis with optimal k selection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with feedstock composition data
    oc_col : str
        Column name for O/C molar ratio
    hc_col : str
        Column name for H/C molar ratio
    family_col : str
        Column name for feedstock family classification
    k_min : int
        Minimum number of clusters to test
    k_max : int
        Maximum number of clusters to test
    random_state : int
        Random seed for reproducibility
    show_plots : bool
        Whether to display plots
    show_tables : bool
        Whether to display summary tables
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    df : pd.DataFrame
        Input dataframe with added 'VK_cluster' column
    results : dict
        Dictionary containing:
        - k_best: optimal number of clusters
        - silhouette_scores: scores for each k tested
        - vk_data: cleaned dataframe with VK coordinates
        - centroids: cluster centroids
        - model: fitted KMeans model
    """
    
    vk_data = df[[oc_col, hc_col, family_col]].dropna().copy()
    
    if len(vk_data) == 0:
        raise ValueError(f"No valid data found in columns {oc_col}, {hc_col}")
    
    if verbose:
        print(f"Van Krevelen clustering analysis on {len(vk_data):,} samples")
        print(f"Testing k = {k_min} to {k_max} clusters\n")
    
    X = vk_data[[oc_col, hc_col]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    silhouette_scores = {}
    models = {}
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = score
        models[k] = kmeans
        
        if verbose:
            print(f"k={k:2d}: silhouette score = {score:.4f}")
    
    k_best = max(silhouette_scores, key=silhouette_scores.get)
    
    if verbose:
        print(f"\n✓ Optimal k = {k_best} (silhouette = {silhouette_scores[k_best]:.4f})")
    
    best_model = models[k_best]
    vk_data['VK_cluster'] = best_model.fit_predict(X_scaled)
    
    centroids_scaled = best_model.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids, columns=[oc_col, hc_col])
    centroids_df.index.name = 'Cluster'
    
    df_with_clusters = df.copy()
    df_with_clusters.loc[vk_data.index, 'VK_cluster'] = vk_data['VK_cluster']
    
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        k_values = list(silhouette_scores.keys())
        scores = list(silhouette_scores.values())
        axes[0].plot(k_values, scores, 'o-', linewidth=2, markersize=8)
        axes[0].axvline(k_best, color='red', linestyle='--', 
                       label=f'Optimal k={k_best}')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[0].set_ylabel('Silhouette Score', fontsize=11)
        axes[0].set_title('Elbow Plot: Silhouette Score vs k', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        scatter = axes[1].scatter(
            vk_data[oc_col], 
            vk_data[hc_col],
            c=vk_data['VK_cluster'],
            cmap='tab10',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        axes[1].scatter(
            centroids[:, 0],
            centroids[:, 1],
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
        axes[1].set_xlabel('O/C (molar)', fontsize=11)
        axes[1].set_ylabel('H/C (molar)', fontsize=11)
        axes[1].set_title(f'Van Krevelen Clusters (k={k_best})', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.colorbar(scatter, ax=axes[1], label='Cluster ID')
        
        plt.tight_layout()
        plt.show()
    
    if show_tables:
        print("\n" + "="*60)
        print("CLUSTER CENTROIDS (Van Krevelen space)")
        print("="*60)
        print(centroids_df.to_string())
        
        print("\n" + "="*60)
        print("CLUSTER SIZE DISTRIBUTION")
        print("="*60)
        cluster_counts = vk_data['VK_cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            pct = 100 * count / len(vk_data)
            print(f"Cluster {cluster_id}: {count:5d} samples ({pct:5.1f}%)")
        
        print("\n" + "="*60)
        print("DOMINANT FAMILY PER CLUSTER")
        print("="*60)
        for cluster_id in sorted(vk_data['VK_cluster'].unique()):
            cluster_data = vk_data[vk_data['VK_cluster'] == cluster_id]
            top_family = cluster_data[family_col].value_counts().head(1)
            if len(top_family) > 0:
                family_name = top_family.index[0]
                family_count = top_family.values[0]
                family_pct = 100 * family_count / len(cluster_data)
                print(f"Cluster {cluster_id}: {family_name} "
                      f"({family_count}/{len(cluster_data)} = {family_pct:.1f}%)")
    
    results = {
        'k_best': k_best,
        'silhouette_scores': silhouette_scores,
        'vk_data': vk_data,
        'centroids': centroids_df,
        'model': best_model,
        'scaler': scaler
    }
    
    return df_with_clusters, results
