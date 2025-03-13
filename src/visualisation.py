import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def plot_feature_importance(model, features, output_dir='results'):
    """Plot feature importance from the model."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    return plt


def plot_stat_distributions(df, output_dir='results'):
    """Plot distributions of Pokemon stats by legendary status."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # Stats to plot
    stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'base_total']

    # Create plots
    for stat in stats:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df,
            x=stat,
            hue='is_legendary',
            multiple='stack',
            palette=['skyblue', 'salmon'],
            bins=20
        )
        plt.title(f'Distribution of {stat.replace("_", " ").title()} by Legendary Status')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{stat}_distribution.png'))

        # Create a boxplot for all stats
    plt.figure(figsize=(14, 8))
    df_melt = pd.melt(
        df[stats + ['is_legendary']],
        id_vars=['is_legendary'],
        var_name='Stat',
        value_name='Value'
    )
    sns.boxplot(x='Stat', y='Value', hue='is_legendary', data=df_melt, palette=['skyblue', 'salmon'])
    plt.title('Pokemon Stats by Legendary Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stats_boxplot.png'))
    return plt