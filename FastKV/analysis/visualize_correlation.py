# analysis/visualize_correlation.py

import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def create_correlation_plot(csv_path: str, output_dir: str, x_column: str, x_label: str):
    """
    Loads correlation data from a CSV and generates a scatter plot with
    a regression line and statistical annotations.
    """
    # --- 1. Load the Data ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found. Please check the path.")
        return
    
    # --- FIX: Use the dynamic x_column name ---
    required_cols = [x_column, 'score']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain '{x_column}' and 'score' columns. Found: {df.columns.tolist()}")
        return
    
    if len(df) < 2:
        print("Error: Need at least 2 data points to create a correlation plot.")
        return

    # --- 2. Perform Statistical Analysis ---
    df.dropna(subset=required_cols, inplace=True)
    
    pearson_corr, p_pearson = pearsonr(df[x_column], df['score'])
    spearman_corr, p_spearman = spearmanr(df[x_column], df['score'])

    print("\n--- Correlation Statistics ---")
    print(f"Pearson's r: {pearson_corr:.4f} (p-value: {p_pearson:.4f})")
    print(f"Spearman's ρ: {spearman_corr:.4f} (p-value: {p_spearman:.4f})")
    print("------------------------------")

    # --- 3. Create the Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- FIX: Use the dynamic x_column name ---
    sns.regplot(
        x=x_column,
        y='score',
        data=df,
        ax=ax,
        scatter_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'w'},
        line_kws={'color': 'firebrick', 'linestyle': '--', 'linewidth': 2}
    )

    # --- 4. Annotate and Style the Plot ---
    dataset_name = os.path.basename(csv_path).replace('correlation_data_', '').replace('tree_correlation_data_', '').replace('.csv', '')
    
    ax.set_title(f"'{x_label}' vs. Model Performance on '{dataset_name}'", fontsize=18, pad=20)
    # --- FIX: Use the dynamic x_label ---
    ax.set_xlabel(f"{x_label} (Predicted Difficulty)", fontsize=14)
    ax.set_ylabel("Ground Truth Model Score (Actual Difficulty)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    stats_text = (
        f"Spearman's ρ = {spearman_corr:.3f}\n"
        f"(p-value = {p_spearman:.3g})\n\n"
        f"Pearson's r = {pearson_corr:.3f}\n"
        f"(p-value = {p_pearson:.3g})\n\n"
        f"N = {len(df)} samples"
    )
    
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

    # --- 5. Save the Figure ---
    output_filename = os.path.join(output_dir, f"correlation_plot_{dataset_name}_{x_column}.png")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Correlation plot saved to: {output_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize the correlation between a difficulty metric and model score.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file.")
    # --- FIX: Add new arguments ---
    parser.add_argument("--x_column", type=str, default="entropy", help="Name of the column to use for the x-axis (e.g., 'entropy', 'tree_entropy').")
    parser.add_argument("--x_label", type=str, default="Prompt Entropy", help="Label to use for the x-axis in the plot title and axes.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the plot. Defaults to the same directory as the CSV file.")
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.csv_file)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_correlation_plot(args.csv_file, args.output_dir, args.x_column, args.x_label)

if __name__ == "__main__":
    main()
