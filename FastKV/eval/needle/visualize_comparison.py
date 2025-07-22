import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import glob
import argparse
import os

def load_results(model, eval_path, expected_answer):
    """Load and process results from a single evaluation path"""
    save_path = os.path.join(f"outputs/{model}/needle", eval_path)
    folder_path = os.path.join(save_path, "result")
    
    if not os.path.exists(folder_path):
        print(f"Warning: Results not found at {folder_path}")
        return pd.DataFrame()

    json_files = glob.glob(f"{folder_path}/*.json")
    data = []

    expected_words = expected_answer.lower().split()

    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            model_response = json_data.get("model_response", "").lower().split()
            
            # Calculate score based on keyword overlap
            score = len(set(model_response).intersection(set(expected_words))) / len(set(expected_words))
            
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    df = pd.DataFrame(data)
    if not df.empty:
        avg_score = df["Score"].mean()
        print(f"{eval_path} Average Score: {avg_score:.3f}")
    
    return df

def create_pivot_table(df):
    """Create pivot table for heatmap"""
    # Define expected context lengths and their labels
    expected_lengths = [16384, 24576, 32768, 40960, 49152, 57344, 65536]
    length_labels = ["16k", "24k", "32k", "40k", "48k", "56k", "64k"]
    
    # Expected depth percentages (matching needle evaluation default: 10 intervals)
    expected_depths = [0, 11, 22, 33, 44, 56, 67, 78, 89, 100]
    
    if df.empty:
        # Create empty pivot table with all expected dimensions
        pivot_table = pd.DataFrame(index=expected_depths, columns=length_labels)
        pivot_table.index.name = "Document Depth"
        pivot_table.columns.name = "Context Length"
        return pivot_table
    
    # Create pivot table from available data
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index()
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    
    # Map actual context lengths to standard labels
    length_mapping = dict(zip(expected_lengths, length_labels))
    
    # Rename columns to standard labels
    column_mapping = {}
    for col in pivot_table.columns:
        # Find closest standard length
        closest_len = min(expected_lengths, key=lambda x: abs(x - col))
        if abs(closest_len - col) < 2000:  # Within 2k tolerance
            column_mapping[col] = length_mapping[closest_len]
        else:
            column_mapping[col] = f"{int(col/1024)}k"
    
    pivot_table = pivot_table.rename(columns=column_mapping)
    
    # Create complete grid with all expected rows and columns
    complete_pivot = pd.DataFrame(index=expected_depths, columns=length_labels)
    complete_pivot.index.name = "Document Depth"
    complete_pivot.columns.name = "Context Length"
    
    # Fill in available data
    for depth in expected_depths:
        for length_label in length_labels:
            if depth in pivot_table.index and length_label in pivot_table.columns:
                complete_pivot.loc[depth, length_label] = pivot_table.loc[depth, length_label]
    
    # Convert to float, ensuring NaN values are properly handled
    complete_pivot = complete_pivot.astype(float)
    
    return complete_pivot

def main(args):
    # Method configurations
    methods = [
        ("gemfilter", "GemFilter"),
        ("fastkv", "FastKV"), 
        ("speculative_prefill", "Speculative Prefill"),
        ("claa", "CLAA")
    ]
    
    # Set high-quality plotting parameters
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    
    # Track if we need to create colorbar
    need_colorbar = True
    
    for idx, (eval_path, method_name) in enumerate(methods):
        # Load results for this method
        df = load_results(args.model, eval_path, args.expected_answer)
        
        if df.empty:
            axes[idx].text(0.5, 0.5, f'No data\nfor {method_name}', 
                          transform=axes[idx].transAxes, ha='center', va='center',
                          fontsize=14, color='red')
            axes[idx].set_title(f'{method_name}\n(No Data)', fontweight='bold')
            continue
            
        # Create pivot table
        pivot_table = create_pivot_table(df)
        
        if pivot_table.empty:
            continue
            
        # Create a mask for missing data (NaN values)
        mask = pivot_table.isna()
        
        # Create heatmap
        im = sns.heatmap(
            pivot_table,
            vmin=0, vmax=1,
            cmap=cmap,
            ax=axes[idx],
            cbar=idx == 3,  # Only show colorbar on last plot
            cbar_kws={'label': 'Retrieval Score', 'shrink': 0.8} if idx == 3 else None,
            linewidths=0.3,
            linecolor='white',
            square=True,
            mask=False,  # Don't mask anything initially
            annot=False
        )
        
        # Add gray squares with X for missing data (OOM cases)
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                if mask.iloc[i, j]:  # If data is missing
                    # Add gray rectangle
                    axes[idx].add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                                     facecolor='lightgray', edgecolor='white', linewidth=0.3))
                    # Add X mark
                    axes[idx].text(j + 0.5, i + 0.5, '✕', ha='center', va='center', 
                                  fontsize=16, color='darkred', fontweight='bold')
        
        # Formatting
        avg_score = df["Score"].mean()
        axes[idx].set_title(f'{method_name}\nAverage Score: {avg_score:.3f}', fontweight='bold', pad=10)
        axes[idx].set_xlabel('Context Length', fontweight='bold')
        
        # Only show y-label on first plot
        if idx == 0:
            axes[idx].set_ylabel('Document Depth (%)', fontweight='bold')
        else:
            axes[idx].set_ylabel('')
            
        # Rotate x-axis labels
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save high-quality outputs
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, "needle.png")
    pdf_path = os.path.join(output_dir, "needle.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nVisualization saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help="Model name or path")
    parser.add_argument("--expected_answer", type=str, 
                       default="eat a sandwich and sit in Dolores Park on a sunny day", 
                       help="Expected answer for scoring")
    args = parser.parse_args()
    main(args)