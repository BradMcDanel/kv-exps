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
    if df.empty:
        return pd.DataFrame()
    
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index()
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    return pivot_table

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
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Needle-in-Haystack Performance Comparison', fontsize=16, fontweight='bold')
    
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
            square=True
        )
        
        # Formatting
        avg_score = df["Score"].mean()
        axes[idx].set_title(f'{method_name}\nAvg: {avg_score:.3f}', fontweight='bold', pad=10)
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