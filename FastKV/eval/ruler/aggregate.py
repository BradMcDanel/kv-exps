# eval/ruler/aggregate.py

"""
Script to aggregate RULER results across different KV cache methods.
Reads results.json files from each method folder and outputs to CSV.
Includes subcategory averages for Retrieval, Multi-hop, Aggregation, and QA.
"""

import os
import json
import csv
from pathlib import Path
import sys

# Define the RULER tasks in order with their display names and categories
RULER_TASKS = [
    # Retrieval tasks (niah variants)
    ("niah_single_1", "NIAH-S1", "Retrieval"),
    ("niah_single_2", "NIAH-S2", "Retrieval"), 
    ("niah_single_3", "NIAH-S3", "Retrieval"),
    ("niah_multikey_1", "NIAH-MK1", "Retrieval"),
    ("niah_multikey_2", "NIAH-MK2", "Retrieval"),
    ("niah_multikey_3", "NIAH-MK3", "Retrieval"),
    ("niah_multivalue", "NIAH-MV", "Retrieval"),
    ("niah_multiquery", "NIAH-MQ", "Retrieval"),
    
    # Multi-hop Tracing
    ("vt", "VT", "Multi-hop"),
    
    # Aggregation tasks
    ("cwe", "CWE", "Aggregation"),
    ("fwe", "FWE", "Aggregation"),
    
    # Question Answering
    ("qa_1", "QA-1", "QA"),
    ("qa_2", "QA-2", "QA"),
]

# Extract components for easy access
TASK_KEYS = [key for key, _, _ in RULER_TASKS]
TASK_NAMES = [name for _, name, _ in RULER_TASKS]
TASK_CATEGORIES = [cat for _, _, cat in RULER_TASKS]

# Define category mappings
CATEGORIES = {
    "Retrieval": [name for _, name, cat in RULER_TASKS if cat == "Retrieval"],
    "Multi-hop": [name for _, name, cat in RULER_TASKS if cat == "Multi-hop"], 
    "Aggregation": [name for _, name, cat in RULER_TASKS if cat == "Aggregation"],
    "QA": [name for _, name, cat in RULER_TASKS if cat == "QA"],
}

def load_results(results_file):
    """Load results from a CSV file, return empty dict if file doesn't exist or is invalid."""
    try:
        results = {}
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Task'] != 'Average':  # Skip the average row
                    results[row['Task']] = float(row['Score'])
        return results
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Warning: Could not load {results_file}: {e}")
        return {}

def parse_method_name(folder_name):
    """
    Parse method name from folder name.
    
    Args:
        folder_name: Name of the folder (e.g., 'fastkv-4096', 'fullkv-2048')
    
    Returns:
        str: method_name
    """
    # Return the folder name as-is to preserve full path information
    return folder_name

def calculate_category_averages(row, task_results):
    """Calculate average scores for each category."""
    category_avgs = {}
    
    for category, task_names in CATEGORIES.items():
        valid_scores = []
        for task_name in task_names:
            if task_name in row and row[task_name] != "":
                score = row[task_name]
                if isinstance(score, (int, float)):
                    valid_scores.append(score)
        
        if valid_scores:
            category_avgs[f"{category}_Avg"] = sum(valid_scores) / len(valid_scores)
        else:
            category_avgs[f"{category}_Avg"] = ""
    
    return category_avgs

def aggregate_ruler_results(base_path="outputs/meta-llama/Llama-3.1-8B-Instruct/ruler"):
    """
    Aggregate RULER results from all method folders.
    
    Args:
        base_path: Path to the ruler results directory
    
    Returns:
        List of dictionaries containing aggregated results
    """
    results = []
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist!")
        return results
    
    # Get all method folders
    method_folders = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))]
    method_folders.sort()  # Sort for consistent ordering
    
    print(f"Found method folders: {method_folders}")
    
    # Check for multiple fullkv entries
    fullkv_folders = [f for f in method_folders if f.startswith("fullkv")]
    if len(fullkv_folders) > 1:
        print(f"Warning: Found multiple fullkv folders: {fullkv_folders}")
    
    for folder_name in method_folders:
        method_path = os.path.join(base_path, folder_name)
        
        # Look for summary.csv in subdirectories (sequence length folders)
        method_results = {}
        for subdir in os.listdir(method_path):
            subdir_path = os.path.join(method_path, subdir)
            if os.path.isdir(subdir_path):
                results_file = os.path.join(subdir_path, "pred", "summary.csv")
                if os.path.exists(results_file):
                    method_results = load_results(results_file)
                    break  # Use the first valid results found
        
        # Parse method name
        method_name = parse_method_name(folder_name)
        
        # Create row for this method
        row = {
            "Method": method_name
        }
        
        # Add results for each task (in specified order)
        valid_scores = []
        for task_key, task_name, category in RULER_TASKS:
            if task_key in method_results:
                score = method_results[task_key]
                row[task_name] = score
                # Collect valid scores for overall average calculation
                if isinstance(score, (int, float)):
                    valid_scores.append(score)
            else:
                row[task_name] = ""  # Empty placeholder for missing tasks
        
        # Calculate category averages
        category_avgs = calculate_category_averages(row, method_results)
        row.update(category_avgs)
        
        # Calculate overall average of valid scores
        if valid_scores:
            row["Overall_Avg"] = sum(valid_scores) / len(valid_scores)
        else:
            row["Overall_Avg"] = ""
        
        results.append(row)
    
    # Sort results: fullkv first, then alphabetically by method name
    def sort_key(result):
        method = result["Method"]
        
        if method == "fullkv":
            return (0, method)  # fullkv always first
        else:
            return (1, method)  # Then alphabetically
    
    results.sort(key=sort_key)
    
    return results

def save_to_csv(results, base_path, output_filename="ruler_results.csv"):
    """Save aggregated results to CSV file in the base path directory."""
    if not results:
        print("No results to save!")
        return
    
    # Create output file path in the base_path directory
    output_file = os.path.join(base_path, output_filename)
    
    # CSV headers: Method + all task names + category averages + overall average
    headers = ["Method"] + TASK_NAMES + [f"{cat}_Avg" for cat in CATEGORIES.keys()] + ["Overall_Avg"]
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {output_file}")
    except IOError as e:
        print(f"Error saving CSV file: {e}")

def print_summary_table(results):
    """Print a formatted summary table to screen."""
    if not results:
        print("No results to display!")
        return
    
    print("\n" + "="*150)
    print("RULER RESULTS SUMMARY")
    print("="*150)
    
    # Calculate column widths
    method_width = max(8, max(len(str(r["Method"])) for r in results))  # Min 8 for "Method"
    score_width = 7   # Fixed width for scores
    
    # Build header row
    header_parts = []
    header_parts.append(f"{'Method':<{method_width}}")
    
    # Add individual task headers
    for task_name in TASK_NAMES:
        header_parts.append(f"{task_name:<{score_width}}")
    
    # Add category average headers
    for category in CATEGORIES.keys():
        abbrev_name = f"{category[:6]}_Av"
        header_parts.append(f"{abbrev_name:<{score_width}}")  # Abbreviated for space
    
    # Add overall average
    header_parts.append(f"{'Overall':<{score_width}}")
    
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))
    
    # Print results for each method
    for result in results:
        row_parts = []
        
        # Method name
        row_parts.append(f"{result['Method']:<{method_width}}")
        
        # Individual task scores
        for task_name in TASK_NAMES:
            value = result[task_name]
            if value == "":
                display_value = "N/A"
            else:
                display_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            row_parts.append(f"{display_value:<{score_width}}")
        
        # Category averages
        for category in CATEGORIES.keys():
            avg_key = f"{category}_Avg"
            avg_value = result[avg_key]
            if avg_value == "":
                avg_display = "N/A"
            else:
                avg_display = f"{avg_value:.2f}" if isinstance(avg_value, (int, float)) else str(avg_value)
            row_parts.append(f"{avg_display:<{score_width}}")
        
        # Overall average
        overall_value = result["Overall_Avg"]
        if overall_value == "":
            overall_display = "N/A"
        else:
            overall_display = f"{overall_value:.2f}" if isinstance(overall_value, (int, float)) else str(overall_value)
        row_parts.append(f"{overall_display:<{score_width}}")
        
        row = " ".join(row_parts)
        print(row)
    
    print("="*150)
    
    # Print category breakdown
    print("\nCATEGORY BREAKDOWN:")
    for category, task_names in CATEGORIES.items():
        print(f"{category}: {', '.join(task_names)}")

def main():
    """Main function to run the aggregation."""
    # Require base path as command line argument
    if len(sys.argv) < 2:
        print("Usage: python3 aggregate.py <base_path>")
        print("Example: python3 aggregate.py outputs/meta-llama/Llama-3.1-8B-Instruct/ruler")
        sys.exit(1)
    
    base_path = sys.argv[1]
    
    print(f"Aggregating RULER results from: {base_path}")
    print()
    
    # Aggregate results
    results = aggregate_ruler_results(base_path)
    
    if results:
        # Print summary table
        print_summary_table(results)
        
        # Save to CSV
        save_to_csv(results, base_path)
        
        print(f"\nProcessed {len(results)} methods across {len(TASK_KEYS)} tasks.")
        print(f"Categories: {', '.join(CATEGORIES.keys())}")
    else:
        print("No results found!")

if __name__ == "__main__":
    main()