# eval/ruler/eval/evaluate.py
import argparse
import json
from pathlib import Path
import pandas as pd

# Import from our own package
from eval.ruler.eval.metrics import string_match_all, string_match_part, TASK_METRICS

def get_metric_map():
    """Maps a base task type to its metric function."""
    return {task: info['metric_fn'] for task, info in TASK_METRICS.items()}

def main():
    parser = argparse.ArgumentParser(description="Evaluate RULER benchmark predictions.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to the prediction directory (e.g., outputs/.../2048/pred).")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        print(f"Error: Prediction directory not found at '{pred_dir}'")
        return

    # Load task configurations to map detailed task to base task
    try:
        with open("eval/ruler/config/task_details.json", "r") as f:
            task_details = json.load(f)
    except FileNotFoundError:
        print("Error: `eval/ruler/config/task_details.json` not found.")
        return

    metric_map = get_metric_map()
    results = {}

    print(f"Evaluating predictions in: {pred_dir}")
    for pred_file in sorted(pred_dir.glob("*.jsonl")):
        task_name = pred_file.stem
        
        if task_name not in task_details:
            print(f"Warning: Task '{task_name}' from prediction file not found in config. Skipping.")
            continue
            
        base_task = task_details[task_name]['base_task']
        metric_fn = metric_map.get(base_task)
        
        if not metric_fn:
            print(f"Warning: No metric found for base task '{base_task}'. Skipping evaluation for '{task_name}'.")
            continue

        # Load predictions and references
        predictions, references = [], []
        with open(pred_file, "r") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                references.append(data["outputs"])
        
        if not predictions:
            print(f"Warning: No predictions found in '{pred_file}'. Skipping.")
            continue

        # Calculate score
        score = metric_fn(predictions, references)
        results[task_name] = score
        print(f"  - Task: {task_name:<20} | Score: {score:.2f}")

    if not results:
        print("No results to summarize.")
        return

    # Create and save a summary CSV
    summary_df = pd.DataFrame(list(results.items()), columns=["Task", "Score"])
    
    # Calculate average score
    avg_score = summary_df["Score"].mean()
    avg_row = pd.DataFrame([{"Task": "Average", "Score": avg_score}])
    summary_df = pd.concat([summary_df, avg_row], ignore_index=True)

    summary_file = pred_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False, float_format='%.2f')
    
    print("\n" + "="*30)
    print("      RULER SUMMARY")
    print("="*30)
    print(summary_df.to_string(index=False))
    print("="*30)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
