#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Adjust as needed for your project structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oof_ml.evaluation.model_evaluator import ModelEvaluator

def find_latest_run(results_root="results_model_45", model_filename="zernike_predictor_best.keras"):
    """
    Searches 'results_root' for subdirectories named 'run_YYYYMMDD-HHMMSS',
    sorts them, and returns:
       1) The path to the 'models/<model_filename>' in the newest run folder
       2) The run folder itself (Path object)

    If not found, returns (None, None).
    """
    root = Path(results_root)
    if not root.is_dir():
        return None, None

    # Find subdirs named run_*
    run_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        return None, None

    # Last one in sorted list is the newest
    latest_run_dir = run_dirs[-1]
    model_path = latest_run_dir / "models" / model_filename
    if model_path.is_file():
        return model_path, latest_run_dir
    # If the model file does not exist, return (None, None)
    return None, None

def main():
    # Attempt to find the latest run and its model
    default_model_path, latest_run_dir = find_latest_run(
        results_root="results_model_45",
        model_filename="zernike_predictor_best.keras"
    )

    if default_model_path is None or latest_run_dir is None:
        # Fallback if no run_* found or no best.keras file
        default_model_path = "zernike_predictor_best.keras"
        default_output_dir = "evaluation_results"
    else:
        # If we found a valid model, we default output_dir to <that_run_dir>/evaluation
        default_output_dir = latest_run_dir / "evaluation"

    parser = argparse.ArgumentParser(description="Evaluate a trained model and produce metrics/plots.")
    parser.add_argument("--model-path", type=str, default=str(default_model_path),
                        help="Path to the trained model file (defaults to latest run_... in 'results_model_45').")
    parser.add_argument("--dataset-path", type=str, default="data/synthetic_45m/test",
                        help="Directory containing NPZ files (test sets).")
    parser.add_argument("--band-filter", type=str, default="a2000",
                        help="Which band(s) to test (e.g., 'a2000+a1100').")
    parser.add_argument("--max-batch-num", type=int, default=900,
                        help="Load batch files with an index < max_batch_num.")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir),
                        help="Directory for saving PDF and summary (defaults to <run_dir>/evaluation if found).")
    parser.add_argument("--remove-focus", action="store_true",
                        help="If set, remove 'FOCUS' from the parameter set.")
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        band_filter=args.band_filter,
        max_batch_num=args.max_batch_num,
        output_dir=args.output_dir,
        remove_focus=args.remove_focus
    )

    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
