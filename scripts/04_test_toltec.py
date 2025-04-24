#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from oof_ml.inference.test_toltec_pipeline import OOFAnalysisPipeline

def main():
    parser = argparse.ArgumentParser(description="Run the OOF Analysis Pipeline on TolTEC data.")
    parser.add_argument("--config-dir", type=str, default="data/toltec/fg131833",
                        help="Directory containing config.yaml for the TolTEC dataset.")
    parser.add_argument("--model-path", type=str, default="results_model/run_20250403-000829/models/zernike_predictor_final.keras",
                        help="Path to the trained model.")
    parser.add_argument("--output-dir", type=str, default="results_toltec",
                        help="Directory to save pipeline outputs.")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[32, 32],
                        help="Height and width of the central crop.")
    # parser.add_argument("--bands", nargs="+", default=["a2000", "a1100"],
    #                    help="List of TolTEC bands to process.")
    args = parser.parse_args()

    pipeline = OOFAnalysisPipeline(
        config_dir=args.config_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        crop_size=tuple(args.crop_size),
        # bands=args.bands
    )
    pipeline.run()
    print(f"Pipeline completed. Results written to {args.output_dir}.")

if __name__ == "__main__":
    main()
