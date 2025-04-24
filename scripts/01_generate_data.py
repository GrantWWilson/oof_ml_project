#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from oof_ml.data_generation.synthetic_data_generator import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PSF dataset batches and basis samples.")
    parser.add_argument("--first-batch", type=int, default=1, help="Starting batch number.")
    parser.add_argument("--total-batches", type=int, default=100, help="Total number of batches to generate.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of samples per batch.")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_45m/train/", 
                        help="Directory for output files.")
    parser.add_argument("--generate-basis", action="store_true", default=True,
                        help="Also generate augmented basis function samples (default: True).")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        first_batch=args.first_batch,
        total_batches=args.total_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    # Generate the standard LHS-sampled batches.
    generator.run()

    # If enabled, also generate the basis function samples.
    if args.generate_basis:
        # Here we generate basis samples for 1-parameter and 2-parameter combinations.
        generator.generate_basis_samples(basis_sizes=[1, 2])
        print("Basis function samples generated.")

if __name__ == "__main__":
    main()
