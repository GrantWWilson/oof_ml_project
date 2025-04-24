#!/usr/bin/env python3
"""
High‑level driver to train a ZernikePredictor model with
  • basis‑vector oversampling
  • per‑sample loss weighting
  • two‑phase curriculum

All run artifacts are placed in   <working‑dir>/run_<timestamp>/.
"""

import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oof_ml.modeling.zernike_predictor import ZernikePredictor

# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train ZernikePredictor with basis‑aware options")

    # ------------------------------------------------------------------ data
    p.add_argument("--dataset-path",      default="data/synthetic_45m/train",
                   help="Root containing *.npz batches (expects optional 'basis/' sub‑folder).")
    p.add_argument("--band-filter",       default="a2000",
                   help="Band(s) to load, e.g. 'a2000' or 'a2000+a1100'.")
    p.add_argument("--max-batch-num", type=int, default=100,
                   help="Only load batches with index < N.")

    # ---------------------------------------------------------------- run dir
    p.add_argument("--working-dir",       default="results_model",
                   help="Top‑level output dir (run_<timestamp> is auto‑created).")

    # ---------------------------------------------------------------- train hp
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--shift-pixels",  type=float, default=2.5,
                   help="Max ± pixel shift for RandomTranslation aug.")

    p.add_argument("--val-fraction",  type=float, default=0.20)
    p.add_argument("--test-fraction", type=float, default=0.10)

    # ---------------------------------------------------- basis / curriculum
    p.add_argument("--basis-oversample",  type=int,   default=4,
                   help="Repeat each basis sample this many times (>=1).")
    p.add_argument("--basis-loss-weight", type=float, default=5.0,
                   help="Per‑sample weight applied to every basis vector.")
    p.add_argument("--curriculum-epochs", type=int,   default=20,
                   help="First N epochs train without basis samples.")

    args = p.parse_args()

    os.makedirs(args.working_dir, exist_ok=True)

    predictor = ZernikePredictor(
        dataset_path       = args.dataset_path,
        working_directory  = args.working_dir,
        band_filter        = args.band_filter,
        batch_size         = args.batch_size,
        epochs             = args.epochs,
        learning_rate      = args.learning_rate,
        shift_pixels       = args.shift_pixels,
        basis_oversample   = args.basis_oversample,
        basis_loss_weight  = args.basis_loss_weight,
        curriculum_epochs  = args.curriculum_epochs,
    )

    predictor.load_data(max_batch_num=args.max_batch_num)
    predictor.prepare_datasets(val_fraction=args.val_fraction,
                               test_fraction=args.test_fraction)
    predictor.build_model()
    predictor.train()
    print("✓ Training finished")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
