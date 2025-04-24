#!/usr/bin/env python3
import os
import sys
import numpy as np
from pathlib import Path

# allow "import oof_ml..."
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from oof_ml.utils.predict_from_npz import predict_from_npz

# ------------------------------------------------------------------
# --- user‑editable paths ------------------------------------------
MODEL_PATH = "results_model_45/run_20250405-150859/models/zernike_predictor_final.keras"
NPZ_PATH   = "data/synthetic_basis_vectors/basis_AST_V_p100.npz"
# ------------------------------------------------------------------

# 1) run inference
pred = predict_from_npz(model_path=MODEL_PATH, npz_path=NPZ_PATH)[0]  # shape (10,)
print(f"Raw prediction vector:\n{pred}\n")

# 2) get parameter names from the .npz
with np.load(NPZ_PATH) as data:
    if "zernike_names" in data:
        names = list(data["zernike_names"]) + ["M2z_offset"]
    else:
        # fallback (hard‑coded order used in training)
        names = [
            "TILT_H","TILT_V","AST_V","AST_O","COMA_H","COMA_V",
            "TRE_O","TRE_V","SPH","M2z_offset"
        ]

# 3) pretty print
print("Predicted Zernike parameters:")
for name, val in zip(names, pred):
    print(f"  {name:10s} : {val: .4f}")
