#!/usr/bin/env python3
"""
Evaluate basis‑vector *.npz files, build the response matrix R,
store R and its inverse next to the trained model, and write a CSV/YAML
summary in evaluation_basis/eval_<timestamp>/.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tensorflow as tf

# ------------------------------------------------------------------
def find_latest_model(results_root="results_model",
                      model_filename="zernike_predictor_final.keras"):
    root = Path(results_root)
    if not root.is_dir():
        return None
    run_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        return None
    latest = run_dirs[-1] / "models" / model_filename
    return latest if latest.is_file() else None

def normalize_by_peak(images):
    """Peak‑normalise each channel (same rule as training)."""
    for i in range(images.shape[0]):
        for c in range(images.shape[-1]):
            mx = images[i, :, :, c].max()
            if mx > 0:
                images[i, :, :, c] /= mx
    return images

# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate basis vectors and build response matrix.")
    parser.add_argument("--basis-dir", type=str, default="data/synthetic_basis_vectors")
    parser.add_argument("--model-path", type=str,
                        default=str(find_latest_model() or "zernike_predictor_best.keras"))
    parser.add_argument("--output-root", type=str, default="evaluation_basis")
    args = parser.parse_args()

    basis_dir  = Path(args.basis_dir)
    model_path = Path(args.model_path)
    out_root   = Path(args.output_root)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    if not basis_dir.is_dir():
        raise FileNotFoundError(f"Basis directory '{basis_dir}' not found.")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # ----------------------------------------------------------
    # pass 1: run predictions, collect ±100 for each parameter
    # ----------------------------------------------------------
    rows = []
    zernike_names = None  # will be set from a training npz.
    # resp will be a dictionary mapping parameter => { "+": prediction, "-": prediction }
    resp = {}

    for npz_file in sorted(basis_dir.glob("*.npz")):
        data   = np.load(npz_file)
        images = normalize_by_peak(data["images"])
        # Expect prediction vector length matches training outputs.
        pred   = model.predict(images, verbose=0)[0]   # shape (n,) where n = len(zernike_names)

        # Parse file name: e.g. "basis_AST_O_p100.npz"
        stem   = npz_file.stem
        tokens = stem.split("_")
        param  = "_".join(tokens[1:-1])
        # map legacy name to training naming (if needed)
        if param == "M2Z_OFFSET":
            param = "M2z_offset"
        sign   = tokens[-1]              # 'p100' or 'm100'
        amp    = +100.0 if sign.startswith("p") else -100.0
        sign_key = "+" if amp > 0 else "-"

        if zernike_names is None:
            # load the true output order from any training npz
            ref_npz = next(Path("data/synthetic_45m/train/").glob("batch_2000_*.npz"))
            zernike_names = list(np.load(ref_npz)["zernike_names"]) + ["M2z_offset"]

        rows.append({"file": npz_file.name, "param": param, "amp": amp,
                     **{n: v for n, v in zip(zernike_names, pred)}})

        resp.setdefault(param, {})[sign_key] = pred

    # Use dynamic size: let n = number of network outputs
    n = len(zernike_names)
    # ----------------------------------------------------------
    # Build R matrix of shape (n, n)
    # ----------------------------------------------------------
    param_order = zernike_names               # network output order
    basis_order = zernike_names               # we assume one-to-one mapping
    R = np.zeros((n, n), dtype=float)

    for i, p in enumerate(basis_order):
        if "+" not in resp.get(p, {}) or "-" not in resp.get(p, {}):
            raise RuntimeError(f"Missing + or – basis vector for parameter '{p}'")
        diff = (resp[p]["+"] - resp[p]["-"]) / 200.0   # shape (n,)
        R[:, i] = diff  # Each column i corresponds to the output differences for parameter p

    print("---- Debug Info: R matrix ----")
    print("R shape:", R.shape)
    print("R[0:5, 0:5] =\n", R[0:5, 0:5])
    cond_R = np.linalg.cond(R)
    print("Condition number of R:", cond_R)

    # Compute pseudo-inverse of R
    R_inv = np.linalg.pinv(R)
    check_eye = R_inv @ R
    print("R_inv @ R shape:", check_eye.shape)
    print("R_inv @ R [0:5,0:5] =\n", check_eye[0:5, 0:5])
    diag_vals = np.diagonal(check_eye)
    off_diag_sum = np.sum(np.abs(check_eye - np.eye(n)))
    print("Diagonal of R_inv @ R:", diag_vals)
    print("Sum of off-diagonal elements:", off_diag_sum)
    close_to_identity = np.allclose(check_eye, np.eye(n), atol=1e-2)
    print("Is R_inv @ R close to Identity (within 1e-2)?", close_to_identity)

    # (Optional) Print per-parameter difference vectors
    for i, p in enumerate(basis_order):
        plus_vec = resp[p]["+"]
        minus_vec = resp[p]["-"]
        diff = (plus_vec - minus_vec) / 200.0
        print(f"\nParam = {p} (row i={i})")
        print(" plus_vec[:5] =", plus_vec[:5])
        print("minus_vec[:5] =", minus_vec[:5])
        print(" diff[:5]     =", diff[:5])
        R[:, i] = diff  # (Redundant if already set above)

    # Save R and R_inv next to the model
    model_dir = model_path.parent
    np.save(model_dir / "R.npy", R)
    np.save(model_dir / "R_inv.npy", R_inv)
    print(f"Saved R and R_inv to {model_dir}")

    # ----------------------------------------------------------
    # Write CSV & YAML manifest
    # ----------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = out_dir / "basis_predictions.csv"
    df.to_csv(csv_path, index=False)

    manifest = {
        "timestamp": ts,
        "model_path": str(model_path),
        "basis_dir": str(basis_dir),
        "num_files": len(rows),
        "zernike_output_names": zernike_names,
        "R_path": str(model_dir / "R.npy"),
        "R_inv_path": str(model_dir / "R_inv.npy")
    }
    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, sort_keys=False)
    print(f"Predictions CSV → {csv_path}")
    print(f"Manifest        → {out_dir/'manifest.yaml'}")

    # ----------------------------------------------------------
    # POST‑PROCESS: apply R_inv to each prediction row and output a fixed CSV.
    # ----------------------------------------------------------
    print("Applying R_inv to all predictions to verify correction ...")
    R_inv = np.load(model_dir / "R_inv.npy")
    pred_cols = zernike_names
    pred_matrix = df[pred_cols].values  # shape (N, n)
    corrected = pred_matrix @ R_inv      # shape (N, n)

    # For debugging: print one specific row correction (if exists)
    if "basis_TILT_H_p100.npz" in df["file"].values:
        mask = df["file"] == "basis_TILT_H_p100.npz"
        row_index = df.index[mask][0]
        raw_pred = pred_matrix[row_index]
        corr_pred = corrected[row_index]
        print("\n--- Example Row Debug ---")
        print("raw_pred:", raw_pred)
        print("corr_pred:", corr_pred)

    # Add corrected columns to DataFrame.
    for i, name in enumerate(pred_cols):
        df[f"corr_{name}"] = corrected[:, i]

    fixed_csv = out_dir / "basis_predictions_fixed.csv"
    df.to_csv(fixed_csv, index=False)
    print(f"Corrected predictions written to {fixed_csv}")

    print("zernike_names (expected order):", zernike_names)
    print("DataFrame column order         :", list(df.columns[:len(zernike_names)]))
    
if __name__ == "__main__":
    main()
