#!/usr/bin/env python3
"""
Convert raw basis_* files (MODEL_0/1/2 fits + zernike.dat + subref.dat)
into a single batch_2000_basis.npz that is compatible with the *tilt‑free*
training setup (only 7 Zernike terms).
"""

import re, argparse
from pathlib import Path
import numpy as np
from astropy.io import fits

# --------------------------------------------------------------------------
# 7‑term order that the new training expects
PARAM_ORDER = [
    "AST_V", "AST_O",
    "COMA_H", "COMA_V",
    "TRE_O",  "TRE_V",
    "SPH",
]

# --------------------------------------------------------------------------
def load_fits_stack(stem):
    imgs = [fits.open(stem + f"_MODEL_{k}.fits")[0].data.astype("float32")
            for k in range(3)]
    img = np.stack(imgs, axis=-1)           # (H,W,3)
    for c in range(3):                      # peak‑normalise each channel
        m = img[..., c].max()
        if m > 0: img[..., c] /= m
    return img

def load_zernike_vec(zdat_path):
    """
    Robustly extract the 7 coefficients we care about from a zernike.dat file.

    Accepts any of the following line styles, ignoring extra tokens:
        AST_V   -42.3
        3  AST_V  -42.3  micron
        AST_V=-42.3
    """
    vals = {k: 0.0 for k in PARAM_ORDER}

    def try_float(tok):
        try:  return float(tok)
        except ValueError: return None

    with open(zdat_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # normalise separators so "AST_V=-42.3" → ["AST_V","-42.3"]
            tokens = []
            for t in line.replace("=", " ").split():
                # split "AST_V=-42.3" that had no spaces
                tokens += t.split("=")

            for i, tok in enumerate(tokens):
                if tok in vals and i + 1 < len(tokens):
                    val = try_float(tokens[i + 1])
                    if val is not None:
                        vals[tok] = val
                        break   # done with this line

    return np.array([vals[k] for k in PARAM_ORDER], dtype="float32")

# --------------------------------------------------------------------------
def main(basis_root):
    basis_root = Path(basis_root)
    stems = sorted({re.sub(r"_MODEL_[012]\.fits$", "", str(p))
                    for p in basis_root.glob("*_MODEL_0.fits")})

    images, zernikes, subrefs, names = [], [], [], []

    for stem in stems:
        images.append(load_fits_stack(stem))
        zernikes.append(load_zernike_vec(stem + "_zernike.dat"))
        m2z = float(open(stem + "_subref.dat").read().split()[0])
        subrefs.append([m2z, 0.0, 0.0])
        names.append(Path(stem).name)

    images   = np.stack(images)
    zernikes = np.stack(zernikes)
    subrefs  = np.stack(subrefs)
    zernike_names = np.array(PARAM_ORDER, dtype="U")

    out_npz = basis_root / "batch_2000_basis.npz"
    np.savez_compressed(
        out_npz,
        images        = images,
        zernikes      = zernikes,
        zernike_names = zernike_names,
        subrefs       = subrefs,
        subref_names  = np.array(["M2Z", "M2X", "M2Y"]),
        jobnames      = np.array(names),
    )
    print(f"Wrote {len(stems)} samples → {out_npz}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-root",
        default="data/synthetic_45m/train/basis",
        help="Directory that holds the raw basis_* files")
    args = parser.parse_args()
    main(args.basis_root)
