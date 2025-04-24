#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oof_ml.utils.data_generation_utils import generate_data_files

# ------------------------------------------------------------------
# Output location
OUTDIR = Path("data/sph_focus_curves/")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Band / simulation constants for a2000
BAND_INFO = {"id": "2000", "wavelength": "0.002", "noise": "0.0"}
# BAND_INFO = {"id": "1100", "wavelength": "0.0011", "noise": "0.0"}

# Zernike order used by the network / simulator
ZERN_NAMES = [
    "TILT_H", "TILT_V",
    "AST_V", "AST_O",
    "COMA_H", "COMA_V",
    "TRE_O", "TRE_V",
    "SPH"
]


def make_single_basis(param_name: str, amplitude: float):
    # Build Zernike param dict (all zeros except possibly one)
    param_dict = {name: 0.0 for name in ZERN_NAMES}
    m2z_offset = 0.0

    if param_name == "M2Z_OFFSET":
        m2z_offset = amplitude
    else:
        param_dict[param_name] = amplitude

    # Call the unified simulator utility
    images = generate_data_files(
        param_dict=param_dict,
        output_dir="temp_basis",          # scratch directory
        jobname=f"basis_{param_name}_{amplitude:+}",
        channel_id=BAND_INFO["id"],
        wavelength=BAND_INFO["wavelength"],
        noise=BAND_INFO["noise"],
        m2z_offset=m2z_offset,
        mapc=4,
        df=250
    )

    # Build zernike array (shape (1, 9))
    zern_arr = np.array([[param_dict[n] for n in ZERN_NAMES]], dtype=np.float32)
    # Build subref array (shape (1, 3))  (M2Z, M2X, M2Y)
    subref_arr = np.array([[m2z_offset, 0.0, 0.0]], dtype=np.float32)

    jobname = f"{param_name}_{'p' if amplitude>0 else 'm'}{np.abs(amplitude)}"

    return images[np.newaxis, ...], zern_arr, subref_arr, jobname


def main():
    # for param in ZERN_NAMES:
    for param in ['SPH']:
        for amp in (-250, -200, 0, 200, 250):
            print(f"Generating basis for {param:>9}  amplitude={amp:+}")
            images, zern_arr, subref_arr, jobname = make_single_basis(param, amp)

            npz_name = f"basis_{jobname}.npz"
            np.savez_compressed(
                OUTDIR / npz_name,
                images=images,               # (1, 32, 32, 3)
                zernikes=zern_arr,           # (1, 9)
                zernike_names=np.array(ZERN_NAMES),
                subrefs=subref_arr,          # (1, 3)
                subref_names=np.array(["M2Z", "M2X", "M2Y"]),
                jobnames=np.array([jobname])
            )
    print(f"\nAll basis vectors written to {OUTDIR}")

if __name__ == "__main__":
    main()
