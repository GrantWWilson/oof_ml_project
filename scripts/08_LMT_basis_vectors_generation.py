#!/usr/bin/env python3
"""
Generate specific basis vector samples matching last night's LMT TolTEC data.
This script produces data for the following basis vectors and amplitudes:
    AST_V: +100
    AST_V: -100
    AST_O: +200
    AST_O: -200
    COMA_V: +200
    COMA_V: -200
    COMA_V and COMA_H: both +75
    COMA_H: +100
    COMA_H: -100

For each test case, the simulation is run for three channels:
    2000, 1400, and 1100

The generated outputs are saved as compressed npz files (with the usual arrays:
    images, zernikes, zernike_names, subrefs, subref_names, jobnames)
in per-channel subdirectories under data/synthetic_basis_vectors_LMT/.
"""

import os
from pathlib import Path
import numpy as np
import sys
# Ensure that the parent directory is on the path so that we can import generate_data_files.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oof_ml.utils.data_generation_utils import generate_data_files

# ------------------------------------------------------------------
# Setup output directories
BASE_OUTDIR = Path("data/synthetic_basis_vectors/LMT")
# We'll place results for each channel in a subdirectory named a<ID>
CHANNELS = ["2000", "1400", "1100"]
for ch in CHANNELS:
    (BASE_OUTDIR / f"a{ch}").mkdir(parents=True, exist_ok=True)

# Define Band Info for each channel id.
BAND_INFOS = {
    "2000": {"wavelength": "0.002",  "noise": "0.0"},
    "1400": {"wavelength": "0.0014", "noise": "0.0"},
    "1100": {"wavelength": "0.0011", "noise": "0.0"}
}

# Zernike parameters expected by the simulator (order is important)
ZERN_NAMES = [
    "TILT_H", "TILT_V",
    "AST_V",  "AST_O",
    "COMA_H", "COMA_V",
    "TRE_O",  "TRE_V",
    "SPH"
]

def make_basis_sample(param_names, amplitude, channel_id, band_info):
    """
    Generate a single basis sample for the provided parameter(s) at the given amplitude.

    :param param_names: tuple (or list) of parameter names to set (e.g., ("AST_V",) or ("COMA_V", "COMA_H")).
    :param amplitude: Amplitude value to set (float).
    :param channel_id: Channel id as string (e.g., "2000").
    :param band_info: Dictionary with wavelength and noise info for the channel.
    :return: Tuple of (images, zernike array, subref array, jobname).
    """
    # Initialize all zernike parameters to zero.
    param_dict = {name: 0.0 for name in ZERN_NAMES}
    m2z_offset = 0.0  # Default value for M2Z offset (not used in these tests)

    # Set the chosen parameter(s) to the test amplitude.
    for p in param_names:
        param_dict[p] = amplitude

    # Build a jobname that encodes the test parameters, amplitude, and channel.
    param_str = "_".join(param_names)
    jobname = f"basis_{param_str}_{'p' if amplitude > 0 else 'm'}{abs(amplitude):.0f}_{channel_id}"
    print(f"Generating basis sample: {jobname} for {param_names} with amplitude {amplitude:+} on channel {channel_id}")

    # Call the simulation utility.
    images = generate_data_files(
        param_dict=param_dict,
        output_dir="temp_basis",  # Temporary scratch directory
        jobname=jobname,
        channel_id=channel_id,
        wavelength=band_info["wavelength"],
        noise=band_info["noise"],
        m2z_offset=m2z_offset
    )
    # Add an extra dimension so images has shape (1, H, W, C)
    images = images[np.newaxis, ...]
    
    # Build the zernike array (shape: (1, 9)).
    zern_arr = np.array([[param_dict[n] for n in ZERN_NAMES]], dtype=np.float32)
    # Build the subref array (shape: (1, 3)), here (M2Z, M2X, M2Y) with only M2Z set.
    subref_arr = np.array([[m2z_offset, 0.0, 0.0]], dtype=np.float32)

    return images, zern_arr, subref_arr, jobname

def main():
    # Define the list of basis test cases as (parameters, amplitude).
    test_cases = [
        (("AST_V",), +100.0),
        (("AST_V",), -100.0),
        (("AST_O",), +200.0),
        (("AST_O",), -200.0),
        (("COMA_V",), +200.0),
        (("COMA_V",), -200.0),
        (("COMA_V", "COMA_H"), +75.0),
        (("COMA_H",), +100.0),
        (("COMA_H",), -100.0)
    ]

    # Loop over each channel and test case.
    for channel_id in CHANNELS:
        out_dir = BASE_OUTDIR / f"a{channel_id}"
        band_info = BAND_INFOS[channel_id]
        for params, amp in test_cases:
            images, zern_arr, subref_arr, jobname = make_basis_sample(params, amp, channel_id, band_info)
            
            # Name the output file to include the test description and channel id.
            npz_name = f"{jobname}.npz"
            np.savez_compressed(
                out_dir / npz_name,
                images=images,               # Expected shape (1, 32, 32, 3) or similar.
                zernikes=zern_arr,           # Shape (1, 9)
                zernike_names=np.array(ZERN_NAMES),
                subrefs=subref_arr,          # Shape (1, 3)
                subref_names=np.array(["M2Z", "M2X", "M2Y"]),
                jobnames=np.array([jobname])
            )
    print(f"\nAll basis vectors written to {BASE_OUTDIR}")

if __name__ == "__main__":
    main()
