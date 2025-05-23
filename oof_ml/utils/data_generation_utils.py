import os
import subprocess
import numpy as np
from astropy.io import fits
from pathlib import Path

from oof_ml.utils.zernike_utils import write_zernike_dat_file

# Some global defaults for the simulation
PIXEL_SIZE = 2.0       # arcsec
NFT = 256              # FFT grid size
MAP_SIZE_XY = 128      # Output image size
NUM_MODELS = 3         # Number of images produced by simulation
CROP_NX = 32           # Final crop size (width)
CROP_NY = 32           # Final crop size (height)
EXECUTABLE = "/work/toltec/wilson/OOF/LMTOOF/bin/create_data_files"
DUMMY_TRANSFER = "/work/toltec/wilson/OOF/LMTOOF/etc/DUMMY"

def crop_center(image, size=(CROP_NX, CROP_NY)):
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    half_size_x, half_size_y = size[0] // 2, size[1] // 2
    return image[
        center_x - half_size_x : center_x + half_size_x,
        center_y - half_size_y : center_y + half_size_y
    ]

def generate_data_files(
    param_dict,
    output_dir="temp_oof_sim",
    jobname="model",
    channel_id="2000",
    wavelength="0.002",
    noise="0.008",
    m2z_offset=0.0,
    mapc=1,
    df=1000
):
    """
    Unified function to:
      1) Write zernike.dat via our known template and param_dict,
      2) Write subref.dat with M2Z offset,
      3) Call create_data_files,
      4) Return a (H, W, NUM_MODELS) np.array of cropped images.

    :param param_dict: dict of the 9 free Zernike terms (TILT_H, TILT_V, AST_V, etc.)
                       Typically, 'FOCUS' is left at zero in the Zernike template,
                       but if you want to override it, add 'FOCUS' to param_dict.
    :param output_dir: Directory to store the intermediate .dat and .fits files.
    :param jobname:    Base name for the output files (e.g. "model").
    :param channel_id: "2000" or "1100" etc.
    :param wavelength: "0.002", "0.0011", etc.
    :param noise:      e.g. "0.008".
    :param m2z_offset: The subreflector defocus offset in microns (like +150.0).
    :return: A float32 ndarray of shape (CROP_NX, CROP_NY, NUM_MODELS),
             or None if the external call fails.
    """
    modified_jobname = f"{jobname}_{channel_id}"
    os.makedirs(output_dir, exist_ok=True)

    NUM_MODELS = 2*mapc+1
    
    zernike_path = os.path.join(output_dir, f"{modified_jobname}_zernike.dat")
    subref_path = os.path.join(output_dir, f"{modified_jobname}_subref.dat")

    # 1) Write zernike.dat
    write_zernike_dat_file(zernike_path, param_dict)

    # 2) Write subreflector file
    with open(subref_path, 'w') as f:
        f.write("0 1 M2Z 0.000000\n")
        f.write("1 0 M2X 0.000000\n")
        f.write("2 0 M2Y 0.000000\n")

    # 3) External executable call
    args = [
        EXECUTABLE,
        modified_jobname,
        channel_id,
        wavelength,
        str(PIXEL_SIZE),
        str(NFT),
        str(MAP_SIZE_XY),
        str(MAP_SIZE_XY),
        os.path.basename(zernike_path),
        os.path.basename(subref_path),
        "1.0",     # gain factor
        "0",       # random phase
        noise,     # noise sigma
        f"{mapc}", # mapc (2*mapc + 1 maps)
        f"{m2z_offset:.2f}", # f0
        f"{df}",    # df
        "0",       # M2X slope
        "0",       # M2Y slope
        "0",       # use transfer function?
        DUMMY_TRANSFER,
        "1"        # Use FFT
    ]
    print(f"[generate_data_files] Running: {' '.join(args)} in {output_dir}")

    try:
        subprocess.run(args, check=True, cwd=output_dir)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: create_data_files failed with return code {e.returncode}")
        return None

    # 4) Load each of the NUM_MODELS fits files, crop, stack
    cropped_data_images = []
    for i in range(NUM_MODELS):
        fits_path = os.path.join(output_dir, f"{modified_jobname}_DATA_{i}.fits")
        if not os.path.exists(fits_path):
            continue
        with fits.open(fits_path) as hdul:
            data = crop_center(hdul[0].data, (CROP_NX, CROP_NY))
            cropped_data_images.append(data.astype(np.float32))
        os.remove(fits_path)

    if not cropped_data_images:
        return None

    return np.stack(cropped_data_images, axis=-1)
