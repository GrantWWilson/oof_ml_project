#!/usr/bin/env python3
import os
import shutil
import argparse
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from astropy.io import fits
from pathlib import Path
from itertools import combinations

from oof_ml.utils.data_generation_utils import generate_data_files

class SyntheticDataGenerator:
    """
    A class to generate synthetic data batches using Latin Hypercube Sampling (LHS)
    and to augment the dataset with additional basis samples.
    
    The zernike.dat file (and subreflector file) are created by our shared
    'generate_data_files' utility.
    """

    # ------------------------------------------------------
    # Unified parameter definitions:
    #   "range":  (low, high) if LHS sampling is desired, else None
    #   "lhs":    Boolean indicating whether we actually sample it
    #   "default": The fallback or fixed value if lhs=False or range=None
    # ------------------------------------------------------
    PARAM_DEFINITIONS = {
        "TILT_H": {"range": (-250, 250), "lhs": False, "default": 0.0},   # Removed from LHS, 4/8/2025
        "TILT_V": {"range": (-250, 250), "lhs": False, "default": 0.0},   # Removed from LHS, 4/8/2025
        "AST_V":  {"range": (-300, 300), "lhs": True,  "default": 0.0},
        "AST_O":  {"range": (-300, 300), "lhs": True,  "default": 0.0},
        "COMA_H": {"range": (-300, 300), "lhs": True,  "default": 0.0},
        "COMA_V": {"range": (-300, 300), "lhs": True,  "default": 0.0},
        "TRE_O":  {"range": (-300, 300), "lhs": True,  "default": 0.0},
        "TRE_V":  {"range": (-300, 300), "lhs": True,  "default": 0.0},
        "SPH":    {"range": (-300, 300), "lhs": True,  "default": 0.0},
    }
    # Note: "M2z_offset" (defocus) is controlled separately during simulation.
    
    # Define the array/band settings, including noise levels.
    ARRAY_BANDS = [
        {"id": "2000", "wavelength": "0.002",  "noise": "0.005"},
        {"id": "1100", "wavelength": "0.0011", "noise": "0.008"},
    ]

    def __init__(self, first_batch=1, total_batches=100, batch_size=1000,
                 output_dir="data/synthetic"):
        self.first_batch = first_batch
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.output_dir = output_dir

        # Main storage location.
        self.DATA_DATASET_PATH = output_dir
        os.makedirs(self.DATA_DATASET_PATH, exist_ok=True)

        # Pre-compute the list of parameters that need LHS sampling.
        self.lhs_param_names = [
            p for p, info in self.PARAM_DEFINITIONS.items()
            if info["lhs"] and info["range"] is not None
        ]
    
    def lhs_sampling(self, num_samples):
        """
        Perform Latin Hypercube Sampling for the parameters that have 'lhs': True.
        
        Returns a dict mapping param_name -> array of length num_samples.
        """
        d = len(self.lhs_param_names)
        sampler = LatinHypercube(d=d)
        sample_points = sampler.random(num_samples)  # shape (num_samples, d)
        result = {}
        for i, param_name in enumerate(self.lhs_param_names):
            low, high = self.PARAM_DEFINITIONS[param_name]["range"]
            vals = low + sample_points[:, i] * (high - low)
            result[param_name] = vals
        return result

    def build_param_dict_for_sample(self, lhs_samples, sample_idx):
        """
        Combine LHS-sampled values (if available) with fixed defaults,
        returning a dictionary of parameter values.
        """
        param_dict = {}
        for p, info in self.PARAM_DEFINITIONS.items():
            if info.get("lhs", False) and info.get("range") is not None:
                param_dict[p] = lhs_samples[p][sample_idx]
            else:
                param_dict[p] = info["default"]
        return param_dict

    def generate_data(self, jobname, output_dir, lhs_params_for_sample,
                      channel_id, wavelength, noise, M2z_offset=0.0):
        """
        Calls the shared utility to produce simulated PSF images.
        """
        modified_jobname = f"{jobname}_{channel_id}"
        return generate_data_files(
            param_dict=lhs_params_for_sample,
            output_dir=output_dir,
            jobname=modified_jobname,
            channel_id=channel_id,
            wavelength=wavelength,
            noise=noise,
            m2z_offset=M2z_offset
        )

    def run(self):
        """
        Main routine to generate batches of synthetic data via LHS.
        """
        total_needed = self.batch_size * self.total_batches
        lhs_samples = self.lhs_sampling(total_needed)
        for batch_idx in range(self.first_batch, self.first_batch + self.total_batches):
            batch_data_images = {band["id"]: [] for band in self.ARRAY_BANDS}
            batch_zernikes = {band["id"]: [] for band in self.ARRAY_BANDS}
            batch_subrefs = {band["id"]: [] for band in self.ARRAY_BANDS}
            batch_jobnames = {band["id"]: [] for band in self.ARRAY_BANDS}
            batch_dir = os.path.join(self.DATA_DATASET_PATH, f"batch_{batch_idx:03d}")
            os.makedirs(batch_dir, exist_ok=True)
            for sample_idx in range(self.batch_size):
                global_sample_idx = ((batch_idx - self.first_batch) * self.batch_size +
                                     sample_idx)
                base_jobname = f"sample_{batch_idx:03d}_{sample_idx:03d}"
                M2z_offset = np.random.uniform(-250, 250)
                param_dict = self.build_param_dict_for_sample(lhs_samples, global_sample_idx)
                for band in self.ARRAY_BANDS:
                    channel_id = band["id"]
                    data_images = self.generate_data(
                        base_jobname,
                        batch_dir,
                        param_dict,
                        channel_id,
                        band["wavelength"],
                        band["noise"],
                        M2z_offset
                    )
                    if data_images is None:
                        continue
                    batch_data_images[channel_id].append(data_images)
                    sample_zernikes = [param_dict[p] for p in self.lhs_param_names]
                    batch_zernikes[channel_id].append(sample_zernikes)
                    batch_subrefs[channel_id].append([M2z_offset, 0.0, 0.0])
                    batch_jobnames[channel_id].append(base_jobname)
            for band in self.ARRAY_BANDS:
                channel_id = band["id"]
                if batch_data_images[channel_id]:
                    outpath = os.path.join(
                        self.DATA_DATASET_PATH,
                        f"batch_{channel_id}_{batch_idx:03d}.npz"
                    )
                    np.savez_compressed(
                        outpath,
                        images=np.array(batch_data_images[channel_id]),
                        zernikes=np.array(batch_zernikes[channel_id]),
                        zernike_names=np.array(self.lhs_param_names),
                        subrefs=np.array(batch_subrefs[channel_id]),
                        subref_names=np.array(["M2Z", "M2X", "M2Y"]),
                        jobnames=np.array(batch_jobnames[channel_id])
                    )
            shutil.rmtree(batch_dir)
            print(f"Saved batch {batch_idx} for bands: {', '.join([b['id'] for b in self.ARRAY_BANDS])}")
        print("Data generation complete.")

    def generate_basis_samples(self, output_subdir="basis_aug",
                               amp_levels=[-150, -125, -100, -75, 75, 100, 125, 150],
                               basis_sizes=[1, 2]):  # use size 1 and 2 combinations
        """
        Generate extra "basis vector" samples where a specified number
        of parameters are set to a given amplitude and the remaining are at default.
        The results are saved as separate .npz files into DATA_DATASET_PATH/output_subdir.
        
        :param output_subdir: Directory inside DATA_DATASET_PATH for these augmented samples.
        :param amp_levels: List of amplitudes to try (in microns).
        :param basis_sizes: A list of how many parameters (from the candidate list) to be nonzero.
                            For example, 1 for isolated basis vectors, 2 for pair‐combinations.
                            (You can include 3 if you like, though starting with 1 and 2 is a good idea.)
        """
        out_dir = Path(self.DATA_DATASET_PATH) / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Candidate parameters: we consider only those that are sampled,
        # plus optionally "M2z_offset". 
        candidate_params = self.lhs_param_names.copy()
        # Optionally, include defocus as a candidate; here we include it.
        if "M2z_offset" not in candidate_params:
            candidate_params.append("M2z_offset")
        
        total_count = 0
        # For each basis size (number of non-zero parameters)
        for size in basis_sizes:
            for param_subset in combinations(candidate_params, size):
                for amp in amp_levels:
                    # Build parameter dictionary: use defaults for all, then set the chosen ones to 'amp'
                    param_dict = {}
                    for p, info in self.PARAM_DEFINITIONS.items():
                        param_dict[p] = info["default"]
                    # For any candidate not in PARAM_DEFINITIONS (e.g., "M2z_offset"), set 0
                    for p in candidate_params:
                        if p not in param_dict:
                            param_dict[p] = 0.0
                    for p in param_subset:
                        param_dict[p] = float(amp)
                    # For these basis samples, we fix M2z_offset to 0 if it isn’t part of the chosen subset.
                    if "M2z_offset" not in param_subset:
                        param_dict["M2z_offset"] = 0.0
                    jobname = "basis_" + "_".join(param_subset) + ("_p" if amp > 0 else "_m") + f"{abs(amp):.0f}"
                    # Use a fixed defocus offset for these evaluations
                    M2z_offset_val = param_dict.get("M2z_offset", 0.0)
                    # Use the "2000" channel
                    result = self.generate_data(
                        jobname=jobname,
                        output_dir=str(out_dir),
                        lhs_params_for_sample=param_dict,
                        channel_id="2000",
                        wavelength="0.002",
                        noise="0.005",
                        M2z_offset=M2z_offset_val
                    )
                    if result is not None:
                        total_count += 1
                        print(f"Generated basis sample: {jobname} with {param_subset} set to {amp}")
        print(f"Generated {total_count} basis sample files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PSF dataset batches with unified parameters and augmented basis samples.")
    parser.add_argument("--first-batch", type=int, default=1, help="Starting batch number for LHS data.")
    parser.add_argument("--total-batches", type=int, default=100, help="Total number of batches to generate.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of samples per batch.")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory for synthetic data.")
    parser.add_argument("--generate-basis", action="store_true", help="Also generate augmented basis samples.")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        first_batch=args.first_batch,
        total_batches=args.total_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    generator.run()
    if args.generate_basis:
        # Generate basis samples for 1 and 2 parameters at various amplitudes.
        generator.generate_basis_samples(basis_sizes=[1, 2])
