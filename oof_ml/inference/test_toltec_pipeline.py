#!/usr/bin/env python3
"""
Test TolTEC pipeline with an 8‑parameter model (tilts removed).
The ML model output is assumed to have 8 values in this order:
    [ AST_V, AST_O, COMA_H, COMA_V, TRE_O, TRE_V, SPH, M2z_offset ]
When generating simulation inputs, TILT_H and TILT_V are set to 0.
"""

import sys
sys.path.insert(0, "/work/toltec/wilson/OOF/LMTOOF/python")

import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.signal import correlate2d
from scipy.ndimage import shift

from TolTECBeamMap3 import TolTECBeamMap
from oof_ml.utils.data_generation_utils import generate_data_files

# ------------------------------------------------------------------
class MultiBandDataLoader:
    """
    Loads TolTEC data for multiple bands.
    """
    def __init__(self, config_dir, crop_size=(32, 32), bands=None):
        self.config_dir = Path(config_dir)
        self.crop_height, self.crop_width = crop_size
        self.bands = bands if bands else ["a2000", "a1400", "a1100"]
        self.images = {}
        self.fp = {}
        self.config = self._load_config()
        self._load_all_bands()

    def _load_config(self):
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _central_crop(self, image, crop_height, crop_width):
        height, width = image.shape[:2]
        start_y = (height - crop_height) // 2
        start_x = (width - crop_width) // 2
        return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    def _get_fits_data_for_band(self, band):
        path = self.config['path']
        obsnums = self.config['obsnums']
        fp = {}
        for obs in obsnums:
            raw_dir = Path(path) / str(obs) / "raw"
            pattern = f"toltec_*{band}*.fits"
            found = list(raw_dir.glob(pattern))
            if not found:
                raise FileNotFoundError(
                    f"No fits found for obsnum={obs}, band={band} with pattern='{pattern}'."
                )
            fits_file = str(found[0])
            fp[obs] = {
                'path': str(raw_dir),
                'file': fits_file,
                'tbm': TolTECBeamMap(fits_file, obs, 0, str(raw_dir), 0, 0),
            }
        return fp

    def _make_test_images_for_band(self, fp):
        images = np.zeros((1, self.crop_height, self.crop_width, 3))
        i = 0
        for obsnum in fp:
            im = fp[obsnum]['tbm'].signal
            im_cropped = self._central_crop(im, self.crop_height, self.crop_width)
            images[0, :, :, i] = im_cropped
            i += 1
        # Renormalize per-channel.
        for i in range(3):
            chan_max = images[0, :, :, i].max()
            if chan_max != 0:
                images[0, :, :, i] /= chan_max
        return images

    def _load_all_bands(self):
        for band in self.bands:
            fp_band = self._get_fits_data_for_band(band)
            self.fp[band] = fp_band
            self.images[band] = self._make_test_images_for_band(fp_band)


class ModelEvaluator:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.band_info_lookup = {
            "a2000": {"id": "2000", "wavelength": "0.002", "noise": "0.0"},
            "a1400": {"id": "1400", "wavelength": "0.0014", "noise": "0.0"},
            "a1100": {"id": "1100", "wavelength": "0.0011", "noise": "0.0"},
        }

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        return tf.keras.models.load_model(str(self.model_path))

    def run_prediction(self, images):
        print("Running predictions on input images...")
        return self.model.predict(images)

    def generate_images_from_prediction(self, prediction, band):
        """
        Given a prediction vector of length 8 (without tilts), generate a model image.
        The 8 numbers are interpreted as:
            [AST_V, AST_O, COMA_H, COMA_V, TRE_O, TRE_V, SPH, M2z_offset]
        We then form a parameter dictionary including tilts set to 0.
        """
        if band not in self.band_info_lookup:
            raise ValueError(f"Unknown band '{band}'.")
        band_info = self.band_info_lookup[band]

        # Unpack prediction: first 7 correspond to the free aberrations; 8th is defocus.
        pred_vals = prediction.astype("float32")
        if len(pred_vals) != 8:
            raise ValueError(f"Expected prediction length 8, got {len(pred_vals)}")
        # Build a full parameter dictionary expected by generate_data_files:
        param_dict = {
            "TILT_H": 0.0,   # tilts are fixed to zero
            "TILT_V": 0.0,
            "AST_V":  float(pred_vals[0]),
            "AST_O":  float(pred_vals[1]),
            "COMA_H": float(pred_vals[2]),
            "COMA_V": float(pred_vals[3]),
            "TRE_O":  float(pred_vals[4]),
            "TRE_V":  float(pred_vals[5]),
            "SPH":    float(pred_vals[6]),
        }
        m2z_offset = float(pred_vals[7])
        model_imgs = generate_data_files(
            param_dict=param_dict,
            output_dir="temp_oof_sim",  # scratch folder
            jobname="model",
            channel_id=band_info["id"],
            wavelength=band_info["wavelength"],
            noise=band_info["noise"],
            m2z_offset=m2z_offset
        )
        return model_imgs


class ImageAligner:
    @staticmethod
    def _find_best_offset(data, model):
        corr = correlate2d(data, model, mode='full')
        y0, x0 = np.unravel_index(np.argmax(corr), corr.shape)
        offset_y = y0 - (model.shape[0] - 1)
        offset_x = x0 - (model.shape[1] - 1)
        return offset_y, offset_x

    def align_images_and_compute_residuals(self, data_images, model_images):
        aligned_model = np.zeros_like(model_images)
        residuals = np.zeros_like(model_images)
        offsets = {}
        metrics = {}
        for i in range(3):
            data_chan = data_images[:, :, i]
            model_chan = model_images[:, :, i]
            offset_y, offset_x = self._find_best_offset(data_chan, model_chan)
            offsets[f"channel_{i}"] = {"offset_y": int(offset_y), "offset_x": int(offset_x)}
            shifted = shift(model_chan, shift=(offset_y, offset_x), order=0, mode='constant', cval=0)
            aligned_model[:, :, i] = shifted
            diff = data_chan - shifted
            residuals[:, :, i] = diff
            mse_val = float(np.mean(diff**2))
            corr_coef = float(np.corrcoef(data_chan.ravel(), shifted.ravel())[0, 1])
            metrics[f"channel_{i}"] = {"mse": mse_val, "pearson_corr": corr_coef}
        return aligned_model, residuals, offsets, metrics


class OOFAnalysisPipeline:
    """
    Runs the overall pipeline:
      1. Loads data for multiple bands.
      2. Runs the ML model (using a2000 data to determine parameters).
      3. Generates model images for each band.
      4. Aligns model images with the data.
      5. Saves results and a manifest.
    """
    def __init__(self, 
                 config_dir='fg131833',
                 model_path='../results_250330/zernike_predictor_best.keras',
                 output_dir='.',
                 crop_size=(32, 32),
                 bands=None):
        self.config_dir = Path(config_dir)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size
        self.bands = bands if bands else ["a2000", "a1400", "a1100"]
        self.data_loader = MultiBandDataLoader(self.config_dir, crop_size=self.crop_size, bands=self.bands)
        self.model_evaluator = ModelEvaluator(self.model_path)
        self.aligner = ImageAligner()
        self.band_results = {}
        self.zernikes = None

    def run(self):
        # Use a2000 data to get the ML-predicted parameters.
        a2000_images = self.data_loader.images["a2000"]  # shape (1,H,W,3)
        prediction_full = self.model_evaluator.run_prediction(a2000_images)
        # The model output should be a vector of length 8.
        self.zernikes = prediction_full[0]
        print("Predicted zernike parameters:", self.zernikes)

        # For each band, generate and align model images.
        for band in self.bands:
            data_images = self.data_loader.images[band][0]
            model_imgs_raw = self.model_evaluator.generate_images_from_prediction(self.zernikes, band)
            in_focus_amplitude = float(model_imgs_raw[:, :, 1].max())
            model_imgs_norm = np.zeros_like(model_imgs_raw)
            for chan_idx in range(3):
                cmax = model_imgs_raw[:, :, chan_idx].max()
                if cmax != 0:
                    model_imgs_norm[:, :, chan_idx] = model_imgs_raw[:, :, chan_idx] / cmax
                else:
                    model_imgs_norm[:, :, chan_idx] = model_imgs_raw[:, :, chan_idx]
            aligned_model, residuals, offsets, metrics = self.aligner.align_images_and_compute_residuals(data_images, model_imgs_norm)
            self.band_results[band] = {
                "data_images": data_images,
                "model_images_raw": model_imgs_raw,
                "model_images_norm": model_imgs_norm,
                "aligned_model": aligned_model,
                "residuals": residuals,
                "offsets": offsets,
                "metrics": metrics,
                "in_focus_gain": in_focus_amplitude,
            }
        self._save_results()

    def _save_results(self):
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'r') as f:
            config_contents = f.read()
        config_dir_name = self.config_dir.name

        # Report the 8 output parameters (tilts have been removed):
        zernike_names = ["AST_V", "AST_O", "COMA_H", "COMA_V", "TRE_O", "TRE_V", "SPH", "M2z_offset"]

        manifest = {
            "description": (
                "Analysis performed for multiple bands (a2000, a1400, a1100) using a model that predicts 8 parameters: "
                "7 aberrations (AST_V, AST_O, COMA_H, COMA_V, TRE_O, TRE_V, SPH) and defocus (M2z_offset). "
                "TILT_H and TILT_V are fixed at zero. Alignment metrics (MSE, Pearson correlation) are computed."
            ),
            "bands": self.bands,
            "zernikes": self.zernikes.tolist(),
            "config_contents": config_contents,
            "config_directory": config_dir_name,
            "zernike_names": zernike_names,
        }
        manifest_path = self.output_dir / "results_manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, sort_keys=False)
        print(f"Manifest saved to {manifest_path}")

        npz_path = self.output_dir / "results.npz"
        np.savez(npz_path, band_results=self.band_results, zernikes=self.zernikes, allow_pickle=True)
        print(f"Results saved to {npz_path}")

if __name__ == "__main__":
    pipeline = OOFAnalysisPipeline(
        config_dir="fg131833",
        model_path="../results_250330/zernike_predictor_best.keras",
        output_dir=".",
        crop_size=(32, 32),
        bands=["a2000", "a1400", "a1100"]
    )
    pipeline.run()
    print("Multi-band analysis complete.")
