# oof_ml/utils/predict_utils.py
"""
Tiny helper to load a trained Keras model and run inference on
one *.npz file produced by our synthetic‑data pipeline.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

__all__ = ["predict_from_npz"]

def _normalise_by_peak(images: np.ndarray) -> np.ndarray:
    """
    Per‑channel peak normalisation (same rule as in training/inference).
    Operates in‑place and returns the array for convenience.
    """
    for i in range(images.shape[0]):
        for c in range(images.shape[-1]):
            mx = images[i, :, :, c].max()
            if mx > 0:
                images[i, :, :, c] /= mx
    return images


def predict_from_npz(model_path: str | Path,
                     npz_path: str | Path,
                     normalise: bool = True) -> np.ndarray:
    """
    Load *model_path* (Keras) and run it on the `images` stored in *npz_path*.

    Parameters
    ----------
    model_path : str or Path
        Trained Keras model (.keras or .h5).
    npz_path   : str or Path
        A *.npz file containing at least an `images` array shaped (N,H,W,C).
    normalise  : bool, default=True
        If True, apply per‑channel peak normalisation before prediction.

    Returns
    -------
    np.ndarray
        Predicted parameter array of shape (N, num_outputs).
    """
    model_path = Path(model_path)
    npz_path   = Path(npz_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file '{npz_path}' not found.")

    # 1) Load model
    model = tf.keras.models.load_model(str(model_path))

    # 2) Load images
    with np.load(npz_path) as data:
        if "images" not in data:
            raise KeyError(f"'images' array not found in {npz_path}")
        images = data["images"].astype("float32")  # ensure float32

    # 3) Optional normalisation
    if normalise:
        images = _normalise_by_peak(images)

    # 4) Predict
    preds = model.predict(images, verbose=0)
    return preds
