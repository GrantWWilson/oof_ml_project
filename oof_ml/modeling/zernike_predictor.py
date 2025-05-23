#!/usr/bin/env python3
"""
ZernikePredictor — now with
  • basis-vector oversampling
  • loss-weighting
  • simple two-phase curriculum
"""

import os, sys, psutil, yaml, math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  -- ENV & THREADING -------------------------------------------------------
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # CPU-only
for var, n in [
    ("OMP_NUM_THREADS",32), ("TF_NUM_INTEROP_THREADS",32),
    ("TF_NUM_INTRAOP_THREADS",32), ("MKL_NUM_THREADS",32),
    ("OPENBLAS_NUM_THREADS",32), ("VECLIB_MAXIMUM_THREADS",32),
    ("NUMEXPR_NUM_THREADS",32)
]:
    os.environ[var] = str(n)

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
class ZernikePredictor:
    # ----------------------------- INIT ------------------------------------
    def __init__(
        self,
        dataset_path       = "data/synthetic_45m/train",
        working_directory  = "results_model",
        band_filter        = "a2000",
        batch_size         = 32,
        epochs             = 100,
        learning_rate      = 1e-3,
        shift_pixels       = 2.5,
        #
        basis_oversample   = 4,      # repeat each basis sample this many times
        basis_loss_weight  = 5.0,    # sample-weight for every basis sample
        curriculum_epochs  = 20      # first N epochs ⇒ no basis data
    ):
        self.dataset_path      = Path(dataset_path)
        self.working_directory = Path(working_directory)
        self.band_filter       = band_filter
        self.batch_size        = batch_size
        self.epochs            = epochs
        self.learning_rate     = learning_rate
        self.shift_pixels      = shift_pixels
        self.basis_oversample  = max(1, int(basis_oversample))
        self.basis_loss_weight = float(basis_loss_weight)
        self.curriculum_epochs = int(curriculum_epochs)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir    = self.working_directory / f"run_{ts}"
        self.models_dir = self.run_dir / "models"
        self.plots_dir  = self.run_dir / "plots"
        self.logs_dir   = self.run_dir / "logs"
        for d in (self.models_dir, self.plots_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Placeholders for data, model, parameter names, etc.
        self.X_data = self.y_data = self.sample_weights = None
        self.zernike_names = None
        self.model  = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

        self.manifest = {
            "timestamp"        : ts,
            "dataset_path"     : str(self.dataset_path),
            "band_filter"      : self.band_filter,
            "batch_size"       : self.batch_size,
            "epochs"           : self.epochs,
            "learning_rate"    : self.learning_rate,
            "shift_pixels"     : self.shift_pixels,
            "basis_oversample" : self.basis_oversample,
            "basis_loss_weight": self.basis_loss_weight,
            "curriculum_epochs": self.curriculum_epochs,
        }

    # ---------------------- helper utilities ------------------------------
    @staticmethod
    def log_mem(tag=""):
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        print(f"[MEM] {tag:>12}: {mem:6.2f} GB")

    def precondition_maps(self, imgs):
        return imgs

    def _norm_relative_to_zero(self, imgs, subrefs):
        """
        Normalize each sample by the peak of the channel whose M2-offset == 0,
        preserving relative amplitudes of the other channels.
        """
        imgs = self.precondition_maps(imgs)
        # imgs: (N, H, W, C), subrefs: (N,1) or (N, C)
        # ensure shape (N, C)
        offsets = subrefs if subrefs.ndim == 2 else np.tile(subrefs, (1, imgs.shape[-1]))
        # find the zero-offset channel index
        if (offsets[0] == 0).any():
            zero_idx = int(np.where(offsets[0] == 0)[0][0])
        else:
            zero_idx = imgs.shape[-1] // 2
        # compute per-sample peaks for each channel
        peaks = imgs.max(axis=(1,2))           # shape (N, C)
        ref_peaks = peaks[:, zero_idx]         # shape (N,)
        ref_peaks[ref_peaks <= 0] = 1.0        # avoid division by zero
        return imgs / ref_peaks[:, None, None, None]

    # ---------------------- DATA LOADING ----------------------------------
    def _collect_npz_files(self, subdir):
        files = []
        for band in self.band_filter.split("+"):
            suf = band.replace("a", "")
            files += sorted(Path(subdir).glob(f"batch_{suf}_*.npz"))
        return files

    def _load_npz_group(self, file_list, weight):
        X, Y, W = [], [], []
        for f in file_list:
            d = np.load(f)
            if self.zernike_names is None:
                self.zernike_names = list(d["zernike_names"]) + ["M2z_offset"]
            imgs = self._norm_relative_to_zero(d["images"], d["subrefs"])
            zern = d["zernikes"]
            m2   = d["subrefs"][:, 0:1]
            lbl  = np.concatenate([zern, m2], axis=1)
            n    = imgs.shape[0]
            X.append(imgs)
            Y.append(lbl)
            W.append(np.full((n,), weight, dtype="float32"))
        if not X:
            return None, None, None
        return np.concatenate(X), np.concatenate(Y), np.concatenate(W)

    def load_data(self, max_batch_num=9999):
        # Load regular (non-basis) files:
        main_files  = [
            f for f in self._collect_npz_files(self.dataset_path)
            if f.stem.split("_")[-1].isdigit()
               and int(f.stem.split("_")[-1]) < max_batch_num
               and "basis" not in f.parts
        ]
        X_main, Y_main, W_main = self._load_npz_group(main_files, 1.0)

        # Load basis files:
        basis_root = self.dataset_path / "basis"
        X_basis = Y_basis = W_basis = None
        if basis_root.is_dir():
            basis_files = [
                f for f in self._collect_npz_files(basis_root)
                if f.stem.split("_")[-1].isdigit()
                   and int(f.stem.split("_")[-1]) < max_batch_num
            ]
            X_basis, Y_basis, W_basis = self._load_npz_group(basis_files, self.basis_loss_weight)
            if self.basis_oversample > 1 and X_basis is not None:
                X_basis = np.repeat(X_basis, self.basis_oversample, axis=0)
                Y_basis = np.repeat(Y_basis, self.basis_oversample, axis=0)
                W_basis = np.repeat(W_basis, self.basis_oversample, axis=0)

        parts = [(X_main, Y_main, W_main), (X_basis, Y_basis, W_basis)]
        X = np.concatenate([p[0] for p in parts if p[0] is not None], axis=0)
        Y = np.concatenate([p[1] for p in parts if p[1] is not None], axis=0)
        W = np.concatenate([p[2] for p in parts if p[2] is not None], axis=0)

        self.X_data, self.y_data, self.sample_weights = X, Y, W
        self.manifest.update({
            "total_samples": int(X.shape[0]),
            "basis_fraction": float((W > 1.0).sum() / len(W)),
        })
        print(f"Loaded {X.shape[0]} samples (basis ≈ {self.manifest['basis_fraction'] * 100:.1f}%)")
        self.input_shape = X.shape[1:]
        self.num_outputs = Y.shape[1]
        self.manifest["input_shape"] = list(self.input_shape)
        self.manifest["num_outputs"] = self.num_outputs

        H, W_img, _ = self.input_shape
        frac = self.shift_pixels / float(min(H, W_img))
        self.data_augment = tf.keras.Sequential([
            layers.RandomTranslation(frac, frac, fill_mode="reflect")
        ])
        self.manifest["translation_fraction"] = frac

    # --------------------- DATASET BUILD ----------------------------------
    def _augment_map(self, img, lbl, wt):
        chans = tf.unstack(img, axis=-1)
        chans = [
            tf.squeeze(self.data_augment(tf.expand_dims(c, -1), training=True), -1)
            for c in chans
        ]
        return tf.stack(chans, axis=-1), lbl, wt

    def prepare_datasets(self, val_fraction=0.2, test_fraction=0.1):
        Xtr, Xtmp, Ytr, Ytmp, Wtr, Wtmp = train_test_split(
            self.X_data, self.y_data, self.sample_weights,
            test_size=val_fraction + test_fraction, random_state=42
        )
        Xv, Xte, Yv, Yte, Wv, Wte = train_test_split(
            Xtmp, Ytmp, Wtmp,
            test_size=test_fraction / (val_fraction + test_fraction), random_state=42
        )

        def make_ds(X, Y, W, augment):
            ds = tf.data.Dataset.from_tensor_slices((X, Y, W))
            if augment:
                ds = ds.map(self._augment_map, tf.data.AUTOTUNE)
            return ds.shuffle(8192).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        self.train_ds_full = make_ds(Xtr, Ytr, Wtr, augment=True)
        mask_main = (Wtr == 1.0)
        self.train_ds_main = make_ds(Xtr[mask_main], Ytr[mask_main], Wtr[mask_main], augment=True)
        self.val_ds = make_ds(Xv, Yv, Wv, augment=False)
        self.test_ds = make_ds(Xte, Yte, Wte, augment=False)

        self.manifest.update({
            "train_samples": int(len(Xtr)),
            "val_samples": int(len(Xv)),
            "test_samples": int(len(Xte))
        })

    # --------------------- MODEL ------------------------------------------
    def build_model(self):
        inp = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, padding="same")(inp); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(64, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(128, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling2D()(x)
        for units in (64, 32):
            x = layers.Dense(units)(x); x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x); x = layers.Dropout(0.2)(x)
        out = layers.Dense(self.num_outputs, activation="linear")(x)
        self.model = Model(inp, out)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse", metrics=["mae"]
        )
        self.manifest["architecture"] = "3×Conv32-64-128 → GAP → Dense(64)/Dense(32) → Dense(num_outputs)"

    # --------------------- TRAIN ------------------------------------------
    def train(self):
        ckpt_best = str(self.models_dir / "zernike_predictor_best.keras")
        ckpt_cb = ModelCheckpoint(ckpt_best, monitor="val_loss", save_best_only=True, verbose=1)
        lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

        history_all = {"loss": [], "val_loss": []}

        # Phase 1: Main-only curriculum (no basis samples)
        n1 = min(self.curriculum_epochs, self.epochs)
        if n1 > 0:
            print(f"\n[Curriculum] Phase-1 : {n1} epochs (no basis samples)")
            h = self.model.fit(self.train_ds_main, validation_data=self.val_ds,
                               epochs=n1, callbacks=[lr_cb, ckpt_cb], verbose=1)
            history_all["loss"] += h.history["loss"]
            history_all["val_loss"] += h.history["val_loss"]

        # Phase 2: Full dataset (including basis samples, weighted)
        n2 = self.epochs - n1
        if n2 > 0:
            print(f"\n[Curriculum] Phase-2 : {n2} epochs (full dataset with basis weights)")
            h = self.model.fit(self.train_ds_full, validation_data=self.val_ds,
                               initial_epoch=n1, epochs=self.epochs,
                               callbacks=[lr_cb, ckpt_cb], verbose=1)
            history_all["loss"] += h.history["loss"]
            history_all["val_loss"] += h.history["val_loss"]

        final_path = str(self.models_dir / "zernike_predictor_final.keras")
        self.model.save(final_path)

        plt.figure()
        plt.plot(history_all["loss"], label="train")
        plt.plot(history_all["val_loss"], label="val")
        plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Loss")
        plt.savefig(self.plots_dir / "loss_curve.png")
        plt.close()

        pd.DataFrame(history_all).to_csv(self.logs_dir / "training_history.csv", index=False)

        self.manifest.update({
            "best_model": ckpt_best,
            "final_model": final_path,
            "final_loss": float(history_all["loss"][-1]),
            "final_val_loss": float(history_all["val_loss"][-1]),
        })
        with open(self.run_dir / "train_manifest.yaml", "w") as f:
            yaml.dump(self.manifest, f, sort_keys=False)
        print("Training complete. Manifest written.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", default="data/synthetic_45m/train")
    p.add_argument("--working-directory", default="results_model")
    p.add_argument(
        "--band-filter",
        default="a2000",
        help="Which array(s) to train on. For multiple, separate with '+', e.g. 'a1100+a2000'"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--shift-pixels", type=float, default=2.5)
    p.add_argument("--max-batch-num", type=int, default=9999)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--basis-oversample", type=int, default=4)
    p.add_argument("--basis-loss-weight", type=float, default=5.0)
    p.add_argument("--curriculum-epochs", type=int, default=20)
    args = p.parse_args()

    predictor = ZernikePredictor(
        dataset_path      = args.dataset_path,
        working_directory = args.working_directory,
        band_filter       = args.band_filter,
        batch_size        = args.batch_size,
        epochs            = args.epochs,
        learning_rate     = args.learning_rate,
        shift_pixels      = args.shift_pixels,
        basis_oversample  = args.basis_oversample,
        basis_loss_weight = args.basis_loss_weight,
        curriculum_epochs = args.curriculum_epochs,
    )
    predictor.load_data(max_batch_num=args.max_batch_num)
    predictor.prepare_datasets(val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    predictor.build_model()
    predictor.train()
