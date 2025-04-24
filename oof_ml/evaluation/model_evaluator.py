#!/usr/bin/env python3
import os
import sys
import yaml
import seaborn as sns
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # So we can write plots without an active display
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             explained_variance_score)
import tensorflow as tf

class ModelEvaluator:
    """
    A class to load a trained model, load test data from specified .npz files
    (filtered by band prefix and max batch num), compute evaluation metrics,
    and generate a PDF with plots.
    """
    def __init__(self,
                 model_path="zernike_predictor_final.keras",
                 dataset_path="data/synthetic",
                 band_filter="a2000",
                 max_batch_num=100,
                 output_dir="evaluation_results",
                 remove_focus=False):
        """
        :param model_path:   Path to the trained model file (.keras).
        :param dataset_path: Path to the .npz data files (test sets).
        :param band_filter:  E.g. "a2000" or "a1100" or "a2000+a1100"
        :param max_batch_num: Only load batch files with an index < max_batch_num.
        :param output_dir:    Root directory to store evaluation results.
        :param remove_focus:  If True, remove "FOCUS" from zernike_names and y_test.
        """
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.band_filter = band_filter
        self.max_batch_num = max_batch_num
        self.output_dir = Path(output_dir)
        self.remove_focus = remove_focus

        # We'll create a unique subfolder for these results with date/time stamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.output_dir / f"eval_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.pdf_path = self.run_dir / "model_evaluation.pdf"
        self.yaml_path = self.run_dir / "evaluation_summary.yaml"

        self.model = None
        self.zernike_names = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

        # We'll store some metadata for a YAML summary
        self.manifest = {
            "timestamp": timestamp,
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "band_filter": self.band_filter,
            "max_batch_num": self.max_batch_num,
            "remove_focus": self.remove_focus,
        }

    def load_model(self):
        """Loads the trained Keras model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        print(f"Loading model from {self.model_path} ...")
        self.model = tf.keras.models.load_model(str(self.model_path))


    def normalize_by_peak(self, images):
        """
        Normalizes each channel in each image by its peak pixel value.
        Expects images with shape (N, H, W, C).
        Also calls precondition_maps() if desired.
        """
        for i in range(images.shape[0]):
            for c in range(images.shape[-1]):
                max_val = images[i, :, :, c].max()
                if max_val > 0:
                    images[i, :, :, c] /= max_val
        return images

        
    def _load_files_for_band(self, band):
        """
        Helper function: loads .npz files for the given band (e.g. 'a2000'),
        with index < self.max_batch_num.
        Returns X_list, y_list, local_zernike_names.
        """
        # e.g. band='a2000' => suffix='2000'
        suffix = band.replace("a", "")
        pattern = f"batch_{suffix}_"

        files = sorted([f for f in os.listdir(self.dataset_path)
                        if f.startswith(pattern) and f.endswith(".npz")])
        selected = []
        for f in files:
            # e.g. "batch_2000_5.npz"
            parts = f.split("_")
            # parts => ["batch","2000","5.npz"]
            idx_str = parts[-1].split(".")[0]  # "5"
            idx = int(idx_str)
            if idx < self.max_batch_num:
                selected.append(f)

        if not selected:
            return [], [], None

        X_list = []
        y_list = []
        local_zernike_names = None

        for filename in selected:
            path_npz = self.dataset_path / filename
            data = np.load(path_npz)
            # zernike_names is an array of length ~9
            if local_zernike_names is None:
                local_zernike_names = data["zernike_names"]
            images = data["images"]  # shape: (samples, H, W, C)
            images = self.normalize_by_peak(images)
            zernikes = data["zernikes"]  # shape: (samples, #Z)
            m2_offset = data["subrefs"][:, 0].reshape(-1, 1)
            labels = np.concatenate([zernikes, m2_offset], axis=1)

            X_list.append(images)
            y_list.append(labels)

        X_comb = np.concatenate(X_list, axis=0)
        y_comb = np.concatenate(y_list, axis=0)
        return [X_comb], [y_comb], local_zernike_names

    def load_test_data(self):
        """
        Loads test data from .npz files using the specified band_filter.
        If band_filter contains '+', we load multiple bands.
        We combine all data into X_test, y_test, and unify zernike names.
        """
        all_X = []
        all_Y = []
        bands = [self.band_filter]
        if "+" in self.band_filter:
            bands = self.band_filter.split("+")

        zernike_names_local = None
        for b in bands:
            X_list, Y_list, local_names = self._load_files_for_band(b)
            if not X_list:
                print(f"WARNING: No test data found for band '{b}' up to batch_{self.max_batch_num-1}.")
                continue
            if zernike_names_local is None:
                zernike_names_local = local_names
            # each X_list, Y_list is a list of arrays, but we just appended them as single big arrays
            all_X.extend(X_list)
            all_Y.extend(Y_list)

        if not all_X:
            raise FileNotFoundError(f"No .npz files found matching band filter '{self.band_filter}' under {self.dataset_path}.")

        self.X_test = np.concatenate(all_X, axis=0)
        self.y_test = np.concatenate(all_Y, axis=0)
        # zernike_names_local is e.g. ["TILT_H", "TILT_V", ..., "SPH"] shape (9,)

        # The final label array is 9 zernikes + 1 M2z_offset => 10 columns
        # We'll define the names accordingly
        self.zernike_names = list(zernike_names_local) + ["M2z_offset"]

        if self.remove_focus and "FOCUS" in self.zernike_names:
            idx_focus = self.zernike_names.index("FOCUS")
            self.zernike_names.pop(idx_focus)
            self.y_test = np.delete(self.y_test, idx_focus, axis=1)

        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
        self.manifest["X_test_shape"] = list(self.X_test.shape)
        self.manifest["y_test_shape"] = list(self.y_test.shape)

    def run_predictions(self):
        """
        Runs the loaded model on X_test, storing the results in self.y_pred.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.X_test is None:
            raise RuntimeError("No test data. Call load_test_data() first.")
        print("Running predictions on test set...")
        self.y_pred = self.model.predict(self.X_test)

    def compute_metrics(self):
        """
        Computes various error metrics, returns a dictionary.
        """
        if self.y_pred is None:
            raise RuntimeError("No predictions found. Call run_predictions() first.")

        y_true = self.y_test
        y_pred = self.y_pred

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

        mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
        mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')
        expl_var = explained_variance_score(y_true, y_pred, multioutput='raw_values')
        medae = np.array([np.median(np.abs(y_true[:, i] - y_pred[:, i])) 
                          for i in range(y_true.shape[1])])
        max_err = np.array([np.max(np.abs(y_true[:, i] - y_pred[:, i])) 
                            for i in range(y_true.shape[1])])
        residuals = y_true - y_pred

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "explained_variance": expl_var,
            "median_ae": medae,
            "max_error": max_err,
            "residuals": residuals
        }
        return metrics

    def generate_plots(self, metrics):
        """
        Generates a PDF containing evaluation plots, storing it in self.pdf_path.
        Uses the same structure as your existing code.
        """
        zernike_names = self.zernike_names
        mae = metrics["mae"]
        mse = metrics["mse"]
        rmse = metrics["rmse"]
        r2 = metrics["r2"]
        expl_var = metrics["explained_variance"]
        medae = metrics["median_ae"]
        max_err = metrics["max_error"]
        residuals = metrics["residuals"]
        y_true = self.y_test
        y_pred = self.y_pred

        with PdfPages(str(self.pdf_path)) as pdf:
            # Page 1: MAE Bar Plot
            fig, ax = plt.subplots(figsize=(16, 9))
            plt.subplots_adjust(bottom=0.25)
            ax.bar(zernike_names, mae)
            ax.set_xlabel("Parameters")
            ax.set_ylabel("Mean Absolute Error")
            ax.set_title("MAE per Parameter")
            ax.tick_params(axis='x', rotation=45)
            description = (
                "MAE (Mean Absolute Error) measures the average absolute difference between the predicted and "
                "actual values. Lower values indicate better performance. A small spread across parameters "
                "suggests consistent predictions."
            )
            plt.figtext(0.5, 0.10, description, wrap=True, horizontalalignment='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: MSE and RMSE
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            plt.subplots_adjust(bottom=0.25)
            axs[0].bar(zernike_names, mse)
            axs[0].set_xlabel("Parameters")
            axs[0].set_ylabel("Mean Squared Error")
            axs[0].set_title("MSE per Parameter")
            axs[0].tick_params(axis='x', rotation=45)

            axs[1].bar(zernike_names, rmse)
            axs[1].set_xlabel("Parameters")
            axs[1].set_ylabel("Root Mean Squared Error")
            axs[1].set_title("RMSE per Parameter")
            axs[1].tick_params(axis='x', rotation=45)

            description = (
                "MSE (Mean Squared Error) squares the errors, penalizing larger deviations more than MAE, while "
                "RMSE (Root Mean Squared Error) converts this back to the original scale. Lower values indicate "
                "a better fit."
            )
            plt.figtext(0.5, 0.10, description, wrap=True, horizontalalignment='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: R2 and Explained Variance
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            plt.subplots_adjust(bottom=0.25)
            axs[0].bar(zernike_names, r2)
            axs[0].set_xlabel("Parameters")
            axs[0].set_ylabel("R² Score")
            axs[0].set_title("R² Score per Parameter")
            axs[0].tick_params(axis='x', rotation=45)

            axs[1].bar(zernike_names, expl_var)
            axs[1].set_xlabel("Parameters")
            axs[1].set_ylabel("Explained Variance")
            axs[1].set_title("Explained Variance per Parameter")
            axs[1].tick_params(axis='x', rotation=45)

            description = (
                "The R² Score indicates the proportion of variance in the data explained by the model, with 1 being ideal. "
                "Explained Variance measures how much variability is captured. High values (near 1) are desirable."
            )
            plt.figtext(0.5, 0.10, description, wrap=True, horizontalalignment='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 4: Median & Max Error
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            plt.subplots_adjust(bottom=0.25)
            axs[0].bar(zernike_names, medae)
            axs[0].set_xlabel("Parameters")
            axs[0].set_ylabel("Median Absolute Error")
            axs[0].set_title("Median Absolute Error per Parameter")
            axs[0].tick_params(axis='x', rotation=45)

            axs[1].bar(zernike_names, max_err)
            axs[1].set_xlabel("Parameters")
            axs[1].set_ylabel("Maximum Error")
            axs[1].set_title("Maximum Error per Parameter")
            axs[1].tick_params(axis='x', rotation=45)

            description = (
                "Median Absolute Error provides a robust central tendency of errors, while Maximum Error indicates "
                "the worst-case scenario. Ideally, both should be low."
            )
            plt.figtext(0.5, 0.10, description, wrap=True, horizontalalignment='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 5: Residual Distribution
            fig, ax = plt.subplots(figsize=(16, 9))
            plt.subplots_adjust(bottom=0.25)
            ax.hist(residuals.flatten(), bins=50, alpha=0.7)
            ax.set_xlabel("Residual Error")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Residual Errors")

            description = (
                "Shows the distribution of (y_true - y_pred). A symmetric distribution centered near zero suggests minimal bias."
            )
            plt.figtext(0.5, 0.10, description, wrap=True, horizontalalignment='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 6: Contour (Hexbin) for each param
            n_terms = len(zernike_names)
            plots_per_page = 4
            n_pages = (n_terms + plots_per_page - 1) // plots_per_page
            idx = 0
            for page in range(n_pages):
                fig, axs = plt.subplots(2, 2, figsize=(16, 9))
                plt.subplots_adjust(bottom=0.20, top=0.90, wspace=0.30, hspace=0.40)
                axs = axs.flatten()
                for i in range(plots_per_page):
                    term_index = page * plots_per_page + i
                    if term_index >= n_terms:
                        axs[i].axis('off')
                        continue
                    term = zernike_names[term_index]
                    ax = axs[i]
                    hb = ax.hexbin(y_true[:, term_index],
                                   y_pred[:, term_index] - y_true[:, term_index],
                                   gridsize=50, cmap='inferno', mincnt=1)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted - Actual")
                    ax.set_title(f"{term}")
                    fig.colorbar(hb, ax=ax)
                fig.suptitle("Hexbin: Actual vs. (Predicted-Actual)", fontsize=16)
                description = (
                    "Each plot compares the actual and residual for a parameter using a hexbin plot. "
                    "Concentration near zero indicates good predictions."
                )
                plt.figtext(0.5, 0.05, description, wrap=True, horizontalalignment='center', fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)

            # Page 7: Correlation Heatmap
            df_predictions = pd.DataFrame(y_pred, columns=zernike_names)
            corr_matrix = df_predictions.corr()

            fig, ax = plt.subplots(figsize=(16, 9))
            plt.subplots_adjust(bottom=0.25)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap of Predicted Parameters")
            description = (
                "Shows pairwise correlations between predicted parameters. High correlations may reveal "
                "underlying couplings or redundancies."
            )
            plt.figtext(0.5, 0.10, description, wrap=True, horizontalalignment='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Evaluation plots saved to {self.pdf_path}")

    def save_manifest(self, metrics):
        """
        Writes a YAML file summarizing the evaluation.
        """
        # E.g. store aggregated stats
        # We'll store average MAE, MSE, etc.
        self.manifest["mean_mae"] = float(np.mean(metrics["mae"]))
        self.manifest["mean_rmse"] = float(np.mean(metrics["rmse"]))
        self.manifest["mean_r2"] = float(np.mean(metrics["r2"]))

        with open(self.yaml_path, 'w') as f:
            yaml.dump(self.manifest, f, sort_keys=False)
        print(f"Evaluation summary saved to {self.yaml_path}")

    def run_evaluation(self):
        """
        Orchestrates the entire workflow:
          1) load model
          2) load test data
          3) predict
          4) compute metrics
          5) produce PDF
          6) produce YAML summary
        """
        self.load_model()
        self.load_test_data()
        self.run_predictions()
        metrics = self.compute_metrics()
        self.generate_plots(metrics)
        self.save_manifest(metrics)

