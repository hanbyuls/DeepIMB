# -*- coding: utf-8 -*-
"""
main.py

Main training and imputation script for DeepIMB.

This script performs the following steps:
1. Loads and preprocesses microbiome data.
2. Identifies confident vs. imputable data points using a Gamma-Normal mixture model.
3. Constructs design matrices based on nearest-neighbor taxa.
4. Runs a hyperparameter sweep to find the best neural network parameters.
5. Retrains a final model on the best parameters.
6. Predicts the values for the impute set and saves the completed data matrix.

Usage:
    python main.py --base_dir . --study Karlsson --condition t2d
"""

# ----------------------------
# Imports
# ----------------------------
import os
import csv
import time
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from tqdm.auto import tqdm

# SciPy for statistical modeling
from scipy.stats import gamma, norm
from scipy.special import digamma
from scipy.optimize import root_scalar
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

# TensorFlow / Keras for the neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ----------------------------
# Environment and Reproducibility Setup
# ----------------------------

def setup_environment(seed: int, gpu_id: str):
    """
    Configures the environment for reproducibility and GPU usage.

    Args:
        seed (int): The random seed for all libraries.
        gpu_id (str): The ID of the GPU to use. Set to "" to force CPU.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress verbose TF messages
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f" Reproducibility seed set to {seed}")

    if gpu_id:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_index = int(gpu_id)
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
                print(f" GPU {gpu_id} configured and memory growth enabled.")
            else:
                print(" No GPUs found. Falling back to CPU.")
        except (ValueError, RuntimeError) as e:
            print(f" GPU config issue: {e}. Falling back to default device.")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print(" CPU forced as per configuration.")

# ----------------------------
# Core Algorithmic Components
# ----------------------------

def gamma_norm_mix(y: np.ndarray, X: np.ndarray):
    """
    EM algorithm to fit a 1D Gamma-Normal mixture model.
    The normal component's mean is dependent on covariates.

    Args:
        y (np.ndarray): The target vector (a single taxon's abundances).
        X (np.ndarray): The covariate matrix.

    Returns:
        dict: A dictionary containing fitted parameters and responsibilities ('d').
    """
    # (The full implementation of your gamma_norm_mix function remains here)
    # This function is complex and self-contained, so no changes are needed.
    def loglik(p, alpha, beta, cov_par, var1, X, y):
        fgam 	 = gamma.pdf(y, a=alpha, scale=1.0 / beta)
        norm_pdf = norm.pdf(y, loc=X @ cov_par, scale=np.sqrt(var1))
        mix 	 = np.clip(p * fgam + (1 - p) * norm_pdf, 1e-12, None)
        return np.sum(np.log10(mix))

    n = y.shape[0]
    alpha_t, beta_t, p_t = 1.0, 10.0, 0.5
    XtX = X.T @ X
    Xty = X.T @ y
    cov_par_t = np.linalg.solve(XtX, Xty)
    var_t 	  = np.sum((y - X @ cov_par_t) ** 2) / n

    def update_gmm_pars(x, wt):
        if np.max(wt) > 1e-5:
            tp_s = wt.sum()
            tp_t = (wt * x).sum()
            tp_u = (wt * np.log(np.clip(x, 1e-12, None))).sum()
            tp_v = -tp_u / tp_s - np.log(tp_s / tp_t)
            if tp_v <= 0:
                alpha = 20.0
            else:
                alpha0 = (3 - tp_v + np.sqrt((tp_v - 3) ** 2 + 24 * tp_v)) / (12 * tp_v)
                if alpha0 >= 20:
                    alpha = 20.0
                else:
                    root = root_scalar(lambda a: np.log(a) - digamma(a) - tp_v,
                                       bracket=[0.9 * alpha0, 1.1 * alpha0],
                                       method="brentq")
                    alpha = float(root.root)
            beta = tp_s / tp_t * alpha
        else:
            alpha, beta = 1e-3, 1e3
        return alpha, beta

    maxitr, itr = 300, 0
    prev_ll = -np.inf
    while True:
        mean_t   = X @ cov_par_t
        dg_t 	 = gamma.pdf(y, a=alpha_t, scale=1.0 / beta_t)
        norm_pdf = norm.pdf(y, loc=mean_t, scale=np.sqrt(var_t))
        denom 	 = np.clip(p_t * dg_t + (1 - p_t) * norm_pdf, 1e-12, None)
        a_hat_t  = np.clip((p_t * dg_t) / denom, 1e-10, 1 - 1e-10)

        p_t1 = a_hat_t.mean()
        w 	 = np.sqrt(1 - a_hat_t)
        Xw   = X * w[:, None]
        yw   = y * w
        try:
            cov_par_t1 = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
        except np.linalg.LinAlgError:
            return {"d": (y < (np.log10(1.01) + 1e-3)).astype(float)}

        var_t1 = np.sum((1 - a_hat_t) * (y - X @ cov_par_t) ** 2) / np.sum(1 - a_hat_t)
        alpha_t1, beta_t1 = update_gmm_pars(y, a_hat_t)
        ll_new = loglik(p_t1, alpha_t1, beta_t1, cov_par_t1, var_t1, X, y)
        if np.isfinite(prev_ll) and (abs(ll_new - prev_ll) < 0.05 or itr >= maxitr):
            p_t, alpha_t, beta_t, cov_par_t, var_t = p_t1, alpha_t1, beta_t1, cov_par_t1, var_t1
            break
        p_t, alpha_t, beta_t, cov_par_t, var_t = p_t1, alpha_t1, beta_t1, cov_par_t1, var_t1
        prev_ll = ll_new
        itr += 1

    eta_hat   = np.linalg.solve(XtX, Xty)
    omega_hat = np.sum((y - X @ eta_hat) ** 2) / n
    norm_ll   = np.sum(np.log10(norm.pdf(y, loc=X @ eta_hat, scale=np.sqrt(omega_hat)) + 1e-12))
    Dev 	  = -2 * norm_ll - (-2 * prev_ll)

    if (p_t == 0) or (alpha_t / beta_t > 1):
        a_hat_t = np.zeros_like(a_hat_t)

    return {
        "p": p_t, "alpha": alpha_t, "beta": beta_t,
        "cov_par": cov_par_t, "var": var_t, "d": a_hat_t,
        "eta": eta_hat, "omega": omega_hat, "Deviance": Dev
    }


# ----------------------------
# Keras Model Components
# ----------------------------

# Define a log10 constant compatible with TensorFlow
LOG10_ONE01 = np.log(1.01) / np.log(10.0)

# Custom loss functions
def wMSE(y_true, y_pred):
    """Weighted Mean Squared Error."""
    weights = tf.cast(y_true, tf.float32) + 1e-8
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

def MSE(y_true, y_pred):
    """Standard Mean Squared Error."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

LOSS_FUNCTIONS = {"wMSE": wMSE, "MSE": MSE}


def build_model(input_shape, learning_rate, dropout_rate, num_layers, loss_function_name):
    """
    Builds and compiles the Keras sequential model.
    """
    model = keras.Sequential()
    neuron_sizes = [2048 // (2 ** i) for i in range(num_layers)]

    model.add(layers.Dense(neuron_sizes[0], activation='relu', input_shape=[input_shape]))
    model.add(layers.Dropout(dropout_rate))

    for size in neuron_sizes[1:]:
        model.add(layers.Dense(size, activation='relu'))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=LOSS_FUNCTIONS[loss_function_name],
        metrics=['mae', 'mse']
    )
    return model


class TqdmEpochCallback(callbacks.Callback):
    """Keras callback for a per-epoch tqdm progress bar."""
    def __init__(self, total_epochs, desc):
        super().__init__()
        self.bar = tqdm(total=total_epochs, desc=desc, leave=False, unit="epoch")
    def on_epoch_end(self, epoch, logs=None):
        self.bar.update(1)
        if logs:
            self.bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})
    def on_train_end(self, logs=None):
        self.bar.close()

# ----------------------------
# Main Workflow Functions
# ----------------------------

def load_and_preprocess_data(data_dir, study, condition, k_neighbors):
    """
    Loads data, preprocesses it, and identifies confidence/impute sets.
    """
    print("\n--- 1. Loading and Preprocessing Data ---")
    # Define filenames
    f_comp = data_dir / f"{study}_{condition}_k_{k_neighbors}_comp_otu.csv"
    f_zi   = data_dir / f"{study}_{condition}_k_{k_neighbors}_zi_otu_mb.csv"
    f_meta = data_dir / f"{study}_meta_data_{condition}.csv"
    f_D    = data_dir / f"{study}_{condition}_D.csv"

    # Load data
    # otable_complete = pd.read_csv(f_comp, index_col=0)
    otable          = pd.read_csv(f_zi,   index_col=0)
    meta_data       = pd.read_csv(f_meta, index_col=0)
    D               = pd.read_csv(f_D,    index_col=0)
    print(f"Loaded OTU table with shape: {otable.shape}")

    # Filter out IGT samples
    mask_keep = meta_data["study_condition"] != "IGT"
    otable    = otable.loc[mask_keep, :]
    meta_data = meta_data.loc[mask_keep, :]

    # Assign data for processing
    y_sim = otable.copy()
    x     = meta_data.copy()

    # Identify taxa with >5% non-zeros
    nz_frac = (y_sim.values > (np.log10(1.01) + 1e-6)).sum(axis=0) / y_sim.shape[0]
    keep_idx = np.where((nz_frac > 0.05))[0]
    y_sim = y_sim.iloc[:, keep_idx]
    D     = D.iloc[keep_idx, keep_idx]
    print(f"Filtered to {y_sim.shape[1]} taxa with >5% non-zeros.")

    # Prepare covariates (drop ID column)
    x = x.drop(columns=[x.columns[0]])

    # Identify confidence/impute sets using the GMM
    print("Identifying confidence/impute sets using Gamma-Normal mixture...")
    t0 = time.time()
    results = [gamma_norm_mix(y_sim.values[:, j], x.values)["d"] for j in tqdm(range(y_sim.shape[1]), desc="Fitting GMM")]
    confidence_set, impute_set = [], []
    for j, d in enumerate(results):
        confidence_set.extend([[i, j] for i in np.where(d < 0.5)[0]])
        impute_set.extend([[i, j] for i in np.where(d > 0.5)[0]])
    print(f"Done in {time.time() - t0:.2f}s. Found {len(confidence_set)} confident and {len(impute_set)} imputable points.")

    return y_sim, x, D, np.array(confidence_set), np.array(impute_set)


def create_design_matrices(y_sim, x, D, confidence_set, impute_set, k_neighbors):
    """
    Generates sparse design matrices for training and testing.
    """
    print("\n--- 2. Creating Design Matrices ---")
    t0 = time.time()
    n, m = y_sim.shape
    p = x.shape[1]
    row_length = m * k_neighbors + n * (n - 1) + n * p
    yv = y_sim.values
    xv = x.values

    # Find k-nearest neighbors for each taxon
    D_array = D.to_numpy()
    close_taxa = []
    for j in range(m):
        order = np.argsort(D_array[:, j], kind="mergesort")
        close_taxa.append(order[order != j][:k_neighbors].tolist())

    def generate_matrix(index_pairs):
        rows, cols, data = [], [], []
        for r, (i, j) in enumerate(tqdm(index_pairs, desc="Building Matrix Rows")):
            # Block 1: Neighboring taxa values for sample i
            block1_cols = list(range(j * k_neighbors, (j + 1) * k_neighbors))
            block1_data = yv[i, close_taxa[j]]

            # Block 2: Other samples' values for taxon j
            block2_cols = list(range(m * k_neighbors + i * (n - 1), m * k_neighbors + (i + 1) * (n - 1)))
            block2_data = yv[np.arange(n) != i, j]

            # Block 3: Covariate values for sample i
            block3_cols = list(range(m * k_neighbors + n * (n - 1) + p * i, m * k_neighbors + n * (n - 1) + p * (i + 1)))
            block3_data = xv[i, :]

            all_cols = block1_cols + block2_cols + block3_cols
            all_data = np.concatenate((block1_data, block2_data, block3_data))

            rows.extend([r] * len(all_cols))
            cols.extend(all_cols)
            data.extend(all_data)

        return coo_matrix((data, (rows, cols)), shape=(len(index_pairs), row_length))

    design_mat_confidence = generate_matrix(confidence_set)
    design_mat_impute = generate_matrix(impute_set)

    print(f"Done in {time.time() - t0:.2f}s.")
    print(f"Confidence matrix shape: {design_mat_confidence.shape}")
    print(f"Impute matrix shape: {design_mat_impute.shape}")

    # Extract response vectors
    response_confidence = np.array([yv[i, j] for i, j in confidence_set])
    response_impute = np.array([yv[i, j] for i, j in impute_set])

    return design_mat_confidence.tocsr(), response_confidence, design_mat_impute.tocsr(), response_impute


def run_hyperparameter_sweep(train_x, train_y, configs, epochs, results_dir):
    """
    Performs a grid search over the provided hyperparameter configurations.
    """
    print("\n--- 3. Running Hyperparameter Sweep ---")
    sweep_csv = results_dir / "hyperparameter_sweep_progress.csv"
    csv_header = ['Learning Rate', 'Dropout Rate', 'Batch Size', 'Num Layers',
                  'Loss Function', 'Val Loss', 'Val MAE', 'Training Time (s)']
    if not sweep_csv.exists():
        with open(sweep_csv, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    results = []
    input_shape = train_x.shape[1]

    with tqdm(total=len(configs), desc="Total Trials") as outer_bar:
        for lr, dr, bs, nl, ls_name in configs:
            t_start = time.time()
            model = build_model(input_shape, lr, dr, nl, ls_name)
            tag = f"LR={lr},DR={dr},BS={bs},NL={nl},L={ls_name}"
            epoch_cb = TqdmEpochCallback(total_epochs=epochs, desc=tag)

            # Manually split the data
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                train_x, train_y, test_size=0.2, random_state=42
            )

            hist = model.fit(
                X_train_split, y_train_split,
                epochs=epochs,
                batch_size=bs,
                validation_data=(X_val_split, y_val_split),
                verbose=0,
                callbacks=[
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    epoch_cb
                ]
            )

            elapsed = time.time() - t_start

            val_loss = float(np.min(hist.history['val_loss']))
            val_mae = float(np.min(hist.history['val_mae']))
            row = (lr, dr, bs, nl, ls_name, val_loss, val_mae, round(elapsed, 3))
            results.append(row)

            # Log to live CSV
            with open(sweep_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)

            outer_bar.set_postfix({"best_val_mae": f"{val_mae:.4f}"})
            outer_bar.update(1)

    results_df = pd.DataFrame(results, columns=csv_header)
    best_params = results_df.loc[results_df['Val MAE'].idxmin()]
    print("\n Best Hyperparameters Found:")
    print(best_params)
    return best_params.to_dict()


def train_and_evaluate_final_model(best_params, train_x, train_y, test_x, test_y, results_dir):
    """
    Trains the final model on all data, evaluates, and returns predictions.
    """
    print("\n--- 4. Training and Evaluating Final Model ---")
    # Extract best params
    lr = float(best_params['Learning Rate'])
    dr = float(best_params['Dropout Rate'])
    bs = int(best_params['Batch Size'])
    nl = int(best_params['Num Layers'])
    ls_name = best_params['Loss Function']

    final_model = build_model(train_x.shape[1], lr, dr, nl, ls_name)
    final_epochs = 300 # Can be set as an argument if desired
    epoch_cb = TqdmEpochCallback(total_epochs=final_epochs, desc="Final Model Training")

    # Manually split the data for final training
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42
    )

    history = final_model.fit(
        X_train_split, y_train_split,
        epochs=final_epochs,
        batch_size=bs,
        validation_data=(X_val_split, y_val_split),
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.CSVLogger(str(results_dir / "final_model_history.csv")),
            epoch_cb
        ]
    )
    print("Final model training complete.")

    # Evaluate on the test set (impute set)
    loss, mae, mse = final_model.evaluate(test_x, test_y, verbose=0)
    print(f"\nFinal Model Evaluation on Impute Set: MAE = {mae:.4f}, Loss = {loss:.4f}")

    # Predict values
    print("Generating predictions for the impute set...")
    predictions = final_model.predict(test_x, batch_size=bs * 4, verbose=0).ravel()
    return predictions


def save_results(predictions, y_sim, confidence_set, impute_set, results_df, results_dir):
    """
    Reconstructs the final imputed matrix and saves all results to disk.
    """
    print("\n--- 5. Saving Final Results ---")
    # Create a DataFrame for all points
    conf_df = pd.DataFrame({
        'i': confidence_set[:, 0],
        'j': confidence_set[:, 1],
        'value': [y_sim.iat[i, j] for i, j in confidence_set]
    })
    impute_df = pd.DataFrame({
        'i': impute_set[:, 0],
        'j': impute_set[:, 1],
        'value': predictions
    })
    
    # Combine, sort, and pivot to reconstruct the matrix
    full_df = pd.concat([conf_df, impute_df], ignore_index=True)
    pivoted_df = full_df.pivot(index='i', columns='j', values='value')

    # Restore original sample and taxa names
    pivoted_df.index = y_sim.index[pivoted_df.index]
    pivoted_df.columns = y_sim.columns[pivoted_df.columns]
    pivoted_df = pivoted_df.reindex(index=y_sim.index, columns=y_sim.columns)

    # Define output paths
    imputed_path = results_dir / "pred_DeepIMB.csv"
    results_path = results_dir / "hyperparameter_tuning_results.csv"

    # Save to CSV
    pivoted_df.to_csv(imputed_path)
    results_df.to_csv(results_path, index=False)

    print(f" Final imputed matrix saved to: {imputed_path}")
    print(f" Hyperparameter results saved to: {results_path}")

# ----------------------------
# Argument Parsing and Main Execution
# ----------------------------

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DeepIMB Imputation Model Training Script")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing 'data' and 'results' folders.")
    parser.add_argument("--study", type=str, required=True, help="Study name (e.g., 'Karlsson').")
    parser.add_argument("--condition", type=str, required=True, help="Study condition (e.g., 't2d').")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of nearest-neighbor taxa to use as features.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for hyperparameter sweep.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use (set to '' for CPU).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    """Main script execution flow."""
    args = parse_args()
    
    # Setup
    setup_environment(args.seed, args.gpu_id)
    BASE_DIR = Path(args.base_dir).resolve()
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Data Loading and Preprocessing ---
    y_sim, x, D, confidence_set, impute_set = load_and_preprocess_data(
        DATA_DIR, args.study, args.condition, args.k_neighbors
    )

    # --- Step 2: Design Matrix Creation ---
    train_x, train_y, test_x, test_y = create_design_matrices(
        y_sim, x, D, confidence_set, impute_set, args.k_neighbors
    )
    
    # --- Step 3: Hyperparameter Sweep ---
    # Define the grid of hyperparameters to search
    param_grid = {
        'learning_rates': [1e-4],
        'dropout_rates': [0.25],
        'batch_sizes': [32],
        'num_layers_list': [2],
        'loss_functions': ['wMSE']
    }
    configs = list(product(*param_grid.values()))
    
    best_params = run_hyperparameter_sweep(train_x, train_y, configs, args.epochs, RESULTS_DIR)
    
    # --- Step 4: Final Model Training and Prediction ---
    predictions = train_and_evaluate_final_model(
        best_params, train_x, train_y, test_x, test_y, RESULTS_DIR
    )

    # --- Step 5: Save All Results ---
    results_df = pd.DataFrame([best_params]) # Save the best params for record
    save_results(predictions, y_sim, confidence_set, impute_set, results_df, RESULTS_DIR)

    print("\n DeepIMB process completed successfully!")


if __name__ == "__main__":
    main()
    
