# DeepIMB

A reproducible implementation of the DeepIMB (Deep-learning Imputation for Microbiome) model. This repository provides the code and environment needed to replicate the results from our paper.

## 1. Installation

This project uses Conda to manage its environment.

1.  **Prerequisites:** You must have Conda (Miniconda or Anaconda) installed.
2.  **Create Environment:** From the root directory of this project, create and activate the Conda environment using the provided file:

    ```bash
    conda env create -f environment.yml
    conda activate deepimb
    ```

## 2. Data Structure

The model expects input data to be placed in a `data/` directory. The script is designed to work with four specific CSV files, where the format is as follows:

-   **OTU Tables (`..._zi_otu_mb.csv`, `..._comp_otu.csv`)**
    -   **Rows**: Samples (e.g., patients, experimental subjects)
    -   **Columns**: Taxa (e.g., microbial species, OTUs)
    -   **Values**: Abundance counts (in this project, on a log10 scale)

-   **Metadata Table (`..._meta_data.csv`)**
    -   **Rows**: Samples, which must correspond to the rows in the OTU tables.
    -   **Columns**: Covariates or sample attributes (e.g., age, sex, disease status).

-   **Distance Matrix (`..._D.csv`)**
    -   **Rows**: Taxa, which must correspond to the columns in the OTU tables.
    -   **Columns**: Taxa, same as the rows.
    -   **Values**: The distance (e.g., phyloge

The structure should be as follows:

```
DeepIMB/
├── data/
│   ├── Karlsson_t2d_k_5_comp_otu.csv
│   ├── Karlsson_t2d_k_5_zi_otu_mb.csv
│   ├── Karlsson_meta_data_t2d.csv
│   └── Karlsson_t2d_D.csv
├── results/
├── environment.yml
└── main.py
```

## 3. Usage

The main script can be run from the command line. Key parameters like the data directory, study name, and GPU can be configured via arguments.

**Example Command:**

```bash
# Ensure the 'deepimb' environment is active
python main.py --base_dir . --study Karlsson --condition t2d --gpu_id 0
```
-   `--base_dir`: Path to the project's root directory (containing `data/` and `results/`).
-   `--study`: The name of the study to process (e.g., "Karlsson").
-   `--condition`: The study condition (e.g., "t2d").
-   `--gpu_id`: The GPU to use for training (set to `""` to force CPU).

## 4. Results

All output files are automatically saved in the `results/` directory. This includes:
-   `pred_DeepIMB.csv`: The final, imputed data matrix with original sample and taxa names.
-   `hyperparameter_tuning_results.csv`: A summary of the best-performing model's parameters.
-   `final_model_history.csv`: A log of the training and validation metrics for the final model.
