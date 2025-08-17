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

The model expects input data to be placed in a `data/` directory within the project root. The structure should be as follows:

```
DeepIMB/
├── data/
│   ├── Karlsson_t2d_k_5_comp_otu.csv
│   ├── Karlsson_t2d_k_5_zi_otu_mb.csv
│   ├── Karlsson_meta_data_t2d.csv
│   └── Karlsson_t2d_D.csv
├── results/
├── environment.yml
└── main.py         # Or your script's name
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

All output files, including the final imputed data matrix, training logs, and hyperparameter tuning results, will be saved in the `results/` directory.