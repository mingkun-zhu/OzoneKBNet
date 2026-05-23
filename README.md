# OzoneKBNet

This repository provides the implementation of **OzoneKBNet**, a retrieval-augmented historical knowledge-base framework for station-level ozone forecasting across urban agglomerations.

OzoneKBNet organizes cross-station historical pollution--meteorology sequences into a city-cluster-level knowledge base, retrieves multi-scale historical analogs, and combines retrieved analog futures with a lightweight direct forecasting branch through reliability-aware residual correction.

Repository URL: https://github.com/mingkun-zhu/OzoneKBNet

## Software Availability

- **Software name:** OzoneKBNet
- **Developers:** Mingkun Zhu et al.
- **Year first available:** 2026
- **Programming language:** Python
- **Repository:** https://github.com/mingkun-zhu/OzoneKBNet
- **Cost:** Free for academic research
- **License:** MIT License
- **Contact:** Please refer to the corresponding author information in the manuscript.

## Main Features

The implementation includes:

- City-cluster historical knowledge-base construction
- Multi-scale retrieval encoder pretraining
- FAISS-based historical analog retrieval
- Candidate re-ranking using embedding similarity and local-trend consistency
- Multi-scale analog future aggregation
- Lightweight direct forecasting branch
- Reliability-aware residual correction
- Two-stage training and cross-year rolling evaluation
- Per-sample and city-level metric export
- Computational profiling for system-level deployment analysis

## Repository Structure

```text
.
├── data_provider/                 # Data loading and sample construction
├── exp/                           # Experiment pipeline
├── models/                        # OzoneKBNet model definition
├── scripts/                       # Example running scripts
├── utils/                         # FAISS, metrics, mining, normalization, and time-series utilities
├── run_ozone_kb.py                # Main entry point
├── profile_computation_server.py  # Computational profiling script
├── README.md
└── .gitignore
```

Runtime outputs such as caches, checkpoints, results, processed datasets, FAISS indices, and model weights are ignored by Git.

## Requirements

The code was developed in Python and depends on PyTorch, FAISS, and common scientific-computing packages.

A typical conda environment can be created as follows:

```bash
conda create -n ozonekb python=3.11 -y
conda activate ozonekb
```

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn tqdm fastdtw faiss-cpu
```

Please install PyTorch according to your local CUDA version from the official PyTorch installation instructions. For CPU-only execution, the CPU version of PyTorch can also be used, although GPU acceleration is recommended for model training.

Main dependencies include:

```text
Python
PyTorch
NumPy
pandas
SciPy
scikit-learn
tqdm
fastdtw
FAISS
```

## Data Availability and Preparation

The raw air-quality monitoring data and ERA5 reanalysis data used in the study are publicly accessible from their original data providers, as described in the manuscript. Due to data redistribution restrictions and file-size considerations, the raw data and full processed datasets are not redistributed in this repository.

Users should download the raw data from the corresponding public sources and prepare processed CSV files following the structure below:

```text
data/
├── data_for_train_2023/
│   └── {city}/
├── data_for_train_2024/
│   └── {city}/
├── data_for_test_2025/
│   └── {city}/
└── data_for_test_2025_samples/
    └── {city}/{station}/01.csv ... 12.csv
```

Each CSV file should contain hourly records of the variables used in the paper:

```text
time
O3
NO2
PM2.5
temperature
relative_humidity
```

The input window length is 96 hours, and the forecasting horizon is 48 hours. Metrics are computed after de-standardization in the physical concentration space.

This repository provides implementation code, running scripts, and data-format instructions for reproducing the experimental pipeline. Complete numerical reproduction requires users to prepare the processed data according to the data sources, quality-control rules, station filtering criteria, and sample construction protocol described in the manuscript.

## Experimental Protocol

The main experiments follow a cross-year temporal transfer protocol:

1. Build the city-cluster historical knowledge base using the knowledge-base year.
2. Pretrain multi-scale retrieval encoders.
3. Construct FAISS indices for historical analog retrieval.
4. Freeze retrieval encoders and indices.
5. Train stage-2 forecasting modules on the following year.
6. Evaluate on fixed 2025 test samples.

In the rolling evaluation setting, the knowledge base is updated to the next year while the stage-2 forecasting modules are not re-trained.

## Usage

### Full pipeline

```bash
python run_ozone_kb.py \
  --mode full_pipeline \
  --city changsanjiao \
  --device cuda
```

### Retrieval encoder pretraining and knowledge-base construction

```bash
bash scripts/pretrain_kb_example.sh
```

### Stage-2 supervised training

```bash
bash scripts/train_stage2_example.sh
```

### Evaluation

```bash
bash scripts/eval_2025_example.sh
```

## Main Arguments

Commonly used arguments include:

```text
--mode          Running mode, such as full_pipeline, pretrain_kb, train_stage2, or eval
--city          Target urban agglomeration
--root_path     Root directory of processed data
--result_root   Directory for saving evaluation results
--checkpoints   Directory for saving model checkpoints
--cache_root    Directory for knowledge-base and FAISS cache files
--device        Running device, such as cuda or cpu
```

Default paths are set to local relative directories:

```text
./data
./results
./checkpoints
./caches
```

## Outputs

The evaluation process exports the following files:

```text
city_metrics.csv
per_sample_metrics.csv
failed_samples.csv
overall_metrics.json
pred_o3.npy
true_o3.npy
```

These files are saved under the specified result directory and can be used to compute city-level, station-level, and per-sample evaluation statistics.

## Computational Profiling

The script `profile_computation_server.py` can be used to profile system-level computational characteristics, including:

- Total neural parameters
- Stage-2 trainable parameters
- Knowledge-base storage
- FAISS index storage
- Stage-2 checkpoint size
- Online inference time per sample
- Peak GPU memory usage

Example:

```bash
python profile_computation_server.py \
  --city sichuanpendi \
  --device cuda
```

The profiling results can be used to summarize the computational statistics reported in the manuscript and supplementary material.

## Reproducibility Notes

The implementation is designed to match the experimental protocol described in the manuscript. Exact numerical reproduction may depend on:

- Availability of the same raw data sources
- The same station quality-control rules
- The same station filtering and regional grouping criteria
- The same train/test sample construction protocol
- The same preprocessing and standardization settings
- Hardware and software environment differences
- Random seed settings

Users are encouraged to use fixed test samples and fixed random seeds when reproducing the reported results.

## Citation

If you use this code, please cite the corresponding paper after publication.

```bibtex
@article{ozonekbnet,
  title   = {Retrieval-Augmented Historical Knowledge-Base Forecasting for Station-Level Ozone Prediction across Urban Agglomerations},
  author  = {Zhu, Mingkun and others},
  journal = {Environmental Modelling & Software},
  year    = {TBD}
}
```

## License

This project is released under the MIT License for academic research use. Please see the `LICENSE` file for details.
