# OzoneKBNet

OzoneKBNet is a retrieval-augmented historical knowledge-base framework for **48-hour station-level ozone forecasting across urban agglomerations**.

The framework organizes historical pollutant–meteorology sequences from multiple monitoring stations into an explicit regional knowledge base. It retrieves multi-scale historical analogs, re-ranks them using representation similarity and local-trend consistency, aggregates their future trajectories, and integrates the retrieval forecast with a lightweight direct forecasting branch through reliability-aware residual correction.

Repository: https://github.com/mingkun-zhu/OzoneKBNet

## Main Features

- Urban-agglomeration-level historical knowledge-base construction
- Multi-scale retrieval encoder pretraining
- FAISS-based historical analog recall
- Candidate re-ranking using embedding similarity and local-trend consistency
- Multi-scale analog-future aggregation
- Horizon-wise multi-scale fusion
- Lightweight LSTM direct forecasting branch
- Reliability-aware, direct-biased residual correction
- Two-stage cross-year training and evaluation pipeline
- Station-restricted retrieval option for knowledge-base scope analysis
- Per-sample and regional metric export
- Computational profiling of model size, storage, latency, and GPU memory
- Complete fixed 2025 test set containing 4,836 station-level samples

## Experimental Protocol

The main experiments use a fixed cross-year temporal-transfer protocol:

1. **2023:** pretrain the multi-scale retrieval encoders and construct the historical knowledge base.
2. **2024:** freeze the retrieval encoders and FAISS indices, then train the Stage-2 forecasting modules.
3. **2025:** evaluate the trained model on the fixed test samples without using any 2025 information for training, normalization, or knowledge-base construction.

The seven urban agglomerations are processed independently. Separate knowledge bases, model checkpoints, and evaluation outputs are used for each region.

## Repository Structure

```text
OzoneKBNet/
├── data/                          # User-prepared training and evaluation data
├── data_provider/                 # Data loading and sample construction
├── exp/                           # Two-stage experiment pipeline
├── models/                        # OzoneKBNet model definition
├── scripts/                       # Example shell scripts
├── test_data/                     # Complete fixed 2025 test set
├── utils/                         # Retrieval, mining, metrics, and preprocessing utilities
├── run_ozone_kb.py                # Main training and evaluation entry point
├── profile_computation_server.py  # Computational profiling entry point
├── requirements.txt
├── LICENSE
├── NOTICE
└── README.md
```

Runtime outputs are written to the following local directories by default:

```text
./caches
./checkpoints
./results
./profile_stats
```

These directories and large runtime assets are excluded from Git tracking.

## Requirements

The implementation requires Python, PyTorch, FAISS, and common scientific-computing packages.

A recommended environment can be created with:

```bash
conda create -n ozonekb python=3.11 -y
conda activate ozonekb
```

Install PyTorch according to the CUDA or CPU configuration of the local system. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The minimal dependency list includes:

```text
NumPy
pandas
SciPy
scikit-learn
tqdm
fastdtw
FAISS
```

GPU acceleration is recommended for retrieval-encoder pretraining and Stage-2 model training.

## Data Organization

### Full training and evaluation pipeline

To run the complete 2023–2025 pipeline, prepare the processed data under a root directory using the following structure:

```text
data/
├── data_for_train_2023/
│   └── {region}/
├── data_for_train_2024/
│   └── {region}/
├── data_for_test_2025/
│   └── {region}/
└── data_for_test_2025_samples/
    └── {region}/{station}/01.csv ... 12.csv
```

The five model input variables are:

```text
O3
NO2
PM2.5
temperature
relative_humidity
```

The input length is 96 hours and the forecasting horizon is 48 hours.

### Released fixed 2025 test set

The repository includes the complete fixed 2025 evaluation set under:

```text
test_data/data_for_test_2025_samples/
```

It contains:

- 403 monitoring stations
- 7 urban agglomerations
- 4,836 station-level samples
- 12 monthly samples per station
- 144 hourly rows per sample
- 96 historical input hours and 48 future target hours

The released regional directory names are:

| Directory | Urban agglomeration | Stations | Samples |
|---|---|---:|---:|
| `shanyang` | Shenyang Metropolitan Area | 35 | 420 |
| `huabei` | North China Plain | 67 | 804 |
| `guanzhong` | Guanzhong Plain | 22 | 264 |
| `sichuanpendi` | Sichuan Basin | 96 | 1,152 |
| `changsha` | Changsha Metropolitan Area | 21 | 252 |
| `zhusanjiao` | Pearl River Delta | 91 | 1,092 |
| `changsanjiao` | Yangtze River Delta | 71 | 852 |
| **Total** | — | **403** | **4,836** |

Each released CSV file has 144 rows and the following columns:

```text
time
O3
NO2
PM2.5
temperature
relative_humidity
part
relative_step
```

See `test_data/README.md` for details.

The fixed test set can be inspected directly. To use it as the evaluation root, pass:

```bash
--root_path ./test_data
```

Evaluation additionally requires the corresponding trained checkpoint, normalization statistics, knowledge-base cache, and FAISS indices.

## Data Availability Scope

The repository redistributes the complete processed fixed 2025 test set used in the manuscript.

It does not redistribute:

- original raw air-quality monitoring records;
- original ERA5 reanalysis files;
- complete 2023 data used for retrieval-encoder pretraining and knowledge-base construction;
- complete 2024 data used for Stage-2 supervised training;
- precomputed knowledge-base embeddings;
- FAISS indices;
- trained model checkpoints.

The raw air-quality and meteorological data should be obtained from the original providers described in the manuscript. Users who wish to reproduce the full training pipeline must prepare the processed 2023 and 2024 datasets according to the data-quality-control, station-filtering, and sample-construction procedures reported in the paper.

## Usage

### Display all command-line arguments

```bash
python run_ozone_kb.py --help
```

### Full pipeline

```bash
python run_ozone_kb.py \
  --mode full_pipeline \
  --city changsanjiao \
  --root_path ./data \
  --device cuda
```

### Retrieval-encoder pretraining and knowledge-base construction

```bash
python run_ozone_kb.py \
  --mode pretrain_kb \
  --city changsanjiao \
  --root_path ./data \
  --device cuda
```

or:

```bash
bash scripts/pretrain_kb_example.sh changsanjiao
```

### Stage-2 supervised training

```bash
python run_ozone_kb.py \
  --mode train_stage2 \
  --city changsanjiao \
  --root_path ./data \
  --device cuda
```

or:

```bash
bash scripts/train_stage2_example.sh changsanjiao
```

### Fixed 2025 evaluation

```bash
python run_ozone_kb.py \
  --mode evaluate \
  --city changsanjiao \
  --root_path ./test_data \
  --checkpoints ./checkpoints \
  --cache_root ./caches \
  --result_root ./results \
  --device cuda
```

or, when all required data and assets use the default paths:

```bash
bash scripts/eval_2025_example.sh changsanjiao
```

### Station-restricted retrieval

The default retrieval scope is the full urban-agglomeration knowledge base. A station-restricted variant can be evaluated with:

```bash
python run_ozone_kb.py \
  --mode evaluate \
  --city changsanjiao \
  --retrieval_scope station \
  --root_path ./test_data \
  --device cuda
```

## Main Arguments

| Argument | Description |
|---|---|
| `--mode` | `pretrain_kb`, `train_stage2`, `evaluate`, or `full_pipeline` |
| `--city` | Regional directory key |
| `--root_path` | Root directory of the processed data |
| `--result_root` | Directory for evaluation outputs |
| `--checkpoints` | Directory for model checkpoints |
| `--cache_root` | Directory for knowledge-base and FAISS caches |
| `--device` | Execution device, such as `cuda` or `cpu` |
| `--seed` | Random seed |
| `--kb_year` | Knowledge-base and retrieval-encoder pretraining year |
| `--train_year` | Stage-2 supervised training year |
| `--test_year` | Fixed evaluation year |
| `--retrieval_scope` | `city` for the full regional KB or `station` for same-station retrieval |
| `--seq_len` | Historical input length |
| `--pred_len` | Forecasting horizon |
| `--scales` | Temporal retrieval scales |
| `--coarse_top_m` | Number of coarse retrieval candidates |
| `--final_top_k` | Number of retained analogs after re-ranking |
| `--batch_size` | Stage-2 training batch size |
| `--branch_mode` | Full model, direct-only, or retrieval-only evaluation |
| `--direct_branch_type` | Direct forecasting branch type |
| `--direct_biased_gamma_max` | Maximum residual-correction strength |

The compatibility argument `--rolling_update` should remain `0` for the fixed 2023/2024/2025 protocol reported in the manuscript.

## Default Experimental Settings

The released code uses the following central settings for the reported framework:

```text
Knowledge-base year:       2023
Stage-2 training year:     2024
Test year:                 2025
Input length:              96 hours
Forecast horizon:          48 hours
Input variables:           5
Temporal scales:           1, 2, 4, 8
Coarse candidates:         200
Final retained analogs:    10
Stage-2 batch size:        64
Random seeds:              42, 43, 44
```

Additional hyperparameters are available through `run_ozone_kb.py --help` and are documented in the supplementary material of the paper.

## Outputs

Evaluation outputs are saved under the specified result directory. Depending on the selected mode, the pipeline may generate:

```text
city_metrics.csv
per_sample_metrics.csv
failed_samples.csv
overall_metrics.json
pred_o3.npy
true_o3.npy
```

The exported files support regional, station-level, sample-level, and high-ozone post-hoc analyses.

## Computational Profiling

The script `profile_computation_server.py` reports:

- total neural parameters;
- Stage-2 trainable parameters;
- knowledge-base storage;
- FAISS-index storage;
- Stage-2 checkpoint size;
- evaluation-style mean latency per sample;
- peak allocated GPU memory.

Example:

```bash
python profile_computation_server.py \
  --cities sichuanpendi \
  --root_path ./test_data \
  --result_root ./results \
  --checkpoints ./checkpoints \
  --cache_root ./caches \
  --out_dir ./profile_stats \
  --device cuda
```

The formal profiling protocol processes samples individually with an effective batch size of one, uses five warm-up samples, synchronizes CUDA around the timed pass, and excludes one-time loading of the model, knowledge base, FAISS indices, and checkpoint.

## Reproducibility Notes

Exact numerical reproduction depends on:

- use of the same raw data sources;
- identical station-quality-control and station-filtering rules;
- identical regional grouping;
- identical preprocessing and normalization statistics;
- identical 2023 knowledge-base construction;
- identical 2024 Stage-2 samples;
- identical fixed 2025 test samples;
- identical random seeds and training settings;
- hardware and software differences.

The complete fixed 2025 test set is included to make the evaluation sample selection transparent and to support direct inspection of the reported test protocol.

## Software Information

- **Software name:** OzoneKBNet
- **Developers:** Mingkun Zhu, Xiaoxia Han, Jinde Wu, Haonan Zhu, and Wenxin Chai
- **Programming language:** Python
- **First public release:** 2026
- **Repository:** https://github.com/mingkun-zhu/OzoneKBNet
- **License:** MIT License

## Citation

The code accompanies the following manuscript:

```bibtex
@article{zhu2026ozonekbnet,
  title  = {A Retrieval-Augmented Historical Knowledge-Base Framework with Residual Correction for Station-Level Ozone Forecasting across Urban Agglomerations},
  author = {Zhu, Mingkun and Han, Xiaoxia and Wu, Jinde and Zhu, Haonan and Chai, Wenxin},
  note   = {Manuscript submitted to Air Quality, Atmosphere \& Health},
  year   = {2026}
}
```

The citation entry will be updated after publication.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Contact

For questions about the implementation or data-processing protocol, please refer to the corresponding-author information in the manuscript.
