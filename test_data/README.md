# Fixed 2025 Test Set

This directory contains the complete fixed 2025 test set used to evaluate OzoneKBNet and all compared methods under the unified no-future-information protocol described in the manuscript.

## Dataset Summary

The released test set contains **4,836 station-level samples** from **403 monitoring stations** distributed across seven Chinese urban agglomerations. Each station contributes 12 fixed monthly samples.

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

## Directory Structure

```text
test_data/
├── README.md
└── data_for_test_2025_samples/
    └── {region}/
        └── {station}/
            ├── 01.csv
            ├── 02.csv
            ├── ...
            └── 12.csv
```

The file name indicates the monthly sample index for each station.

## Sample Format

Every sample CSV contains **144 consecutive hourly records** and **8 columns**:

| Column | Description |
|---|---|
| `time` | Timestamp of the hourly observation |
| `O3` | Ozone concentration |
| `NO2` | Nitrogen dioxide concentration |
| `PM2.5` | Fine particulate matter concentration |
| `temperature` | Air temperature |
| `relative_humidity` | Relative humidity |
| `part` | Indicator distinguishing the historical input segment from the future target segment |
| `relative_step` | Relative hourly index within the sample |

Each sample consists of:

- the first **96 hours** as the multivariate historical input;
- the following **48 hours** as the ozone forecasting target.

All 4,836 released sample files have the same shape of **144 rows × 8 columns**.

## Experimental Role

These files are the fixed 2025 test samples used for:

- evaluation of OzoneKBNet;
- comparison with supervised forecasting baselines;
- comparison with zero-shot time-series foundation models;
- comparison with retrieval-augmented forecasting baselines;
- component ablation studies;
- backbone generality analysis;
- high-ozone subset analysis.

The 2025 test samples were not used for knowledge-base construction, normalization-statistics computation, retrieval-encoder pretraining, or Stage-2 supervised training.

## Data Scope

This release contains the complete processed test set used in the manuscript. It does not include:

- the original raw air-quality monitoring records;
- the original ERA5 reanalysis files;
- the complete 2023 data used for retrieval-encoder pretraining and knowledge-base construction;
- the complete 2024 data used for Stage-2 supervised training;
- precomputed knowledge-base embeddings;
- FAISS indices;
- model checkpoints.

The raw air-quality and meteorological data should be obtained from the original providers described in the manuscript.

## Repository

OzoneKBNet repository:

https://github.com/mingkun-zhu/OzoneKBNet

## Citation

Please cite the corresponding OzoneKBNet paper when using these fixed test samples or the accompanying implementation.
