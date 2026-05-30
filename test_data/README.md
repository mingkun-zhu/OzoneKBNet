# Test Data

This directory provides the fixed 2025 test samples used for evaluating OzoneKBNet.

The files are provided to facilitate the review process, check the required CSV schema, and illustrate the sample-level evaluation protocol. They are not the full raw or processed training datasets used in the manuscript.

## File structure

test_data/
└── data_for_test_2025_samples/
    └── {city}/
        └── {station}/
            └── 01.csv ... 12.csv

## CSV columns

Each CSV file contains 144 hourly rows, including a 96-hour input window and a 48-hour target window.

Expected columns:

time
O3
NO2
PM2.5
temperature
relative_humidity
part
relative_step

The first 96 rows are marked as input, and the following 48 rows are marked as target.

The full raw and processed training datasets, knowledge-base files, FAISS indices, and model checkpoints are not redistributed in this repository due to data redistribution restrictions and file-size considerations. The raw air-quality monitoring data and ERA5 reanalysis data are publicly accessible from their original data providers, as described in the manuscript.
