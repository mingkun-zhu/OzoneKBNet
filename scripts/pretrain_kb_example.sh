#!/usr/bin/env bash
set -euo pipefail
CITY=${1:-changsanjiao}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python run_ozone_kb.py --mode pretrain_kb --city "$CITY" --device cuda
