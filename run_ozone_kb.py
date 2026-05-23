#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from exp.exp_ozone_kb_forecast import ExpOzoneKBForecast


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OzoneKBNet two-stage pipeline")
    p.add_argument("--mode", type=str, required=True, choices=["pretrain_kb", "train_stage2", "evaluate", "full_pipeline"])
    p.add_argument("--root_path", type=str, default="./data")
    p.add_argument("--result_root", type=str, default="./results")
    p.add_argument("--checkpoints", type=str, default="./checkpoints")
    p.add_argument("--cache_root", type=str, default="./caches")
    p.add_argument("--model", type=str, default="OzoneKBNet")
    p.add_argument("--task_name", type=str, default="ozone_kb_forecast")
    p.add_argument("--city", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--kb_year", type=int, default=2023)
    p.add_argument("--train_year", type=int, default=2024)
    p.add_argument("--test_year", type=int, default=2025)
    p.add_argument("--rolling_update", type=int, default=1)

    p.add_argument("--enc_in", type=int, default=5)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=48)
    p.add_argument("--scales", nargs="+", type=int, default=[1, 2, 4, 8])

    p.add_argument("--encoder_hidden", type=int, default=64)
    p.add_argument("--encoder_layers", type=int, default=4)
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--horizon_emb_dim", type=int, default=16)
    p.add_argument("--rank_gate_hidden", type=int, default=64)
    p.add_argument("--alpha_mlp_hidden", type=int, default=64)
    p.add_argument("--final_gate_hidden", type=int, default=64)
    p.add_argument("--l_dir", type=int, default=4)
    p.add_argument("--c_dir", type=int, default=64)
    p.add_argument("--c_u", type=int, default=32)

    p.add_argument("--positive_top_p", type=int, default=3)
    p.add_argument("--negative_top_q", type=int, default=12)
    p.add_argument("--candidate_min_gap_hours", type=int, default=72)
    p.add_argument("--pretrain_candidate_pool_cap", type=int, default=2000)
    p.add_argument("--delta_pre_hours", type=int, default=24)
    p.add_argument("--delta_dedup_hours", type=int, default=24)
    p.add_argument("--coarse_top_m", type=int, default=200)
    p.add_argument("--final_top_k", type=int, default=10)
    p.add_argument("--local_trend_L", type=int, default=24)

    p.add_argument("--pretrain_batch_size", type=int, default=128)
    p.add_argument("--pretrain_learning_rate", type=float, default=1e-3)
    p.add_argument("--pretrain_weight_decay", type=float, default=1e-4)
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--pretrain_patience", type=int, default=10)
    p.add_argument("--tau_supcon", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--peak_quantile", type=float, default=0.90)
    p.add_argument("--use_peak_loss", type=int, default=1)

    p.add_argument("--samples_per_month", type=int, default=5)
    p.add_argument("--train_per_month", type=int, default=4)
    p.add_argument("--monthly_min_gap_days", type=int, default=4)

    p.add_argument("--lambda_dtw", type=float, default=1/3)
    p.add_argument("--lambda_l2", type=float, default=1/3)
    p.add_argument("--lambda_peak", type=float, default=1/3)

    p.add_argument("--force_retrain", type=int, default=0)
    p.add_argument("--force_rebuild_kb", type=int, default=0)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = build_parser()
    parser.add_argument("--branch_mode", type=str, default="full", choices=["full", "direct_only", "retrieval_only"], help="Use full fusion, direct branch only, or retrieval branch only.")
    parser.add_argument("--direct_branch_type", type=str, default="lstm", choices=["tcn", "lstm"], help="Direct branch type.")
    parser.add_argument("--direct_lstm_hidden", type=int, default=128, help="Hidden size for LSTM direct branch.")
    parser.add_argument("--direct_lstm_layers", type=int, default=1, help="Number of LSTM layers for direct branch.")
    parser.add_argument("--direct_lstm_dropout", type=float, default=0.1, help="Dropout for LSTM direct branch.")
    parser.add_argument("--direct_biased_gate", type=int, default=1, help="Use direct-biased residual gate: direct + gamma*(retrieval-direct).")
    parser.add_argument("--direct_biased_gamma_max", type=float, default=0.3, help="Maximum retrieval residual weight for direct-biased gate.")
    args = parser.parse_args()
    # === FORCE_EXPERIMENT_SETTINGS_15_22 ===
    args.kb_year = 2023
    args.train_year = 2024
    args.test_year = 2025
    args.rolling_update = False
    args.branch_mode = "full"
    args.direct_branch_type = "lstm"
    args.direct_lstm_hidden = 128
    args.direct_lstm_layers = 1
    args.direct_lstm_dropout = 0.1
    args.direct_biased_gate = 1
    args.direct_biased_gamma_max = 0.2
    args.coarse_top_m = 200
    args.final_top_k = 10
    set_seed(args.seed)

    exp = ExpOzoneKBForecast(args)
    if args.mode == "pretrain_kb":
        exp.pretrain_retrieval_encoder(args.kb_year)
        exp.build_or_load_kb(args.kb_year)
    elif args.mode == "train_stage2":
        exp.train_stage2()
    elif args.mode == "evaluate":
        exp.evaluate_2025()
    elif args.mode == "full_pipeline":
        exp.pretrain_retrieval_encoder(args.kb_year)
        exp.build_or_load_kb(args.kb_year)
        exp.train_stage2()
        exp.evaluate_2025()
