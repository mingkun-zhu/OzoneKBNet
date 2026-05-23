#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile computational characteristics for the server-side OzoneKBNet code.

Default cities:
    sichuanpendi  # largest city cluster
    changsha      # smallest city cluster

It reports:
1) neural parameter counts
2) checkpoint/cache/storage size
3) KB scale statistics
4) online inference time using model(x), consistent with evaluate_2025
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from utils.faiss_utils import build_index_flat_ip

from run_ozone_kb import build_parser, set_seed
from exp.exp_ozone_kb_forecast import ExpOzoneKBForecast
from data_provider.ozone_kb_data import list_sample_files, load_one_test_sample


CITY_SHORT = {
    "shanyang": "SMA",
    "huabei": "NCP",
    "guanzhong": "GZP",
    "sichuanpendi": "SCB",
    "changsha": "CSA",
    "zhusanjiao": "PRD",
    "changsanjiao": "YRD",
}


DEFAULT_CITIES = [
    "shanyang",        # SMA
    "huabei",          # NCP
    "guanzhong",       # GZP
    "sichuanpendi",    # SCB
    "changsha",        # CSA
    "zhusanjiao",      # PRD
    "changsanjiao",    # YRD
]


def file_size_mb(path: Path) -> float:
    if path.exists() and path.is_file():
        return path.stat().st_size / (1024 ** 2)
    return float("nan")


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
    }


def make_args(city: str, cli_args) -> argparse.Namespace:
    """
    Build args consistent with run_ozone_kb.py official __main__ settings.
    """
    parser = build_parser()

    base_argv = [
        "--mode", "evaluate",
        "--city", city,
        "--root_path", cli_args.root_path,
        "--result_root", cli_args.result_root,
        "--checkpoints", cli_args.checkpoints,
        "--cache_root", cli_args.cache_root,
        "--model", cli_args.model,
        "--device", cli_args.device,
        "--batch_size", str(cli_args.batch_size),
        "--learning_rate", "1e-4",
    ]

    args = parser.parse_args(base_argv)

    # The following settings mirror the FORCE_EXPERIMENT_SETTINGS block
    # in the server run_ozone_kb.py.
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

    # Make sure these attrs exist even if imported parser does not include them.
    if not hasattr(args, "loader_workers"):
        args.loader_workers = cli_args.loader_workers

    return args

def repair_kb_indices(exp):
    """
    Rebuild FAISS indices from the embeddings stored in the loaded KB bundle.
    This avoids index/meta mismatch caused by stale or overwritten index_scale_*.faiss files.
    """
    if exp.model.kb is None:
        raise RuntimeError("KB has not been attached to the model.")

    for scale in exp.model.scales:
        k = str(scale)
        kb = exp.model.kb[k]

        n_emb = int(kb["embeddings"].shape[0])
        n_meta = len(kb["meta"])
        n_x = int(kb["x_std"].shape[0])
        n_y = int(kb["y_std"].shape[0])

        print(
            f"[KB-Check] scale={scale} "
            f"emb={n_emb} meta={n_meta} x={n_x} y={n_y}",
            flush=True,
        )

        if not (n_emb == n_meta == n_x == n_y):
            raise RuntimeError(
                f"KB internal mismatch at scale={scale}: "
                f"emb={n_emb}, meta={n_meta}, x={n_x}, y={n_y}"
            )

        kb["index"] = build_index_flat_ip(kb["embeddings"])

    # Re-attach KB to refresh GPU tensors if needed.
    exp.model.set_kb(exp.model.kb)
    print("[KB-Check] FAISS indices rebuilt from current KB embeddings.", flush=True)

def read_kb_pickle_summary(kb_path: Path) -> Dict[str, Any]:
    """
    Read the raw KB pickle without rebuilding anything.
    The saved bundle stores embeddings, x_std, y_std, meta, and index_path per scale.
    """
    out: Dict[str, Any] = {
        "kb_file_mb": file_size_mb(kb_path),
        "kb_num_scales": 0,
        "kb_windows_sum_over_scales": 0,
        "kb_array_total_mb": 0.0,
        "faiss_index_total_mb": 0.0,
    }

    scale_rows = []

    if not kb_path.exists():
        return out, pd.DataFrame(scale_rows)

    with kb_path.open("rb") as f:
        obj = pickle.load(f)

    out["kb_num_scales"] = len(obj)

    for scale, bundle in obj.items():
        emb = bundle.get("embeddings", None)
        x_std = bundle.get("x_std", None)
        y_std = bundle.get("y_std", None)
        meta = bundle.get("meta", [])
        index_path = Path(bundle.get("index_path", kb_path.parent / f"index_scale_{scale}.faiss"))

        emb_mb = emb.nbytes / (1024 ** 2) if hasattr(emb, "nbytes") else float("nan")
        x_mb = x_std.nbytes / (1024 ** 2) if hasattr(x_std, "nbytes") else float("nan")
        y_mb = y_std.nbytes / (1024 ** 2) if hasattr(y_std, "nbytes") else float("nan")
        arr_mb = np.nansum([emb_mb, x_mb, y_mb])
        faiss_mb = file_size_mb(index_path)

        n_windows = int(emb.shape[0]) if hasattr(emb, "shape") else len(meta)

        out["kb_windows_sum_over_scales"] += n_windows
        out["kb_array_total_mb"] += arr_mb
        if not np.isnan(faiss_mb):
            out["faiss_index_total_mb"] += faiss_mb

        scale_rows.append({
            "scale": int(scale),
            "num_windows": n_windows,
            "embeddings_shape": str(getattr(emb, "shape", None)),
            "x_std_shape": str(getattr(x_std, "shape", None)),
            "y_std_shape": str(getattr(y_std, "shape", None)),
            "meta_len": len(meta),
            "embeddings_mb": emb_mb,
            "x_std_mb": x_mb,
            "y_std_mb": y_mb,
            "array_total_mb": arr_mb,
            "faiss_index_mb": faiss_mb,
            "index_path": str(index_path),
        })

    return out, pd.DataFrame(scale_rows)


def get_cache_file_sizes(exp: ExpOzoneKBForecast, args) -> Dict[str, float]:
    """
    Storage of pretrain checkpoint, stage2 checkpoint, stats, pair cache,
    and optional stage2 retrieval caches.
    """
    kb_year = args.kb_year
    train_year = args.train_year

    paths = {
        "pretrain_ckpt_mb": exp._pretrain_ckpt_path(kb_year),
        "stage2_ckpt_mb": exp._stage2_ckpt_path(),
        "stats_mb": exp._stats_path(kb_year),
        "pair_cache_mb": exp._pair_cache_path(kb_year),
        "stage2_cache_train_mb": exp._stage2_retrieval_cache_path("train", kb_year),
        "stage2_cache_val_mb": exp._stage2_retrieval_cache_path("val", kb_year),
    }

    return {k: file_size_mb(v) for k, v in paths.items()}


def profile_online_inference(exp: ExpOzoneKBForecast, args, warmup: int, max_samples: int | None) -> Dict[str, Any]:
    """
    Profile the same online inference path as evaluate_2025:
        x -> model(x) -> y_hat

    This includes query encoding, FAISS search, re-ranking, fusion,
    direct LSTM branch, and final gate.
    """
    device = exp.device
    model = exp.model
    model.eval()

    kb_year = args.train_year if args.rolling_update else args.kb_year
    stats = exp.get_or_build_stats(kb_year)

    samples = [s for s in list_sample_files(exp.paths["test_samples_2025"]) if s["city"] == exp.city]
    if max_samples is not None:
        samples = samples[:max_samples]

    if len(samples) == 0:
        return {
            "num_profile_samples": 0,
            "online_time_total_sec": float("nan"),
            "online_time_ms_per_sample": float("nan"),
            "peak_gpu_memory_mb": float("nan"),
        }

    def run_one(item):
        x, _ = load_one_test_sample(item["path"])
        x_std = stats.standardize_x(x)
        x_t = torch.tensor(x_std[None, :], dtype=torch.float32, device=device)
        with torch.inference_mode():
            outputs = model(x_t)
            y_hat_std = outputs["y_hat"].detach().cpu().numpy()[0]
            _ = stats.destandardize_y(y_hat_std)

    # warmup
    for item in samples[:min(warmup, len(samples))]:
        run_one(item)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for i, item in enumerate(samples, start=1):
        run_one(item)
        if i % 100 == 0 or i == len(samples):
            print(f"[Profile] {exp.city} inference {i}/{len(samples)}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    peak_gpu = float("nan")
    if device.type == "cuda":
        peak_gpu = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    total = t1 - t0
    return {
        "num_profile_samples": int(len(samples)),
        "online_time_total_sec": float(total),
        "online_time_ms_per_sample": float(total / max(len(samples), 1) * 1000.0),
        "peak_gpu_memory_mb": peak_gpu,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    p.add_argument("--out_dir", type=str, default="./results/profile_stats")
    p.add_argument("--root_path", type=str, default="./data")
    p.add_argument("--result_root", type=str, default="./results")
    p.add_argument("--checkpoints", type=str, default="./checkpoints")
    p.add_argument("--cache_root", type=str, default="./caches")
    p.add_argument("--model", type=str, default="OzoneKBNet")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--loader_workers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=None, help="For quick debugging only. Omit for formal profiling.")
    cli_args = p.parse_args()

    out_dir = Path(cli_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    all_scale_rows: List[pd.DataFrame] = []

    for city in cli_args.cities:
        print("\n" + "=" * 100)
        print(f"Profiling city={city}")
        print("=" * 100)

        args = make_args(city, cli_args)
        set_seed(args.seed)

        exp = ExpOzoneKBForecast(args)

        kb_year = args.train_year if args.rolling_update else args.kb_year
        kb_path = exp._kb_path(kb_year)

        # Load KB and stage2 checkpoint exactly as evaluate_2025 does.
        exp.build_or_load_kb(kb_year)
        repair_kb_indices(exp)
        exp._load_best_stage2()

        # Match the stage-2 setting: retrieval encoders are frozen.
        exp.model.freeze_retrieval_encoders(True)
        exp.model.eval()

        param_info = count_params(exp.model)
        kb_info, scale_df = read_kb_pickle_summary(kb_path)
        cache_sizes = get_cache_file_sizes(exp, args)
        infer_info = profile_online_inference(
            exp=exp,
            args=args,
            warmup=cli_args.warmup,
            max_samples=cli_args.max_samples,
        )

        kb_windows_per_scale = (
                kb_info["kb_windows_sum_over_scales"] / max(kb_info["kb_num_scales"], 1)
        )

        row = {
            "city": city,
            "city_short": CITY_SHORT.get(city, city),
            "kb_year": kb_year,
            **param_info,
            **cache_sizes,
            **kb_info,
            **infer_info,
            "kb_windows_per_scale": int(kb_windows_per_scale),
        }

        all_rows.append(row)

        if len(scale_df):
            scale_df.insert(0, "city", city)
            scale_df.insert(1, "city_short", CITY_SHORT.get(city, city))
            all_scale_rows.append(scale_df)

        print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(all_rows)

    if len(summary) > 0:
        numeric_cols = summary.select_dtypes(include=[np.number]).columns.tolist()
        avg_row = {c: "" for c in summary.columns}
        avg_row["city"] = "Avg."
        avg_row["city_short"] = "Avg."
        for c in numeric_cols:
            avg_row[c] = summary[c].mean()
        summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)

    scale_detail = pd.concat(all_scale_rows, ignore_index=True) if all_scale_rows else pd.DataFrame()

    summary_path = out_dir / "computational_summary_all7.csv"
    scale_path = out_dir / "kb_scale_detail_all7.csv"
    latex_path = out_dir / "computational_latex_rows_all7.txt"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    scale_detail.to_csv(scale_path, index=False, encoding="utf-8-sig")

    def fmt(x, digits=3):
        try:
            if pd.isna(x):
                return "--"
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)

    lines = []
    for _, r in summary.iterrows():
        total_params_m = float(r["total_params"]) / 1e6
        trainable_params_m = float(r["trainable_params"]) / 1e6

        kb_windows = r["kb_windows_per_scale"]
        if pd.isna(kb_windows):
            kb_windows_str = "--"
        else:
            kb_windows_str = f"{int(kb_windows):,}"

        lines.append(
            f"{r['city_short']} & "
            f"{kb_windows_str} & "
            f"{total_params_m:.3f}M & "
            f"{trainable_params_m:.3f}M & "
            f"{fmt(r['kb_file_mb'], 2)} MB & "
            f"{fmt(r['faiss_index_total_mb'], 2)} MB & "
            f"{fmt(r['stage2_ckpt_mb'], 2)} MB & "
            f"{fmt(r['online_time_ms_per_sample'], 2)} ms & "
            f"{fmt(r['peak_gpu_memory_mb'], 2)} MB \\\\"
        )
    latex_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Saved:")
    print(summary_path)
    print(scale_path)
    print(latex_path)
    print("=" * 100)
    print("\nLaTeX rows:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()