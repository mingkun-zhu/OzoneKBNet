
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from data_provider.ozone_kb_data import (
    PretrainPairDataset,
    Stage2SampleDataset,
    build_stage2_monthly_samples,
    list_sample_files,
    load_city_windows_from_year_dir,
    load_one_test_sample,
    normalize_city_name,
)
from models.OzoneKBNet import OzoneKBNet
from utils.kb_utils import encode_windows_for_scale, load_kb_bundle, save_kb_bundle
from utils.metrics import mae, mse, rmse
from utils.mining import PairCache, build_pair_cache
from utils.normalization import CityYearStats, compute_city_year_stats
from utils.ts_utils import get_scale_stride


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        if improved:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False


class Stage2CachedDataset(Dataset):
    def __init__(self, items, cache_dict: Dict[str, torch.Tensor]) -> None:
        self.items = items
        self.cache_dict = cache_dict

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        out = {
            "x": torch.as_tensor(item.x, dtype=torch.float32),
            "y_std": torch.as_tensor(item.y_std, dtype=torch.float32),
            "y_phys": torch.as_tensor(item.y_phys, dtype=torch.float32),
        }
        for k, v in self.cache_dict.items():
            out[k] = v[idx]
        return out


class ExpOzoneKBForecast:
    def __init__(self, args) -> None:
        self.args = args
        self.city = normalize_city_name(args.city)
        self.device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        self.model = OzoneKBNet(args).to(self.device)
        self.paths = self._build_paths()
        self.cpu_threads = os.cpu_count() or 1
        self.loader_workers = int(getattr(args, "loader_workers", 8))
        self.pin_memory = bool(self.device.type == "cuda")
        print(
            f"[Init] cpu_threads={self.cpu_threads} loader_workers={self.loader_workers} pin_memory={self.pin_memory}",
            flush=True,
        )
        print(
            f"[Init] model={args.model} city={self.city} device={self.device} "
            f"kb_year={self.args.kb_year} train_year={self.args.train_year} test_year={self.args.test_year}",
            flush=True,
        )

    def _build_paths(self) -> Dict[str, Path]:
        root = Path(self.args.root_path)
        result_root = Path(self.args.result_root) / self.args.model / self.city
        ckpt_root = Path(self.args.checkpoints) / self.args.model / self.city
        cache_root = Path(self.args.cache_root) / self.args.model / self.city
        return {
            "train_2023": root / "data" / "data_for_train_2023" / self.city,
            "train_2024": root / "data" / "data_for_train_2024" / self.city,
            "test_2025": root / "data" / "data_for_test_2025" / self.city,
            "test_samples_2025": root / "data" / "data_for_test_2025_samples",
            "result_root": result_root,
            "ckpt_root": ckpt_root,
            "cache_root": cache_root,
        }

    def _stats_path(self, year: int) -> Path:
        return self.paths["cache_root"] / f"stats_{self.city}_{year}.json"

    def _pair_cache_path(self, year: int) -> Path:
        return self.paths["cache_root"] / f"pair_cache_{self.city}_{year}.pkl"

    def _pretrain_ckpt_path(self, year: int) -> Path:
        return self.paths["ckpt_root"] / f"pretrain_encoder_{self.city}_{year}.pt"

    def _kb_path(self, year: int) -> Path:
        return self.paths["cache_root"] / f"kb_{self.city}_{year}.pkl"

    def _stage2_ckpt_path(self) -> Path:
        return self.paths["ckpt_root"] / f"stage2_{self.city}.pt"

    def _stage2_retrieval_cache_path(self, split: str, kb_year: int) -> Path:
        return self.paths["cache_root"] / f"stage2_retrieval_cache_{self.city}_{kb_year}_{self.args.train_year}_{split}.pt"

    def _make_loader(self, dataset: Dataset, batch_size: int, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.loader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.loader_workers > 0),
            prefetch_factor=(2 if self.loader_workers > 0 else None),
        )

    def get_or_build_stats(self, year: int) -> CityYearStats:
        p = self._stats_path(year)
        if p.exists():
            print(f"[Stats] load existing stats: {p}", flush=True)
            return CityYearStats.load(p)
        year_dir = self.paths[f"train_{year}"] if year in (2023, 2024) else None
        if year_dir is None:
            raise ValueError(f"Unsupported stats year: {year}")
        print(f"[Stats] build stats city={self.city} year={year} from={year_dir}", flush=True)
        stats = compute_city_year_stats(self.city, year, year_dir)
        stats.save(p)
        print(f"[Stats] saved stats: {p}", flush=True)
        return stats

    def build_pretrain_windows(self, year: int):
        year_dir = self.paths[f"train_{year}"]
        stride = get_scale_stride(1)
        print(
            f"[PretrainWindows] build windows city={self.city} year={year} stride={stride} "
            f"min_gap_hours={self.args.delta_pre_hours} from={year_dir}",
            flush=True,
        )
        windows = load_city_windows_from_year_dir(
            city=self.city,
            year=year,
            city_year_dir=year_dir,
            stride=stride,
            min_gap_hours=self.args.delta_pre_hours,
        )
        print(f"[PretrainWindows] built {len(windows)} windows", flush=True)
        return windows

    def get_or_build_pair_cache(self, year: int, windows) -> PairCache:
        p = self._pair_cache_path(year)
        if p.exists():
            print(f"[PairCache] load existing cache: {p}", flush=True)
            return PairCache.load(p)
        print(f"[PairCache] build new cache for city={self.city} year={year}", flush=True)
        cache = build_pair_cache(
            city=self.city,
            year=year,
            windows=windows,
            positive_top_p=self.args.positive_top_p,
            negative_top_q=self.args.negative_top_q,
            candidate_pool_cap=self.args.pretrain_candidate_pool_cap,
            candidate_min_gap_hours=self.args.candidate_min_gap_hours,
            lambda_dtw=self.args.lambda_dtw,
            lambda_l2=self.args.lambda_l2,
            lambda_peak=self.args.lambda_peak,
        )
        cache.save(p)
        print(f"[PairCache] saved cache: {p}", flush=True)
        return cache

    def pretrain_retrieval_encoder(self, year: int) -> Path:
        print(f"[Pretrain] start pretraining retrieval encoder for year={year}", flush=True)
        stats = self.get_or_build_stats(year)
        windows = self.build_pretrain_windows(year)
        pair_cache = self.get_or_build_pair_cache(year, windows)
        ckpt_path = self._pretrain_ckpt_path(year)
        if ckpt_path.exists() and not getattr(self.args, "force_retrain", False):
            print(f"[Pretrain] skip, existing checkpoint found: {ckpt_path}", flush=True)
            return ckpt_path

        dataset = PretrainPairDataset(windows, pair_cache.records, stats=stats, scales=self.args.scales)
        n_val = max(1, int(0.1 * len(dataset)))
        n_train = max(1, len(dataset) - n_val)
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.args.seed),
        )
        print(f"[Pretrain] dataset size={len(dataset)} train={len(train_set)} val={len(val_set)}", flush=True)
        train_loader = self._make_loader(train_set, self.args.pretrain_batch_size, True)
        val_loader = self._make_loader(val_set, self.args.pretrain_batch_size, False)

        self.model.freeze_retrieval_encoders(False)
        optimizer = torch.optim.Adam(
            self.model.encoders.parameters(),
            lr=self.args.pretrain_learning_rate,
            weight_decay=self.args.pretrain_weight_decay,
        )
        stopper = EarlyStopping(self.args.pretrain_patience, mode="min")
        best_state = None
        best_val = None

        for epoch in range(1, self.args.pretrain_epochs + 1):
            t0 = time.time()
            self.model.train()
            train_losses = []
            print(f"[Pretrain] epoch {epoch}/{self.args.pretrain_epochs} start lr={optimizer.param_groups[0]['lr']:.8f}", flush=True)
            for i, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                out = self.model.pretrain_forward(batch["anchor_x"], batch["positive_x"], batch["negative_x"])
                loss = self.model.supcon_loss(out, tau=self.args.tau_supcon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                if i % 100 == 0 or i == len(train_loader):
                    print(f"[Pretrain] epoch={epoch} batch={i}/{len(train_loader)} loss={float(loss.detach().cpu()):.6f}", flush=True)

            self.model.eval()
            val_losses = []
            with torch.inference_mode():
                for batch in val_loader:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    out = self.model.pretrain_forward(batch["anchor_x"], batch["positive_x"], batch["negative_x"])
                    loss = self.model.supcon_loss(out, tau=self.args.tau_supcon)
                    val_losses.append(float(loss.detach().cpu()))
            val_loss = float(np.mean(val_losses))
            train_loss = float(np.mean(train_losses))
            if stopper.step(val_loss):
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in self.model.encoders.state_dict().items()}
                print(f"[Pretrain] new best checkpoint at epoch={epoch} val_loss={val_loss:.6f}", flush=True)
            print(
                f"[Pretrain] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_val_loss={(best_val if best_val is not None else float('nan')):.6f} time={time.time()-t0:.1f}s",
                flush=True,
            )
            if stopper.should_stop:
                print(f"[Pretrain] early stopping at epoch={epoch}", flush=True)
                break

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoders": best_state}, ckpt_path)
        print(f"[Pretrain] saved checkpoint: {ckpt_path}", flush=True)
        return ckpt_path

    def load_pretrained_encoders(self, year: int) -> None:
        ckpt_path = self._pretrain_ckpt_path(year)
        print(f"[Pretrain] load checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.encoders.load_state_dict(ckpt["encoders"])

    def build_or_load_kb(self, year: int) -> Dict:
        kb_path = self._kb_path(year)
        if kb_path.exists() and not getattr(self.args, "force_rebuild_kb", False):
            print(f"[KB] load existing KB bundle: {kb_path}", flush=True)
            kb = load_kb_bundle(kb_path)
            self.model.set_kb(kb)
            print("[KB] loaded into model", flush=True)
            return kb

        print(f"[KB] build new KB bundle for year={year}", flush=True)
        stats = self.get_or_build_stats(year)
        windows = self.build_pretrain_windows(year)
        self.load_pretrained_encoders(year)
        self.model.freeze_retrieval_encoders(True)
        self.model.eval()
        kb = {}
        for scale in self.args.scales:
            t0 = time.time()
            print(f"[KB] encode scale={scale}", flush=True)
            kb[str(scale)] = encode_windows_for_scale(self.model, windows, stats=stats, scale=scale)
            print(f"[KB] scale={scale} done in {time.time()-t0:.1f}s", flush=True)
        save_kb_bundle(kb, kb_path)
        self.model.set_kb(kb)
        print(f"[KB] saved bundle: {kb_path}", flush=True)
        return kb

    def _compute_stage2_loss(self, outputs: Dict[str, torch.Tensor], y_std: torch.Tensor, y_phys: torch.Tensor, peak_threshold: float) -> torch.Tensor:
        y_hat = outputs["y_hat"]
        pred_loss = torch.mean((y_hat - y_std) ** 2)
        if not self.args.use_peak_loss:
            return pred_loss
        peak_mask = (y_phys > peak_threshold).float()
        weights = 1.0 + (self.args.gamma - 1.0) * peak_mask
        peak_loss = torch.mean(weights * (y_hat - y_std) ** 2)
        return pred_loss + self.args.eta * peak_loss

    def build_or_load_stage2_retrieval_cache(self, items, split: str, kb_year: int) -> Dict[str, torch.Tensor]:
        p = self._stage2_retrieval_cache_path(split, kb_year)
        if p.exists() and not getattr(self.args, "force_rebuild_stage2_cache", False):
            print(f"[Stage2Cache] load existing {split} retrieval cache: {p}", flush=True)
            return torch.load(p, map_location="cpu")

        print(f"[Stage2Cache] build {split} retrieval cache: samples={len(items)} path={p}", flush=True)
        self.model.eval()
        self.model.freeze_retrieval_encoders(True)
        cache_accum: Dict[str, List[torch.Tensor]] = {}

        with torch.inference_mode():
            for i, item in enumerate(items, start=1):
                x_t = torch.as_tensor(item.x, dtype=torch.float32, device=self.device)
                sample_cache = self.model.build_stage2_cache_for_x(x_t)
                for k, v in sample_cache.items():
                    cache_accum.setdefault(k, []).append(v.cpu())
                if i % 100 == 0 or i == len(items):
                    print(f"[Stage2Cache] {split} done {i}/{len(items)}", flush=True)

        cache_tensors = {k: torch.stack(v, dim=0) for k, v in cache_accum.items()}
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_tensors, p)
        print(f"[Stage2Cache] saved {split} retrieval cache: {p}", flush=True)
        return cache_tensors

    def train_stage2(self) -> Path:
        print("[Stage2] start training", flush=True)
        self.pretrain_retrieval_encoder(self.args.kb_year)
        self.build_or_load_kb(self.args.kb_year)
        self.model.freeze_retrieval_encoders(True)
        stats = self.get_or_build_stats(self.args.kb_year)

        train_items, val_items = build_stage2_monthly_samples(
            city=self.city,
            year_dir=self.paths[f"train_{self.args.train_year}"],
            stats=stats,
            samples_per_month=self.args.samples_per_month,
            train_per_month=self.args.train_per_month,
            min_gap_days=self.args.monthly_min_gap_days,
            seed=self.args.seed,
        )
        print(
            f"[Stage2] built monthly samples: train_items={len(train_items)} val_items={len(val_items)} "
            f"batch_size={self.args.batch_size}",
            flush=True,
        )

        train_cache = self.build_or_load_stage2_retrieval_cache(train_items, "train", self.args.kb_year)
        val_cache = self.build_or_load_stage2_retrieval_cache(val_items, "val", self.args.kb_year)

        train_ds = Stage2CachedDataset(train_items, train_cache)
        val_ds = Stage2CachedDataset(val_items, val_cache)
        train_loader = self._make_loader(train_ds, self.args.batch_size, True)
        val_loader = self._make_loader(val_ds, self.args.batch_size, False)

        print(
            f"[Stage2] loader config: batch_size={self.args.batch_size} num_workers={self.loader_workers} pin_memory={self.pin_memory}",
            flush=True,
        )
        print(f"[Stage2] train_batches={len(train_loader)} val_batches={len(val_loader)} train_epochs={self.args.train_epochs}", flush=True)
        print("[Stage2] validation policy: every epoch for first 5 epochs, then every 2 epoch(s)", flush=True)

        all_train_targets = np.concatenate([item.y_phys.reshape(-1) for item in train_items], axis=0)
        peak_threshold = float(np.quantile(all_train_targets, self.args.peak_quantile))
        print(f"[Stage2] peak_threshold(q={self.args.peak_quantile})={peak_threshold:.6f}", flush=True)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=self.args.min_lr
        )
        stopper = EarlyStopping(self.args.patience, mode="min")
        best_state = None
        best_metric = None

        for epoch in range(1, self.args.train_epochs + 1):
            run_val = True if epoch <= 5 else (epoch % 2 == 0)
            epoch_t0 = time.time()
            print(f"[Stage2] epoch {epoch}/{self.args.train_epochs} start lr={optimizer.param_groups[0]['lr']:.8f} run_val={run_val}", flush=True)

            self.model.train()
            train_t0 = time.time()
            train_losses = []
            for i, batch in enumerate(train_loader, start=1):
                x = batch["x"].to(self.device, non_blocking=True)
                y_std = batch["y_std"].to(self.device, non_blocking=True)
                y_phys = batch["y_phys"].to(self.device, non_blocking=True)
                cache_batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                    if k not in {"x", "y_std", "y_phys"}
                }
                outputs = self.model.forward_from_stage2_cache(x, cache_batch)
                loss = self._compute_stage2_loss(outputs, y_std, y_phys, peak_threshold)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                if i % 10 == 0 or i == len(train_loader):
                    print(f"[Stage2] epoch={epoch} batch={i}/{len(train_loader)} loss={float(loss.detach().cpu()):.6f}", flush=True)
            train_time = time.time() - train_t0
            train_loss = float(np.mean(train_losses))

            val_time = 0.0
            if run_val:
                val_t0 = time.time()
                val_metric = self.validate_stage2(val_loader, stats, use_cached=True)
                val_time = time.time() - val_t0
                scheduler.step(val_metric)
                if stopper.step(val_metric):
                    best_metric = val_metric
                    best_state = {
                        "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                        "peak_threshold": peak_threshold,
                        "val_rmse": val_metric,
                    }
                    print(f"[Stage2] new best checkpoint at epoch={epoch} val_rmse={val_metric:.6f}", flush=True)
            else:
                val_metric = float("nan")
                print(f"[Stage2] epoch={epoch} skip validation by policy", flush=True)

            best_val_str = f"{best_metric:.6f}" if best_metric is not None else "nan"
            print(
                f"[Stage2] epoch={epoch} train_loss={train_loss:.6f} val_rmse={val_metric:.6f} "
                f"best_val_rmse={best_val_str} train_time={train_time:.1f}s val_time={val_time:.1f}s total_time={time.time()-epoch_t0:.1f}s",
                flush=True,
            )

            if stopper.should_stop:
                print(f"[Stage2] early stopping at epoch={epoch}", flush=True)
                break

        ckpt_path = self._stage2_ckpt_path()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_path)
        print(f"[Stage2] saved best checkpoint: {ckpt_path}", flush=True)
        return ckpt_path

    def validate_stage2(self, loader: DataLoader, stats: CityYearStats, use_cached: bool = False) -> float:
        self.model.eval()
        preds, trues = [], []
        with torch.inference_mode():
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                y_phys = batch["y_phys"].detach().cpu().numpy()
                if use_cached:
                    cache_batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                        if k not in {"x", "y_std", "y_phys"}
                    }
                    outputs = self.model.forward_from_stage2_cache(x, cache_batch)
                else:
                    outputs = self.model(x)
                y_hat_std = outputs["y_hat"].detach().cpu().numpy()
                y_hat_phys = stats.destandardize_y(y_hat_std)
                preds.append(y_hat_phys)
                trues.append(y_phys)
        p = np.concatenate(preds, axis=0)
        t = np.concatenate(trues, axis=0)
        return rmse(t, p)

    def _load_best_stage2(self):
        ckpt = torch.load(self._stage2_ckpt_path(), map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        return ckpt

    def evaluate_2025(self) -> Path:
        kb_year = self.args.train_year if self.args.rolling_update else self.args.kb_year
        print(f"[Eval] start evaluate_2025 with kb_year={kb_year} rolling_update={self.args.rolling_update}", flush=True)
        if self.args.rolling_update:
            self.pretrain_retrieval_encoder(self.args.train_year)
        self.build_or_load_kb(kb_year)
        self._load_best_stage2()
        self.model.eval()
        stats = self.get_or_build_stats(kb_year)

        model_root = self.paths["result_root"]
        model_root.mkdir(parents=True, exist_ok=True)

        good_meta = []
        failed_meta = []
        all_preds = []
        all_trues = []

        samples = [s for s in list_sample_files(self.paths["test_samples_2025"]) if s["city"] == self.city]
        print(f"[Eval] total test samples discovered: {len(samples)}", flush=True)
        for i, item in enumerate(samples, start=1):
            try:
                x, y = load_one_test_sample(item["path"])
                x_std = stats.standardize_x(x)
                x_t = torch.tensor(x_std[None, ...], dtype=torch.float32, device=self.device)
                with torch.inference_mode():
                    outputs = self.model(x_t)
                pred_std = outputs["y_hat"].detach().cpu().numpy()[0]
                pred_phys = stats.destandardize_y(pred_std)
                true_phys = y.astype(np.float32)
                all_preds.append(pred_phys.reshape(-1))
                all_trues.append(true_phys.reshape(-1))
                good_meta.append({
                    "city": item["city"],
                    "station": item["station"],
                    "sample_id": item["sample_id"],
                    "file_path": str(item["path"]),
                    "mae": mae(true_phys, pred_phys),
                    "mse": mse(true_phys, pred_phys),
                    "rmse": rmse(true_phys, pred_phys),
                })
            except Exception as e:
                failed_meta.append({
                    "city": item["city"],
                    "station": item["station"],
                    "sample_id": item["sample_id"],
                    "file_path": str(item["path"]),
                    "reason": f"inference_error: {repr(e)}",
                })
            if i % 100 == 0 or i == len(samples):
                print(f"[Eval] processed {i}/{len(samples)} test samples", flush=True)

        per_sample_df = pd.DataFrame(good_meta)
        failed_df = pd.DataFrame(failed_meta)
        per_sample_df.to_csv(model_root / "per_sample_metrics.csv", index=False)
        failed_df.to_csv(model_root / "failed_samples.csv", index=False)

        if len(all_preds) == 0:
            overall = {"model": self.args.model, "city": self.city, "num_success_samples": 0, "num_failed_samples": int(len(failed_meta))}
            with open(model_root / "overall_metrics.json", "w", encoding="utf-8") as f:
                json.dump(overall, f, ensure_ascii=False, indent=2)
            print(f"[Eval] no successful samples. saved overall to {model_root / 'overall_metrics.json'}", flush=True)
            return model_root

        preds_arr = np.stack(all_preds, axis=0)
        trues_arr = np.stack(all_trues, axis=0)
        np.save(model_root / "pred_o3.npy", preds_arr)
        np.save(model_root / "true_o3.npy", trues_arr)

        city_df = pd.DataFrame([{
            "city": self.city,
            "num_samples": int(preds_arr.shape[0]),
            "mae": mae(trues_arr, preds_arr),
            "mse": mse(trues_arr, preds_arr),
            "rmse": rmse(trues_arr, preds_arr),
        }])
        city_df.to_csv(model_root / "city_metrics.csv", index=False)

        overall = {
            "model": self.args.model,
            "city": self.city,
            "num_success_samples": int(preds_arr.shape[0]),
            "num_failed_samples": int(len(failed_meta)),
            "mae": mae(trues_arr, preds_arr),
            "mse": mse(trues_arr, preds_arr),
            "rmse": rmse(trues_arr, preds_arr),
        }
        with open(model_root / "overall_metrics.json", "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)

        print(f"[Eval] saved per-sample metrics: {model_root / 'per_sample_metrics.csv'}", flush=True)
        print(f"[Eval] saved city metrics: {model_root / 'city_metrics.csv'}", flush=True)
        print(f"[Eval] saved overall metrics: {model_root / 'overall_metrics.json'}", flush=True)
        print(f"[Eval] overall={json.dumps(overall, ensure_ascii=False)}", flush=True)
        return model_root
