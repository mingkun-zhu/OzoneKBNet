from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.normalization import CityYearStats, FEATURE_COLS
from utils.ts_utils import (
    PRED_LEN,
    SEQ_LEN,
    WindowMeta,
    WindowSample,
    get_scale_stride,
    make_windows_from_station,
    read_station_csv,
)

CITY_ALIASES = {
    "shenyang": "shanyang",
    "sichuanpend": "sichuanpendi",
    "zhusanjia": "zhusanjiao",
}


def normalize_city_name(name: str) -> str:
    return CITY_ALIASES.get(name, name)


def load_city_windows_from_year_dir(
    city: str,
    year: int,
    city_year_dir: Path,
    stride: int,
    min_gap_hours: Optional[int] = None,
) -> List[WindowSample]:
    city = normalize_city_name(city)
    out: List[WindowSample] = []
    for fp in sorted(city_year_dir.glob("*.csv")):
        if fp.suffix.lower() != ".csv":
            continue
        out.extend(make_windows_from_station(city, year, fp, stride=stride, min_gap_hours=min_gap_hours))
    return out


class PretrainPairDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[WindowSample],
        pair_records: Sequence,
        stats: CityYearStats,
        scales: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        self.windows = list(windows)
        self.pairs = list(pair_records)
        self.stats = stats
        self.scales = tuple(scales)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rec = self.pairs[idx]
        anchor = self.windows[rec.anchor_idx]
        pos = self.windows[rec.positive_indices[0]]
        neg = self.windows[rec.negative_indices[0]]

        def std_x(x: np.ndarray) -> np.ndarray:
            return self.stats.standardize_x(x)

        return {
            "anchor_x": torch.tensor(std_x(anchor.x), dtype=torch.float32),
            "positive_x": torch.tensor(std_x(pos.x), dtype=torch.float32),
            "negative_x": torch.tensor(std_x(neg.x), dtype=torch.float32),
        }


@dataclass
class Stage2Item:
    city: str
    station: str
    start_time: str
    x: np.ndarray
    y_std: np.ndarray
    y_phys: np.ndarray


class Stage2SampleDataset(Dataset):
    def __init__(self, items: Sequence[Stage2Item]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return {
            "city": item.city,
            "station": item.station,
            "start_time": item.start_time,
            "x": torch.tensor(item.x, dtype=torch.float32),
            "y_std": torch.tensor(item.y_std, dtype=torch.float32),
            "y_phys": torch.tensor(item.y_phys, dtype=torch.float32),
        }


def build_stage2_monthly_samples(
    city: str,
    year_dir: Path,
    stats: CityYearStats,
    samples_per_month: int = 5,
    train_per_month: int = 4,
    min_gap_days: int = 4,
    seed: int = 2026,
) -> Tuple[List[Stage2Item], List[Stage2Item]]:
    city = normalize_city_name(city)
    rng = np.random.default_rng(seed)
    train_items: List[Stage2Item] = []
    val_items: List[Stage2Item] = []

    for fp in sorted(year_dir.glob("*.csv")):
        station = fp.stem
        df = read_station_csv(fp)
        by_month = {m: [] for m in range(1, 13)}
        total = len(df)
        for start in range(0, total - (SEQ_LEN + PRED_LEN) + 1):
            start_time = pd.Timestamp(df.iloc[start]["time"])
            by_month[int(start_time.month)].append((start, start_time))

        for month in range(1, 13):
            candidates = by_month[month]
            if not candidates:
                continue
            chosen: List[Tuple[int, pd.Timestamp]] = []
            for min_gap in [min_gap_days, 3, 2, 1]:
                shuffled = candidates.copy()
                rng.shuffle(shuffled)
                tmp: List[Tuple[int, pd.Timestamp]] = []
                for start, ts in shuffled:
                    ok = True
                    for _, prev in tmp:
                        if abs((ts - prev).total_seconds()) < min_gap * 86400:
                            ok = False
                            break
                    if ok:
                        tmp.append((start, ts))
                    if len(tmp) >= samples_per_month:
                        break
                if len(tmp) >= min(samples_per_month, len(candidates)) or min_gap == 1:
                    chosen = tmp
                    break
            chosen = sorted(chosen, key=lambda x: x[1])
            for i, (start, ts) in enumerate(chosen[:samples_per_month]):
                x_df = df.iloc[start:start + SEQ_LEN]
                y_df = df.iloc[start + SEQ_LEN:start + SEQ_LEN + PRED_LEN]
                x = stats.standardize_x(x_df[FEATURE_COLS].to_numpy(dtype=np.float32))
                y_phys = y_df[["O3"]].to_numpy(dtype=np.float32)
                y_std = stats.standardize_y(y_phys)
                item = Stage2Item(city=city, station=station, start_time=str(ts), x=x, y_std=y_std, y_phys=y_phys)
                if i < train_per_month:
                    train_items.append(item)
                else:
                    val_items.append(item)
    return train_items, val_items


def list_sample_files(data_root: Path):
    items = []
    for city_dir in sorted(data_root.iterdir()):
        if not city_dir.is_dir() or city_dir.name.startswith("_"):
            continue
        city = normalize_city_name(city_dir.name)
        for station_dir in sorted(city_dir.iterdir()):
            if not station_dir.is_dir():
                continue
            for sample_fp in sorted(station_dir.glob("*.csv")):
                items.append({
                    "city": city,
                    "station": station_dir.name,
                    "sample_id": sample_fp.stem,
                    "path": sample_fp,
                })
    return items


def load_one_test_sample(fp: Path):
    df = pd.read_csv(fp)
    missing_cols = [c for c in ["time"] + FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "part" in df.columns:
        x_df = df[df["part"] == "input"].copy()
        y_df = df[df["part"] == "target"].copy()
    else:
        x_df = df.iloc[:SEQ_LEN].copy()
        y_df = df.iloc[SEQ_LEN:SEQ_LEN + PRED_LEN].copy()
    if len(x_df) != SEQ_LEN or len(y_df) != PRED_LEN:
        raise ValueError(f"bad sample length: input={len(x_df)}, target={len(y_df)}")
    x = x_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = y_df[["O3"]].to_numpy(dtype=np.float32)
    return x, y
