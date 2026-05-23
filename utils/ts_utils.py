from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .normalization import FEATURE_COLS

SEQ_LEN = 96
PRED_LEN = 48
SCALES = (1, 2, 4, 8)


def average_pool_sequence(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr.copy()
    n, c = arr.shape
    usable = (n // factor) * factor
    arr = arr[:usable]
    return arr.reshape(usable // factor, factor, c).mean(axis=1)


def average_pool_target(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr.copy()
    usable = (len(arr) // factor) * factor
    arr = arr[:usable]
    return arr.reshape(usable // factor, factor, 1).mean(axis=1).reshape(-1, 1)


def get_scale_stride(scale: int) -> int:
    if scale in (1, 2):
        return 6
    return 12


def month_to_season(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


@dataclass
class WindowMeta:
    city: str
    station: str
    start_idx: int
    start_time: str
    month: int
    season: int
    year: int


@dataclass
class WindowSample:
    x: np.ndarray  # [96,5]
    y: np.ndarray  # [48,1]
    meta: WindowMeta


def read_station_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["time"] + FEATURE_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[need].copy()
    df["time"] = pd.to_datetime(df["time"])
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def make_windows_from_station(
    city: str,
    year: int,
    station_path: Path,
    stride: int,
    min_gap_hours: Optional[int] = None,
) -> List[WindowSample]:
    df = read_station_csv(station_path)
    station = station_path.stem
    windows: List[WindowSample] = []
    total = len(df)
    last_kept_ts: Optional[pd.Timestamp] = None
    for start in range(0, total - (SEQ_LEN + PRED_LEN) + 1, stride):
        x_df = df.iloc[start:start + SEQ_LEN]
        y_df = df.iloc[start + SEQ_LEN:start + SEQ_LEN + PRED_LEN]
        start_time = pd.Timestamp(x_df.iloc[0]["time"])
        if min_gap_hours is not None and last_kept_ts is not None:
            if (start_time - last_kept_ts).total_seconds() < min_gap_hours * 3600:
                continue
        x = x_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = y_df[["O3"]].to_numpy(dtype=np.float32)
        meta = WindowMeta(
            city=city,
            station=station,
            start_idx=start,
            start_time=str(start_time),
            month=int(start_time.month),
            season=month_to_season(int(start_time.month)),
            year=year,
        )
        windows.append(WindowSample(x=x, y=y, meta=meta))
        last_kept_ts = start_time
    return windows


def normalize_score_array(scores: np.ndarray) -> np.ndarray:
    mu = float(scores.mean())
    sigma = float(scores.std(ddof=0))
    return (scores - mu) / (sigma + 1e-8)
