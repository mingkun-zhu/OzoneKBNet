from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

FEATURE_COLS = ["O3", "NO2", "PM2.5", "temperature", "relative_humidity"]
TARGET_COL = "O3"


@dataclass
class CityYearStats:
    city: str
    year: int
    feature_mean: Dict[str, float]
    feature_std: Dict[str, float]
    target_mean: float
    target_std: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CityYearStats":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def standardize_x(self, x: np.ndarray) -> np.ndarray:
        out = x.astype(np.float32).copy()
        for i, col in enumerate(FEATURE_COLS):
            out[:, i] = (out[:, i] - self.feature_mean[col]) / (self.feature_std[col] + 1e-8)
        return out

    def standardize_y(self, y: np.ndarray) -> np.ndarray:
        return ((y.astype(np.float32) - self.target_mean) / (self.target_std + 1e-8)).astype(np.float32)

    def destandardize_y(self, y: np.ndarray) -> np.ndarray:
        return (y.astype(np.float32) * (self.target_std + 1e-8) + self.target_mean).astype(np.float32)


def _iter_city_csvs(city_dir: Path) -> Iterable[Path]:
    for fp in sorted(city_dir.glob("*.csv")):
        if fp.is_file() and fp.suffix.lower() == ".csv":
            yield fp


def compute_city_year_stats(city: str, year: int, city_dir: Path) -> CityYearStats:
    frames: List[pd.DataFrame] = []
    for fp in _iter_city_csvs(city_dir):
        df = pd.read_csv(fp, usecols=FEATURE_COLS)
        for col in FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df.dropna())
    if not frames:
        raise FileNotFoundError(f"No CSV files found for city={city} in {city_dir}")
    full = pd.concat(frames, ignore_index=True)
    feature_mean = {c: float(full[c].mean()) for c in FEATURE_COLS}
    feature_std = {c: float(full[c].std(ddof=0) + 1e-8) for c in FEATURE_COLS}
    return CityYearStats(
        city=city,
        year=year,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=feature_mean[TARGET_COL],
        target_std=feature_std[TARGET_COL],
    )
