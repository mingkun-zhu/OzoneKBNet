from __future__ import annotations

from pathlib import Path
import pickle
from typing import Dict, List, Sequence

import numpy as np
import torch

from utils.faiss_utils import build_index_flat_ip, save_faiss_index, load_faiss_index
from utils.normalization import CityYearStats
from utils.ts_utils import WindowSample, average_pool_sequence, average_pool_target


def encode_windows_for_scale(model, windows: Sequence[WindowSample], stats: CityYearStats, scale: int, batch_size: int = 256):
    xs_std = [stats.standardize_x(w.x) for w in windows]
    embs: List[np.ndarray] = []
    device = next(model.parameters()).device
    for i in range(0, len(xs_std), batch_size):
        batch = xs_std[i:i + batch_size]
        batch_t = torch.tensor(np.stack(batch, axis=0), dtype=torch.float32, device=device)
        with torch.inference_mode():
            emb = model.encode_scale(batch_t, scale).detach().cpu().numpy()
        embs.append(emb)
    embeddings = np.concatenate(embs, axis=0).astype(np.float32)
    x_std_scale = np.stack([average_pool_sequence(stats.standardize_x(w.x), scale) for w in windows], axis=0).astype(np.float32)
    y_std_scale = np.stack([average_pool_target(stats.standardize_y(w.y), scale) for w in windows], axis=0).astype(np.float32)
    meta = [
        {
            "city": w.meta.city,
            "station": w.meta.station,
            "start_idx": w.meta.start_idx,
            "start_time": w.meta.start_time,
            "month": w.meta.month,
            "season": w.meta.season,
            "year": w.meta.year,
        }
        for w in windows
    ]
    return {
        "embeddings": embeddings,
        "x_std": x_std_scale,
        "y_std": y_std_scale,
        "meta": meta,
        "index": build_index_flat_ip(embeddings),
    }


def save_kb_bundle(kb_bundle: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for scale, bundle in kb_bundle.items():
        scale_path = path.parent / f"index_scale_{scale}.faiss"
        save_faiss_index(bundle["index"], scale_path)
        serializable[scale] = {
            "embeddings": bundle["embeddings"],
            "x_std": bundle["x_std"],
            "y_std": bundle["y_std"],
            "meta": bundle["meta"],
            "index_path": str(scale_path),
        }
    with path.open("wb") as f:
        pickle.dump(serializable, f)


def load_kb_bundle(path: Path) -> Dict:
    with path.open("rb") as f:
        obj = pickle.load(f)
    kb = {}
    for scale, bundle in obj.items():
        kb[str(scale)] = {
            "embeddings": bundle["embeddings"],
            "x_std": bundle["x_std"],
            "y_std": bundle["y_std"],
            "meta": bundle["meta"],
            "index": load_faiss_index(Path(bundle["index_path"])),
        }
    return kb
