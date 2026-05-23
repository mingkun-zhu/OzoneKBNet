from __future__ import annotations

from pathlib import Path
import pickle
from typing import Tuple

import faiss
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def build_index_flat_ip(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    emb = l2_normalize(embeddings)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def search_index(index: faiss.IndexFlatIP, query: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    q = l2_normalize(query.astype(np.float32))
    return index.search(q, topk)


def save_faiss_index(index: faiss.IndexFlatIP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))
