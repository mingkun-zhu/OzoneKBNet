from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Sequence, Optional

import numpy as np
from fastdtw import fastdtw

from .ts_utils import WindowSample


@dataclass
class PairRecord:
    anchor_idx: int
    positive_indices: List[int]
    negative_indices: List[int]


@dataclass
class PairCache:
    city: str
    year: int
    records: List[PairRecord]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "PairCache":
        with path.open("rb") as f:
            return pickle.load(f)


# =========================
# Worker global state
# =========================
_WORKER_START_TIMES: Optional[np.ndarray] = None
_WORKER_SEASONS: Optional[np.ndarray] = None
_WORKER_YS: Optional[List[np.ndarray]] = None
_WORKER_YEAR: Optional[int] = None


def _set_worker_state(
    start_times: np.ndarray,
    seasons: np.ndarray,
    ys: List[np.ndarray],
    year: int,
) -> None:
    global _WORKER_START_TIMES, _WORKER_SEASONS, _WORKER_YS, _WORKER_YEAR
    _WORKER_START_TIMES = start_times
    _WORKER_SEASONS = seasons
    _WORKER_YS = ys
    _WORKER_YEAR = year


def _peak_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    ta = int(np.argmax(a))
    tb = int(np.argmax(b))
    pa = float(np.max(a))
    pb = float(np.max(b))
    dt = abs(ta - tb) / max(len(a) - 1, 1)
    dm = abs(pa - pb) / (abs(pa) + abs(pb) + 1e-8)
    return 0.5 * dt + 0.5 * dm


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    da, _ = fastdtw(a.reshape(-1).tolist(), b.reshape(-1).tolist())
    return float(da)


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))


def _normalize_minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax - vmin < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v - vmin) / (vmax - vmin + 1e-8)


def _vectorized_l2(anchor_y: np.ndarray, candidate_mat: np.ndarray) -> np.ndarray:
    # anchor_y: [48], candidate_mat: [N, 48]
    return np.linalg.norm(candidate_mat - anchor_y[None, :], axis=1).astype(np.float32)


def _vectorized_peak(anchor_y: np.ndarray, candidate_mat: np.ndarray) -> np.ndarray:
    ta = int(np.argmax(anchor_y))
    pa = float(np.max(anchor_y))

    tb = np.argmax(candidate_mat, axis=1).astype(np.float32)
    pb = np.max(candidate_mat, axis=1).astype(np.float32)

    dt = np.abs(tb - float(ta)) / max(candidate_mat.shape[1] - 1, 1)
    dm = np.abs(pb - pa) / (np.abs(pb) + abs(pa) + 1e-8)
    return (0.5 * dt + 0.5 * dm).astype(np.float32)


def _build_one_anchor_record(
    anchor_idx: int,
    positive_top_p: int,
    negative_top_q: int,
    candidate_pool_cap: int,
    candidate_min_gap_hours: int,
    lambda_dtw: float,
    lambda_l2: float,
    lambda_peak: float,
    dtw_shortlist_near: int,
    dtw_shortlist_far: int,
) -> Optional[PairRecord]:
    start_times = _WORKER_START_TIMES
    seasons = _WORKER_SEASONS
    ys = _WORKER_YS
    year = _WORKER_YEAR

    if start_times is None or seasons is None or ys is None or year is None:
        raise RuntimeError("Worker state has not been initialized.")

    anchor_time = start_times[anchor_idx]
    anchor_season = seasons[anchor_idx]
    anchor_y = ys[anchor_idx]

    same_season_idx = np.where(seasons == anchor_season)[0]
    if same_season_idx.size == 0:
        return None

    deltas = np.abs((start_times[same_season_idx] - anchor_time).astype("timedelta64[h]").astype(np.int64))
    valid_mask = (same_season_idx != anchor_idx) & (deltas > candidate_min_gap_hours)
    valid_idx = same_season_idx[valid_mask]

    if valid_idx.size == 0:
        return None

    if valid_idx.size > candidate_pool_cap:
        rng = np.random.default_rng(anchor_idx + year)
        valid_idx = rng.choice(valid_idx, size=candidate_pool_cap, replace=False)

    candidate_mat = np.stack([ys[j] for j in valid_idx.tolist()], axis=0).astype(np.float32)

    # Step 1: cheap distances for all valid candidates
    l2s = _vectorized_l2(anchor_y, candidate_mat)
    peaks = _vectorized_peak(anchor_y, candidate_mat)

    l2s_norm = _normalize_minmax(l2s)
    peaks_norm = _normalize_minmax(peaks)

    # Cheap pre-score for shortlist
    cheap_scores = lambda_l2 * l2s_norm + lambda_peak * peaks_norm
    order = np.argsort(cheap_scores)

    near_take = min(dtw_shortlist_near, len(order))
    far_take = min(dtw_shortlist_far, len(order))

    shortlist_pos = np.concatenate([order[:near_take], order[-far_take:]], axis=0)
    shortlist_pos = np.unique(shortlist_pos)

    shortlist_valid_idx = valid_idx[shortlist_pos]
    shortlist_mat = candidate_mat[shortlist_pos]
    shortlist_l2 = l2s[shortlist_pos]
    shortlist_peak = peaks[shortlist_pos]

    # Step 2: DTW only on shortlist
    dtws = np.array(
        [_dtw_distance(anchor_y, shortlist_mat[i]) for i in range(shortlist_mat.shape[0])],
        dtype=np.float32,
    )

    dtws_norm = _normalize_minmax(dtws)
    shortlist_l2_norm = _normalize_minmax(shortlist_l2)
    shortlist_peak_norm = _normalize_minmax(shortlist_peak)

    final_scores = (
        lambda_dtw * dtws_norm
        + lambda_l2 * shortlist_l2_norm
        + lambda_peak * shortlist_peak_norm
    )

    final_order = np.argsort(final_scores)

    positives = shortlist_valid_idx[final_order[:positive_top_p]].tolist()
    negatives = shortlist_valid_idx[final_order[::-1][:negative_top_q]].tolist()

    return PairRecord(
        anchor_idx=anchor_idx,
        positive_indices=positives,
        negative_indices=negatives,
    )


def build_pair_cache(
    city: str,
    year: int,
    windows: Sequence[WindowSample],
    positive_top_p: int,
    negative_top_q: int,
    candidate_pool_cap: int = 2000,
    candidate_min_gap_hours: int = 72,
    lambda_dtw: float = 1 / 3,
    lambda_l2: float = 1 / 3,
    lambda_peak: float = 1 / 3,
    dtw_shortlist_near: int = 64,
    dtw_shortlist_far: int = 64,
    n_jobs: Optional[int] = None,
    log_every: int = 100,
) -> PairCache:
    """
    Build pair cache for supervised contrastive pretraining.

    Major speedups over the old version:
    1) parallel over anchors
    2) vectorized L2 / peak distances
    3) DTW only on shortlist, not on all candidates
    """
    if n_jobs is None:
        n_jobs = min(8, max(1, (os.cpu_count() or 1) - 2))

    start_times = np.array([np.datetime64(w.meta.start_time) for w in windows])
    seasons = np.array([w.meta.season for w in windows])
    ys = [w.y.astype(np.float32).reshape(-1) for w in windows]

    total_anchors = len(windows)
    print(
        f"[PairCache] city={city}, year={year}, anchors={total_anchors}, "
        f"n_jobs={n_jobs}, candidate_pool_cap={candidate_pool_cap}, "
        f"dtw_shortlist=({dtw_shortlist_near}+{dtw_shortlist_far})",
        flush=True,
    )

    records: List[PairRecord] = []

    # Keep BLAS/OpenMP from over-parallelizing inside each process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Put data into global state; with fork on Linux this avoids repeatedly passing huge arrays
    _set_worker_state(start_times, seasons, ys, year)

    anchor_indices = list(range(total_anchors))

    if n_jobs <= 1:
        for i, anchor_idx in enumerate(anchor_indices, start=1):
            rec = _build_one_anchor_record(
                anchor_idx=anchor_idx,
                positive_top_p=positive_top_p,
                negative_top_q=negative_top_q,
                candidate_pool_cap=candidate_pool_cap,
                candidate_min_gap_hours=candidate_min_gap_hours,
                lambda_dtw=lambda_dtw,
                lambda_l2=lambda_l2,
                lambda_peak=lambda_peak,
                dtw_shortlist_near=dtw_shortlist_near,
                dtw_shortlist_far=dtw_shortlist_far,
            )
            if rec is not None:
                records.append(rec)
            if i % log_every == 0 or i == total_anchors:
                print(f"[PairCache] done {i}/{total_anchors} anchors", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futures = [
                ex.submit(
                    _build_one_anchor_record,
                    anchor_idx,
                    positive_top_p,
                    negative_top_q,
                    candidate_pool_cap,
                    candidate_min_gap_hours,
                    lambda_dtw,
                    lambda_l2,
                    lambda_peak,
                    dtw_shortlist_near,
                    dtw_shortlist_far,
                )
                for anchor_idx in anchor_indices
            ]

            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
                done += 1
                if done % log_every == 0 or done == total_anchors:
                    print(f"[PairCache] done {done}/{total_anchors} anchors", flush=True)

    records.sort(key=lambda r: r.anchor_idx)
    print(f"[PairCache] finished: valid_records={len(records)}", flush=True)
    return PairCache(city=city, year=year, records=records)