
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.faiss_utils import search_index


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def _trim(self, x: torch.Tensor, length: int) -> torch.Tensor:
        return x[..., :length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        out = self.conv1(x)
        out = self._trim(out, length)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self._trim(out, length)
        out = self.act(out)
        out = self.dropout(out)
        return x + out


class TCNEncoder(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 64, layers: int = 4, emb_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualTCNBlock(hidden, kernel_size=3, dilation=2 ** i, dropout=dropout) for i in range(layers)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.proj(out)
        for blk in self.blocks:
            out = blk(out)
        out = self.head(out)
        out = F.normalize(out, p=2, dim=-1)
        return out


class UpsampleRefine(nn.Module):
    def __init__(self, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        return residual + out


class DirectForecastBranch(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 64, layers: int = 4, pred_len: int = 48, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = TCNEncoder(in_dim=in_dim, hidden=hidden, layers=layers, emb_dim=hidden, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.mlp(feat)
        return out.unsqueeze(-1)



class LSTMDirectBranch(nn.Module):
    """Lightweight LSTM direct forecasting branch.

    Input:
        x: [B, 96, C]
    Output:
        y: [B, pred_len, 1]
    """
    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        feat = h_n[-1]
        y = self.head(feat).unsqueeze(-1)
        return y


class OzoneKBNet(nn.Module):
    """
    v6:
    - keep the original online forward for evaluation/debugging
    - add stage2 retrieval cache support:
      cache only frozen retrieval outputs (z-scored s1/s2/s3 and candidate y sequences)
      so final training logic remains learnable and precision is preserved
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.branch_mode = getattr(cfg, "branch_mode", "full")
        self.scales = tuple(cfg.scales)
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.horizon_emb = nn.Embedding(cfg.pred_len, cfg.horizon_emb_dim)
        self.encoders = nn.ModuleDict({str(r): TCNEncoder(
            in_dim=cfg.enc_in,
            hidden=cfg.encoder_hidden,
            layers=cfg.encoder_layers,
            emb_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
        ) for r in self.scales})
        self.rank_gate = nn.Sequential(
            nn.Linear(cfg.enc_in + 3, cfg.rank_gate_hidden),
            nn.GELU(),
            nn.Linear(cfg.rank_gate_hidden, 3),
        )
        self.alpha_mlp = nn.Sequential(
            nn.Linear(cfg.enc_in + cfg.horizon_emb_dim + 4, cfg.alpha_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.alpha_mlp_hidden, 4),
        )
        self.final_gate = nn.Sequential(
            nn.Linear(cfg.enc_in + 12 + 1 + cfg.horizon_emb_dim, cfg.final_gate_hidden),
            nn.GELU(),
            nn.Linear(cfg.final_gate_hidden, 1),
        )
        self.direct_branch = DirectForecastBranch(
            in_dim=cfg.enc_in,
            hidden=cfg.c_dir,
            layers=cfg.l_dir,
            pred_len=cfg.pred_len,
            dropout=cfg.dropout,
        )
        self.upsamplers = nn.ModuleDict({
            str(r): UpsampleRefine(hidden=cfg.c_u, dropout=cfg.dropout) for r in self.scales if r != 1
        })
        self.kb: Optional[Dict] = None
        self.kb_device: Optional[torch.device] = None
        self.freeze_retrieval_encoders(False)



        self._maybe_replace_direct_branch()

    def _make_direct_only_output(self, direct: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return direct-only outputs with the same keys as full mode."""
        b = direct.shape[0]
        device = direct.device
        dtype = direct.dtype
        alpha = torch.zeros(b, self.pred_len, len(self.scales), device=device, dtype=dtype)
        beta = torch.zeros(b, self.pred_len, 1, device=device, dtype=dtype)
        return {
            "y_hat": direct,
            "y_rag": torch.zeros_like(direct),
            "y_dir": direct,
            "alpha": alpha,
            "beta": beta,
        }

    def _select_final_branch(
        self,
        rag_t: torch.Tensor,
        direct_t: torch.Tensor,
        beta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select/fuse branches.

        Normal full gate:
            final = beta * retrieval + (1-beta) * direct

        Direct-biased residual gate:
            final = direct + gamma * (retrieval - direct),
            where gamma in [0, gamma_max].
        """
        if self.branch_mode == "retrieval_only":
            return rag_t, torch.ones_like(beta_t)
        if self.branch_mode == "direct_only":
            return direct_t, torch.zeros_like(beta_t)

        if bool(getattr(self.cfg, "direct_biased_gate", 0)):
            gamma_max = float(getattr(self.cfg, "direct_biased_gamma_max", 0.30))
            gamma = gamma_max * beta_t.clamp(0.0, 1.0)
            final_t = direct_t + gamma * (rag_t - direct_t)
            return final_t, gamma

        return beta_t * rag_t + (1.0 - beta_t) * direct_t, beta_t
    def _maybe_replace_direct_branch(self) -> None:
        """Replace the default TCN direct branch with LSTM when requested."""
        direct_type = getattr(self.cfg, "direct_branch_type", "tcn")
        if direct_type != "lstm":
            return

        input_dim = int(getattr(self.cfg, "enc_in", 5))
        pred_len = int(getattr(self.cfg, "pred_len", getattr(self, "pred_len", 48)))
        hidden = int(getattr(self.cfg, "direct_lstm_hidden", 128))
        layers = int(getattr(self.cfg, "direct_lstm_layers", 1))
        dropout = float(getattr(self.cfg, "direct_lstm_dropout", 0.1))

        self.direct_branch = LSTMDirectBranch(
            input_dim=input_dim,
            pred_len=pred_len,
            hidden=hidden,
            num_layers=layers,
            dropout=dropout,
        )


    def freeze_retrieval_encoders(self, freeze: bool = True) -> None:
        for p in self.encoders.parameters():
            p.requires_grad = not freeze

    def pool_to_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return x
        b, t, c = x.shape
        usable = (t // scale) * scale
        x = x[:, :usable, :]
        return x.reshape(b, usable // scale, scale, c).mean(dim=2)

    @staticmethod
    def _zscore_torch(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean()
        sigma = x.std(unbiased=False)
        return (x - mu) / (sigma + 1e-8)

    @staticmethod
    def _hours_between(a: str, b: str) -> int:
        a_ts = np.datetime64(a)
        b_ts = np.datetime64(b)
        return int((a_ts - b_ts).astype("timedelta64[h]").astype(int))

    def _ensure_kb_device(self, device: torch.device) -> None:
        if self.kb is None:
            raise RuntimeError("KB has not been attached. Call set_kb(...) first.")
        if self.kb_device == device:
            return
        for scale in self.scales:
            kb = self.kb[str(scale)]
            kb["embeddings_t"] = torch.as_tensor(kb["embeddings"], dtype=torch.float32, device=device)
            kb["x_std_t"] = torch.as_tensor(kb["x_std"], dtype=torch.float32, device=device)
            kb["y_std_t"] = torch.as_tensor(kb["y_std"], dtype=torch.float32, device=device)
        self.kb_device = device

    def set_kb(self, kb_bundle: Dict) -> None:
        self.kb = kb_bundle
        self.kb_device = None
        device = next(self.parameters()).device
        self._ensure_kb_device(device)

    def encode_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        pooled = self.pool_to_scale(x, scale)
        return self.encoders[str(scale)](pooled)

    def pretrain_forward(self, anchor_x: torch.Tensor, positive_x: torch.Tensor, negative_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        for r in self.scales:
            a = self.encode_scale(anchor_x, r)
            p = self.encode_scale(positive_x, r)
            n = self.encode_scale(negative_x, r)
            out[str(r)] = {"anchor": a, "positive": p, "negative": n}
        return out

    def supcon_loss(self, batch_dict: Dict[str, Dict[str, torch.Tensor]], tau: float) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for r in self.scales:
            d = batch_dict[str(r)]
            a, p, n = d["anchor"], d["positive"], d["negative"]
            sim_ap = torch.sum(a * p, dim=-1) / tau
            sim_an = torch.sum(a * n, dim=-1) / tau
            logits = torch.stack([sim_ap, sim_an], dim=1)
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = loss + F.cross_entropy(logits, targets)
        return loss

    def _upsample_to_48(self, v: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return v
        up = F.interpolate(v.transpose(1, 2), size=self.pred_len, mode="linear", align_corners=False).transpose(1, 2)
        return self.upsamplers[str(scale)](up)

    def _local_trend_scores_torch(self, q_scale_t: torch.Tensor, cand_x_t: torch.Tensor, scale: int) -> torch.Tensor:
        l_r = max(self.cfg.local_trend_L // scale, 1)
        q_tail = q_scale_t[-l_r:, :]
        c_tail = cand_x_t[:, -l_r:, :]
        dq = torch.diff(q_tail, dim=0).reshape(-1)
        dc = torch.diff(c_tail, dim=1).reshape(c_tail.shape[0], -1)
        if dq.numel() == 0 or dc.numel() == 0:
            return torch.zeros(cand_x_t.shape[0], dtype=q_scale_t.dtype, device=q_scale_t.device)
        dq_norm = torch.norm(dq) + 1e-8
        dc_norm = torch.norm(dc, dim=1) + 1e-8
        dot = torch.sum(dc * dq.unsqueeze(0), dim=1)
        return dot / (dq_norm * dc_norm)

    def _dedup_candidate_indices(self, idxs: np.ndarray, meta_list: List[Dict]) -> List[int]:
        dedup: List[int] = []
        seen: List[Tuple[int, Dict]] = []
        for j in idxs.tolist():
            if j < 0:
                continue
            meta = meta_list[j]
            allow = True
            for _, kept_meta in seen:
                if meta["station"] == kept_meta["station"]:
                    delta = abs(self._hours_between(meta["start_time"], kept_meta["start_time"]))
                    if delta < self.cfg.delta_dedup_hours:
                        allow = False
                        break
            if allow:
                seen.append((j, meta))
                dedup.append(j)
        return dedup

    def _retrieve_one_scale(self, q_std_t: torch.Tensor, q_emb_t: torch.Tensor, q_feat_t: torch.Tensor, scale: int):
        if self.kb is None:
            raise RuntimeError("KB has not been attached.")
        kb = self.kb[str(scale)]
        device = q_std_t.device

        q_emb_np = q_emb_t.detach().cpu().numpy().astype(np.float32)
        _, idxs = search_index(kb["index"], q_emb_np[None, :], topk=self.cfg.coarse_top_m)
        idxs = idxs[0]
        cand_indices = self._dedup_candidate_indices(idxs, kb["meta"])
        if len(cand_indices) == 0:
            raise RuntimeError(f"No candidates left after dedup for scale={scale}")

        idx_t = torch.as_tensor(cand_indices, dtype=torch.long, device=device)
        cand_emb_t = kb["embeddings_t"].index_select(0, idx_t)
        cand_x_t = kb["x_std_t"].index_select(0, idx_t)
        cand_y_t = kb["y_std_t"].index_select(0, idx_t)

        cosine_scores = torch.sum(cand_emb_t * q_emb_t.unsqueeze(0), dim=1)
        embed_scores = -torch.sum((cand_emb_t - q_emb_t.unsqueeze(0)) ** 2, dim=1)
        q_scale_t = self.pool_to_scale(q_std_t.unsqueeze(0), scale)[0]
        trend_scores = self._local_trend_scores_torch(q_scale_t, cand_x_t, scale)

        s1 = self._zscore_torch(cosine_scores)
        s2 = self._zscore_torch(embed_scores)
        s3 = self._zscore_torch(trend_scores)

        return self._finish_one_scale_from_scores(q_feat_t, scale, s1, s2, s3, cand_y_t)

    def _finish_one_scale_from_scores(
        self,
        q_feat_t: torch.Tensor,
        scale: int,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
        cand_y_t: torch.Tensor,
    ):
        h_t = torch.cat([q_feat_t, torch.stack([s1.mean(), s2.mean(), s3.mean()], dim=0)], dim=0)
        g = F.softmax(self.rank_gate(h_t), dim=-1)
        w1 = F.softmax(s1, dim=0)
        w2 = F.softmax(s2, dim=0)
        w3 = F.softmax(s3, dim=0)
        weights_t = g[0] * w1 + g[1] * w2 + g[2] * w3

        topk = min(self.cfg.final_top_k, weights_t.numel())
        sel_w_t, sel_pos_t = torch.topk(weights_t, k=topk, dim=0, largest=True, sorted=True)
        sel_w_t = sel_w_t / (sel_w_t.sum() + 1e-8)
        sel_y_t = cand_y_t.index_select(0, sel_pos_t)
        agg_t = torch.sum(sel_y_t * sel_w_t[:, None, None], dim=0)

        if sel_w_t.numel() == 1:
            m_r = sel_w_t[0]
            h_r = torch.zeros((), dtype=sel_w_t.dtype, device=sel_w_t.device)
            v_r = torch.zeros((), dtype=sel_w_t.dtype, device=sel_w_t.device)
        else:
            m_r = sel_w_t[0] - sel_w_t[1]
            h_r = -(sel_w_t * torch.log(sel_w_t + 1e-8)).sum()
            sel_up_t = self._upsample_to_48(sel_y_t, scale)
            v_r = sel_up_t.var(dim=0, unbiased=False).mean()
        z_t = torch.stack([m_r, h_r, v_r], dim=0)
        return agg_t, z_t

    @torch.inference_mode()
    def build_stage2_cache_for_x(self, x_std_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Cache only frozen retrieval-side outputs.
        Returns per-scale:
        - s1/s2/s3: z-scored similarity vectors padded to [M]
        - cand_y: candidate future sequences padded to [M, Lr, 1]
        - mask: valid candidates in the padded arrays [M]
        """
        if self.kb is None:
            raise RuntimeError("KB has not been attached.")
        if x_std_t.ndim != 2:
            raise ValueError(f"x_std_t must be [T,C], got {tuple(x_std_t.shape)}")
        device = x_std_t.device
        self._ensure_kb_device(device)

        cache: Dict[str, torch.Tensor] = {}
        max_m = int(self.cfg.coarse_top_m)

        for scale in self.scales:
            kb = self.kb[str(scale)]
            q_emb_t = self.encode_scale(x_std_t.unsqueeze(0), scale)[0]
            q_emb_np = q_emb_t.detach().cpu().numpy().astype(np.float32)
            _, idxs = search_index(kb["index"], q_emb_np[None, :], topk=self.cfg.coarse_top_m)
            idxs = idxs[0]
            cand_indices = self._dedup_candidate_indices(idxs, kb["meta"])
            if len(cand_indices) == 0:
                raise RuntimeError(f"No candidates left after dedup for scale={scale}")

            idx_t = torch.as_tensor(cand_indices, dtype=torch.long, device=device)
            cand_emb_t = kb["embeddings_t"].index_select(0, idx_t)
            cand_x_t = kb["x_std_t"].index_select(0, idx_t)
            cand_y_t = kb["y_std_t"].index_select(0, idx_t)

            cosine_scores = torch.sum(cand_emb_t * q_emb_t.unsqueeze(0), dim=1)
            embed_scores = -torch.sum((cand_emb_t - q_emb_t.unsqueeze(0)) ** 2, dim=1)
            q_scale_t = self.pool_to_scale(x_std_t.unsqueeze(0), scale)[0]
            trend_scores = self._local_trend_scores_torch(q_scale_t, cand_x_t, scale)

            s1 = self._zscore_torch(cosine_scores)
            s2 = self._zscore_torch(embed_scores)
            s3 = self._zscore_torch(trend_scores)

            nc = s1.numel()
            Lr = cand_y_t.shape[1]
            s1_pad = torch.zeros(max_m, dtype=torch.float32, device=device)
            s2_pad = torch.zeros(max_m, dtype=torch.float32, device=device)
            s3_pad = torch.zeros(max_m, dtype=torch.float32, device=device)
            y_pad = torch.zeros(max_m, Lr, 1, dtype=torch.float32, device=device)
            mask = torch.zeros(max_m, dtype=torch.bool, device=device)

            take = min(max_m, nc)
            s1_pad[:take] = s1[:take]
            s2_pad[:take] = s2[:take]
            s3_pad[:take] = s3[:take]
            y_pad[:take] = cand_y_t[:take]
            mask[:take] = True

            cache[f"s1_{scale}"] = s1_pad.cpu()
            cache[f"s2_{scale}"] = s2_pad.cpu()
            cache[f"s3_{scale}"] = s3_pad.cpu()
            cache[f"cand_y_{scale}"] = y_pad.cpu()
            cache[f"mask_{scale}"] = mask.cpu()

        return cache

    def forward_from_stage2_cache(self, x: torch.Tensor, cache_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Use cached frozen retrieval outputs for stage2 training.
        This should preserve final accuracy because only frozen parts are cached.
        """
        device = x.device
        q_feat = x.mean(dim=1)
        direct = self.direct_branch(x)
        if self.branch_mode == "direct_only":
            return self._make_direct_only_output(direct)
        batch_size = x.size(0)
        horizon_idx = torch.arange(self.pred_len, device=device)
        horizon_emb = self.horizon_emb(horizon_idx)

        all_final, all_rag, all_direct, all_alpha, all_beta = [], [], [], [], []

        for b in range(batch_size):
            q_feat_t = q_feat[b]
            scale_preds = []
            z_parts = []

            for scale in self.scales:
                s1_full = cache_batch[f"s1_{scale}"][b].to(device)
                s2_full = cache_batch[f"s2_{scale}"][b].to(device)
                s3_full = cache_batch[f"s3_{scale}"][b].to(device)
                cand_y_full = cache_batch[f"cand_y_{scale}"][b].to(device)
                mask_full = cache_batch[f"mask_{scale}"][b].to(device).bool()

                if mask_full.sum() == 0:
                    raise RuntimeError(f"Cached retrieval has zero valid candidates for scale={scale}")

                s1 = s1_full[mask_full]
                s2 = s2_full[mask_full]
                s3 = s3_full[mask_full]
                cand_y = cand_y_full[mask_full]

                agg_t, z_t = self._finish_one_scale_from_scores(q_feat_t, scale, s1, s2, s3, cand_y)
                up = self._upsample_to_48(agg_t.unsqueeze(0), scale)[0]
                scale_preds.append(up)
                z_parts.append(z_t)

            scale_preds_t = torch.stack(scale_preds, dim=0)
            q_feat_rep = q_feat_t.unsqueeze(0).expand(self.pred_len, -1)
            scale_vals = torch.stack([scale_preds_t[i, :, 0] for i in range(len(self.scales))], dim=1)
            u_t = torch.cat([q_feat_rep, horizon_emb, scale_vals], dim=1)
            alpha_t = F.softmax(self.alpha_mlp(u_t), dim=-1)
            rag_t = torch.sum(alpha_t.unsqueeze(-1) * scale_preds_t.permute(1, 0, 2), dim=1)

            direct_t = direct[b]
            branch_diff = torch.mean(torch.abs(rag_t - direct_t))
            z_gate = torch.cat([q_feat_t] + z_parts + [branch_diff.unsqueeze(0)], dim=0)

            gate_in = torch.cat([z_gate.unsqueeze(0).expand(self.pred_len, -1), horizon_emb], dim=1)
            beta_t = torch.sigmoid(self.final_gate(gate_in))
            final_t, beta_t = self._select_final_branch(rag_t, direct_t, beta_t)

            all_final.append(final_t)
            all_rag.append(rag_t)
            all_direct.append(direct_t)
            all_alpha.append(alpha_t)
            all_beta.append(beta_t)

        return {
            "y_hat": torch.stack(all_final, dim=0),
            "y_rag": torch.stack(all_rag, dim=0),
            "y_dir": torch.stack(all_direct, dim=0),
            "alpha": torch.stack(all_alpha, dim=0),
            "beta": torch.stack(all_beta, dim=0),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.kb is None:
            raise RuntimeError("KB has not been attached. Call set_kb(...) first.")
        self._ensure_kb_device(x.device)

        device = x.device
        q_feat = x.mean(dim=1)
        direct = self.direct_branch(x)
        if self.branch_mode == "direct_only":
            return self._make_direct_only_output(direct)
        batch_size = x.size(0)
        horizon_idx = torch.arange(self.pred_len, device=device)
        horizon_emb = self.horizon_emb(horizon_idx)

        all_final, all_rag, all_direct, all_alpha, all_beta = [], [], [], [], []

        for b in range(batch_size):
            qb = x[b:b + 1]
            q_feat_t = q_feat[b]
            scale_preds = []
            z_parts = []

            for scale in self.scales:
                q_emb_t = self.encode_scale(qb, scale)[0]
                agg_t, z_t = self._retrieve_one_scale(qb[0], q_emb_t, q_feat_t, scale)
                up = self._upsample_to_48(agg_t.unsqueeze(0), scale)[0]
                scale_preds.append(up)
                z_parts.append(z_t)

            scale_preds_t = torch.stack(scale_preds, dim=0)
            q_feat_rep = q_feat_t.unsqueeze(0).expand(self.pred_len, -1)
            scale_vals = torch.stack([scale_preds_t[i, :, 0] for i in range(len(self.scales))], dim=1)
            u_t = torch.cat([q_feat_rep, horizon_emb, scale_vals], dim=1)
            alpha_t = F.softmax(self.alpha_mlp(u_t), dim=-1)
            rag_t = torch.sum(alpha_t.unsqueeze(-1) * scale_preds_t.permute(1, 0, 2), dim=1)

            direct_t = direct[b]
            branch_diff = torch.mean(torch.abs(rag_t - direct_t))
            z_gate = torch.cat([q_feat_t] + z_parts + [branch_diff.unsqueeze(0)], dim=0)

            gate_in = torch.cat([z_gate.unsqueeze(0).expand(self.pred_len, -1), horizon_emb], dim=1)
            beta_t = torch.sigmoid(self.final_gate(gate_in))
            final_t, beta_t = self._select_final_branch(rag_t, direct_t, beta_t)

            all_final.append(final_t)
            all_rag.append(rag_t)
            all_direct.append(direct_t)
            all_alpha.append(alpha_t)
            all_beta.append(beta_t)

        return {
            "y_hat": torch.stack(all_final, dim=0),
            "y_rag": torch.stack(all_rag, dim=0),
            "y_dir": torch.stack(all_direct, dim=0),
            "alpha": torch.stack(all_alpha, dim=0),
            "beta": torch.stack(all_beta, dim=0),
        }
