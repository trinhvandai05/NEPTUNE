"""Autoregressive full-catalog evaluation.

The state is rolled through each user's whole history with the same
event_step() used in training; every VAL and TEST event is scored against the
entire usable catalog (items the user already consumed are excluded) BEFORE it
is assimilated.  No ANN, no sampled candidates: sampled ranking metrics are
inconsistent estimators of full-catalog ones and every claim here is about the
tail.

Outputs are kept PER USER, because the H1 statistic is a paired per-user
difference Delta_u, not a global mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..heads.ranking import rank_against_catalog
from ..training.event_step import event_step

VAL, TEST = 1, 2


class MetricAccumulator:
    """Per-(user, split) sums of NDCG@k / Recall@k, split by cold / warm target."""

    def __init__(self, n_users: int, ks=(10, 50)):
        self.ks = tuple(ks)
        self.n = np.zeros((n_users, 3))
        self.n_cold = np.zeros((n_users, 3))
        self.sums = {f"{m}@{k}": np.zeros((n_users, 3)) for k in self.ks for m in ("ndcg", "recall")}
        self.cold_ndcg = np.zeros((n_users, 3))
        self.warm_ndcg = np.zeros((n_users, 3))

    def add(self, users: np.ndarray, splits: np.ndarray, ranks: np.ndarray, cold: np.ndarray):
        ranks = ranks.astype(np.float64)
        np.add.at(self.n, (users, splits), 1.0)
        np.add.at(self.n_cold, (users, splits), cold.astype(np.float64))
        for k in self.ks:
            hit = ranks <= k
            ndcg = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
            np.add.at(self.sums[f"ndcg@{k}"], (users, splits), ndcg)
            np.add.at(self.sums[f"recall@{k}"], (users, splits), hit.astype(np.float64))
            if k == self.ks[0]:
                np.add.at(self.cold_ndcg, (users, splits), np.where(cold, ndcg, 0.0))
                np.add.at(self.warm_ndcg, (users, splits), np.where(cold, 0.0, ndcg))

    def frame(self, split: int) -> pd.DataFrame:
        n = self.n[:, split]
        keep = np.flatnonzero(n > 0)
        k0 = self.ks[0]
        out = {"u": keep, "n_events": n[keep].astype(np.int64),
               "n_cold": self.n_cold[keep, split].astype(np.int64)}
        for name, arr in self.sums.items():
            out[name] = arr[keep, split] / n[keep]
        nc = self.n_cold[keep, split]
        nw = n[keep] - nc
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"ndcg@{k0}_cold"] = np.where(nc > 0, self.cold_ndcg[keep, split] / nc, np.nan)
            out[f"ndcg@{k0}_warm"] = np.where(nw > 0, self.warm_ndcg[keep, split] / nw, np.nan)
        return pd.DataFrame(out)


@torch.no_grad()
def evaluate_neptune(model, batcher, X: torch.Tensor, cold_mask: np.ndarray, n_users: int,
                     cfg, device) -> dict:
    model.eval()
    E_all = model.item_embeddings(X)
    N = E_all.shape[0]
    cold_t = torch.as_tensor(cold_mask, device=device)
    acc = MetricAccumulator(n_users, cfg.eval.ks)
    chunk = cfg.eval.chunk_size

    for g in range(len(batcher)):
        cpu = batcher.batch(g)
        items = cpu["items"].to(device)
        dt = cpu["dt"].to(device)
        mask = cpu["mask"].to(device)
        mask_np, split_np = cpu["mask"].numpy(), cpu["split"].numpy()
        users_np = cpu["users"].numpy()
        dt_colmax = cpu["dt"].max(dim=0).values.numpy()
        B, L = items.shape
        state = model.new_state(B, device)
        seen = torch.zeros(B, N, dtype=torch.bool, device=device)

        for t in range(L):
            rows_np = np.flatnonzero(mask_np[:, t] & (split_np[:, t] >= VAL))

            def head(st, rows_np=rows_np, t=t):
                if len(rows_np) == 0:
                    return None
                rows = torch.as_tensor(rows_np, device=device)
                sub = st.index(rows)
                tgt = items[rows, t]
                tscore = model.score(sub, E_all[tgt].unsqueeze(1), E_all)[:, 0]
                ranks = rank_against_catalog(lambda lo, hi: model.score(sub, E_all[lo:hi], E_all),
                                             tgt, tscore, seen[rows], N, chunk)
                return ranks.cpu().numpy(), cold_t[tgt].cpu().numpy()

            state, out, _ = event_step(model, state, E_all, X, items[:, t], dt[:, t], mask[:, t],
                                       float(dt_colmax[t]), head)
            if out is not None:
                ranks, cold = out
                acc.add(users_np[rows_np], split_np[rows_np, t].astype(np.int64), ranks, cold)
            act = np.flatnonzero(mask_np[:, t])
            if len(act):
                act_t = torch.as_tensor(act, device=device)
                seen[act_t, items[act_t, t]] = True

    return {VAL: acc.frame(VAL), TEST: acc.frame(TEST)}
