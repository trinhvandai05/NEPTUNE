"""SASRec and SASRec+Psi -- the controls that make C-III falsifiable.

Both are CONTENT-based (same ItemEncoder architecture, no item-ID table), use
the same warm-item negative sampler, the same sampled-softmax loss, the same
full-catalog autoregressive evaluator and the same sealed test protocol.
SASRec+Psi subtracts exactly NEPTUNE's satiation field: same AtomBank, same
deposit rule, same learned (a_psi, lambda_phi) heads, same sigma.  If
SASRec+Psi matches NEPTUNE, the satiation field -- not the measure-valued state
-- was doing the work.

Context semantics (v1.2 -- fixes two v1.1 defects found in review)
  v1.1 trained on NON-overlapping chunks: later targets in a chunk lost all
  history before the chunk, and the hidden state at position W-1 never received
  gradient (it only fed a detached carry) -- yet evaluation read exactly that
  position for every event with >= W items of history.  The control was being
  scored through an untrained positional embedding, i.e. artificially weakened.

  v1.2, shared by train and eval (see `train_windows` / `eval_context_len`):
  * a context of n items occupies positions 0..n-1, the prediction reads
    position n-1, and n <= W-1, so every position read at eval is trained;
  * train: windows of length W with stride W/2; each target t is scored in
    exactly one window and sees >= min(t, W/2) items of context;
  * eval: each scored event sees its last min(t, W-1) items.
  Train contexts span W/2..W-1 and eval uses W-1: eval sits at the long end of
  the training distribution.  Exact per-target identity would need one
  transformer pass per target (W x the cost); that is not attempted.
Only a causal mask is used: with right padding, valid positions never attend
to padding, and a key-padding mask would create all-masked rows (NaN).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..content.encoder import ItemEncoder
from ..evaluation.full_catalog import MetricAccumulator
from ..heads.ranking import rank_against_catalog, sampled_softmax_ce
from ..state.satiation import AtomBank, SatiationParams, contributions, deposit, prune, psi_from_contrib
from ..training.batching import BucketedBatcher
from ..training.trainer import BatchStats, fit_and_evaluate

VAL = 1


def train_windows(L: int, W: int) -> list:
    """[(start, t_lo, t_hi)]: window items[start:start+W], targets t in [t_lo, t_hi).
    Every t in 1..L-1 is a target exactly once; t - start >= min(t, W//2);
    the position read for target t is t - start - 1 <= W - 2."""
    half = W // 2
    out = [(0, 1, min(W, L))] if L > 1 else []
    k = 1
    while k * half + half < L:
        s = k * half
        out.append((s, s + half, min(s + W, L)))
        k += 1
    return out


def eval_context_len(t: int, W: int) -> int:
    return min(t, W - 1)


class SASRec(nn.Module):
    def __init__(self, cfg, d_x: int):
        super().__init__()
        m, b = cfg.model, cfg.baseline
        self.d, self.W = m.d, b.max_len
        self.use_psi = b.kind == "sasrec_psi"
        self.sigma, self.R, self.atom_eps = m.sigma, m.R, m.atom_eps
        self.encoder = ItemEncoder(d_x, m.d, m.encoder_hidden)
        self.pos = nn.Embedding(b.max_len, m.d)           # positions, never item IDs
        layer = nn.TransformerEncoderLayer(m.d, b.n_heads, 4 * m.d, b.dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, b.n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(m.d)
        self.satiation = SatiationParams(d_x, m.satiation_hidden) if self.use_psi else None

    def item_embeddings(self, X):
        return self.encoder(X)

    def hidden(self, E_all: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        L = items.shape[1]
        x = E_all[items] + self.pos(torch.arange(L, device=items.device))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=items.device), 1)
        return self.ln(self.blocks(x, mask=causal))

    def base_scores(self, h: torch.Tensor, E_query: torch.Tensor) -> torch.Tensor:
        if E_query.dim() == 2:
            return h @ E_query.T
        return torch.einsum("bd,bqd->bq", h, E_query)

    def psi(self, bank: AtomBank, now, E_all, E_query):
        if E_query.dim() == 2:
            E_query = E_query.unsqueeze(0).expand(bank.item.shape[0], -1, -1)
        return psi_from_contrib(E_query, E_all[bank.item], contributions(bank, now), self.sigma)[0]


def _train_batch(model, opt, sched, cpu, X, sampler, cfg, device) -> BatchStats:
    model.train()
    items = cpu["items"].to(device)
    dt = cpu["dt"].to(device)
    mask = cpu["mask"].to(device)
    maskf = mask.float()
    n_col = cpu["mask"].sum(0).numpy()
    B, L = items.shape
    S, W = cfg.train.n_negatives, model.W
    bank = AtomBank.empty(B, model.R, device)
    now = torch.zeros(B, device=device)

    def advance_and_deposit(t, bank, now):
        active = mask[:, t]
        now = torch.where(active, now + dt[:, t], now)
        if model.use_psi:
            bank = prune(bank, now, active, model.atom_eps)
        return bank, now

    def deposit_item(t, bank, now):
        if model.use_psi:
            a, lam = model.satiation(X[items[:, t]])
            bank, _ = deposit(bank, items[:, t], a, lam, now, mask[:, t], model.atom_eps)
        return bank

    bank, now = advance_and_deposit(0, bank, now)          # event 0 is never a target
    bank = deposit_item(0, bank, now)
    tot, n_ev, n_win = 0.0, 0, 0
    for start, t_lo, t_hi in train_windows(L, W):
        E_all = model.item_embeddings(X)
        H = model.hidden(E_all, items[:, start:min(start + W, L)])
        ce_sum = torch.zeros((), device=device)
        n_scored = int(n_col[t_lo:t_hi].sum())
        for t in range(t_lo, t_hi):
            bank, now = advance_and_deposit(t, bank, now)
            item = items[:, t]
            neg = sampler.sample(B, S)
            cand = torch.cat([item.unsqueeze(1), neg], 1)
            Ec = E_all[cand]
            s = model.base_scores(H[:, t - start - 1], Ec)
            if model.use_psi:
                s = s - model.psi(bank, now, E_all, Ec)
            ce_sum = ce_sum + (sampled_softmax_ce(s, item, neg) * maskf[:, t]).sum()
            bank = deposit_item(t, bank, now)
        bank = bank.detach()
        if n_scored == 0:
            continue
        loss = ce_sum / n_scored
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
        opt.step()
        sched.step()
        tot += float(loss.detach())
        n_ev += n_scored
        n_win += 1
    return BatchStats(loss=tot / max(n_win, 1), n_events=n_ev, n_windows=max(n_win, 1),
                      ess=float("nan"), dis=0.0, evicted_rate=0.0)


@torch.no_grad()
def _evaluate(model, art, X, cfg, device, seqs) -> dict:
    model.eval()
    E_all = model.item_embeddings(X)
    N, W = E_all.shape[0], model.W
    cold_t = torch.as_tensor(art.cold, device=device)
    acc = MetricAccumulator(art.n_users, cfg.eval.ks)
    batcher = BucketedBatcher(seqs, cfg.eval.batch_size, mode="eval",
                              ordinal=cfg.data.ordinal_time, clip_days=cfg.data.clip_days)
    per = cfg.baseline.eval_windows_per_batch
    for g in range(len(batcher)):
        cpu = batcher.batch(g)
        items_np, mask_np, split_np = cpu["items"].numpy(), cpu["mask"].numpy(), cpu["split"].numpy()
        users_np = cpu["users"].numpy()
        items = cpu["items"].to(device)
        dt = cpu["dt"].to(device)
        mask = cpu["mask"].to(device)
        B, L = items_np.shape
        # 1) hidden state for every scored event from its own left-aligned window
        er, et = np.nonzero(mask_np & (split_np >= VAL))
        hid = torch.zeros(len(er), model.d, device=device)
        for a in range(0, len(er), per):
            rr, tt = er[a:a + per], et[a:a + per]
            win = np.zeros((len(rr), W), np.int64)
            last = np.zeros(len(rr), np.int64)
            for k, (r, t) in enumerate(zip(rr, tt)):
                n = eval_context_len(int(t), W)
                assert n >= 1, "scored event without history"
                seq = items_np[r, t - n:t]
                win[k, :len(seq)] = seq
                last[k] = len(seq) - 1
            H = model.hidden(E_all, torch.as_tensor(win, device=device))
            hid[a:a + len(rr)] = H[torch.arange(len(rr), device=device), torch.as_tensor(last, device=device)]
        lookup = {(int(r), int(t)): k for k, (r, t) in enumerate(zip(er, et))}
        # 2) walk the sequence for Psi and the seen-mask, scoring in order
        bank = AtomBank.empty(B, model.R, device)
        now = torch.zeros(B, device=device)
        seen = torch.zeros(B, N, dtype=torch.bool, device=device)
        for t in range(L):
            active = mask[:, t]
            now = torch.where(active, now + dt[:, t], now)
            if model.use_psi:
                bank = prune(bank, now, active, model.atom_eps)
            rows_np = np.flatnonzero(mask_np[:, t] & (split_np[:, t] >= VAL))
            if len(rows_np):
                rows = torch.as_tensor(rows_np, device=device)
                h = hid[torch.as_tensor([lookup[(int(r), t)] for r in rows_np], device=device)]
                sub_bank, sub_now = bank.index(rows), now[rows]
                tgt = items[rows, t]

                def score(E_q, h=h, sub_bank=sub_bank, sub_now=sub_now):
                    s = model.base_scores(h, E_q)
                    if model.use_psi:
                        s = s - model.psi(sub_bank, sub_now, E_all, E_q)
                    return s

                tscore = score(E_all[tgt].unsqueeze(1))[:, 0]
                ranks = rank_against_catalog(lambda lo, hi: score(E_all[lo:hi]), tgt, tscore, seen[rows],
                                             N, cfg.eval.chunk_size)
                acc.add(users_np[rows_np], split_np[rows_np, t].astype(np.int64),
                        ranks.cpu().numpy(), cold_t[tgt].cpu().numpy())
            if model.use_psi:
                a, lam = model.satiation(X[items[:, t]])
                bank, _ = deposit(bank, items[:, t], a, lam, now, active, model.atom_eps)
            act = np.flatnonzero(mask_np[:, t])
            if len(act):
                act_t = torch.as_tensor(act, device=device)
                seen[act_t, items[act_t, t]] = True
    return {1: acc.frame(1), 2: acc.frame(2)}


def run_baseline(cfg, art, run_dir, device, *, experiment, resume=True, log=print) -> dict:
    return fit_and_evaluate(cfg, art, run_dir, device, experiment=experiment,
                            build_model=lambda c, X: SASRec(c, X.shape[1]),
                            train_batch=_train_batch, evaluate=_evaluate, resume=resume, log=log)
