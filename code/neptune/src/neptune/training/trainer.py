"""Training loop.

`fit_and_evaluate` is generic: NEPTUNE and the SASRec controls plug in their
own (build_model, train_batch, evaluate) functions but share scheduling,
checkpointing, resume, logging and -- critically -- the sealing of test
results.  Validation per-user results go to  peruser_val.parquet ; test
per-user results go to  sealed/peruser_test.parquet , which the selection code
never opens.  summary.json is written LAST: its existence marks a finished run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..common import set_seed, write_json_atomic
from ..heads.negatives import NegativeSampler
from ..heads.ranking import sampled_softmax_ce
from ..logging.sink import LongSink
from ..model import NeptuneH1
from . import checkpoint as ckpt
from .batching import BucketedBatcher, build_sequences
from .event_step import event_step


def warmup_cosine(total_steps: int, warmup: int):
    total = max(total_steps, 1)

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = min(1.0, (step - warmup) / max(total - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return f


@dataclass
class BatchStats:
    loss: float
    n_events: int
    n_windows: int
    ess: float
    dis: float
    evicted_rate: float
    ess_pen: float = 0.0


# ------------------------------------------------------------ NEPTUNE pieces

def build_neptune(cfg, X: torch.Tensor):
    return NeptuneH1(cfg, X, seed=cfg.seed)


def neptune_train_batch(model, optimizer, scheduler, cpu: dict, X: torch.Tensor,
                        sampler: NegativeSampler, cfg, device) -> BatchStats:
    model.train()
    items = cpu["items"].to(device, non_blocking=True)
    dt = cpu["dt"].to(device, non_blocking=True)
    mask = cpu["mask"].to(device, non_blocking=True)
    maskf = mask.float()
    n_col = cpu["mask"].sum(0).numpy()
    dt_colmax = cpu["dt"].max(0).values.numpy()
    B, L = items.shape
    S, T = cfg.train.n_negatives, cfg.train.bptt
    want_dis = cfg.reg.lambda_dis > 0

    state = model.new_state(B, device)
    tot_loss, tot_ev, n_win, dis_acc, ev_acc, ess_acc = 0.0, 0, 0, 0.0, 0.0, 0.0
    for lo in range(0, L, T):
        hi = min(lo + T, L)
        n_events = int(n_col[lo:hi].sum())
        if n_events == 0:
            break
        E_all = model.item_embeddings(X)          # re-embed: the encoder moved last step
        ce_sum = torch.zeros((), device=device)
        dis_sum = torch.zeros((), device=device)
        ess_sum = torch.zeros((), device=device)
        evicted = torch.zeros((), device=device)
        for t in range(lo, hi):
            item, active = items[:, t], mask[:, t]

            def head(st, item=item, t=t):
                neg = sampler.sample(B, S)
                cand = torch.cat([item.unsqueeze(1), neg], dim=1)
                ce = sampled_softmax_ce(model.score(st, E_all[cand], E_all), item, neg)
                return (ce * maskf[:, t]).sum()

            state, ce_t, aux = event_step(model, state, E_all, X, item, dt[:, t], active,
                                          float(dt_colmax[t]), head, want_dis=want_dis)
            ce_sum = ce_sum + ce_t
            # ESS guard on EVERY active event: a user whose sequence ends inside the
            # window must still be penalized for collapse (v1.1 only checked hi-1)
            ess_sum = ess_sum + (model.ess_penalty(state) * maskf[:, t]).sum()
            if want_dis:
                dis_sum = dis_sum + aux["dis_sum"]
            evicted = evicted + aux["evicted"].float().sum()

        loss = ce_sum / n_events
        if cfg.reg.lambda_ess > 0:
            loss = loss + cfg.reg.lambda_ess * ess_sum / n_events
        if want_dis:
            loss = loss + cfg.reg.lambda_dis * dis_sum / n_events
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
        optimizer.step()
        scheduler.step()
        state = state.detach()

        tot_loss += float((ce_sum / n_events).detach())
        dis_acc += float(dis_sum.detach()) / n_events
        ev_acc += float(evicted.detach()) / n_events
        ess_acc += float(ess_sum.detach()) / n_events
        tot_ev += n_events
        n_win += 1
    return BatchStats(loss=tot_loss / max(n_win, 1), n_events=tot_ev, n_windows=n_win,
                      ess=float(state.ess.mean()), dis=dis_acc / max(n_win, 1),
                      evicted_rate=ev_acc / max(n_win, 1), ess_pen=ess_acc / max(n_win, 1))


@torch.no_grad()
def item_manifold_stats(model, X: torch.Tensor, n_pairs: int = 200_000) -> dict:
    """D17 item-manifold diagnostics.

    GATED (acceptance.d17_max_mean_cos): mean pairwise <e_i, e_j> over i != j,
    exact in O(N d) for unit vectors: (||sum e||^2 - N) / (N (N - 1)).  It detects
    collapse TOWARD ONE DIRECTION (a cone) -- the OPEN-1 mechanism.  It is blind
    to e.g. an antipodal +v / -v collapse (mean cos ~ 0), so it is not a general
    "no collapse" certificate.
    DESCRIPTIVE (not gated; freezing a criterion on them requires an amendment
    BEFORE the pilot): effective rank (participation ratio of the covariance),
    mean |cos| and cos quantiles over a fixed random sample of pairs.
    """
    was = model.training
    model.eval()
    E = model.item_embeddings(X).double()
    model.train(was)
    N = E.shape[0]
    s = E.sum(0)
    mean_cos = float((s @ s - N) / (N * (N - 1)))
    ev = torch.linalg.eigvalsh(torch.cov(E.T)).clamp_min(0)
    eff_rank = float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30))
    g = torch.Generator(device="cpu").manual_seed(0)
    i = torch.randint(0, N, (n_pairs,), generator=g)
    j = torch.randint(0, N - 1, (n_pairs,), generator=g)
    j = j + (j >= i).long()                                     # i != j
    cos = (E[i.to(E.device)] * E[j.to(E.device)]).sum(-1)
    q = torch.quantile(cos.float().cpu(), torch.tensor([0.05, 0.5, 0.95]))
    return {"mean_pairwise_cos": mean_cos, "effective_rank": eff_rank,
            "mean_abs_cos": float(cos.abs().mean()), "cos_q05": float(q[0]),
            "cos_q50": float(q[1]), "cos_q95": float(q[2])}


def neptune_evaluate(model, art, X, cfg, device, seqs):
    from ..evaluation.full_catalog import evaluate_neptune
    eval_b = BucketedBatcher(seqs, cfg.eval.batch_size, mode="eval",
                             ordinal=cfg.data.ordinal_time, clip_days=cfg.data.clip_days)
    return evaluate_neptune(model, eval_b, X, art.cold, art.n_users, cfg, device)


# ---------------------------------------------------------------- generic

def fit_and_evaluate(cfg, art, run_dir: Path, device, *, experiment: str,
                     build_model=build_neptune, train_batch=neptune_train_batch,
                     evaluate=neptune_evaluate, resume: bool = True, log=print) -> dict:
    run_dir = Path(run_dir)
    set_seed(cfg.seed)
    X = torch.from_numpy(art.X).to(device)
    model = build_model(cfg, X).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    seqs = build_sequences(art.events, art.n_users)
    train_b = BucketedBatcher(seqs, cfg.train.batch_size, mode="train", ordinal=cfg.data.ordinal_time,
                              clip_days=cfg.data.clip_days, max_len=cfg.train.max_len, seed=cfg.seed)
    steps_per_epoch = train_b.steps_per_epoch(cfg.train.bptt)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(cfg.train.epochs * steps_per_epoch, cfg.train.warmup_steps))
    sampler = NegativeSampler(art.warm, art.popularity, cfg.train.negatives, device, cfg.seed)

    ck_path = run_dir / "checkpoint_last.pt"
    epoch0, pos0, step = 0, 0, 0
    if resume and ck_path.exists():
        st = ckpt.load(ck_path, model=model, optimizer=opt, scheduler=sched, sampler=sampler,
                       map_location=device)
        epoch0, pos0, step = st["epoch"], st["batch_pos"], st["step"]
        log(f"[run] resumed at epoch {epoch0}, batch {pos0}, step {step}")

    arm = run_dir.name
    sink = LongSink(run_dir / "metrics.jsonl", {"experiment": experiment, "arm": arm,
                                                "dataset": "ml25m", "seed": cfg.seed})
    throughput, d17_history = [], []
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def save(epoch, pos):
        ckpt.save(ck_path, model=model, optimizer=opt, scheduler=sched, sampler=sampler,
                  epoch=epoch, batch_pos=pos, step=step)

    for epoch in range(epoch0, cfg.train.epochs):
        order = train_b.order(epoch)
        start = pos0 if epoch == epoch0 else 0
        t0, n_ev = time.time(), 0
        for pos in range(start, len(order)):
            bs = train_batch(model, opt, sched, train_b.batch(int(order[pos])), X, sampler, cfg, device)
            step += bs.n_windows
            n_ev += bs.n_events
            sink.log("loss", bs.loss, epoch=epoch, split="train")
            sink.log("ess_mean", bs.ess, epoch=epoch, split="train")
            sink.log("dissipation", bs.dis, epoch=epoch, split="train")
            sink.log("atom_evicted_rate", bs.evicted_rate, epoch=epoch, split="train")
            if not math.isfinite(bs.loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {pos}")
            if (pos + 1) % cfg.train.ckpt_every_batches == 0:
                save(epoch, pos + 1)
        dur = max(time.time() - t0, 1e-9)
        throughput.append(n_ev / dur)
        sink.log("events_per_sec", n_ev / dur, epoch=epoch, split="train")
        d17 = item_manifold_stats(model, X)
        d17_history.append(d17)
        for k, v in d17.items():
            sink.log(f"D17_{k}", v, epoch=epoch, split="train")
        log(f"[run] {arm} epoch {epoch}: {n_ev/dur:,.0f} events/s, {dur:.0f}s")
        save(epoch + 1, 0)

    t0 = time.time()
    frames = evaluate(model, art, X, cfg, device, seqs)
    eval_s = time.time() - t0
    frames[1].to_parquet(run_dir / "peruser_val.parquet", index=False)
    (run_dir / "sealed").mkdir(exist_ok=True)
    frames[2].to_parquet(run_dir / "sealed" / "peruser_test.parquet", index=False)
    sink.close()

    metric = cfg.eval.primary_metric
    val = frames[1]
    summary = {
        "run": arm, "experiment": experiment, "claim_eligible": cfg.claim_eligible,
        "profile": cfg.profile, "prereg_sha256": cfg.prereg_sha256,
        "val": {"metric": metric, "value": float(val[metric].mean()) if len(val) else float("nan"),
                "n_users": int(len(val))},
        "test_sealed": True, "n_test_users": int(len(frames[2])),
        "train_events_per_sec_median": float(np.median(throughput)) if throughput else None,
        "eval_seconds": eval_s, "steps": step,
        "D17_final": item_manifold_stats(model, X),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if str(device).startswith("cuda") else None,
    }
    write_json_atomic(run_dir / "summary.json", summary)
    return summary
