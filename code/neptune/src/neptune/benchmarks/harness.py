"""Throughput benchmarks with a vectorization oracle.

Stages (events/second, B users advanced by one event = B events):
  kernel   log-kernel + ambient gradient on [B, M, d] x [B, M, d]   (primitive)
  state    one state step: Psi + grad at particles, candidate density,
           interaction gradient, replicator                         (forward)
  rk4      model.evolve() for one event, J = 1                      (forward)
  train    event_step() with sampled-softmax loss over a window of T
           events + backward + optimizer step                        (fwd+bwd)

Oracle: the SAME computation on B*T independent copies of a fixed dummy state
in one launch -- what perfect vectorization over the time axis would give.
E_vec = valid / oracle measures how much is lost to the sequential time loop
(the loop is required for causality; the oracle is not a valid training
algorithm).  The ratchet thresholds are enforced only on the preregistered
reference GPU; elsewhere they are reported as informational.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..dynamics.mass import replicator_step
from ..heads.negatives import NegativeSampler
from ..heads.ranking import sampled_softmax_ce
from ..model import NeptuneH1
from ..state.kernel import kernel_ambient_grad, log_kernel, project
from ..state.satiation import contributions, psi_from_contrib
from ..training.event_step import event_step


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_fn(fn, *, device, warmup: int, iters: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    per_iter = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(device)
        per_iter.append((time.perf_counter() - t0) / iters)
    a = np.asarray(per_iter)
    return {"median_s": float(np.median(a)), "p10_s": float(np.quantile(a, 0.1)),
            "p90_s": float(np.quantile(a, 0.9))}


def _setup(cfg, B: int, n_items: int, d_x: int, device, seed: int = 0):
    torch.manual_seed(seed)
    X = torch.randn(n_items, d_x, device=device)
    model = NeptuneH1(cfg, X, seed=seed).to(device)
    with torch.no_grad():
        E = model.item_embeddings(X)
    state = model.new_state(B, device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    with torch.no_grad():                    # fill the atom bank so Psi costs what it will cost
        for _ in range(cfg.model.R):
            item = torch.randint(0, n_items, (B,), device=device)
            state, _ = model.assimilate(state, item, E, X, active)
            state.time = state.time + 1.0
    return model, X, E, state, active


def _replicate(state, times: int):
    idx = torch.arange(state.z.shape[0], device=state.z.device).repeat(times)
    return state.index(idx)


def run_benchmarks(cfg, bench: dict, device, stages=("kernel", "state", "rk4", "train")) -> dict:
    B, T = int(bench["batch_users"]), int(bench["window"])
    kw = {"device": device, "warmup": int(bench["warmup_iters"]), "iters": int(bench["iters"]),
          "repeats": int(bench["repeats"])}
    model, X, E, state, active = _setup(cfg, B, int(bench["n_items"]), 64, device)
    S, h = cfg.train.n_negatives, cfg.model.h
    dt = torch.ones(B, device=device)
    out = {"B": B, "T": T, "M": cfg.model.M, "d": cfg.model.d, "R": cfg.model.R, "S": S}

    if "kernel" in stages:
        z, p = state.z.contiguous(), state.p

        def f_kernel():
            log_kernel(z, z, h)
            kernel_ambient_grad(z, z, p, h)
        r = time_fn(f_kernel, **kw)
        out["kernel"] = {**r, "events_per_sec": B / r["median_s"]}

    if "state" in stages:
        cand = torch.randint(0, E.shape[0], (B, S + 1), device=device)

        @torch.no_grad()
        def f_state():
            e_atom = E[state.atoms.item]
            c = contributions(state.atoms, state.time)
            _, g = psi_from_contrib(state.z, e_atom, c, model.sigma, want_grad=True)
            project(state.z, g + kernel_ambient_grad(state.z, state.z, state.p, h))
            model.score(state, E[cand], E)
            replicator_step(state.logp, torch.zeros_like(state.logp), 1.0, 0.1, dt)
        r = time_fn(f_state, **kw)
        out["state"] = {**r, "events_per_sec": B / r["median_s"]}

    if "rk4" in stages:
        @torch.no_grad()
        def f_rk4(st=state, a=active, d=dt):
            model.evolve(st, E, d, a, 1)
        r = time_fn(f_rk4, **kw)
        out["rk4"] = {**r, "events_per_sec": B / r["median_s"]}
        big = _replicate(state, T)
        a_big = torch.ones(B * T, dtype=torch.bool, device=device)
        d_big = torch.ones(B * T, device=device)

        @torch.no_grad()
        def f_rk4_oracle():
            model.evolve(big, E, d_big, a_big, 1)
        r = time_fn(f_rk4_oracle, **kw)
        out["rk4_oracle"] = {**r, "events_per_sec": B * T / r["median_s"]}
        out["rk4_E_vec"] = out["rk4"]["events_per_sec"] / out["rk4_oracle"]["events_per_sec"]

    if "train" in stages:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sampler = NegativeSampler(np.ones(E.shape[0], bool), np.ones(E.shape[0]), "uniform", device, 0)
        items = torch.randint(0, E.shape[0], (B, T), device=device)
        base = state.detach()

        def f_train():
            E_all = model.item_embeddings(X)
            st, loss = base, torch.zeros((), device=device)
            for t in range(T):
                it = items[:, t]

                def head(s, it=it):
                    neg = sampler.sample(B, S)
                    cand = torch.cat([it.unsqueeze(1), neg], 1)
                    return sampled_softmax_ce(model.score(s, E_all[cand], E_all), it, neg).sum()
                st, l, _ = event_step(model, st, E_all, X, it, dt, active, 1.0, head, want_dis=True)
                loss = loss + l
            opt.zero_grad(set_to_none=True)
            (loss / (B * T)).backward()
            opt.step()
        r = time_fn(f_train, **kw)
        out["train"] = {**r, "events_per_sec": B * T / r["median_s"]}

        big = _replicate(base, T)
        a_big = torch.ones(B * T, dtype=torch.bool, device=device)
        d_big = torch.ones(B * T, device=device)
        it_big = items.T.reshape(-1)

        def f_train_oracle():
            E_all = model.item_embeddings(X)

            def head(s):
                neg = sampler.sample(B * T, S)
                cand = torch.cat([it_big.unsqueeze(1), neg], 1)
                return sampled_softmax_ce(model.score(s, E_all[cand], E_all), it_big, neg).sum()
            _, l, _ = event_step(model, big, E_all, X, it_big, d_big, a_big, 1.0, head, want_dis=True)
            opt.zero_grad(set_to_none=True)
            (l / (B * T)).backward()
            opt.step()
        r = time_fn(f_train_oracle, **kw)
        out["train_oracle"] = {**r, "events_per_sec": B * T / r["median_s"]}
        out["train_E_vec"] = out["train"]["events_per_sec"] / out["train_oracle"]["events_per_sec"]
        if str(device).startswith("cuda"):
            out["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


def ratchet(results: dict, thresholds: dict, gpu_name, reference: str) -> dict:
    on_ref = bool(gpu_name) and reference.lower() in str(gpu_name).lower()
    checks = {}
    for stage, need in thresholds.items():
        if stage in results:
            got = results[stage]["events_per_sec"]
            checks[stage] = {"events_per_sec": got, "required": need, "pass": got >= need}
    return {"reference_hardware": reference, "device_gpu": gpu_name, "enforced": on_ref,
            "checks": checks, "all_pass": all(c["pass"] for c in checks.values())}
