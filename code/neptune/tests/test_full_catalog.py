import numpy as np
import torch

from conftest import tiny_model
from neptune.evaluation.full_catalog import MetricAccumulator
from neptune.heads.ranking import rank_against_catalog


def test_chunked_rank_equals_unchunked_and_excludes_seen(cfg):
    model, X = tiny_model(cfg, n_items=53)
    E = model.item_embeddings(X).detach()
    st = model.new_state(3, "cpu")
    seen = torch.zeros(3, 53, dtype=torch.bool)
    seen[0, :10] = True
    tgt = torch.tensor([20, 5, 40])
    ts = model.score(st, E[tgt].unsqueeze(1), E)[:, 0]
    full = model.score(st, E, E)
    not_target = torch.ones(3, 53, dtype=torch.bool)
    not_target[torch.arange(3), tgt] = False
    ref = ((full > ts[:, None]) & ~seen & not_target).sum(1) + 1
    for chunk in (1, 7, 53, 100):
        r = rank_against_catalog(lambda lo, hi: model.score(st, E[lo:hi], E), tgt, ts, seen, 53, chunk)
        assert torch.equal(r, ref), (chunk, r, ref)
    unseen = rank_against_catalog(lambda lo, hi: model.score(st, E[lo:hi], E), tgt, ts,
                                  torch.zeros_like(seen), 53, 16)
    assert unseen[0] >= ref[0]


def test_metric_accumulator_values():
    acc = MetricAccumulator(3, (10, 50))
    acc.add(np.array([0, 0, 1]), np.array([2, 2, 2]), np.array([1, 11, 3]), np.array([False, True, False]))
    f = acc.frame(2).set_index("u")
    assert np.isclose(f.loc[0, "ndcg@10"], (1.0 + 0.0) / 2)
    assert np.isclose(f.loc[1, "ndcg@10"], 1 / np.log2(4))
    assert np.isclose(f.loc[0, "recall@50"], 1.0) and f.loc[0, "n_cold"] == 1
    assert np.isclose(f.loc[0, "ndcg@10_cold"], 0.0) and np.isclose(f.loc[0, "ndcg@10_warm"], 1.0)
    assert 2 not in f.index and acc.frame(1).empty
