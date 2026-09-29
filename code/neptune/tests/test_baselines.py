import torch

from neptune.baselines.sasrec import SASRec
from neptune.config import build_run_config
from neptune.pipeline.artifacts import load_artifacts
from neptune.pipeline.runner import execute


def test_sasrec_is_causal_and_id_free(prereg, profile):
    cfg = build_run_config(prereg, profile, {"baseline.kind": "sasrec", "baseline.dropout": 0.1})
    torch.manual_seed(0)
    X = torch.randn(30, 10)
    m = SASRec(cfg, 10).eval()
    E = m.item_embeddings(X)
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 9, 9]])
    ha, hb = m.hidden(E, a), m.hidden(E, b)
    assert torch.allclose(ha[:, :3], hb[:, :3], atol=1e-6)       # future items cannot leak back
    for mod in m.modules():
        assert not (isinstance(mod, torch.nn.Embedding) and mod.num_embeddings == 30)


def test_baselines_run_through_the_same_protocol(data_dir, prereg, profile, tmp_path):
    from conftest import keep_gate
    keep_gate(tmp_path, prereg)
    for kind in ("sasrec", "sasrec_psi"):
        s = execute(prereg, profile, {"baseline.kind": kind, "baseline.dropout": 0.1, "model.sigma": 0.25},
                    data_dir=data_dir, out_root=tmp_path, device="cpu", experiment="B01_SASREC",
                    log=lambda *x: None)
        assert s["test_sealed"] and 0 <= s["val"]["value"] <= 1
    runs = sorted(p.parent.name for p in (tmp_path / "runs" / profile.name).rglob("summary.json"))
    assert len(runs) == 2
    assert all(p.parent.parent.name == prereg.sha12
               for p in (tmp_path / "runs" / profile.name).rglob("summary.json"))
    assert any(r.startswith("sasrec_psi") for r in runs) and any(r.startswith("sasrec_do") for r in runs)


def test_sasrec_windows_cover_targets_once_with_trained_positions():
    from neptune.baselines.sasrec import eval_context_len, train_windows
    for L in (1, 2, 5, 16, 17, 40, 203):
        for W in (4, 16):
            seen = []
            for start, lo, hi in train_windows(L, W):
                assert hi - start <= W
                for t in range(lo, hi):
                    seen.append(t)
                    assert t - start >= min(t, W // 2)           # enough context
                    assert 0 <= t - start - 1 <= W - 2           # position read is trained
            assert sorted(seen) == list(range(1, L))             # every target exactly once
            for t in range(1, L):
                n = eval_context_len(t, W)
                assert 1 <= n <= W - 1 and n - 1 <= W - 2        # eval reads trained positions only
