import torch

from conftest import tiny_model
from neptune.dynamics.mass import replicator_step
from neptune.dynamics.rk4 import n_substeps
from neptune.state.kernel import kernel, kernel_ambient_grad, log_kernel, renormalize
from neptune.state.satiation import AtomBank, contributions, deposit, psi_from_contrib


def test_encoder_unit_norm_and_no_item_embedding(cfg):
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X)
    assert torch.allclose(E.norm(dim=-1), torch.ones(len(E)), atol=1e-5)
    n_items = X.shape[0]
    for m in model.modules():
        assert not (isinstance(m, torch.nn.Embedding) and m.num_embeddings == n_items)
    E.sum().backward()
    assert model.encoder.net[0].weight.grad.abs().sum() > 0


def test_kernel_bounds_and_gradient():
    torch.manual_seed(0)
    x = renormalize(torch.randn(3, 5, 8, dtype=torch.float64)).requires_grad_(True)
    y = renormalize(torch.randn(3, 7, 8, dtype=torch.float64))
    w = torch.rand(3, 7, dtype=torch.float64)
    assert (log_kernel(x, y, 0.3) <= 0).all()
    k = kernel(x, y, 0.3)
    assert (k <= 1).all() and (k > 0).all()
    (g_auto,) = torch.autograd.grad((kernel(x, y, 0.3) * w.unsqueeze(1)).sum(), x)
    assert torch.allclose(g_auto, kernel_ambient_grad(x, y, w, 0.3), atol=1e-8)
    z = renormalize(torch.randn(4, 8)) * (1 + 1e-7)
    assert (log_kernel(z, z, 0.2) <= 0).all()                # cosine clamp


def test_replicator_simplex_and_gauge():
    torch.manual_seed(0)
    logp = torch.log_softmax(torch.randn(4, 6), -1)
    f, dt = torch.randn(4, 6), torch.ones(4)
    out = replicator_step(logp, f, 1.0, 0.1, dt)
    assert torch.allclose(out.exp().sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(out, replicator_step(logp + 3.0, f + 5.0, 1.0, 0.1, dt), atol=1e-6)
    assert torch.equal(replicator_step(torch.zeros(4, 1), f[:, :1], 1.0, 0.1, dt), torch.zeros(4, 1))


def test_satiation_decay_prune_evict():
    B, R, d = 2, 3, 4
    bank = AtomBank.empty(B, R, "cpu")
    act = torch.tensor([True, False])
    for k in range(3):
        bank, ev = deposit(bank, torch.tensor([k, k]), torch.ones(B), torch.full((B,), 0.5),
                           torch.full((B,), float(k)), act, 1e-3)
        assert not ev.any()
    assert bank.mask[0].all() and not bank.mask[1].any()        # inactive row untouched
    c = contributions(bank, torch.tensor([2.0, 2.0]))
    assert torch.allclose(c[0], torch.exp(-0.5 * torch.tensor([2.0, 1.0, 0.0])), atol=1e-6)
    bank2, ev = deposit(bank, torch.tensor([9, 9]), torch.ones(B), torch.ones(B), torch.full((B,), 3.0), act, 1e-3)
    assert ev[0] and bank2.item[0].tolist().count(9) == 1 and 0 not in bank2.item[0].tolist()
    bank3, ev = deposit(bank, torch.tensor([9, 9]), torch.ones(B), torch.ones(B), torch.full((B,), 100.0), act, 1e-3)
    assert not ev[0] and bank3.mask[0].sum() == 1                # all old atoms pruned by decay


def test_psi_gradient_matches_autograd():
    torch.manual_seed(0)
    pts = renormalize(torch.randn(2, 5, 6, dtype=torch.float64)).requires_grad_(True)
    ea = renormalize(torch.randn(2, 3, 6, dtype=torch.float64))
    c = torch.rand(2, 3, dtype=torch.float64)
    psi, g = psi_from_contrib(pts, ea, c, 0.4, want_grad=True)
    (ga,) = torch.autograd.grad(psi.sum(), pts)
    assert torch.allclose(ga, g, atol=1e-8)


def test_rk4_keeps_particles_on_sphere(cfg):
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X).detach()
    st = model.new_state(3, "cpu")
    act = torch.ones(3, dtype=torch.bool)
    for k in range(5):
        st, _ = model.evolve(st, E, torch.ones(3), act, model.n_substeps(1.0))
        st, _ = model.assimilate(st, torch.tensor([k, k + 1, k + 2]), E, X, act)
    assert torch.allclose(st.z.norm(dim=-1), torch.ones(3, cfg.model.M), atol=1e-5)
    assert torch.isfinite(st.logp).all()
    assert model.n_substeps(1.0) == 1                            # ordinal H1: J = 1
    assert n_substeps(10.0, 2.5) == 4 and n_substeps(2.5, 2.5) == 1


def test_inactive_rows_are_frozen(cfg):
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X).detach()
    st = model.new_state(2, "cpu")
    act = torch.tensor([True, False])
    st2, _ = model.evolve(st, E, torch.ones(2), act, 1)
    assert torch.equal(st2.z[1], st.z[1]) and st2.time[1] == 0 and st2.time[0] == 1


def test_ess_regularizer_zero_at_M1(prereg, profile):
    from neptune.config import build_run_config
    c1 = build_run_config(prereg, profile, {"model.M": 1, "model.h": 0.3, "model.sigma": 0.25})
    model, _ = tiny_model(c1)
    assert model.reg_ess(model.new_state(3, "cpu")).item() == 0.0
