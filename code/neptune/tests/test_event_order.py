import pytest
import torch

from conftest import tiny_model
from neptune.training.event_step import OrderingError, event_step


def _roll(model, X, E, items, B=2):
    st = model.new_state(B, "cpu")
    act = torch.ones(B, dtype=torch.bool)
    for it in items:
        st, _, _ = event_step(model, st, E, X, torch.tensor([it] * B), torch.ones(B), act, 1.0)
    return st, act


def test_no_label_leakage(cfg):
    """Same history, different current target -> the state the heads see is identical."""
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X).detach()
    st, act = _roll(model, X, E, [3, 7, 11])
    seen = []
    for target in (5, 30):
        event_step(model, st, E, X, torch.tensor([target, target]), torch.ones(2), act, 1.0,
                   head_fn=lambda s: seen.append((s.z.clone(), s.logp.clone(), s.atoms.item.clone(),
                                                  s.atoms.mask.clone(), s.last_item.clone())))
    for a, b in zip(*seen):
        assert torch.equal(a, b)
    assert seen[0][4].tolist() == [11, 11]                        # previous item, not the target


def test_current_item_absent_from_pre_observation_atoms(cfg):
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X).detach()
    st, act = _roll(model, X, E, [1, 2])
    captured = {}
    nxt, _, _ = event_step(model, st, E, X, torch.tensor([33, 33]), torch.ones(2), act, 1.0,
                           head_fn=lambda s: captured.setdefault("items", s.atoms.item[s.atoms.mask].tolist()))
    assert 33 not in captured["items"]
    assert 33 in nxt.atoms.item[nxt.atoms.mask].tolist()


def test_evolve_that_deposits_early_is_caught(cfg, monkeypatch):
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X).detach()
    st, act = _roll(model, X, E, [1])
    real = model.evolve

    def leaky(state, E_all, dt, active, J, want_dis=False):
        new, aux = real(state, E_all, dt, active, J, want_dis)
        new.ema = new.ema + 0.0                                   # a new tensor object
        return new, aux
    monkeypatch.setattr(model, "evolve", leaky)
    with pytest.raises(OrderingError):
        event_step(model, st, E, X, torch.tensor([2, 2]), torch.ones(2), act, 1.0)


def test_head_in_place_mutation_is_caught(cfg):
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X).detach()
    st, act = _roll(model, X, E, [1])

    def bad_head(s):
        s.atoms.mask[0, 0] = True
    with pytest.raises(OrderingError):
        event_step(model, st, E, X, torch.tensor([2, 2]), torch.ones(2), act, 1.0, head_fn=bad_head)
