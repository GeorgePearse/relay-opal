"""The estimator never touches autograd: every backward entry point is tripwired while it runs."""

import pytest
import torch

from relay_opal import RelayEstimator, RelaySettings


class AutogradUsed(AssertionError):
    pass


@pytest.fixture
def tripwire(monkeypatch):
    def boom(*a, **k):
        raise AutogradUsed("autograd entry point called inside the estimator")

    monkeypatch.setattr(torch.autograd, "grad", boom)
    monkeypatch.setattr(torch.autograd, "backward", boom)
    monkeypatch.setattr(torch.Tensor, "backward", boom)
    monkeypatch.setattr(torch.autograd.functional, "jacobian", boom)
    monkeypatch.setattr(torch.autograd.functional, "jvp", boom)
    monkeypatch.setattr(torch.func, "grad", boom)
    monkeypatch.setattr(torch.func, "jvp", boom)
    monkeypatch.setattr(torch.func, "vjp", boom)


def test_estimate_runs_with_autograd_tripwired(tiny, tripwire):
    spec, x, y = tiny
    est = RelayEstimator(spec, RelaySettings(routes=64, tokens=32), seed=0)
    with torch.autograd.grad_mode.set_grad_enabled(False):
        grads = est.gradients(x, y)
    assert all(not g.requires_grad and g.grad_fn is None for g in grads.values())
    assert all(not p.requires_grad for p in spec.model.parameters())


def test_estimate_does_not_build_a_graph_even_with_grad_enabled(tiny):
    spec, x, y = tiny
    est = RelayEstimator(spec, RelaySettings(routes=64, tokens=32), seed=0)
    with torch.enable_grad():
        grads = est.gradients(x, y)
    assert all(g.grad_fn is None for g in grads.values())


def test_tripwire_itself_fires(tripwire):
    with pytest.raises(AutogradUsed):
        torch.ones(1, requires_grad=True).sum().backward()
