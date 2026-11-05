"""Hessian against PyHessian 0.1 on a small MLP with a fixed batch (top eigenvalue within 1% at PyHessian's
tolerance; the trace within the Hutchinson standard error at matched probe counts, since the probes differ) and
against the dense Hessian of a tiny model (eigenvalues exact, trace within its standard error)."""

import numpy as np
import pytest
import torch
from torch import nn

from relay_opal import _signals
from relay_opal.signals import Hessian

pyhessian = pytest.importorskip("pyhessian")


def hutchinson(hess, probes, seed=0):
    """(mean, standard error) of v.Hv over Rademacher probes, every probe used."""
    samples = np.array(_signals.trace(hess.hvp, hess.dim, probes, 0.0, seed))
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples)))


def mlp(seed, width=32, depth=2, inputs=8, classes=4):
    torch.manual_seed(seed)
    layers, d = [], inputs
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, classes))
    return nn.Sequential(*layers)


def batch(seed, n=64, inputs=8, classes=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, inputs, generator=g), torch.randint(0, classes, (n,), generator=g)


def dense_hessian(model, x, y):
    params = [p for p in model.parameters()]
    names = [n for n, _ in model.named_parameters()]
    flat = torch.cat([p.detach().reshape(-1) for p in params])

    def loss_of(theta):
        i, values = 0, {}
        for n, p in zip(names, params):
            values[n] = theta[i : i + p.numel()].view(p.shape)
            i += p.numel()
        out = torch.func.functional_call(model, values, (x,))
        return nn.functional.cross_entropy(out, y)

    return torch.autograd.functional.hessian(loss_of, flat)


@pytest.fixture(scope="module")
def small():
    model, (x, y) = mlp(0), batch(1)
    loss_fn = lambda: nn.functional.cross_entropy(model(x), y)
    return model, x, y, loss_fn


def test_matches_dense_hessian_exactly():
    model, (x, y) = mlp(3, width=6, depth=1, inputs=3, classes=2), batch(4, n=16, inputs=3, classes=2)
    h = dense_hessian(model, x, y).double()
    evals = torch.linalg.eigvalsh(h)
    hess = Hessian(lambda: nn.functional.cross_entropy(model(x), y), list(model.parameters()))
    assert hess.dim == h.shape[0]
    top = hess.top_eigenvalues(k=3, max_iter=500, tol=1e-9)
    assert top[0] == pytest.approx(float(evals[-1]), rel=1e-4)
    assert top[1] == pytest.approx(float(evals[-2]), rel=1e-3)
    lanczos = hess.top_eigenvalues(k=3, max_iter=hess.dim, method="lanczos")
    assert lanczos[0] == pytest.approx(float(evals[-1]), rel=1e-4)
    mean, err = hutchinson(hess, 1000)
    assert abs(mean - float(h.trace())) < 4 * err, (mean, err, float(h.trace()))
    density = hess.density(n_probes=4, iters=hess.dim, num_bins=2000)
    step = density["grid"][1] - density["grid"][0]
    assert float((density["density"] * step).sum()) == pytest.approx(1.0, rel=1e-6)
    assert hess.edge_of_stability(0.1) == pytest.approx(float(evals[-1]) * 0.05, rel=1e-4)


def test_top_eigenvalue_and_trace_match_pyhessian(small):
    model, x, y, loss_fn = small
    torch.manual_seed(0)
    ref = pyhessian.hessian(model, nn.functional.cross_entropy, data=(x, y), cuda=False)
    ref_top, _ = ref.eigenvalues(maxIter=100, tol=1e-3, top_n=1)
    ref_samples = np.array(ref.trace(maxIter=100, tol=0.0))
    ref_trace, ref_err = float(ref_samples.mean()), float(ref_samples.std(ddof=1) / np.sqrt(len(ref_samples)))
    hess = Hessian(loss_fn, list(model.parameters()))
    top = hess.top_eigenvalues(k=1, max_iter=100, tol=1e-3, seed=0)[0]
    assert top == pytest.approx(ref_top[0], rel=0.01)
    products = hess.products
    assert products <= 100
    exact = float(dense_hessian(model, x, y).trace())
    ours, err = hutchinson(hess, 100)
    assert len(ref_samples) == 100 and hess.products == products + 100
    assert abs(ours - exact) < 4 * err, (ours, err, exact)
    assert abs(ours - ref_trace) < 4 * (err + ref_err), (ours, err, ref_trace, ref_err)
    print(f"top {top:.6f} vs pyhessian {ref_top[0]:.6f}")
    print(f"trace {ours:.4f}+-{err:.4f} vs pyhessian {ref_trace:.4f}+-{ref_err:.4f}, exact {exact:.4f}")


def test_hvp_runs_through_the_callback(small):
    model, x, y, loss_fn = small
    hess = Hessian(loss_fn, list(model.parameters()))
    v = np.random.default_rng(0).standard_normal(hess.dim).astype(np.float32)
    hv = hess.hvp(v)
    assert hv.shape == (hess.dim,) and hv.dtype == np.float32
    assert hess.products == 1
