"""The Rust extension builds, imports and answers on closed-form inputs; the numerics are tested in cargo test and
against WeightWatcher and PyHessian in the other signals tests."""

import numpy as np
import pytest

from relay_opal import _signals


def test_esd_is_squared_singular_values():
    m = np.array([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    e = _signals.esd(m.ravel(), 3, 2)
    assert np.allclose(e, [9.0, 16.0])
    assert np.allclose(_signals.esd_from_svals(np.array([4.0, 3.0])), [9.0, 16.0])


def test_power_law_fit_recovers_pareto_alpha():
    n, alpha = 4000, 3.0
    data = ((np.arange(n) + 0.5) / n) ** (-1.0 / (alpha - 1.0))
    fit = _signals.power_law_fit(data)
    assert abs(fit["alpha"] - alpha) < 0.05
    m = _signals.layer_metrics(data, n)
    assert m["num_evals"] == n and m["alpha"] == fit["alpha"]
    assert m["alpha_weighted"] == pytest.approx(fit["alpha"] * np.log10(data.max()))


def test_layer_lrs_keeps_the_budget():
    lrs = _signals.layer_lrs(np.array([2.0, 6.0, 3.0, 4.5]), 0.1, "linear_map", 0.5, 1.5)
    assert lrs.mean() == pytest.approx(0.1)
    assert lrs[0] < lrs[2] < lrs[3] < lrs[1]
    with pytest.raises(ValueError):
        _signals.layer_lrs(np.array([2.0]), 0.1, "sigmoid")


def test_window_closed_form():
    w = _signals.Window(4, 1)
    f32 = lambda *x: np.array(x, dtype=np.float32)
    assert w.before_step(0, f32(-2.0), f32(0.0), 1.0) is None
    w.after_step(0, f32(1.0))
    row = w.before_step(0, f32(0.0), f32(1.0), 0.0)
    assert row["grad_norm"] == 2.0 and row["alpha"] == pytest.approx(0.0) and row["snr"] is None


def test_hessian_drivers_on_a_diagonal():
    d = np.array([5, -3, 2, 1, 0.5, 0.1], dtype=np.float32)
    hvp = lambda v: v * d
    values, vectors = _signals.top_eigenvalues(hvp, 6, top_n=2, max_iter=200, tol=1e-8)
    assert values == pytest.approx([5.0, -3.0], abs=1e-2)
    assert abs(vectors[0][0]) > 0.99
    assert np.mean(_signals.trace(hvp, 6, max_probes=4, tol=0.0)) == pytest.approx(d.sum(), abs=1e-4)
    assert _signals.lanczos_eigenvalues(hvp, 6, top_n=1, iters=6) == pytest.approx([5.0], abs=1e-4)
    density, grid = _signals.density_on_grid(_signals.slq(hvp, 6, n_probes=2, iters=6), num_bins=1000)
    assert density.sum() * (grid[1] - grid[0]) == pytest.approx(1.0)
    assert _signals.edge_of_stability(4.0, 0.5) == 1.0


def test_hvp_errors_propagate():
    with pytest.raises(ZeroDivisionError):
        _signals.trace(lambda v: 1 / 0, 6)
