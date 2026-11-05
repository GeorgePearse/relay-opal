"""Instruments against closed forms on a 1-D quadratic loss = c/2 (x - x*)^2 under SGD: the gradient is c (x - x*),
the step is -lr c (x - x*), so Cockpit's local step size is alpha = lr c - 1 (0 lands on the minimum, +1 overshoots
to the starting loss), the cosine between consecutive gradients is sign(1 - lr c), the update ratio is
lr c |x - x*| / |x|, and the step-wise SNR over two gradients g0, g1 is ((g0 + g1) / (g0 - g1))^2."""

import pytest
import torch
from torch import nn

from relay_opal.signals import Instruments

C, X_STAR, X0 = 4.0, 1.0, 3.0


def run(lr, steps, window=8):
    x = nn.Parameter(torch.tensor([X0]))
    opt = torch.optim.SGD([dict(params=[x], name="x")], lr=lr)
    ins = Instruments(opt, window=window)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * C * (x - X_STAR) ** 2
        loss.sum().backward()
        ins.before_step(loss.item())
        opt.step()
        ins.after_step()
    return ins.table()


@pytest.mark.parametrize("lr_c, alpha", [(1.0, 0.0), (2.0, 1.0), (0.5, -0.5), (3.0, 2.0)])
def test_local_step_alpha_is_lr_times_curvature_minus_one(lr_c, alpha):
    t = run(lr_c / C, 3)
    rows = t["alpha"].to_list()
    assert rows[0] == pytest.approx(alpha, abs=1e-5)
    if lr_c == 1.0:
        assert rows[1] is None  # on the minimum, the next step has zero length
    else:
        assert rows[1] == pytest.approx(alpha, abs=1e-5)


def test_first_row_closed_forms():
    lr = 0.5 / C
    t = run(lr, 3)
    assert t.columns == [
        "step",
        "group",
        "grad_norm",
        "update_ratio",
        "grad_cosine",
        "snr",
        "alpha",
        "loss",
        "loss_next",
    ]
    assert t.height == 2 and t["step"].to_list() == [0, 1] and t["group"][0] == "x"
    g0 = C * (X0 - X_STAR)
    g1 = C * (1 - lr * C) * (X0 - X_STAR)
    assert t["grad_norm"][0] == pytest.approx(abs(g0))
    assert t["update_ratio"][0] == pytest.approx(lr * C * abs(X0 - X_STAR) / abs(X0))
    assert t["grad_cosine"][0] == pytest.approx(1.0)
    assert t["loss"][0] == pytest.approx(0.5 * C * (X0 - X_STAR) ** 2)
    assert t["loss_next"][0] == pytest.approx(0.5 * C * (1 - lr * C) ** 2 * (X0 - X_STAR) ** 2)
    assert t["snr"][0] is None  # one gradient in the window at step 0
    assert t["snr"][1] == pytest.approx(((g0 + g1) / (g0 - g1)) ** 2)


def test_cosine_sign_flips_past_the_minimum():
    assert run(1.5 / C, 2)["grad_cosine"][0] == pytest.approx(-1.0)
    assert abs(run(1.0 / C, 2)["grad_cosine"][0]) < 1e-6


def test_window_bounds_the_snr_history():
    t = run(0.5 / C, 6, window=2)
    ratios = [C * (1 - 0.5) ** k * (X0 - X_STAR) for k in range(6)]
    assert t["snr"][0] is None
    for i in range(1, 5):
        g0, g1 = ratios[i - 1], ratios[i]
        assert t["snr"][i] == pytest.approx(((g0 + g1) / (g0 - g1)) ** 2, rel=1e-5)
