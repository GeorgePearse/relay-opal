"""TempBalance on a ResNet-shaped list of layers with given alphas (the paper gives no table of alphas, so these are
illustrative: hot early layers near 2, cold late layers near 5): the lr ordering follows the alphas, the budget is
preserved, the first and last layers keep the base rate as in the reference code, and the unscaled linear map is
the paper's equation (2)."""

import json
import math
from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn

from relay_opal.signals import TempBalance, scan, scan_delta

FIXTURE = Path(__file__).parent / "fixtures" / "signals_tempbalance.json"
LAYERS = {
    "conv1": 2.1,
    "layer1.0.conv1": 2.4,
    "layer1.0.conv2": 2.2,
    "layer2.0.conv1": 3.1,
    "layer2.0.conv2": 2.9,
    "layer3.0.conv1": 3.8,
    "layer3.0.conv2": 4.4,
    "layer4.0.conv1": 5.2,
    "layer4.0.conv2": 5.6,
    "fc": 3.0,
}
BASE = 0.1


def resnet_like():
    model = nn.Sequential()
    for name in LAYERS:
        mod = nn.Linear(16, 16, bias=False) if name == "fc" else nn.Conv2d(16, 16, 3, bias=False)
        model.add_module(name.replace(".", "_"), mod)
    return model


def scan_frame():
    names = [n.replace(".", "_") for n in LAYERS]
    return pl.DataFrame(dict(name=names, head=[None] * len(names), alpha=list(LAYERS.values())))


def test_ordering_budget_and_exclusions():
    model = resnet_like()
    opt = torch.optim.SGD(TempBalance.groups_from(model, BASE), momentum=0.9)
    tb = TempBalance(opt, model, BASE)
    out = tb.step(scan_frame())
    lr = dict(zip(out["name"].to_list(), out["lr"].to_list()))
    assert lr["conv1"] == BASE and lr["fc"] == BASE
    balanced = {n: v for n, v in lr.items() if n not in ("conv1", "fc")}
    alphas = {n.replace(".", "_"): a for n, a in LAYERS.items()}
    ordered = sorted(balanced, key=lambda n: alphas[n])
    assert [balanced[n] for n in ordered] == sorted(balanced.values())
    assert sum(balanced.values()) / len(balanced) == pytest.approx(BASE)
    assert balanced["layer1_0_conv2"] < BASE < balanced["layer4_0_conv2"]
    for g in opt.param_groups:
        assert g["lr"] == lr[g["name"]]
    if not FIXTURE.exists():
        FIXTURE.write_text(json.dumps(dict(alphas=alphas, base_lr=BASE, lr=lr)))


def test_unscaled_linear_map_is_equation_two():
    model = resnet_like()
    opt = torch.optim.SGD(TempBalance.groups_from(model, BASE))
    tb = TempBalance(opt, model, BASE, exclude=set(), rescale=False, lr_min_ratio=0.5, lr_max_ratio=1.5)
    out = tb.step(scan_frame())
    a = list(LAYERS.values())
    amin, amax = min(a), max(a)
    for alpha, lr in zip(a, out["lr"].to_list()):
        assert lr == pytest.approx(BASE * ((alpha - amin) / (amax - amin) * (1.5 - 0.5) + 0.5))


def test_rules_and_schedule():
    model = resnet_like()
    opt = torch.optim.SGD(TempBalance.groups_from(model, BASE))
    for rule in ("sqrt", "log2", "step"):
        tb = TempBalance(opt, model, BASE, rule=rule, exclude=set())
        out = tb.step(scan_frame(), base_lr=0.05)
        assert out["lr"].mean() == pytest.approx(0.05)
        assert out["lr"][0] < out["lr"][8]
    with pytest.raises(ValueError):
        TempBalance(torch.optim.SGD(model.parameters(), lr=BASE), model, BASE)
    assert TempBalance.temperature(0.1, 128, 0.9) == pytest.approx(0.1 / 128 / 0.1)


def test_groups_from_covers_every_parameter_and_scan_rows_feed_step():
    model = resnet_like()
    groups = TempBalance.groups_from(model, BASE)
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    opt = torch.optim.SGD(groups)
    torch.manual_seed(0)
    tb = TempBalance(opt, model, BASE)
    out = tb.step(scan(model, exclude_output_heads=False, device_svd=False))
    assert out.height == len(LAYERS) and out["lr"].is_finite().all()


def test_scan_delta_keeps_layer_rows_and_heads():
    model = resnet_like()
    torch.manual_seed(0)
    prev = scan(model, exclude_output_heads=False, device_svd=False)
    with torch.no_grad():
        model[0].weight.mul_(2.0)
    cur = scan(model, exclude_output_heads=False, device_svd=False)
    d = scan_delta(prev, cur)
    assert d.height == cur.height and d.filter(pl.col("head").is_null()).height == cur.height
    assert abs(d.filter(pl.col("name") == "conv1")["alpha_delta"][0]) < 1e-9
    assert d.filter(pl.col("name") == "conv1")["log_spectral_norm_delta"][0] == pytest.approx(math.log10(4.0))
    assert (d["alpha_delta_z"].abs() < 1e-9).all()
