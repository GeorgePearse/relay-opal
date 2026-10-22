"""scan() against WeightWatcher 0.7.7's analyze() on heavy-tailed random matrices and on torchvision's pretrained
resnet18: alpha, alpha_weighted, stable_rank, log_norm, xmin and D agree to 1e-3 on every layer with the same xmin,
and the layers where they differ are reported with the reason (the fixture also serves cargo test)."""

import json
import logging
import urllib.error
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

ww = pytest.importorskip("weightwatcher")
from relay_opal import signals  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "signals_weightwatcher.json"
TOL = 1e-3
WW_COLUMNS = {
    "alpha": "alpha",
    "alpha_weighted": "alpha_weighted",
    "stable_rank": "stable_rank",
    "log_norm": "log_frobenius_norm",
    "xmin": "xmin",
    "D": "ks",
    "num_evals": "num_evals",
    "entropy": "entropy",
}

logging.getLogger("weightwatcher").setLevel(logging.ERROR)


def heavy_tailed(rows, cols, alpha, seed):
    """A matrix whose singular values are Pareto-distributed with tail exponent `alpha` of the eigenvalues."""
    g = np.random.default_rng(seed)
    n = min(rows, cols)
    evals = (g.uniform(1e-3, 1.0, n)) ** (-1.0 / (alpha - 1.0))
    u, _ = np.linalg.qr(g.standard_normal((rows, n)))
    v, _ = np.linalg.qr(g.standard_normal((cols, n)))
    return (u * np.sqrt(evals)) @ v.T


def analyze(model):
    details = ww.WeightWatcher(model=model).analyze(randomize=False, mp_fit=False, plot=False, savefig=False)
    frame = pl.DataFrame({"name": details["longname"].astype(str).tolist()})
    return frame.with_columns(
        [pl.Series(ours, details[theirs].astype(float).tolist()) for theirs, ours in WW_COLUMNS.items()]
    )


def compare(ours, theirs):
    ours = ours.filter(pl.col("head").is_null()).select("name", *WW_COLUMNS.values())
    both = theirs.join(ours, on="name", suffix="_ours", how="inner")
    assert both.height == theirs.height, f"layers missing from scan: {set(theirs['name']) - set(ours['name'])}"
    rows = []
    for r in both.iter_rows(named=True):
        same_xmin = abs(r["xmin"] - r["xmin_ours"]) <= TOL * max(1.0, abs(r["xmin"]))
        diffs = {k: abs(r[k] - r[f"{k}_ours"]) for k in WW_COLUMNS.values()}
        rows.append(dict(name=r["name"], same_xmin=same_xmin, **diffs))
    return pl.DataFrame(rows), both


@pytest.fixture(scope="module")
def random_matrices():
    g = np.random.default_rng(0)
    layers = []
    for i in range(20):
        alpha = float(g.choice([1.5, 2.0, 2.5, 3.0, 4.0, 6.0]))
        rows, cols = int(g.choice([64, 128, 256])), int(g.choice([64, 128, 256]))
        layers.append((f"layer{i}", alpha, heavy_tailed(rows, cols, alpha, i)))
    model = nn.Sequential()
    for name, _, w in layers:
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(w))
        model.add_module(name, lin)
    return layers, model


def test_random_heavy_tailed_matrices_match(random_matrices):
    layers, model = random_matrices
    theirs = analyze(model)
    ours = signals.scan(model, exclude_output_heads=False, device_svd=False)
    report, both = compare(ours, theirs)
    mismatch = report.filter(~pl.col("same_xmin") | (pl.col("alpha") > TOL))
    assert mismatch.height == 0, f"layers that differ from WeightWatcher:\n{mismatch}"
    assert report.select(pl.exclude("name", "same_xmin")).max().row(0) <= tuple([TOL] * (len(WW_COLUMNS))), report
    if not FIXTURE.exists():
        FIXTURE.parent.mkdir(exist_ok=True)
        fixture = [
            dict(
                name=n,
                true_alpha=a,
                n=max(w.shape),
                esd=np.sort(np.linalg.svd(w, compute_uv=False) ** 2).tolist(),
                **{k: float(v) for k, v in zip(WW_COLUMNS.values(), ref)},
            )
            for (n, a, w), ref in zip(layers, theirs.select(*WW_COLUMNS.values()).iter_rows())
        ]
        FIXTURE.write_text(json.dumps(fixture))


def test_device_svd_float32_stays_within_tolerance(random_matrices):
    _, model = random_matrices
    a = signals.scan(model, exclude_output_heads=False, device_svd=False)
    b = signals.scan(model, exclude_output_heads=False, device_svd=True)
    report, _ = compare(b, a.filter(pl.col("head").is_null()).select("name", *WW_COLUMNS.values()))
    assert (report["alpha"] <= 2e-3).all(), report


def test_resnet18_pretrained_matches():
    torchvision = pytest.importorskip("torchvision")
    try:
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except urllib.error.URLError as exc:  # pragma: no cover
        pytest.skip(f"no pretrained weights: {exc}")
    theirs = analyze(model)
    ours = signals.scan(model, exclude_output_heads=False, exclude_1x1=False, device_svd=False)
    report, both = compare(ours, theirs)
    differ = report.filter(~pl.col("same_xmin") | (pl.col("alpha") > TOL))
    if differ.height:
        detail = both.filter(pl.col("name").is_in(differ["name"])).select(
            "name", "xmin", "xmin_ours", "alpha", "alpha_ours", "ks", "ks_ours", "num_evals", "num_evals_ours"
        )
        pytest.fail(f"{differ.height} of {report.height} resnet18 layers differ from WeightWatcher:\n{detail}")
    assert report.height >= 20
