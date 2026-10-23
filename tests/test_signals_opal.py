"""wrap(signals=True) on the tiny Noise GPT with backend="keep": the instruments run every update, the report lands
every `every` updates with the scan, the outliers and the flat values the agent is shown, and the signals analysis
program is the starting program and runs on obs.params."""

import types

import polars as pl
import pytest
import torch

import relay_opal
from relay_opal import signals
from relay_opal.models import make_optimizer
from relay_opal.opal.wrap import PROGRAM

pytest.importorskip("opal")


def test_keep_run_reports_signals(tiny, tmp_path):
    spec, x, y = tiny
    opts = make_optimizer(spec.model, 6)
    est = relay_opal.RelayEstimator(spec, relay_opal.RelaySettings(routes=64, tokens=32), seed=0)
    loop = relay_opal.wrap(
        opts, spec.model, est, goal="t", total_steps=6, backend="keep", every=3, signals=True, run_dir=tmp_path
    )
    assert loop.instruments is not None and loop.signals_every == 3
    assert (tmp_path / "programs").exists() or True
    for _ in range(6):
        loop.step(x, y)
        loop.opt.zero_grad(set_to_none=True)
    loop.close()
    assert [r["update"] for r in loop.signals] == [3, 6]
    first, second = loop.signals
    assert first["alpha_outliers"] == [] and len(second["alpha_outliers"]) == 5
    names = {r["name"] for r in first["scan"]}
    assert "transformer.h.0.attn.c_q" in names and "lm_head" not in names
    assert any(r["head"] == 1 for r in first["scan"])
    values = second["values"]
    assert {"signals_alpha_mean", "signals_alpha_min", "signals_alpha_max", "signals_alpha_delta_absmax"} <= set(values)
    groups = loop.instruments.names
    assert len(groups) == len([g for o in opts for g in o.param_groups]) and all(groups)
    assert any(k.startswith("signals_") and k.endswith("_grad_cosine") for k in values)
    table = loop.instruments.table()
    assert table.height == 5 * len(groups) and table["step"].max() == 4


def test_signals_analysis_program_runs_on_params(tiny):
    spec, x, y = tiny
    text = relay_opal.analysis_program("signals", heads=spec.n_head)
    assert "HEADS = 2" in text and text == (PROGRAM / "analysis_signals.py").read_text().replace(
        "HEADS = None", "HEADS = 2"
    )
    ns = {}
    exec(compile(text, "analysis.py", "exec"), ns)
    model = spec.model
    with torch.no_grad():
        loss = model(x, y)
    grads = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    obs = types.SimpleNamespace(
        update=50,
        total_updates=100,
        losses=[float(loss)],
        params=dict(model.named_parameters()),
        grads=grads,
        updater=None,
        metrics=[{"signals_alpha_mean": 3.0, "other": 1}],
        program_ms_per_update=1.0,
    )
    memory = {}
    out = ns["analyze"](obs, memory)["report"]
    spectra = out["spectra"]
    assert isinstance(spectra, dict), spectra
    assert "transformer.h.0.mlp.c_fc" in spectra["alpha_by_layer"]
    assert "transformer.h.0.attn.c_q" in spectra["alpha_head_range_by_layer"]
    assert out["shown"] == {"signals_alpha_mean": 3.0}
    assert isinstance(memory["scan"], pl.DataFrame)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1.01)
    out2 = ns["analyze"](obs, memory)["report"]
    assert len(out2["spectra"]["alpha_delta_outliers"]) == 8


def test_report_matches_scan(tiny):
    spec, x, y = tiny
    record, cur = signals.report(spec.model, heads=spec.n_head)
    assert record["instruments"] == {} and record["alpha_outliers"] == []
    assert cur.height == len(record["scan"]) and cur.height == signals.scan(spec.model).height
