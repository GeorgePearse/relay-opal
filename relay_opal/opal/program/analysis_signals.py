"""Analysis: reports what you want to see at your next decision. Runs on rank 0 only, every EVERY updates, after the
gradient estimate and before the update. It never changes the update; the updater computes its own signals.

obs.update, obs.total_updates, obs.losses (mean loss of each update since the previous call), obs.params and
obs.grads (name -> tensor), obs.updater (the running updater object), obs.metrics (values the training script chose
to show you since the previous call), obs.program_ms_per_update (measured cost of the updater), obs.activations
(only if enabled). memory is a dict that persists across calls.
Return {"report": <small JSON>} to keep it for your next decision, {"wake": "<reason>"} to ask for a decision now.

Starting program: the loss summary, gradient and weight norms per parameter role and per block, the estimator
settings in force, and the per-layer signals of relay_opal.signals (Rust numerics behind a torch glue):

- weight spectra (WeightWatcher): for every weight matrix, the power-law exponent alpha of its eigenvalue spectrum
  (small alpha = heavy-tailed, hot, well-trained; large alpha = cold, under-trained; read deltas and outliers between
  layers rather than the absolute band on a run this short), alpha_weighted, stable rank, and per attention head the
  spread of alpha across heads. alpha_delta_outliers are the layers (and heads) whose alpha moved most since the
  previous call relative to the others (z-score across layers). Gradient-free, so Relay cannot distort it.
- instruments (Cockpit-style, from the optimizer's own gradients, logged by the training script as signals_* values
  per parameter group, averaged since the previous call): grad_norm, update_ratio (step size over parameter norm),
  grad_cosine (consecutive gradients: near 1 steady descent, near -1 oscillation), snr (step-wise gradient
  signal-to-noise over the last few steps), alpha_step (Cockpit's local step size: -1 the step barely moved, 0 it
  landed on the fitted minimum along its direction, +1 it overshot to the loss it started from). Under Relay these
  inherit the estimator's error.
- sharpness (PyHessian-style top Hessian eigenvalue) and edge_of_stability = lambda_max * lr / 2 appear only when
  the training script computes them (backprop arms; Relay has no autograd).

To act on per-layer alpha, relay_opal.signals.TempBalance gives the TempBalance rule for per-group learning rates;
the updater can import relay_opal.signals and call `_signals.layer_lrs` on the alphas it reads from this report.
"""

import math
import re

import polars as pl
import torch

from relay_opal import signals

EVERY = 50
HEADS = None  # attention head count for the per-head split; the training script sets it
TOP_K = 8


def _role(name):
    return re.sub(r"\.\d+\.", ".*.", name)


def _block(name):
    m = re.search(r"\.(\d+)\.", name)
    return m.group(1) if m else None


def _spectra(obs, memory):
    params = {n: p.detach() for n, p in obs.params.items()}
    cur = signals.scan(params, heads=HEADS, device_svd=True)
    prev = memory.get("scan")
    memory["scan"] = cur
    layers = cur.filter(pl.col("head").is_null())
    heads = cur.filter(pl.col("head").is_not_null())
    out = dict(
        alpha_by_layer={r["name"]: round(r["alpha"], 3) for r in layers.iter_rows(named=True)},
        alpha_weighted_by_layer={r["name"]: round(r["alpha_weighted"], 3) for r in layers.iter_rows(named=True)},
        stable_rank_by_layer={r["name"]: round(r["stable_rank"], 2) for r in layers.iter_rows(named=True)},
        alpha_mean=round(float(layers["alpha"].mean()), 4) if layers.height else None,
        alpha_std=round(float(layers["alpha"].std()), 4) if layers.height > 1 else None,
    )
    if heads.height:
        spread = heads.group_by("name").agg(pl.col("alpha").min().alias("lo"), pl.col("alpha").max().alias("hi"))
        out["alpha_head_range_by_layer"] = {
            r["name"]: [round(r["lo"], 3), round(r["hi"], 3)] for r in spread.iter_rows(named=True)
        }
    if prev is not None:
        d = signals.scan_delta(prev, cur).with_columns(pl.col("alpha_delta_z").abs().alias("z"))
        top = d.sort("z", descending=True).head(TOP_K)
        out["alpha_delta_outliers"] = [
            dict(
                name=r["name"],
                head=r["head"],
                alpha=round(r["alpha"], 3),
                delta=round(r["alpha_delta"], 4),
                z=round(r["alpha_delta_z"], 2),
            )
            for r in top.iter_rows(named=True)
        ]
        out["alpha_delta_mean"] = round(float(d.filter(pl.col("head").is_null())["alpha_delta"].mean()), 5)
    return out


def analyze(obs, memory):
    names = [n for n, g in obs.grads.items() if g is not None]
    if not names:
        return None
    gn = torch.stack(torch._foreach_norm([obs.grads[n].float() for n in names])).tolist()
    wn = torch.stack(torch._foreach_norm([obs.params[n].detach().float() for n in names])).tolist()
    roles, blocks = {}, {}
    for n, g, w in zip(names, gn, wn):
        r = roles.setdefault(_role(n), [0.0, 0.0])
        r[0] += g * g
        r[1] += w * w
        b = _block(n)
        if b is not None:
            blocks[b] = blocks.get(b, 0.0) + g * g
    losses = [x for x in obs.losses if math.isfinite(x)]
    settings_fn = getattr(obs.updater, "estimator_settings", None)
    try:
        settings = settings_fn(obs.update, obs.total_updates) if settings_fn else None
    except Exception as exc:  # the training script reports the same failure to you
        settings = f"error: {exc}"
    shown = {}
    for m in obs.metrics or []:
        for k, v in m.items():
            if k.startswith(("fidelity_", "estimator_", "signals_", "sharpness", "edge_of_stability")):
                shown[k] = v
    report = dict(
        update=obs.update,
        loss_mean=round(sum(losses) / len(losses), 5) if losses else None,
        loss_min=round(min(losses), 5) if losses else None,
        loss_max=round(max(losses), 5) if losses else None,
        nonfinite_losses=len(obs.losses) - len(losses),
        grad_norm_total=round(math.sqrt(sum(g * g for g in gn)), 6),
        grad_norm_by_role={k: round(math.sqrt(v[0]), 6) for k, v in roles.items()},
        weight_norm_by_role={k: round(math.sqrt(v[1]), 4) for k, v in roles.items()},
        grad_norm_by_block={k: round(math.sqrt(v), 6) for k, v in sorted(blocks.items(), key=lambda kv: int(kv[0]))},
        program_ms_per_update=obs.program_ms_per_update,
        estimator_settings=settings,
        shown=shown,
    )
    try:
        report["spectra"] = _spectra(obs, memory)
    except Exception as exc:
        report["spectra"] = f"error: {type(exc).__name__}: {exc}"
    return {"report": report}
