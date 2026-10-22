"""Analysis: reports what you want to see at your next decision. Runs on rank 0 only, every EVERY updates, after the
gradient estimate and before the update. It never changes the update; the updater computes its own signals.

obs.update, obs.total_updates, obs.losses (mean loss of each update since the previous call), obs.params and
obs.grads (name -> tensor), obs.updater (the running updater object), obs.metrics (values the training script chose
to show you since the previous call: the fidelity_* cosines and estimator_s_* seconds of the latest fidelity check,
and estimator_* notes), obs.program_ms_per_update (measured cost of the updater), obs.activations (only if enabled).
memory is a dict that persists across calls.
Return {"report": <small JSON>} to keep it for your next decision, {"wake": "<reason>"} to ask for a decision now.

Starting program: loss summary, gradient and weight norms per parameter role (block indices merged) and per block,
the estimator settings in force at this update, and the latest fidelity and timing values the training script showed.
"""

import math
import re

import torch

EVERY = 50


def _role(name):
    return re.sub(r"\.\d+\.", ".*.", name)


def _block(name):
    m = re.search(r"\.(\d+)\.", name)
    return m.group(1) if m else None


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
            if k.startswith(("fidelity_", "estimator_")):
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
        estimator_shown=shown,
    )
    return {"report": report}
