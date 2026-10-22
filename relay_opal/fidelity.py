"""Per-site gradient cosine between Relay's estimate and float32 autograd on one batch (check_relay.py as a function).

Analysis only: autograd runs here on a float32 copy of the model to produce the reference, never in the estimator.
"""

import copy
import re
import time

import torch
from torch.nn import functional as F

from .relay.estimator import RelayEstimator
from .relay.spec import site_types

BLOCK_TYPES = ("q", "k", "v", "attn_proj", "gate", "fc", "mlp_proj")


def reference_gradients(model, x, y):
    """Autograd of the summed next-token loss, float32 throughout, name -> gradient."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        ref = copy.deepcopy(model).float().requires_grad_(True)
        params = list(ref.named_parameters())
        with torch.enable_grad():
            loss = ref(x, y, loss_reduction="sum")
            grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
        return {n: (torch.zeros_like(p) if g is None else g).detach().float() for (n, p), g in zip(params, grads)}
    finally:
        torch.set_float32_matmul_precision(precision)


def _flat(d, names):
    return torch.cat([d[n].flatten().float() for n in names])


def cosines(estimate, reference, types):
    """Cosine per site type, per block, for the block matrices together and for every parameter."""
    names = [n for n in reference if n in estimate]
    groups = {"block_matrices": [n for n in names if types.get(n) in BLOCK_TYPES]}
    for kind in BLOCK_TYPES:
        groups[kind] = [n for n in names if types.get(n) == kind]
    for kind in ("wte", "value_embeds", "lm_head", "lambdas"):
        groups[kind] = [n for n in names if types.get(n) == kind]
    blocks = sorted({m.group(1) for n in groups["block_matrices"] if (m := re.search(r"\.(\d+)\.", n))}, key=int)
    for i in blocks:
        groups[f"block_{i}"] = [n for n in groups["block_matrices"] if f".{i}." in n]
    groups["all"] = names
    out = {}
    for g, ns in groups.items():
        if ns:
            out[g] = round(F.cosine_similarity(_flat(estimate, ns), _flat(reference, ns), dim=0).item(), 4)
    return out


def check(estimator: RelayEstimator, x, y):
    """Relay's estimate on (x, y) with the estimator's current settings against float32 autograd on the same batch.

    Returns dict(cosine={group: value}, seconds={site type: seconds}, estimate_seconds=..., reference_seconds=...).
    """
    model = estimator.model
    t0 = time.perf_counter()
    reference = reference_gradients(model, x, y)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    profile = estimator.profile
    estimator.profile = True
    try:
        estimate = estimator.gradients(x, y)
    finally:
        estimator.profile = profile
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    types = site_types(estimator.spec.sites)
    return dict(
        cosine=cosines(estimate, reference, types),
        seconds={k: round(v, 4) for k, v in estimator.times.items()},
        estimate_seconds=round(t2 - t1, 4),
        reference_seconds=round(t1 - t0, 4),
    )
