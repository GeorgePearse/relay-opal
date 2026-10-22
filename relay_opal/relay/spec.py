"""How a model exposes Relay's sites.

Relay measures credit at fixed places of a pre-norm transformer stack: a token embedding read through an rms norm,
blocks whose input is a learned mix of the previous output and the normed embedding (resid_lambdas, x0_lambdas),
rotary attention with normed queries and keys, optional value embeddings scaled by a sigmoid gate, an relu^2 MLP,
and a soft-capped linear head after a final norm. A RelaySpec maps a model's modules onto those sites and runs the
model's own forward so the estimator can cache what it needs. relay_opal.models.noise_gpt is the reference.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class BlockSites:
    block: str
    c_q: str
    c_k: str
    c_v: str
    attn_proj: str
    c_fc: str
    mlp_proj: str
    window: int
    ve_gate: str | None = None
    value_embeds: str | None = None


@dataclass
class Sites:
    wte: str
    lm_head: str
    resid_lambdas: str
    x0_lambdas: str
    blocks: list


class RelaySpec(Protocol):
    model: torch.nn.Module
    sites: Sites
    vocab_size: int
    width: int
    hidden: int
    n_head: int
    head_dim: int
    soft_cap: float
    gate_channels: int

    def rotary(self, length: int):
        """(cos, sin), each [1, length, 1, head_dim / 2], in the dtype the model applies them in."""

    def attention(self, q, k, v, window):
        """Causal windowed attention over [batch, token, head, dim] tensors, as the model computes it."""

    def losses(self, x, y):
        """The model's forward on (x, y), as training runs it (autocast and all), returning per-token losses
        [batch, token]. Module hooks registered by the estimator must fire inside this call."""


def site_types(sites: Sites) -> dict:
    """Parameter name -> site type, for freezing and for the fidelity report."""
    out = {sites.wte + ".weight": "wte", sites.lm_head + ".weight": "lm_head"}
    out[sites.resid_lambdas] = out[sites.x0_lambdas] = "lambdas"
    for blk in sites.blocks:
        out[blk.c_q + ".weight"] = "q"
        out[blk.c_k + ".weight"] = "k"
        out[blk.c_v + ".weight"] = "v"
        out[blk.attn_proj + ".weight"] = "attn_proj"
        out[blk.c_fc + ".weight"] = "fc"
        out[blk.mlp_proj + ".weight"] = "mlp_proj"
        if blk.ve_gate:
            out[blk.ve_gate + ".weight"] = "gate"
        if blk.value_embeds:
            out[blk.value_embeds + ".weight"] = "value_embeds"
    return out
