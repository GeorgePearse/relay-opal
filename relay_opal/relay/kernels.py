"""Relay's measured and closed-form steps, ported from qlabs-eng/noise PR #20 (train/val/run.py).

Every function here is a plain tensor function: no autograd, no module state. Kernels compile on CUDA only
(RELAY_OPAL_COMPILE=0 turns that off); on CPU they run eagerly, which the tests use.
"""

import os

import torch
from torch.nn import functional as F

EPS = torch.finfo(torch.float32).eps


def maybe_compile(fn=None, **kw):
    def wrap(f):
        if torch.cuda.is_available() and os.environ.get("RELAY_OPAL_COMPILE", "1") != "0":
            return torch.compile(f, **kw)
        return f

    return wrap(fn) if fn is not None else wrap


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)


HADAMARDS = {}


def hadamard(d, device, dtype):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1."""
    key = (d, str(device), dtype)
    if key not in HADAMARDS:
        if d & (d - 1):
            raise ValueError(f"Orthogonal draws need a power-of-two width, not {d}.")
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[key] = H.to(dtype)
    return HADAMARDS[key]


@maybe_compile
def coordinate_differences(raw, y, sigma, cap):
    """Finite differences of each token's loss along each of its own logits (the record's coordinate head): every
    moved loss is a full forward evaluation of the soft-capped cross-entropy, written in closed form."""
    capped = lambda v: cap * torch.tanh(v / cap)
    z, up, down = capped(raw.float()), capped(raw.float() + sigma), capped(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


@maybe_compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, width, cap):
    """Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x of
    the clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,)
    |D_j|^2. The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(cap * torch.tanh(base * inv0 / cap)).sum(-1, keepdim=True).log() - cap * torch.tanh(
        base_y * inv0 / cap
    )
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(cap * torch.tanh(z / cap)).sum(-1).log() - cap * torch.tanh((base_y + steps * dirs_y) * inv / cap)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@maybe_compile(dynamic=False)
def readout_losses(x, weight, y, cap):
    """The head's loss at the streams x (n, W), measured: norm, linear, soft cap, cross-entropy."""
    logits = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS), weight)
    logits = cap * torch.tanh(logits / cap)
    return F.cross_entropy(logits, y.clamp(min=0), reduction="none") * (y >= 0)


@maybe_compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary(moved, cos, sin)).to(a.dtype)


@maybe_compile(dynamic=False)
def head_scores(out, baseline, hub):
    change = out.float() - baseline.float().unsqueeze(0)
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@maybe_compile(dynamic=False)
def value_scores(change, hub):
    return (change.float() * hub.float().permute(0, 2, 1, 3).unsqueeze(3)).sum(-1)


@maybe_compile(dynamic=False)
def heads_first(move):
    n, b, t, h, hd = move.shape
    return move.to(torch.bfloat16).permute(1, 3, 2, 0, 4).reshape(b, h, t, n * hd)


@maybe_compile(dynamic=False)
def code_estimate(credited, a):
    return (credited.permute(3, 0, 2, 1).unsqueeze(-1) * a.float()).sum(0)


@maybe_compile(dynamic=False)
def hidden_scores(ut, st, Hk, cols, g):
    """The record's hidden step: each token's top-k units move along a sign-flipped Hadamard cycle through relu^2
    and the output columns, scored against the block-output credit g."""
    moved = st.unsqueeze(-1) * Hk
    change = F.relu(ut.unsqueeze(1) + moved).square() - F.relu(ut.unsqueeze(1) - moved).square()
    stream_change = torch.bmm(change.to(torch.bfloat16), cols)
    scored = torch.bmm(stream_change, g.to(torch.bfloat16).unsqueeze(-1)).squeeze(-1).float()
    return scored / (2 * st)


def norm_difference(x, delta):
    """norm(x + delta) - norm(x), exactly: x (m, W) clean, delta (m, P, W) moves."""
    width = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    dots = torch.einsum("md,mpd->mp", x, delta)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / width + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


def norm_difference_measured(x, delta):
    """The same difference by evaluating the norm at every moved stream."""
    return norm(x.unsqueeze(1) + delta) - norm(x).unsqueeze(1)


@maybe_compile(dynamic=False)
def key_scores(qp, kp, k_raw, steps, cos, sin, A, C, basis):
    """Single-key experiments, each evaluated exactly: the raw key at position t moves by steps_t e_j and nothing
    else moves, so each query's attention changes in one term, o'_s - o_s = (a' - a)(v_t - o_s) / (1 - a + a'),
    a' = a exp(dl). Scored against each query's target (C[s, t] = hub_s . (v_t - o_s)): one local step, as the
    record's coordinate head scores single-logit moves. qp, kp, k_raw (h, T, d); steps (h, T, 1); cos, sin
    (T, d/2); A, C (h, T, T), A zero outside the window; basis (J, d). Returns (h, T, J)."""
    d = k_raw.shape[-1]
    moved = k_raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    x1, x2 = moved[..., : d // 2], moved[..., d // 2 :]
    moved = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    dk = moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)
    dl = torch.einsum("hsd,hjtd->hjst", qp, dk) / d**0.5
    change = A[:, None] * torch.expm1(dl)
    return (change / (1 + change) * C[:, None]).sum(2).transpose(1, 2) / steps


@maybe_compile(dynamic=False)
def mlp_cycle(move, mid, idx, hs, d_out, w_fc, w_out_t, measured_norm, bf16, scale):
    """Moves of a block's mid stream through its MLP alone (exact differences: rms norm in closed form or measured,
    relu^2 on each token's routed units, the rest held at their clean outputs), each scored against the measured
    credit d_out of the block's output. move (m, P, W); mid, d_out (m, W); idx, hs (m, k); scale (m, P)."""
    dt = torch.bfloat16 if bf16 else torch.float32
    dz = norm_difference_measured(mid, move) if measured_norm else norm_difference(mid, move)
    dh = torch.bmm(dz.to(dt), w_fc[idx].to(dt).transpose(1, 2)).float()
    h = hs.unsqueeze(1)
    up, down = F.relu(h + dh), F.relu(h)
    dout = move + torch.bmm(((up - down) * (up + down)).to(dt), w_out_t[idx].to(dt)).float()
    return (dout * d_out.unsqueeze(1)).sum(-1) / scale


@maybe_compile(dynamic=False)
def mlp_cycle_full(move, mid, hs, d_out, w_fc, w_out_t, measured_norm, bf16, scale):
    """mlp_cycle with every unit carrying the move (routes = 0): hs (m, H)."""
    dt = torch.bfloat16 if bf16 else torch.float32
    dz = norm_difference_measured(mid, move) if measured_norm else norm_difference(mid, move)
    dh = (dz.to(dt) @ w_fc.to(dt).T).float()
    h = hs.unsqueeze(1)
    up, down = F.relu(h + dh), F.relu(h)
    dout = move + (((up - down) * (up + down)).to(dt) @ w_out_t.to(dt)).float()
    return (dout * d_out.unsqueeze(1)).sum(-1) / scale


@maybe_compile(dynamic=False)
def norm_cycle(move, x, target, measured, scale):
    """Moves of x through its parameter-free rms norm, scored against the credit of norm(x)."""
    dz = norm_difference_measured(x, move) if measured else norm_difference(x, move)
    return (dz * target.unsqueeze(1)).sum(-1) / scale


@maybe_compile(dynamic=False)
def hidden_all(hpre, steps, g_units):
    """Every MLP unit's credit by its own central difference through relu^2 (elementwise; its exact slope away from
    the kink), read against the block-output credit through the unit's output column (g_units = d_out @ W_out)."""
    up, down = F.relu(hpre + steps).square(), F.relu(hpre - steps).square()
    return (up - down) / (2 * steps) * g_units
