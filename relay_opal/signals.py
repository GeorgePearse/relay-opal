"""Per-layer training signals: weight spectra (WeightWatcher), per-layer learning rates from them (TempBalance),
per-step instruments from the gradients the optimizer already holds (Cockpit) and sharpness by Hessian-vector
products (PyHessian). The numerics are Rust (relay_opal._signals); torch does the glue: collecting weights,
gradients, parameters, losses, and the Hessian-vector product closure.

scan(model) -> polars.DataFrame               one row per layer (and per attention head), every spectrum metric
scan_delta(prev, cur) -> polars.DataFrame     what moved between two scans, with the per-layer outliers
TempBalance(optimizer, model, base_lr)        .step(scan_df) sets each layer group's lr from its alpha
Instruments(optimizer, window=8)              .before_step(loss) / .after_step(); .table() of steps x groups
Hessian(loss_fn, params)                      .top_eigenvalues(k), .trace(), .density(), .edge_of_stability(lr)
"""

import math
import re

import numpy as np
import polars as pl
import torch
from torch import nn

try:
    from . import _signals as rs
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "relay_opal._signals (the Rust extension) is not built for this interpreter. Install Rust "
        "(https://rustup.rs) and run `pip install -e .` in the relay-opal checkout, or install a released wheel."
    ) from exc

METRICS = (
    "alpha",
    "alpha_weighted",
    "xmin",
    "ks",
    "sigma",
    "lambda_max",
    "stable_rank",
    "log_spectral_norm",
    "log_frobenius_norm",
    "log_alpha_norm",
    "entropy",
    "num_evals",
    "num_pl_spikes",
    "rank_loss",
    "weak_rank_loss",
)
DELTA_METRICS = ("alpha", "alpha_weighted", "stable_rank", "log_spectral_norm", "log_frobenius_norm", "entropy", "ks")
MIN_EVALS = 8
QKV = re.compile(r"(^|\.)(c_q|c_k|c_v|q_proj|k_proj|v_proj|wq|wk|wv|query|key|value|qkv|c_attn|in_proj)$")
OUTPUT_HEAD = re.compile(r"(^|\.)(lm_head|head|fc|classifier|output|out_proj_head)$")
HEAD_ATTRS = ("n_head", "num_heads", "n_heads", "heads")
EMBEDDING = re.compile(r"embed|wte|wpe|tok_emb|pos_emb")


def _svals(w, device_svd):
    if device_svd:
        return torch.linalg.svdvals(w.detach().float()).to("cpu", torch.float64).numpy()
    m = w.detach().to("cpu", torch.float32).contiguous()
    return np.sqrt(rs.esd(m.numpy().ravel(), m.shape[0], m.shape[1]))


def _metrics(mats, device_svd, hill_xmin_pos=None, scale=1.0):
    """One ESD pooled over the matrices (WeightWatcher's pool=True for conv slices), the fit and metrics."""
    n = max(max(m.shape) for m in mats)
    esd, rank_loss = [], 0
    for m in mats:
        s = _svals(m, device_svd) * scale
        esd.append(s * s)
        rank_loss += len(s) - rs.matrix_rank(s, max(m.shape))
    esd = np.sort(np.concatenate(esd))
    return rs.layer_metrics(esd, n, rank_loss, hill_xmin_pos)


def _heads_of(model, name):
    """n_head from the nearest ancestor module that declares it."""
    parts = name.split(".")
    for depth in range(len(parts) - 1, -1, -1):
        mod = model.get_submodule(".".join(parts[:depth])) if depth else model
        for attr in HEAD_ATTRS:
            h = getattr(mod, attr, None)
            if isinstance(h, int) and h > 0:
                return h
    return None


def _split_heads(name, kind, w, heads, out):
    rows = w.shape[0] // heads
    blocks = [w[i * rows : (i + 1) * rows] for i in range(heads)]
    for i, b in enumerate(blocks):
        out.append((name, kind, i, [b]))
    out.append((name, kind, None, blocks))


def _from_params(params, include, exclude_output_heads, exclude_1x1, per_head, heads):
    """layer_matrices over a name -> tensor mapping (an OPAL analysis program's obs.params): 2-D tensors are
    linear layers, 4-D ones conv2d; `heads` is the head count for the attention projections, since no module
    declares it."""
    names = [n for n, t in params.items() if torch.is_tensor(t) and t.dim() in (2, 4) and not EMBEDDING.search(n)]
    last = next((n for n in reversed(names) if params[n].dim() == 2), None)
    out = []
    for name in names:
        layer = name[: -len(".weight")] if name.endswith(".weight") else name
        if include is not None and not include(layer):
            continue
        w = params[name]
        if w.dim() == 2:
            if exclude_output_heads and (OUTPUT_HEAD.search(layer) or name == last):
                continue
            if (
                per_head
                and heads
                and QKV.search(layer)
                and w.shape[0] % heads == 0
                and w.shape[0] // heads >= MIN_EVALS
            ):
                _split_heads(layer, "attention", w, heads, out)
            else:
                out.append((layer, "linear", None, [w]))
        else:
            kh, kw = w.shape[2], w.shape[3]
            if exclude_1x1 and kh * kw == 1:
                continue
            out.append((layer, "conv2d", None, [w[:, :, i, j] for i in range(kh) for j in range(kw)]))
    return out


def layer_matrices(model, *, include=None, exclude_output_heads=True, exclude_1x1=True, per_head=True, heads=None):
    """The matrices scan() analyses: (name, kind, head, [matrices]) per layer, with WeightWatcher's extraction
    rules. Linear: the weight. Conv2d (out, in, kh, kw): kh*kw slices (out, in), pooled. Attention projections
    (c_q, c_k, c_v, q_proj, qkv, in_proj, ...): split per head along the output rows, one entry per head and one
    pooled entry for the layer, the head count read from the nearest module that declares n_head (or `heads` when
    `model` is a name -> tensor mapping). Embeddings, norms and biases are not matrices here."""
    if isinstance(include, str):
        include = re.compile(include).search
    if isinstance(model, dict):
        return _from_params(model, include, exclude_output_heads, exclude_1x1, per_head, heads)
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    last_linear = linears[-1] if linears else None
    out = []
    for name, mod in model.named_modules():
        if include is not None and not include(name):
            continue
        if isinstance(mod, nn.Linear):
            w = mod.weight
            if exclude_output_heads and (OUTPUT_HEAD.search(name) or name == last_linear):
                continue
            heads = _heads_of(model, name) if per_head and QKV.search(name) else None
            if heads and w.shape[0] % heads == 0 and w.shape[0] // heads >= MIN_EVALS:
                _split_heads(name, "attention", w, heads, out)
            else:
                out.append((name, "linear", None, [w]))
        elif isinstance(mod, nn.Conv2d):
            w = mod.weight
            if w.dim() != 4:
                continue
            kh, kw = w.shape[2], w.shape[3]
            if exclude_1x1 and kh * kw == 1:
                continue
            out.append((name, "conv2d", None, [w[:, :, i, j] for i in range(kh) for j in range(kw)]))
        elif isinstance(mod, nn.MultiheadAttention) and per_head:
            w = mod.in_proj_weight
            if w is None:
                continue
            e = mod.embed_dim
            for proj, block in zip("qkv", (w[:e], w[e : 2 * e], w[2 * e :])):
                rows = e // mod.num_heads
                heads = [block[i * rows : (i + 1) * rows] for i in range(mod.num_heads)]
                for i, h in enumerate(heads):
                    out.append((f"{name}.{proj}", "attention", i, [h]))
                out.append((f"{name}.{proj}", "attention", None, heads))
    return out


def scan(
    model,
    *,
    include=None,
    exclude_output_heads=True,
    exclude_1x1=True,
    per_head=True,
    device_svd=True,
    hill_xmin_pos=None,
    conv2d_norm=True,
    heads=None,
):
    """WeightWatcher's per-layer metrics for every weight matrix of `model` (a module, or a name -> tensor mapping
    with `heads` the attention head count), one row per layer and per attention head (head=None is the layer's
    pooled ESD). device_svd runs torch.linalg.svdvals on the tensor's device (float32)
    and hands the singular values to Rust; otherwise Rust does the SVD on the CPU in float64. Layers with fewer than
    8 pooled eigenvalues have no fit and are left out, as in WeightWatcher. conv2d_norm scales every conv slice by
    sqrt(1/2) as WeightWatcher (conv2d_norm=True) and TempBalance (conv_norm=0.5) do; it moves xmin, lambda_max and
    the log norms, not alpha. hill_xmin_pos switches the fit to TempBalance's Hill estimator (xmin at the ESD's
    index len/hill_xmin_pos)."""
    rows = []
    with torch.no_grad():
        for name, kind, head, mats in layer_matrices(
            model,
            include=include,
            exclude_output_heads=exclude_output_heads,
            exclude_1x1=exclude_1x1,
            per_head=per_head,
            heads=heads,
        ):
            if sum(min(m.shape) for m in mats) < MIN_EVALS:
                continue
            scale = math.sqrt(0.5) if conv2d_norm and kind == "conv2d" else 1.0
            m = _metrics(mats, device_svd, hill_xmin_pos, scale)
            if m is None:
                continue
            shape = list(mats[0].shape) + ([len(mats)] if len(mats) > 1 else [])
            rows.append(dict(name=name, kind=kind, head=head, shape=shape, **m))
    schema = {"name": pl.Utf8, "kind": pl.Utf8, "head": pl.Int64, "shape": pl.List(pl.Int64)}
    schema.update(
        {
            k: (pl.Int64 if k in ("num_evals", "num_pl_spikes", "rank_loss", "weak_rank_loss") else pl.Float64)
            for k in METRICS
        }
    )
    return pl.DataFrame(rows, schema=schema)


def scan_delta(prev, cur):
    """Per-layer changes between two scans joined on (name, head): each delta metric as `<metric>_delta`, with the
    current value, and `<metric>_delta_z`, the delta's z-score across layers (the outliers between layers that the
    short-run reading of WeightWatcher asks for; 0 when the deltas' spread is below 1e-6), heads excluded from the
    z-score."""
    cols = ["name", "head", "kind", *DELTA_METRICS]
    p = prev.select(cols).rename({m: f"{m}_prev" for m in DELTA_METRICS})
    d = cur.select(cols).join(p, on=["name", "head", "kind"], how="inner", nulls_equal=True)
    d = d.with_columns([(pl.col(m) - pl.col(f"{m}_prev")).alias(f"{m}_delta") for m in DELTA_METRICS])
    layer = pl.col("head").is_null()
    for m in DELTA_METRICS:
        col = pl.col(f"{m}_delta")
        mean, std = col.filter(layer).mean(), col.filter(layer).std()
        d = d.with_columns(pl.when(std > 1e-6).then((col - mean) / std).otherwise(0.0).alias(f"{m}_delta_z"))
    return d.drop([f"{m}_prev" for m in DELTA_METRICS])


class TempBalance:
    """Per-layer learning rates from per-layer alpha (TempBalance). The optimizer needs one param group per layer
    with a "name" key, which groups_from(model, base_lr) builds; groups whose name is not in the scan (norms,
    biases, embeddings) and excluded names keep base_lr. The reference code leaves the first and the last layer
    alone; by default so does this (exclude=None). rule: linear_map (the paper's equation 2), sqrt, log2, step;
    rescale keeps the mean rate over the balanced layers at base_lr."""

    def __init__(
        self,
        optimizer,
        model,
        base_lr,
        rule="linear_map",
        lr_min_ratio=0.5,
        lr_max_ratio=1.5,
        exclude=None,
        rescale=True,
    ):
        self.optimizer, self.model, self.base_lr = optimizer, model, base_lr
        self.rule, self.lr_min_ratio, self.lr_max_ratio, self.rescale = rule, lr_min_ratio, lr_max_ratio, rescale
        self.groups = [g for g in optimizer.param_groups if "name" in g]
        if not self.groups:
            raise ValueError("TempBalance needs one param group per layer with a 'name' key: see groups_from")
        self.names = [g["name"] for g in self.groups]
        if exclude is None:
            layers = layer_matrices(model, exclude_output_heads=False, exclude_1x1=False, per_head=False)
            scanned = [n for n, *_ in layers if n in self.names]
            exclude = {scanned[0], scanned[-1]} if scanned else set()
        elif isinstance(exclude, str):
            pat = re.compile(exclude)
            exclude = {n for n in self.names if pat.search(n)}
        self.exclude = set(exclude)
        self.history = []

    @staticmethod
    def groups_from(model, base_lr, **kw):
        """Param groups for a torch optimizer: one per module that owns parameters directly, named after it,
        plus one "other" group for parameters owned by the model itself."""
        groups = []
        for name, mod in model.named_modules():
            params = [p for p in mod.parameters(recurse=False) if p.requires_grad]
            if params:
                groups.append(dict(params=params, lr=base_lr, name=name or "other", **kw))
        return groups

    def step(self, scan_df, base_lr=None):
        """Set each named group's lr from the scan's layer rows (head is null). base_lr overrides the constructor's
        for this call (a schedule). Returns a frame of (name, alpha, lr)."""
        base = self.base_lr if base_lr is None else base_lr
        layers = scan_df.filter(pl.col("head").is_null()).select("name", "alpha")
        alpha_of = dict(zip(layers["name"].to_list(), layers["alpha"].to_list()))
        alphas = np.array([alpha_of.get(n, math.nan) for n in self.names], dtype=np.float64)
        excl = [n in self.exclude for n in self.names]
        lrs = rs.layer_lrs(alphas, base, self.rule, self.lr_min_ratio, self.lr_max_ratio, excl, self.rescale)
        for g, lr in zip(self.groups, lrs):
            g["lr"] = float(lr)
        out = pl.DataFrame(dict(name=self.names, alpha=alphas, lr=lrs))
        self.history.append(out)
        return out

    @staticmethod
    def temperature(lr, batch_size, momentum=0.0):
        return rs.temperature(lr, batch_size, momentum)


def _flat(tensors, grad=False):
    parts = []
    for p in tensors:
        t = p.grad if grad else p
        if t is None:
            t = torch.zeros_like(p)
        parts.append(t.detach().reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return torch.cat(parts).to("cpu", torch.float32).numpy()


class Instruments:
    """Cockpit's cheap instruments per param group and step, from the optimizer's own gradients and parameters:
    gradient norm, update-to-parameter ratio (from the update actually applied), cosine between consecutive
    gradients, the step-wise SNR over the last `window` step gradients, and Cockpit's local step size alpha (-1 the
    step stayed put, 0 it landed on the fitted minimum, +1 it overshot to its starting loss). Call before_step(loss)
    after backward and before optimizer.step(), after_step() after it. The row for step t is complete at step t+1's
    before_step; table() has the complete rows. Each call copies every group's gradients and parameters to the CPU
    as float32 once; the history (window x parameters) lives in Rust."""

    def __init__(self, optimizer, window=8, names=None):
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        self.groups = [g for o in optimizers for g in o.param_groups]
        self.names = list(names) if names else [g.get("name", str(i)) for i, g in enumerate(self.groups)]
        if len(self.names) != len(self.groups):
            raise ValueError(f"{len(self.names)} names for {len(self.groups)} param groups")
        self.window = rs.Window(window, len(self.groups))
        self.step = 0
        self.rows = []

    def before_step(self, loss):
        loss = float(loss)
        for i, g in enumerate(self.groups):
            row = self.window.before_step(i, _flat(g["params"], grad=True), _flat(g["params"]), loss)
            if row is not None:
                self.rows.append(dict(step=self.step - 1, group=self.names[i], **row))
        self.step += 1

    def after_step(self):
        for i, g in enumerate(self.groups):
            self.window.after_step(i, _flat(g["params"]))

    @staticmethod
    def names_from(model, optimizers):
        """One name per param group across `optimizers`: the parameter role (block indices merged) when a group
        holds one role, else group<i>."""
        roles = {id(p): re.sub(r"\.\d+\.", ".*.", n) for n, p in model.named_parameters()}
        opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        names = []
        for i, g in enumerate([g for o in opts for g in o.param_groups]):
            kinds = sorted({roles.get(id(p), "?") for p in g["params"]})
            names.append(kinds[0] if len(kinds) == 1 else f"group{i}")
        return names

    def summary(self, since=0):
        """Per-group means of the rows from step `since` on, as {group: {metric: mean}}, for a log line."""
        t = self.table().filter(pl.col("step") >= since)
        cols = ["grad_norm", "update_ratio", "grad_cosine", "snr", "alpha"]
        out = {}
        for r in t.group_by("group").agg([pl.col(c).mean() for c in cols]).iter_rows(named=True):
            out[r["group"]] = {c: r[c] for c in cols if r[c] is not None and math.isfinite(r[c])}
        return out

    def table(self):
        schema = dict(
            step=pl.Int64,
            group=pl.Utf8,
            grad_norm=pl.Float64,
            update_ratio=pl.Float64,
            grad_cosine=pl.Float64,
            snr=pl.Float64,
            alpha=pl.Float64,
            loss=pl.Float64,
            loss_next=pl.Float64,
        )
        return pl.DataFrame(self.rows, schema=schema)


class Hessian:
    """Sharpness of loss_fn() (a closure over a fixed batch) with respect to params, by PyHessian's drivers over an
    autograd Hessian-vector product. The gradient graph is built once (create_graph=True) and every product is one
    extra backward pass. Vectors round-trip as flat float32 numpy on the CPU: one copy of the parameters each way
    per product, fine up to about 100M parameters at a decision point, not per step."""

    def __init__(self, loss_fn, params):
        self.params = [p for p in params if p.requires_grad]
        self.shapes = [p.shape for p in self.params]
        self.dim = sum(p.numel() for p in self.params)
        self.device = self.params[0].device
        loss = loss_fn()
        self.grads = torch.autograd.grad(loss, self.params, create_graph=True)
        self.loss = float(loss.detach())
        self.products = 0
        self._top = None

    def hvp(self, v):
        vs, i = [], 0
        vt = torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
        for p, shape in zip(self.params, self.shapes):
            vs.append(vt[i : i + p.numel()].view(shape).to(p.dtype))
            i += p.numel()
        hv = torch.autograd.grad(self.grads, self.params, grad_outputs=vs, retain_graph=True)
        self.products += 1
        return torch.cat([h.reshape(-1) for h in hv]).to("cpu", torch.float32).numpy()

    def top_eigenvalues(self, k=1, max_iter=100, tol=1e-3, seed=0, method="power"):
        """The k largest eigenvalues, largest first. method="power" is PyHessian's deflated power iteration;
        "lanczos" runs max_iter Lanczos steps and returns the top k Ritz values."""
        if method == "lanczos":
            values = rs.lanczos_eigenvalues(self.hvp, self.dim, k, max_iter, seed)
        else:
            values, _ = rs.top_eigenvalues(self.hvp, self.dim, k, max_iter, tol, seed)
        self._top = values[0]
        return values

    def trace(self, max_probes=100, tol=1e-3, seed=0):
        """Hutchinson's estimate (the mean of v.Hv over Rademacher probes, PyHessian's stopping rule)."""
        return float(np.mean(rs.trace(self.hvp, self.dim, max_probes, tol, seed)))

    def density(self, n_probes=1, iters=100, seed=0, num_bins=10000, sigma_squared=1e-5, overhead=0.01):
        """The Gaussian-smoothed spectral density on a grid, PyHessian's density() then density_generate():
        a frame with columns grid and density."""
        runs = rs.slq(self.hvp, self.dim, n_probes, iters, seed)
        density, grid = rs.density_on_grid(runs, num_bins, sigma_squared, overhead)
        return pl.DataFrame(dict(grid=grid, density=density))

    def edge_of_stability(self, lr, **kw):
        """lambda_max * lr / 2 with the top eigenvalue (computed now unless top_eigenvalues already ran)."""
        if self._top is None:
            self.top_eigenvalues(**kw)
        return rs.edge_of_stability(self._top, lr)


def report(model, instruments=None, prev=None, since=0, heads=None, top=5):
    """One record of every signal at a check: the instruments' per-group means since step `since`, the scan (name,
    kind, head, alpha, alpha_weighted, stable_rank, ks per row), the alpha range over layers, and against the
    previous scan `prev` the `top` rows whose alpha moved most relative to the other layers. Returns (record, scan)
    so the caller keeps the scan for the next call; record["values"] holds the flat numbers for a metrics log."""
    values = {}
    summary = instruments.summary(since=since) if instruments is not None else {}
    for group, metrics in summary.items():
        for k, v in metrics.items():
            values[f"signals_{group}_{k}"] = round(float(v), 6)
    cur = scan(model, heads=heads)
    layers = cur.filter(pl.col("head").is_null())
    if layers.height:
        values["signals_alpha_mean"] = round(float(layers["alpha"].mean()), 4)
        values["signals_alpha_min"] = round(float(layers["alpha"].min()), 4)
        values["signals_alpha_max"] = round(float(layers["alpha"].max()), 4)
    outliers = []
    if prev is not None:
        d = scan_delta(prev, cur).with_columns(pl.col("alpha_delta_z").abs().alias("z"))
        outliers = [
            dict(name=r["name"], head=r["head"], alpha=r["alpha"], delta=r["alpha_delta"], z=r["alpha_delta_z"])
            for r in d.sort("z", descending=True).head(top).iter_rows(named=True)
        ]
        if d.height:
            values["signals_alpha_delta_absmax"] = round(float(d["alpha_delta"].abs().max()), 5)
    record = dict(
        instruments=summary,
        scan=cur.select("name", "kind", "head", "alpha", "alpha_weighted", "stable_rank", "ks").to_dicts(),
        alpha_outliers=outliers,
        values=values,
    )
    return record, cur


__all__ = ["Hessian", "Instruments", "TempBalance", "layer_matrices", "report", "scan", "scan_delta"]
