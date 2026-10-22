"""Relay: backward induction over depth for every path, block-local measured credit (qlabs-eng/noise PR #20).

Measured from the top block down, every later credit is complete when a site is measured, so a site's moves only
need to cross the block directly above it and be scored against that block's measured output credit:
  top stream      the readout probe, a full sign-flipped Hadamard cycle through the head
  block i mid     a full cycle of moves of the mid stream through block i's MLP, scored against d_out_i
  attention       queries, keys, values and gate one local step from c_proj's view of d_mid_i (CutBI's layer step)
  block i input   d_in_i = d_mid_i + a full cycle of moves of the input through its norm, scored against u_i
  MLP hidden      every unit by its own central difference through relu^2, against d_out_i @ W_out (PR #15)
  embedding       a full cycle of moves of the raw embedding through its norm, scored against the credit of x0
  mixing scalars  their effect on the block inputs (linear)
Weight gradients are each site's credit times its cached input. Nothing here calls autograd.
"""

import time

import torch
from torch import nn
from torch.nn import functional as F

from . import kernels as K
from .kernels import EPS, hadamard, norm
from .settings import RelaySettings
from .spec import site_types


class Capture:
    """The clean forward's cached activations, and the orthogonal draws."""

    def __init__(self, spec, seed=0):
        self.spec, self.model = spec, spec.model
        self.device = spec.device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.frames = {}
        names = [spec.sites.wte, spec.sites.lm_head]
        for blk in spec.sites.blocks:
            names += [blk.c_q, blk.c_k, blk.c_v, blk.attn_proj, blk.c_fc, blk.mlp_proj]
            names += [n for n in (blk.ve_gate, blk.value_embeds) if n]
        self.modules = {n: self.model.get_submodule(n) for n in names}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
        self.capturing = False
        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer, blk in enumerate(spec.sites.blocks):
            self.hooks.append(self.model.get_submodule(blk.block).register_forward_pre_hook(self.block_hook(layer)))

    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output

        return hook

    def block_hook(self, layer):
        def hook(module, args):
            if self.capturing:
                self.blocks[layer] = args[0]

        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def signs(self, *shape, shared=False):
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1

    def orthogonal_noise(self, site, draws, chunk, n, b, t, d, dtype=None):
        """Draws chunk * n .. chunk * n + n of this step's `draws` at `site`: rows of a sign-flipped Hadamard matrix,
        orthogonal within each cycle."""
        if site not in self.frames:
            cycles = -(-draws // d)
            self.frames[site] = (
                self.signs(cycles, d, shared=True),
                torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator),
            )
        patterns, offset = self.frames[site]
        index = chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        a = hadamard(d, self.device, self.dtype)[rows] * patterns[index // d][:, None, None, :] * self.signs(n, b, t, 1)
        return a if dtype in (None, self.dtype) else a.to(dtype)

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()
        self.capturing = True
        try:
            self.clean_loss = self.spec.losses(x, y)
        finally:
            self.capturing = False
        self.x0 = norm(self.outputs[self.spec.sites.wte].float())

    def head_error(self, sigma):
        raw = self.outputs[self.spec.sites.lm_head][..., : self.spec.vocab_size]
        error = K.coordinate_differences(raw, self.y, sigma, self.spec.soft_cap)
        return F.pad(error, (0, self.outputs[self.spec.sites.lm_head].shape[-1] - raw.shape[-1]))

    def reassemble(self, site, error):
        """Weight gradient from output error and cached input."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]).float())
        return torch.einsum("btd,bti->di", error.float(), x)


class RelayEstimator:
    """Fills every parameter's .grad with Relay's estimate of the gradient of the mean next-token loss.

    spec        a RelaySpec (relay_opal.models.NoiseGPTSpec for the Noise GPT)
    settings    a RelaySettings; `routes` is the number of MLP units per token carrying a move (512 = H/4 for the
                Noise GPT, the submitted setting), the rest are the script's defaults
    seed        the sign draws
    profile     synchronise and time each site type (self.times, in seconds, reset at every estimate)
    """

    def __init__(self, spec, settings=None, seed=0, profile=False):
        self.spec, self.model = spec, spec.model
        self.s = (settings or RelaySettings()).validate(spec.hidden)
        self.seed, self.profile = seed, profile
        self.capture = Capture(spec, seed)
        self.device = self.capture.device
        self.generator = torch.Generator(self.device).manual_seed(seed + 17)
        self.width, self.hidden = spec.width, spec.hidden
        self.types = site_types(spec.sites)
        self.times, self._last = {}, 0.0
        self.maps = {}
        self.named = dict(self.model.named_parameters())

    def close(self):
        self.capture.close()

    def configure(self, settings):
        """Swap the knobs; validated before anything runs."""
        self.s = settings.validate(self.spec.hidden)
        return self

    # ----- helpers -----------------------------------------------------------------------------------------

    def tick(self, name):
        if not self.profile:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + now - self._last
        self._last = now

    def frozen(self, kind):
        return kind in self.s.freeze

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self, d=None):
        d = d or self.width
        return hadamard(d, self.device, torch.float32) * self.signs(d)

    def cycle(self, fn, x, per_token, shared=()):
        """A full sign-flipped Hadamard cycle of moves of x (n, W), each scored by fn: the measured credit of x.
        With settings.cycles > 1, the mean over independent cycles."""
        s = self.s
        steps = s.chain_sigma * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        credit = torch.zeros_like(x)
        for _ in range(s.cycles):
            H = self.basis()
            for t0 in range(0, len(x), s.tokens):
                sl = slice(t0, t0 + s.tokens)
                move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
                scale = steps[sl].expand(-1, self.width)
                credit[sl] += fn(move, x[sl], *[a[sl] for a in per_token], *shared, scale) @ H / self.width
        return credit / s.cycles

    def probe(self, x, y):
        """Each token's credit at the top stream, measured along a full cycle through the head."""
        s, spec = self.s, self.spec
        weight = self.named[spec.sites.lm_head + ".weight"].float()[: spec.vocab_size]
        n, vocab = x.shape[0], weight.shape[0]
        steps = s.probe_sigma * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        credit = torch.zeros(n, self.width, device=self.device)
        for _ in range(s.cycles):
            H = self.basis()
            if s.head == "measured":
                per = max(1, s.probe_chunk // (self.width * vocab))
                for t0 in range(0, n, per):
                    xs, ys = x[t0 : t0 + per], y[t0 : t0 + per]
                    m = xs.shape[0]
                    moved = (xs.unsqueeze(1) + steps[t0 : t0 + per].unsqueeze(-1) * H.unsqueeze(0)).reshape(
                        -1, self.width
                    )
                    yrep = ys.unsqueeze(1).expand(m, self.width).reshape(-1)
                    losses = K.readout_losses(moved, weight, yrep, spec.soft_cap).view(m, self.width)
                    clean = K.readout_losses(xs, weight, ys, spec.soft_cap).unsqueeze(1)
                    derivative = (losses - clean) / steps[t0 : t0 + per]
                    credit[t0 : t0 + per] += derivative @ H / self.width
                continue
            C = (weight @ H.T).T.contiguous()
            dir_sq = H.T.square().sum(0)
            per = max(1, s.probe_chunk // (self.width * vocab))
            for t0 in range(0, n, per):
                xs, ys = x[t0 : t0 + per], y[t0 : t0 + per]
                derivative = K.cycle_derivatives(
                    xs @ weight.T,
                    C,
                    ys,
                    steps[t0 : t0 + per],
                    xs.square().sum(-1, keepdim=True),
                    xs @ H.T,
                    dir_sq,
                    self.width,
                    spec.soft_cap,
                )
                credit[t0 : t0 + per] += derivative @ H / self.width
        return credit / s.cycles

    # ----- attention internals (CutBI's local step) ------------------------------------------------------

    def block_tensors(self, i):
        d, spec = self.capture, self.spec
        blk = spec.sites.blocks[i]
        b, t = d.batch, d.length
        h, hd = spec.n_head, spec.head_dim
        q_raw, k_raw, v_raw = (d.outputs[n].float().view(b, t, h, hd) for n in (blk.c_q, blk.c_k, blk.c_v))
        values = v_raw
        if blk.value_embeds:
            ve = d.outputs[blk.value_embeds].float().view(b, t, h, hd)
            z = d.outputs[blk.ve_gate].float()
            values = v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
        return q_raw, k_raw, values

    def attention_maps(self, i):
        """Layer i's clean attention A[b, h, s, t] and the matched filter K[b, h, t, s] = A[s, t] / max_t' A[s, t']."""
        if i in self.maps:
            return self.maps[i]
        d, spec = self.capture, self.spec
        blk = spec.sites.blocks[i]
        t, hd = d.length, spec.head_dim
        cos, sin = (z.float() for z in spec.rotary(t))
        q_raw, k_raw, _ = self.block_tensors(i)
        q, k = (norm(K.apply_rotary(z, cos, sin)) for z in (q_raw, k_raw))
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / hd**0.5
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= blk.window)), float("-inf"))
        A = logits.softmax(-1)
        del logits
        Km = (A / A.amax(-1, keepdim=True)).transpose(-1, -2)
        self.maps[i] = (A.to(d.dtype), Km.to(d.dtype).contiguous())
        return self.maps[i]

    def query_key_errors(self, i, kind, draws, target):
        """Query or key credit by the record's local step: draws move every position's raw query (or key), the
        attention reruns, each query's output change is scored against its target; key scores reach the moved
        positions through the matched filter."""
        d, spec = self.capture, self.spec
        blk = spec.sites.blocks[i]
        b, t = d.batch, d.length
        h, hd = spec.n_head, spec.head_dim
        cos, sin = (z.float() for z in spec.rotary(t))
        sigma = 0.1
        n = chunk = min(64, draws)
        dt = d.dtype
        site = blk.c_q if kind == "q" else blk.c_k
        A, Km = self.attention_maps(i)
        q_raw, k_raw, values = self.block_tensors(i)
        qp, kp = (norm(K.apply_rotary(z, cos, sin)) for z in (q_raw, k_raw))
        baseline = spec.attention(qp.to(dt), kp.to(dt), values.to(dt), blk.window)
        rep = lambda z_: z_.to(dt).unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
        fixed, vals = rep(kp if kind == "q" else qp), rep(values)
        raw = q_raw if kind == "q" else k_raw
        estimate = torch.zeros(b, t, h, hd, device=self.device)
        for c in range(max(1, draws // chunk)):
            a = d.orthogonal_noise(site, draws, c, n, b, t, self.width).view(n, b, t, h, hd)
            moved = K.moved_heads(raw, a, sigma, cos, sin)
            qa, ka = (moved, fixed) if kind == "q" else (fixed, moved)
            out = spec.attention(qa, ka, vals, blk.window).view(n, b, t, h, hd)
            score = K.head_scores(out, baseline, target.view(b, t, h, hd))
            if kind == "k":
                score = (Km.to(dt) @ score.to(dt)).float()
            estimate += K.code_estimate(score, a)
        return (estimate / (draws * sigma)).reshape(b, t, self.width)

    def exact_key_errors(self, i, target):
        """The record's single-key experiments, evaluated only on the query x key blocks where the causal window
        lets a query see a key (everywhere else the clean attention weight, hence every change, is zero)."""
        d, spec, s = self.capture, self.spec, self.s
        blk = spec.sites.blocks[i]
        b, t = d.batch, d.length
        h, hd = spec.n_head, spec.head_dim
        B = s.key_block if t % s.key_block == 0 and t > s.key_block else t
        cos, sin = (z[0, :t, 0].float() for z in spec.rotary(t))
        rot = lambda z: torch.cat(
            [z[..., : hd // 2] * cos + z[..., hd // 2 :] * sin, z[..., : hd // 2] * -sin + z[..., hd // 2 :] * cos], -1
        )
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        eye = torch.eye(hd, device=self.device)
        window = blk.window
        allowed = ((lag >= 0) & (lag <= window)).float()
        pairs = [
            (q0, k0)
            for q0 in range(0, t, B)
            for k0 in range(0, t, B)
            if q0 + B - 1 >= k0 and q0 - (k0 + B - 1) <= window
        ]
        q_all, k_all, v_all = self.block_tensors(i)
        credit_all = torch.zeros(b, h, t, hd, device=self.device)
        for r in range(b):
            q_raw, k_raw, values = (z[r].transpose(0, 1) for z in (q_all, k_all, v_all))
            qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
            logits = (qp @ kp.transpose(1, 2)) / hd**0.5
            A = torch.softmax(logits.masked_fill(allowed == 0, float("-inf")), -1)
            hub = target[r].view(t, h, hd).transpose(0, 1)
            C = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed
            steps = s.key_sigma * k_raw.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            for j0 in range(0, hd, s.key_chunk):
                j1 = min(j0 + s.key_chunk, hd)
                for q0, k0 in pairs:
                    qs, ks = slice(q0, q0 + B), slice(k0, k0 + B)
                    credit_all[r, :, ks, j0:j1] += K.key_scores(
                        qp[:, qs],
                        kp[:, ks],
                        k_raw[:, ks],
                        steps[:, ks],
                        cos[ks],
                        sin[ks],
                        A[:, qs, ks],
                        C[:, qs, ks],
                        eye[j0:j1],
                    )
        return credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)

    def exact_values(self, i, target):
        """Values and gate, the local step in closed form: a value move at t reaches query s's output as A[s, t]
        times the move, so the value credit is sum_s A[s, t] target_s; a gate move scales the value embedding by the
        gate's slope (the record's central difference)."""
        d, spec = self.capture, self.spec
        blk = spec.sites.blocks[i]
        b, t = d.batch, d.length
        h, hd = spec.n_head, spec.head_dim
        A, _ = self.attention_maps(i)
        credit = (A.float().transpose(-1, -2) @ target.view(b, t, h, hd).transpose(1, 2)).transpose(1, 2)
        errors = {blk.c_v: credit.reshape(b, t, self.width)}
        if blk.value_embeds and not self.frozen("gate"):
            z = d.outputs[blk.ve_gate].float()
            ve = d.outputs[blk.value_embeds].float().view(b, t, h, hd)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[blk.ve_gate] = slope * (ve * credit).sum(-1)
        return errors

    def sampled_values(self, i, kind, draws, target):
        """Values ('v') or value-embedding gates ('gate') by the record's sampled local step: every position's value
        moves along its own code, the attention (linear in its values) carries the moves to every query's output,
        scored per head against the target and carried back by the matched filter."""
        d, spec = self.capture, self.spec
        blk = spec.sites.blocks[i]
        b, t = d.batch, d.length
        h, hd = spec.n_head, spec.head_dim
        bf = torch.bfloat16
        n = min(64, draws)
        A, Km = self.attention_maps(i)
        hub = target.view(b, t, h, hd)
        site = blk.c_v if kind == "v" else blk.ve_gate
        width = h if kind == "gate" else self.width
        w = width // h
        estimate = torch.zeros(b, t, h, w, device=self.device)
        if kind == "gate":
            z = d.outputs[blk.ve_gate].float()
            ve = d.outputs[blk.value_embeds].float().view(b, t, h, hd)
        for chunk in range(max(1, draws // n)):
            a = d.orthogonal_noise(site, draws, chunk, n, b, t, width, dtype=torch.float32)
            if kind == "v":
                move = a.view(n, b, t, h, hd)
            else:
                eps = 0.1
                slope = (torch.sigmoid(z + eps * a) - torch.sigmoid(z - eps * a)) / eps
                move = slope.unsqueeze(-1) * ve
            change = (A.to(bf) @ K.heads_first(move)).view(b, h, t, n, hd)
            score = K.value_scores(change, hub)
            credited = (Km.to(bf) @ score.to(bf)).float()
            estimate += K.code_estimate(credited, a.view(n, b, t, h, w))
        return {site: (estimate / draws).reshape(b, t, width)}

    def layer_internals(self, i, target, gradients):
        """Layer i's query, key, value, gate and value-embedding gradients, one local step from its attention-output
        credit target (b, t, W). Returns the credit u (n, W) of the layer's normed input stream: each error through
        its projection."""
        d, s, spec = self.capture, self.s, self.spec
        blk = spec.sites.blocks[i]
        b, t = d.batch, d.length
        errors = {}
        if not self.frozen("q"):
            errors[blk.c_q] = self.query_key_errors(i, "q", s.query_draws, target)
        self.tick("q")
        if not self.frozen("k"):
            if s.keys == "exact":
                errors[blk.c_k] = self.exact_key_errors(i, target)
            else:
                errors[blk.c_k] = self.query_key_errors(i, "k", s.key_draws, target)
        self.tick("k")
        if not self.frozen("v"):
            if s.values == "closed_form":
                errors.update(self.exact_values(i, target))
            else:
                errors.update(self.sampled_values(i, "v", s.value_draws, target))
                if blk.value_embeds and not self.frozen("gate"):
                    errors.update(self.sampled_values(i, "gate", s.gate_draws, target))
        self.tick("v")
        u = torch.zeros(b * t, self.width, device=self.device)
        for site, error in errors.items():
            gradients[site + ".weight"] = d.reassemble(site, error)
            w = d.modules[site].weight.float()
            u[:, : w.shape[1]] += error.reshape(b * t, -1) @ w
        if blk.value_embeds and blk.c_v in errors and not self.frozen("value_embeds"):
            gate = 2 * torch.sigmoid(d.outputs[blk.ve_gate].float())
            ve_error = (errors[blk.c_v].view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
            gradients[blk.value_embeds + ".weight"] = d.reassemble(blk.value_embeds, ve_error)
        return u

    def hidden_credit(self, u, g, out_weight):
        s = self.s
        if s.hidden == "all":
            steps = s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            return K.hidden_all(u, steps, g @ out_weight.float())
        k = s.hidden_topk
        Hk = hadamard(k, self.device, torch.float32) * self.signs(k)
        top = u.topk(k, dim=-1, sorted=False).indices
        uk = u.gather(1, top)
        steps = s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        columns = out_weight.to(torch.bfloat16).T.contiguous()
        credit = torch.zeros_like(u)
        per = max(1, 2**26 // (k * self.width))
        for t0 in range(0, u.shape[0], per):
            ut, st, tt, gt = uk[t0 : t0 + per], steps[t0 : t0 + per], top[t0 : t0 + per], g[t0 : t0 + per]
            cols = F.embedding(tt, columns)
            derivative = K.hidden_scores(ut, st, Hk, cols, gt)
            credit[t0 : t0 + per].scatter_(1, tt, derivative @ Hk / k)
        return credit

    # ----- the estimate ------------------------------------------------------------------------------------

    @torch.no_grad()
    def gradients(self, x, y):
        """Relay's estimate of the gradient of the summed loss over the batch, name -> tensor (float32)."""
        d, s, spec = self.capture, self.s, self.spec
        sites = spec.sites
        self.times, self.maps = {}, {}
        if self.profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._last = time.perf_counter()
        d.capture(x, y)
        self.tick("forward")
        L = len(sites.blocks)
        b, t = x.shape
        yf = y.flatten()
        W = self.width
        gradients = {}
        if not self.frozen("lm_head"):
            gradients[sites.lm_head + ".weight"] = d.reassemble(sites.lm_head, d.head_error(s.head_sigma))
        self.tick("lm_head")
        e = d.outputs[sites.wte].float().flatten(0, 1)
        x0 = norm(e)
        measured_norm, bf16 = s.norm == "measured", s.mlp_dtype == "bf16"
        xins, mids, outs, routes, w_fc, w_out_t = [], [], [], [], [], []
        for i, blk in enumerate(sites.blocks):
            xin = d.blocks[i].float().flatten(0, 1)
            mid = xin + d.outputs[blk.attn_proj].float().flatten(0, 1)
            wfc, wout = self.named[blk.c_fc + ".weight"].float(), self.named[blk.mlp_proj + ".weight"].float()
            hpre = F.linear(norm(mid), wfc)
            r = F.relu(hpre).square()
            out = mid + F.linear(r, wout)
            if s.routes:
                top = (r * wout.norm(dim=0)).topk(s.routes, dim=-1).indices
                routes.append((top, hpre.gather(1, top)))
            else:
                routes.append((None, hpre))
            xins.append(xin), mids.append(mid), outs.append(out)
            w_fc.append(wfc), w_out_t.append(wout.T.contiguous())
        self.tick("routes")
        g_top = self.probe(outs[-1], yf)
        self.tick("top")
        lam_r, lam_0 = self.named[sites.resid_lambdas].float(), self.named[sites.x0_lambdas].float()
        d_in = [None] * L
        scalars = torch.zeros(2, L, device=self.device)
        for i in reversed(range(L)):
            blk = sites.blocks[i]
            d_out = g_top if i == L - 1 else lam_r[i + 1] * d_in[i + 1]
            if s.routes:
                d_mid = self.cycle(
                    K.mlp_cycle,
                    mids[i],
                    [routes[i][0], routes[i][1], d_out],
                    [w_fc[i], w_out_t[i], measured_norm, bf16],
                )
            else:
                d_mid = self.cycle(
                    K.mlp_cycle_full, mids[i], [routes[i][1], d_out], [w_fc[i], w_out_t[i], measured_norm, bf16]
                )
            self.tick("mid")
            if not self.frozen("attn_proj"):
                gradients[blk.attn_proj + ".weight"] = d_mid.T @ d.inputs[blk.attn_proj].flatten(0, 1)
            target = (d_mid @ self.named[blk.attn_proj + ".weight"].float()).view(b, t, W)
            u = self.layer_internals(i, target, gradients)
            d_in[i] = d_mid + self.cycle(K.norm_cycle, xins[i], [u], [measured_norm])
            self.tick("input")
            if not self.frozen("mlp_proj"):
                gradients[blk.mlp_proj + ".weight"] = d_out.T @ d.inputs[blk.mlp_proj].flatten(0, 1)
            if not self.frozen("fc"):
                hidden = self.hidden_credit(
                    d.outputs[blk.c_fc].float().flatten(0, 1), d_out, self.named[blk.mlp_proj + ".weight"]
                )
                gradients[blk.c_fc + ".weight"] = hidden.T @ d.inputs[blk.c_fc].flatten(0, 1)
            self.tick("hidden")
            prev = x0 if i == 0 else outs[i - 1]
            scalars[:, i] = torch.stack([(d_in[i] * prev).sum(), (d_in[i] * x0).sum()])
        if not self.frozen("wte"):
            d_x0 = lam_r[0] * d_in[0] + sum(lam_0[i] * d_in[i] for i in range(L))
            emb = self.cycle(K.norm_cycle, e, [d_x0], [measured_norm])
            gradients[sites.wte + ".weight"] = d.reassemble(sites.wte, emb.view(b, t, W))
        self.tick("wte")
        if not self.frozen("lambdas"):
            gradients[sites.resid_lambdas], gradients[sites.x0_lambdas] = scalars[0], scalars[1]
        for name, p in self.named.items():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients

    @torch.no_grad()
    def estimate(self, x, y, accumulate=False, scale=1.0):
        """Fill .grad on every parameter with scale * the estimate of the mean-loss gradient over this batch
        (accumulate: add to the existing .grad). Returns the batch's clean mean loss as a 0-d tensor."""
        gradients = self.gradients(x, y)
        valid = (y != -1).sum().clamp(min=1)
        for name, p in self.named.items():
            g = (gradients[name] * (scale / valid)).to(p.dtype)
            if not torch.isfinite(g).all():
                raise FloatingPointError(f"Non-finite estimate at {name}")
            if accumulate and p.grad is not None:
                p.grad.add_(g)
            else:
                p.grad = g
        return self.capture.clean_loss[y != -1].mean()
