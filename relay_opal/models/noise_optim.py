"""The Noise record's optimizer (ANVIL II on block matrices, Adam elsewhere), from the Relay script."""

import torch

POLAR_MARGIN = 1.1
MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)
RUN = (1194, 240, 50, 514)


def polar(g):
    """Approximate orthogonalization with six quintic maps."""
    tall = g.size(0) > g.size(1)
    x = g.T if tall else g
    x = (x / (POLAR_MARGIN * x.norm() + 1e-6)).bfloat16()
    for a, b, c in MAPS:
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).float()


class Anvil(torch.optim.Optimizer):
    """ANVIL II from modded-nanogpt PR #360."""

    def __init__(
        self,
        params,
        lr,
        weight_decay,
        steps,
        lane_beta=0.9,
        beta_min=0.85,
        beta_max=0.93,
        slow_beta=0.98,
        fast_beta=0.85,
        fast_weight=0.4385,
    ):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.t = 0

    def rail_beta(self):
        lo, hi, cool = self.beta_min, self.beta_max, self.total - self.cooldown
        if self.t < self.warmup:
            return lo + (hi - lo) * self.t / self.warmup
        if self.t > cool:
            return hi - (hi - lo) * (self.t - cool) / self.cooldown
        return hi

    @torch.no_grad()
    def step(self, closure=None):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                lanes = 1 if p.size(0) >= p.size(1) else 0
                state = self.state[p]
                if not state:
                    state.update(
                        fast=torch.zeros_like(g),
                        slow=torch.zeros_like(g),
                        energy=torch.zeros_like(g.mean(lanes, keepdim=True)),
                    )
                fast, slow, energy = state["fast"], state["slow"], state["energy"]
                fast.lerp_(g, 1 - fast_beta)
                slow.lerp_(g, 1 - self.slow_beta)
                update = polar(g.lerp(weight * fast + (1 - weight) * slow, beta))
                energy.lerp_(update.square().mean(lanes, keepdim=True), 1 - self.lane_beta)
                gained = update * energy.clamp_min(1e-10).rsqrt()
                update = gained * (update.norm() / gained.norm().clamp_min(1e-10))
                w = p.float()
                decay = (slow * w >= 0) * w * (wd * lr)
                p.copy_(w - lr * max(1, p.size(0) / p.size(1)) ** 0.5 * (update + decay))
        self.t += 1

    def state_dict(self):
        state = super().state_dict()
        state["t"] = self.t
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.t = state_dict.pop("t", 0)
        super().load_state_dict(state_dict)


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


def make_optimizer(model, steps, lr=0.003, betas=(0.7, 0.95), anvil_lr=0.023, anvil_wd=2.25):
    """ANVIL II on block matrices, Adam elsewhere (embeddings 300x the rate, mixing scalars 30x). Returns the two
    torch optimizers in a list, in the order OPAL's wrap takes them."""
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith("transformer.h.") and parameter.ndim == 2 and "ve_gate" not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ("wte" in name or "value_embeds" in name) else 30 if "lambdas" in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW(
        [{"params": ps, "lr": lr * scale} for scale, ps in groups.items()], betas=betas, eps=1e-8, weight_decay=0.0
    )
    return [Anvil(matrices, anvil_lr, anvil_wd, steps), adam]
