"""Updater: turns gradients into parameter updates. Runs on every rank after the gradients are estimated.

Starting program = the training script's first optimizer (ANVIL II on the block matrices, user_optimizers[0]) called
as it is, and the second one (AdamW on everything else, user_optimizers[1]) replaced by a hand-written PTX AdamW
kernel (relay_opal.ptx_seeds.ADAMW_PTX, one thread per element, grid-stride) with this program's own float32
moments. It reads each group's learning rate from the user AdamW at every step, so the training script's scheduler
still applies. relay_opal.ptx_seeds.adamw_reference is the eager twin it matches bit for bit at the start.

You may replace any part of this file, including the PTX: write kernels inline as strings, load them with
relay_opal.ptx.PtxModule(source) and launch them with module.launch(name, grid, block, [tensors, ints, floats]).
Nothing checks a kernel before it runs on the parameters.
"""

import estimator  # noqa: F401  (an edit to estimator.py installs a new updater with it)
import torch

from relay_opal.ptx_seeds import adamw


class Updater:
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opts = spec["user_optimizers"]
        self.scheds = spec["user_schedulers"]
        self.anvil, self.adam = self.opts[0], self.opts[1]
        self.clip = spec.get("clip_grad_norm")
        self.estimator_settings = estimator.settings
        self.estimator_kernels = getattr(estimator, "KERNELS", {})
        self.t = 0
        self.state = {}
        if previous is not None and getattr(previous, "state", None):
            self.t = int(getattr(previous, "t", 0))
            self.state = {k: {n: v.clone() for n, v in s.items()} for k, s in previous.state.items()}

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, grads, update):
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_([p for _, p in self.named], self.clip)
        self.anvil.step()
        self.t += 1
        for group in self.adam.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(id(p), {})
                if not st:
                    st["m"], st["v"] = (
                        torch.zeros_like(p, dtype=torch.float32),
                        torch.zeros_like(p, dtype=torch.float32),
                    )
                adamw(p, p.grad, st["m"], st["v"], group["lr"], (b1, b2), group["eps"], group["weight_decay"], self.t)

    def state_dict(self):
        order = [p for g in self.adam.param_groups for p in g["params"]]
        return dict(
            anvil=self.anvil.state_dict(),
            scheds=[s.state_dict() for s in self.scheds],
            t=self.t,
            adam=[{k: v for k, v in self.state.get(id(p), {}).items()} for p in order],
        )

    def load_state_dict(self, state):
        self.anvil.load_state_dict(state["anvil"])
        for s, st in zip(self.scheds, state["scheds"]):
            s.load_state_dict(st)
        self.t = int(state["t"])
        order = [p for g in self.adam.param_groups for p in g["params"]]
        self.state = {id(p): dict(st) for p, st in zip(order, state["adam"]) if st}


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
