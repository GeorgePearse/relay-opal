"""Updater: turns gradients into parameter updates. Runs on every rank after the gradients are estimated.

Starting program = the training script's own optimizer, unchanged (its full settings are in
reference/starting_optimizer.txt). The gradients come from Relay, whose knobs are set by estimator.py (imported here,
so an edit to estimator.py installs a new updater with it). You may replace any part of this file.
"""

import estimator
import torch


class Updater:
    # True: the training script's scheduler.step() calls still step the user's schedulers (as in the plain run).
    # A program that sets its own learning rates should leave this False.
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opts = spec["user_optimizers"]
        self.scheds = spec["user_schedulers"]
        self.clip = spec.get("clip_grad_norm")
        self.estimator_settings = estimator.settings  # read by the training script before every estimate

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, grads, update):
        # grads[name] is the same tensor as param.grad (None if the parameter got no gradient this update).
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_([p for _, p in self.named], self.clip)
        for o in self.opts:
            o.step()

    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts], scheds=[s.state_dict() for s in self.scheds])

    def load_state_dict(self, state):
        for o, s in zip(self.opts, state["opts"]):
            o.load_state_dict(s)
        for o, s in zip(self.scheds, state["scheds"]):
            o.load_state_dict(s)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
