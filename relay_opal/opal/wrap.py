"""The hybrid: Relay fills the gradients, OPAL's wrapped optimizer consumes them, and the agent owns a third program,
estimator.py, whose knobs are Relay's.

OPAL versions, swaps and reverts the modules its updater imports, so estimator.py rides on that: updater.py imports
it, the training loop reads settings(update, total_updates) (and KERNELS, replacement step functions) from the
running updater before each estimate, and the fidelity check feeds the agent through opt.log(show_agent=True).
Settings that fail validation or whose estimate raises are reverted to the last settings that passed their trial
window, and the agent is told. In mode="ptx" the starting programs carry PTX kernels, the prompt states that nothing
protects the agent from its own kernels, every check reports the consequences (NaN and inf counts, weight-norm
drift, seconds per site type, load errors), and the agent gets checkpoints it can restore from.
"""

import copy
import json
import tempfile
import time
from pathlib import Path

import opal
import torch

from ..fidelity import check
from ..relay.settings import RelaySettings

PROGRAM = Path(__file__).parent / "program"
PROMPT = Path(__file__).parent / "prompt.md"
PROMPT_PTX = Path(__file__).parent / "prompt_ptx.md"

ESTIMATOR_TEMPLATE = '''"""Estimator: Relay's knobs for the next update.

The training script calls settings(update, total_updates) before every gradient estimate and runs Relay with what it
returns (a key left out keeps the starting value; an invalid value or a failing estimate reverts the whole dict to
the last settings that worked, and you are told).

Starting program = the settings the run was launched with, constant over training. Write any schedule as a function
of update and total_updates. The prompt lists every key, what it costs and what it does to fidelity.
"""

STARTING = {starting}


def settings(update, total_updates):
    return dict(STARTING)
'''

ESTIMATOR_PTX_TEMPLATE = '''"""Estimator: Relay's knobs for the next update, and its step functions.

The training script calls settings(update, total_updates) before every gradient estimate and runs Relay with what it
returns (a key left out keeps the starting value; an invalid value or a failing estimate reverts the whole dict to
the last settings that worked, and you are told). It also installs KERNELS: a dict from a step name of
relay_opal.relay.kernels to a replacement with the same signature, in Python, Triton or PTX. Nothing checks them.

Starting program = the settings the run was launched with, and Relay's hidden-unit step as the PTX kernel
relay_opal.ptx_seeds.HIDDEN_ALL_PTX (bit-identical to its eager twin at the start). The PTX source is importable
from relay_opal.ptx_seeds; copy it here and edit it, or write your own with relay_opal.ptx.PtxModule.
"""

from relay_opal.ptx_seeds import hidden_all

STARTING = {starting}

KERNELS = {{"hidden_all": hidden_all}}


def settings(update, total_updates):
    return dict(STARTING)
'''


def count_nonfinite(tensors):
    nan = inf = 0
    for t in tensors:
        if t is None or not t.is_floating_point():
            continue
        nan += int(torch.isnan(t).sum())
        inf += int(torch.isinf(t).sum())
    return nan, inf


def _flatten_state(obj, out):
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_state(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_state(v, out)
    return out


class RelayOpal:
    """Returned by relay_opal.wrap: step(x, y) estimates, checks fidelity when due, and steps the wrapped optimizer.

    opt          the OPAL AgentOptimizer (opt.log, opt.state_dict, opt.close, opt.updater)
    estimator    the RelayEstimator; its settings and kernels follow the agent's estimator.py
    settings     the settings in force for the next estimate
    """

    def __init__(self, opt, estimator, fidelity_every, trial_steps, log, every, checkpoint_every, checkpoint_dir):
        self.opt, self.estimator = opt, estimator
        self.fidelity_every, self.trial_steps = fidelity_every, trial_steps
        self.every, self.checkpoint_every = every, checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.settings = self.fallback = estimator.s
        self.kernels = {}
        self.trial_until = 0
        self.history = []
        self.fidelity = []
        self.checkpoints = {}
        self._log = log
        self._wall = None
        self._update_seconds = None
        self._norm_at_check = None
        self.named = dict(estimator.model.named_parameters())

    @property
    def update(self):
        return self.opt.update

    def _notify(self, text):
        self._log(f"relay_opal: {text}")
        self.history.append(dict(update=self.update, note=text))

    def say(self, text):
        """A note the agent reads at its next decision (what `opal say` writes)."""
        run = getattr(self.opt, "run", None)
        if run is not None:
            with (run.dir / "notes.txt").open("a") as f:
                f.write(text.strip() + "\n")

    def _program(self, update):
        """Settings and kernels from the running updater's estimator module; errors fall back and are reported."""
        up = self.opt.updater
        fn = getattr(up, "estimator_settings", None)
        kernels = getattr(up, "estimator_kernels", None)
        if kernels is None:
            kernels = self.kernels
        if fn is None:
            return self.fallback, kernels, None
        try:
            values = fn(update, self.opt.total)
            if not isinstance(values, dict):
                raise TypeError(f"settings() must return a dict, not {type(values).__name__}")
            merged = {**self.fallback.to_dict(), **values}
            return RelaySettings.from_dict(merged).validate(self.estimator.spec.hidden), kernels, None
        except Exception as exc:
            return self.fallback, kernels, f"{type(exc).__name__}: {exc}"[:400]

    def step(self, x, y, micro_batches=1):
        """One update: Relay's estimate on (x, y) (split into micro_batches along the batch dimension and averaged),
        a fidelity check on the first micro-batch when due, the consequence report when due, then the wrapped
        optimizer's step. Returns the loss."""
        update = self.update + 1
        settings, kernels, error = self._program(update)
        if error:
            self.opt.log(estimator_settings_rejected=1.0, show_agent=True)
            self.say(f"update {update}: estimator settings rejected ({error}); keeping {self.fallback.to_dict()}")
            self._notify(f"update {update}: estimator settings rejected ({error}); keeping {self.fallback.to_dict()}")
        if settings != self.settings:
            self._notify(f"update {update}: estimator settings now {settings.to_dict()}")
            self.history.append(dict(update=update, settings=settings.to_dict()))
            self.settings = settings
            self.trial_until = update + self.trial_steps
        elif settings != self.fallback and update > self.trial_until:
            self.fallback = settings
        if kernels is not self.kernels and dict(kernels) != self.kernels:
            self.history.append(dict(update=update, kernels=sorted(kernels)))
            self._notify(f"update {update}: estimator kernels now {sorted(kernels)}")
        self.kernels = dict(kernels)
        t0 = time.perf_counter()
        try:
            loss = self._estimate(x, y, settings, micro_batches)
        except Exception as exc:
            if settings == self.fallback and not self.kernels:
                raise
            text = f"{type(exc).__name__}: {str(exc)[:1500]}"
            self.opt.log(estimator_settings_reverted=1.0, show_agent=True)
            self.say(
                f"update {update}: the estimate failed ({text}); "
                f"reverting to {self.fallback.to_dict()} and the eager kernels"
            )
            self._notify(f"update {update}: the estimate failed under {settings.to_dict()} ({text[:300]}); reverting")
            self.settings, self.kernels = self.fallback, {}
            loss = self._estimate(x, y, self.fallback, micro_batches)
        self._wall = time.perf_counter() - t0
        if self.fidelity_every and (update == 1 or update % self.fidelity_every == 0):
            self.check_fidelity(x[: max(1, x.shape[0] // micro_batches)], y[: max(1, y.shape[0] // micro_batches)])
        if self.every and update % self.every == 0:
            self.report_consequences(update)
        t1 = time.perf_counter()
        out = self.opt.step(loss=loss)
        self._update_seconds = time.perf_counter() - t1
        if self.checkpoint_every and update % self.checkpoint_every == 0:
            self.checkpoint(update)
        self._restore_if_requested(update)
        return out

    def _estimate(self, x, y, settings, micro_batches):
        self.estimator.configure(settings, kernels=self.kernels)
        total = torch.zeros((), device=x.device)
        n = max(1, micro_batches)
        for m, (xs, ys) in enumerate(zip(x.chunk(n), y.chunk(n))):
            total += self.estimator.estimate(xs, ys, accumulate=m > 0, scale=1.0 / n)
        return total / n

    def check_fidelity(self, x, y):
        """Relay against float32 autograd on (x, y), with the settings in force; shown to the agent and recorded."""
        report = check(self.estimator, x, y)
        values = {f"fidelity_{k}": v for k, v in report["cosine"].items()}
        values.update({f"estimator_s_{k}": v for k, v in report["seconds"].items()})
        values["estimator_s_total"] = report["estimate_seconds"]
        if self._wall is not None:
            values["estimator_wall_s_per_update"] = round(self._wall, 4)
        self.opt.log(values, show_agent=True)
        self.fidelity.append(dict(update=self.update + 1, settings=self.settings.to_dict(), **report))
        return report

    def report_consequences(self, update):
        """What the programs did to the run since the previous check: non-finite counts in parameters, gradients and
        the updater's state, weight-norm drift, seconds of the estimate and of the update."""
        params = list(self.named.values())
        grads = [p.grad for p in params]
        state = []
        try:
            _flatten_state(self.opt.updater.state_dict(), state)
        except Exception as exc:
            self.say(f"update {update}: the updater's state_dict() raised {type(exc).__name__}: {str(exc)[:300]}")
        nan_p, inf_p = count_nonfinite(params)
        nan_g, inf_g = count_nonfinite(grads)
        nan_s, inf_s = count_nonfinite(state)
        norm = float(torch.sqrt(sum((p.float() ** 2).sum() for p in params)))
        drift = None if self._norm_at_check is None else norm - self._norm_at_check
        self._norm_at_check = norm
        values = dict(
            params_nan=nan_p,
            params_inf=inf_p,
            grads_nan=nan_g,
            grads_inf=inf_g,
            state_nan=nan_s,
            state_inf=inf_s,
            weight_norm=round(norm, 4),
            weight_norm_drift=None if drift is None else round(drift, 4),
            estimate_seconds=None if self._wall is None else round(self._wall, 4),
            update_seconds=None if self._update_seconds is None else round(self._update_seconds, 4),
        )
        self.opt.log({k: v for k, v in values.items() if v is not None}, show_agent=True)
        self.history.append(dict(update=update, consequences=values))
        return values

    def checkpoint(self, update):
        """Parameters, the updater's state and the estimator's settings, kept in host memory (and on disk when a
        checkpoint_dir was given) for request_restore and for the restart after a crash."""
        state = dict(
            update=update,
            params={n: p.detach().to("cpu", copy=True) for n, p in self.named.items()},
            updater=copy.deepcopy(_to_cpu(self.opt.updater.state_dict())),
            settings=self.settings.to_dict(),
        )
        self.checkpoints = {update: state}  # the latest only in memory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state, opt=self.opt.state_dict()), self.checkpoint_dir / f"ckpt_{update}.pt")
            torch.save(dict(state, opt=self.opt.state_dict()), self.checkpoint_dir / "latest.pt")

    def restore(self, update):
        """Restore the latest checkpoint taken at or before `update`: parameters, updater state, settings. The
        programs stay as they are. Returns the restored checkpoint's update, or None if there is none."""
        candidates = [u for u in self.checkpoints if u <= update]
        if self.checkpoint_dir:
            candidates += [
                int(p.stem.split("_")[1])
                for p in self.checkpoint_dir.glob("ckpt_*.pt")
                if int(p.stem.split("_")[1]) <= update
            ]
        if not candidates:
            return None
        at = max(candidates)
        state = self.checkpoints.get(at) or torch.load(
            self.checkpoint_dir / f"ckpt_{at}.pt", map_location="cpu", weights_only=False
        )
        with torch.no_grad():
            for n, p in self.named.items():
                p.copy_(state["params"][n].to(p.device))
        try:
            self.opt.updater.load_state_dict(state["updater"])
        except Exception as exc:
            self.say(
                f"restore({update}): the parameters are back at update {at}, but the running updater could not load "
                f"the saved state ({type(exc).__name__}: {str(exc)[:300]}); its state is as it was"
            )
        self.settings = self.fallback = RelaySettings.from_dict(state["settings"])
        self.opt.log(restored_to_update=at, show_agent=True)
        self._notify(f"restored the checkpoint of update {at} (requested {update})")
        return at

    def _restore_if_requested(self, update):
        from ..ptx import take_restore_request

        wanted = take_restore_request()
        if wanted is not None:
            at = self.restore(wanted)
            if at is None:
                self.say(f"update {update}: request_restore({wanted}) found no checkpoint at or before {wanted}")

    def log(self, values=None, **kw):
        return self.opt.log(values, **kw)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state):
        return self.opt.load_state_dict(state)

    def close(self, interrupted=False):
        self.estimator.close()
        return self.opt.close(interrupted=interrupted)


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def starting_program(settings, folder, mode="python"):
    """Write the three starting program files (updater.py, estimator.py, analysis.py) into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    updater = "updater_ptx.py" if mode == "ptx" else "updater.py"
    (folder / "updater.py").write_text((PROGRAM / updater).read_text())
    (folder / "analysis.py").write_text((PROGRAM / "analysis.py").read_text())
    starting = json.dumps(settings.to_dict(), indent=4).replace("true", "True").replace("false", "False")
    template = ESTIMATOR_PTX_TEMPLATE if mode == "ptx" else ESTIMATOR_TEMPLATE
    (folder / "estimator.py").write_text(template.format(starting=starting))
    return folder


def wrap(
    optimizer,
    model,
    estimator,
    *,
    goal,
    total_steps,
    fidelity_every=0,
    trial_steps=20,
    scheduler=None,
    task_facts=None,
    program_dir=None,
    mode="python",
    checkpoint_every=0,
    checkpoint_dir=None,
    log=print,
    **opal_kwargs,
):
    """Relay + OPAL. optimizer: the training script's torch optimizer(s); model: the model estimator estimates for;
    estimator: a RelayEstimator; goal, total_steps, scheduler and every other keyword go to opal.wrap unchanged
    (backend="keep" is bit-identical to Relay + the plain optimizer; the agent needs max_cost_usd).
    fidelity_every: run the autograd fidelity check on the current batch at update 1 and every N updates, and show it
    to the agent (0: never). trial_steps: updates new estimator settings run on trial before a failure reverts to the
    starting settings rather than to them. task_facts: extra facts for the agent's prompt. mode="ptx": the starting
    programs carry PTX kernels (relay_opal.ptx_seeds), the prompt tells the agent it has complete control and no
    protection, every check reports consequences, and checkpoints every checkpoint_every updates (in memory, and
    under checkpoint_dir) back request_restore and the restart after a crash."""
    s = estimator.s
    program = starting_program(s, program_dir or tempfile.mkdtemp(prefix="relay-opal-program-"), mode)
    facts = (
        f"The gradients are Relay's forward-only estimates, with routes={s.routes} of {estimator.spec.hidden} MLP "
        f"units, keys={s.keys!r}, values={s.values!r}, hidden={s.hidden!r} at the start (estimator.py)."
    )
    every = opal_kwargs.get("every", 50)
    if fidelity_every:
        facts += f" The fidelity check runs at update 1 and every {fidelity_every} updates."
    if mode == "ptx":
        facts += (
            f" The consequence report runs every {every} updates; CHECKPOINT_EVERY = {checkpoint_every or 'never'}."
        )
    if task_facts:
        facts = str(task_facts).strip() + "\n" + facts
    prompt = PROMPT_PTX if mode == "ptx" else PROMPT
    text = prompt.read_text().replace("CHECKPOINT_EVERY", str(checkpoint_every or "no"))
    prompt_file = Path(program) / "prompt.md"
    prompt_file.write_text(text)
    opt = opal.wrap(
        optimizer,
        model,
        scheduler=scheduler,
        total_steps=total_steps,
        goal=goal,
        program=str(program),
        prompt=str(prompt_file),
        task_facts=facts,
        **opal_kwargs,
    )
    return RelayOpal(
        opt,
        estimator,
        fidelity_every,
        trial_steps,
        log,
        every if mode == "ptx" else 0,
        checkpoint_every,
        checkpoint_dir,
    )
