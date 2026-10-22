"""The hybrid: Relay fills the gradients, OPAL's wrapped optimizer consumes them, and the agent owns a third program,
estimator.py, whose knobs are Relay's.

OPAL versions, swaps and reverts the modules its updater imports, so estimator.py rides on that: updater.py imports
it, the training loop reads settings(update, total_updates) from the running updater before each estimate, and the
fidelity check feeds the agent through opt.log(show_agent=True). Settings that fail validation or whose estimate
raises are reverted to the last settings that passed their trial window, and the agent is told.
"""

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


class RelayOpal:
    """Returned by relay_opal.wrap: step(x, y) estimates, checks fidelity when due, and steps the wrapped optimizer.

    opt          the OPAL AgentOptimizer (opt.log, opt.state_dict, opt.close, opt.updater)
    estimator    the RelayEstimator; its settings follow the agent's estimator.py
    settings     the settings in force for the next estimate
    """

    def __init__(self, opt, estimator, fidelity_every, trial_steps, log):
        self.opt, self.estimator = opt, estimator
        self.fidelity_every, self.trial_steps = fidelity_every, trial_steps
        self.settings = self.fallback = estimator.s
        self.trial_until = 0
        self.history = []
        self.fidelity = []
        self._log = log
        self._wall = None

    @property
    def update(self):
        return self.opt.update

    def _notify(self, text):
        self._log(f"relay_opal: {text}")
        self.history.append(dict(update=self.update, note=text))

    def _settings_from_program(self, update):
        fn = getattr(self.opt.updater, "estimator_settings", None)
        if fn is None:
            return self.fallback, None
        try:
            values = fn(update, self.opt.total)
            if not isinstance(values, dict):
                raise TypeError(f"settings() must return a dict, not {type(values).__name__}")
            merged = {**self.fallback.to_dict(), **values}
            return RelaySettings.from_dict(merged).validate(self.estimator.spec.hidden), None
        except Exception as exc:
            return self.fallback, f"{type(exc).__name__}: {exc}"[:400]

    def step(self, x, y, micro_batches=1):
        """One update: Relay's estimate on (x, y) (split into micro_batches along the batch dimension and averaged),
        a fidelity check on the first micro-batch when due, then the wrapped optimizer's step. Returns the loss."""
        update = self.update + 1
        settings, error = self._settings_from_program(update)
        if error:
            self.opt.log(estimator_settings_rejected=1.0, show_agent=True)
            self._notify(f"update {update}: estimator settings rejected ({error}); keeping {self.fallback.to_dict()}")
        if settings != self.settings:
            self._notify(f"update {update}: estimator settings now {settings.to_dict()}")
            self.history.append(dict(update=update, settings=settings.to_dict()))
            self.settings = settings
            self.trial_until = update + self.trial_steps
        elif settings != self.fallback and update > self.trial_until:
            self.fallback = settings
        t0 = time.perf_counter()
        try:
            loss = self._estimate(x, y, settings, micro_batches)
        except Exception as exc:
            if settings == self.fallback:
                raise
            self.opt.log(estimator_settings_reverted=1.0, show_agent=True)
            self._notify(
                f"update {update}: the estimate failed under {settings.to_dict()} ({type(exc).__name__}: "
                f"{str(exc)[:300]}); reverting to {self.fallback.to_dict()}"
            )
            self.settings = self.fallback
            loss = self._estimate(x, y, self.fallback, micro_batches)
        self._wall = time.perf_counter() - t0
        if self.fidelity_every and (update == 1 or update % self.fidelity_every == 0):
            self.check_fidelity(x[: max(1, x.shape[0] // micro_batches)], y[: max(1, y.shape[0] // micro_batches)])
        return self.opt.step(loss=loss)

    def _estimate(self, x, y, settings, micro_batches):
        self.estimator.configure(settings)
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

    def log(self, values=None, **kw):
        return self.opt.log(values, **kw)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state):
        return self.opt.load_state_dict(state)

    def close(self, interrupted=False):
        self.estimator.close()
        return self.opt.close(interrupted=interrupted)


def starting_program(settings, folder):
    """Write the three starting program files (updater.py, estimator.py, analysis.py) into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "updater.py").write_text((PROGRAM / "updater.py").read_text())
    (folder / "analysis.py").write_text((PROGRAM / "analysis.py").read_text())
    starting = json.dumps(settings.to_dict(), indent=4).replace("true", "True").replace("false", "False")
    (folder / "estimator.py").write_text(ESTIMATOR_TEMPLATE.format(starting=starting))
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
    log=print,
    **opal_kwargs,
):
    """Relay + OPAL. optimizer: the training script's torch optimizer(s); model: the model estimator estimates for;
    estimator: a RelayEstimator; goal, total_steps, scheduler and every other keyword go to opal.wrap unchanged
    (backend="keep" is bit-identical to Relay + the plain optimizer; the agent needs max_cost_usd).
    fidelity_every: run the autograd fidelity check on the current batch at update 1 and every N updates, and show it
    to the agent (0: never). trial_steps: updates new estimator settings run on trial before a failure reverts to the
    starting settings rather than to them. task_facts: extra facts for the agent's prompt."""
    s = estimator.s
    program = starting_program(s, program_dir or tempfile.mkdtemp(prefix="relay-opal-program-"))
    facts = (
        f"The gradients are Relay's forward-only estimates, with routes={s.routes} of {estimator.spec.hidden} MLP "
        f"units, keys={s.keys!r}, values={s.values!r}, hidden={s.hidden!r} at the start (estimator.py)."
    )
    if fidelity_every:
        facts += f" The fidelity check runs at update 1 and every {fidelity_every} updates."
    if task_facts:
        facts = str(task_facts).strip() + "\n" + facts
    opt = opal.wrap(
        optimizer,
        model,
        scheduler=scheduler,
        total_steps=total_steps,
        goal=goal,
        program=str(program),
        prompt=str(PROMPT),
        task_facts=facts,
        **opal_kwargs,
    )
    return RelayOpal(opt, estimator, fidelity_every, trial_steps, log)
