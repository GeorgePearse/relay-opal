import hashlib
import json

import pytest
import torch

import relay_opal
from relay_opal import RelayEstimator, RelaySettings
from relay_opal.models import GPTConfig, NoiseGPTSpec, make_model

from .conftest import TINY

STEPS = 6


def digest(model):
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        a = p.detach().cpu().contiguous()
        h.update(json.dumps([name, str(a.dtype), list(a.shape)]).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def batches(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        yield (
            torch.randint(0, TINY.vocab_size, (2, 64), generator=g),
            torch.randint(0, TINY.vocab_size, (2, 64), generator=g),
        )


def fresh():
    torch.manual_seed(0)
    model = make_model(torch.device("cpu"), 0, GPTConfig(**vars(TINY)))
    spec = NoiseGPTSpec(model)
    est = RelayEstimator(spec, RelaySettings(routes=64, tokens=32), seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, est, opt


def test_keep_backend_is_bit_identical_to_plain_relay(tmp_path):
    model, est, opt = fresh()
    losses = []
    for x, y in batches(STEPS):
        loss = est.estimate(x, y)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    plain = digest(model)

    model, est, opt = fresh()
    loop = relay_opal.wrap(
        opt,
        model,
        est,
        goal="a tiny GPT on random tokens; judged by nothing",
        total_steps=STEPS,
        backend="keep",
        summarize=False,
        run_dir=tmp_path / "run",
        log=lambda *_: None,
    )
    wrapped = []
    for x, y in batches(STEPS):
        wrapped.append(loop.step(x, y).item())
        loop.opt.zero_grad(set_to_none=True)
    loop.close()
    assert wrapped == losses
    assert digest(model) == plain


def test_micro_batches_average(tmp_path):
    model, est, opt = fresh()
    x, y = next(batches(1))
    est.estimate(x, y)
    whole = {n: p.grad.clone() for n, p in model.named_parameters()}
    model, est, opt = fresh()
    loop = relay_opal.wrap(opt, model, est, goal="g", total_steps=1, backend="keep", summarize=False, run_dir=tmp_path)
    loop._estimate(x, y, est.s, micro_batches=2)
    for n, p in model.named_parameters():  # other draws per micro-batch: the same gradient up to the estimate's noise
        if p.numel() > 2:
            cos = torch.nn.functional.cosine_similarity(p.grad.flatten(), whole[n].flatten(), dim=0)
            assert cos > 0.95, n
            assert abs(p.grad.norm() / whole[n].norm() - 1) < 0.2, n


NEW_ESTIMATOR = """
def settings(update, total_updates):
    return dict(routes=32, freeze=["k"]) if update >= 4 else dict(routes=32)
"""

BAD_ESTIMATOR = """
def settings(update, total_updates):
    return dict(routes=-5)
"""


def run_with_agent(tmp_path, agent, every=2, schedule=(2,), fidelity_every=0, steps=STEPS):
    model, est, opt = fresh()
    notes = []
    loop = relay_opal.wrap(
        opt,
        model,
        est,
        goal="a tiny GPT on random tokens; judged by nothing",
        total_steps=steps,
        backend=agent,
        every=every,
        schedule=list(schedule),
        summarize=False,
        fidelity_every=fidelity_every,
        run_dir=tmp_path / "run",
        log=notes.append,
    )
    seen = []
    for x, y in batches(steps):
        loop.step(x, y)
        loop.opt.zero_grad(set_to_none=True)
        seen.append(loop.settings.to_dict())
    loop.close()
    return loop, seen, notes


def test_agent_edit_to_estimator_takes_effect_next_update(tmp_path):
    def agent(workspace, message, system):
        (workspace / "estimator.py").write_text(NEW_ESTIMATOR)
        return "narrower routes, then frozen keys"

    loop, seen, notes = run_with_agent(tmp_path, agent)
    assert [s["routes"] for s in seen] == [64, 64, 32, 32, 32, 32]
    assert [s["freeze"] for s in seen] == [[], [], [], ["k"], ["k"], ["k"]]
    assert any("estimator settings now" in n for n in notes)


def test_invalid_settings_are_rejected_and_reported(tmp_path):
    def agent(workspace, message, system):
        (workspace / "estimator.py").write_text(BAD_ESTIMATOR)
        return "oops"

    loop, seen, notes = run_with_agent(tmp_path, agent)
    assert all(s["routes"] == 64 for s in seen)
    assert any("rejected" in n for n in notes)
    assert any("estimator_settings_rejected" in m for m in loop.opt.metrics)


def test_failing_estimate_reverts_to_fallback(tmp_path, monkeypatch):
    def agent(workspace, message, system):
        (workspace / "estimator.py").write_text(NEW_ESTIMATOR)
        return "narrower routes"

    original = RelayEstimator.configure

    def configure(self, settings):
        if settings.routes == 32:
            raise RuntimeError("pretend out of memory")
        return original(self, settings)

    monkeypatch.setattr(RelayEstimator, "configure", configure)
    loop, seen, notes = run_with_agent(tmp_path, agent)
    assert all(s["routes"] == 64 for s in seen)
    assert any("reverting" in n for n in notes)


def test_fidelity_is_shown_to_the_agent(tmp_path):
    messages = []

    def agent(workspace, message, system):
        messages.append(message)
        return "keeping"

    loop, seen, notes = run_with_agent(tmp_path, agent, every=2, schedule=(4,), fidelity_every=2)
    assert [f["update"] for f in loop.fidelity] == [1, 2, 4, 6]
    assert loop.fidelity[0]["cosine"]["block_matrices"] > 0.9
    shown = [k for m in loop.opt.metrics for k in m]
    assert "fidelity_block_matrices" in shown and "estimator_s_total" in shown
    assert messages and "fidelity_block_matrices" in messages[0]


def test_prompt_and_program_mention_the_estimator():
    text = relay_opal.opal.PROMPT.read_text()
    assert "estimator.py" in text and "fidelity_block_matrices" in text
    for field in ("{goal}", "{param_table}", "{starting_summary}", "{task_facts_section}"):
        assert field in text
    folder = relay_opal.opal.starting_program(RelaySettings(), pytest.importorskip("tempfile").mkdtemp())
    assert "import estimator" in (folder / "updater.py").read_text()
    ns = {}
    exec((folder / "estimator.py").read_text(), ns)
    assert ns["settings"](1, 10)["routes"] == 512
