"""The parts of mode="ptx" that run without a GPU: starting program files, the consequence report, checkpoints and
request_restore, the contract in the prompt."""

import torch

import relay_opal
from relay_opal import RelaySettings
from relay_opal.opal.wrap import PROMPT_PTX, starting_program
from relay_opal.ptx import request_restore

from .test_opal import batches, fresh


def test_ptx_starting_program_and_prompt(tmp_path):
    folder = starting_program(RelaySettings(routes=64), tmp_path / "program", mode="ptx")
    updater = (folder / "updater.py").read_text()
    estimator = (folder / "estimator.py").read_text()
    assert "adamw" in updater and "import estimator" in updater
    assert "KERNELS" in estimator and "hidden_all" in estimator
    text = PROMPT_PTX.read_text()
    for needle in ("PtxModule", "request_restore", "sm_90", "ptxas", "KERNELS", "CHECKPOINT_EVERY", "{goal}"):
        assert needle in text


def test_consequence_report_checkpoint_and_restore(tmp_path):
    model, est, opt = fresh()
    notes = []
    loop = relay_opal.wrap(
        opt,
        model,
        est,
        goal="g",
        total_steps=6,
        backend="keep",
        every=2,
        summarize=False,
        mode="python",
        checkpoint_every=2,
        checkpoint_dir=tmp_path / "ckpt",
        run_dir=tmp_path / "run",
        log=notes.append,
    )
    loop.every = 2  # the consequence report every 2 updates, as mode="ptx" sets it
    seen = {}
    for i, (x, y) in enumerate(batches(6), 1):
        if i == 4:
            seen["before"] = {n: p.detach().clone() for n, p in model.named_parameters()}
        loop.step(x, y)
        loop.opt.zero_grad(set_to_none=True)
        if i == 4:
            seen["at4"] = {n: p.detach().clone() for n, p in model.named_parameters()}
            request_restore(4)  # restores the checkpoint of update 4 after update 5
    loop.close()
    report = [h["consequences"] for h in loop.history if "consequences" in h]
    assert [r["params_nan"] for r in report] == [0, 0, 0]
    assert report[1]["weight_norm_drift"] is not None and report[0]["update_seconds"] is not None
    assert (tmp_path / "ckpt" / "ckpt_4.pt").exists() and (tmp_path / "ckpt" / "latest.pt").exists()
    restored = [h for h in loop.history if "restored the checkpoint of update 4" in h.get("note", "")]
    assert restored
    assert any("restored_to_update" in m for m in loop.opt.metrics)
    # after the restore at update 5, update 6 started from the update-4 parameters, not from update 5's
    state = torch.load(tmp_path / "ckpt" / "ckpt_4.pt", map_location="cpu", weights_only=False)
    name = "transformer.h.0.mlp.c_fc.weight"
    assert torch.equal(state["params"][name], seen["at4"][name])


def test_restore_without_checkpoint_is_reported(tmp_path):
    model, est, opt = fresh()
    loop = relay_opal.wrap(
        opt, model, est, goal="g", total_steps=2, backend="keep", summarize=False, run_dir=tmp_path / "run", log=print
    )
    for x, y in batches(2):
        request_restore(1)
        loop.step(x, y)
        loop.opt.zero_grad(set_to_none=True)
    loop.close()
    assert "found no checkpoint" in (tmp_path / "run" / "notes.txt").read_text()
