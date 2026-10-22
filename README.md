# relay-opal

Relay is a forward-only credit-assignment estimator for pre-norm transformers, from Emanuele Rodolà's entry to the
Noise speedrun (qlabs-eng/noise PR #20). It never runs a backward pass: per update and per token it measures credit
from the top of the stack down with full sign-flipped Hadamard cycles of forward moves (through the head, through each
block's MLP scored against that block's already-measured output credit, through each block's input norm), takes the
attention internals by one local step from the block's mid-stream credit (queries by a full cycle per head, keys by
single-key experiments, values and gates in closed form), every MLP hidden unit by its own central difference, and
forms weight gradients as credit times cached input. On the Noise Track 1 GPT it matched backprop's validation loss
with the same optimizer at about four times the step time, with gradient cosine 0.977 at init and 0.998 at the end.
This package extracts that script into a library with an adapter protocol, ships the Track 1 GPT as the reference
adapter, and adds a per-site fidelity check against float32 autograd.

OPAL (xingjian-bai/opal) wraps a PyTorch optimizer so that a coding agent rewrites two programs while the model trains:
the updater that turns gradients into parameter changes and the analysis that measures what the agent wants to see.
relay-opal gives that agent a third program, `estimator.py`, whose knobs are Relay's (routes per token, cycles per
update, measured or closed-form steps per site type, exact or sampled keys, which sites to freeze), and feeds it the
fidelity report (cosine per site type, seconds per site type), so it can trade gradient fidelity against step time
against the loss trend. OPAL is a dependency, not a fork: its program model already versions, swaps and reverts the
modules an updater imports, so `estimator.py` rides on that.

## Layers

```
batch ──▶ model forward (cached activations) ──▶ RelayEstimator.estimate ──grads──▶ OPAL updater.py ──▶ parameters
                                                       ▲                                   ▲
                                      estimator.py ────┘  (knobs, read before every update) │
                                                                                            │
fidelity.check (float32 autograd on one batch, every N updates) ──cosine, seconds──▶ agent ─┴─▶ analysis.py
```

## Install

```bash
pip install git+https://github.com/GeorgePearse/relay-opal   # Python >= 3.10, torch >= 2.5, opal-optimizer 0.3.6
```

For the Track 1 example: `pip install "relay-opal[track1] @ git+https://github.com/GeorgePearse/relay-opal"`.

## Usage

```python
import relay_opal
from relay_opal.models import GPTConfig, NoiseGPTSpec, make_model, make_optimizer

model = make_model("cuda", seed=0, config=GPTConfig())                      # the Noise Track 1 GPT
estimator = relay_opal.RelayEstimator(NoiseGPTSpec(model), relay_opal.RelaySettings(routes=512), seed=0)
loop = relay_opal.wrap(make_optimizer(model, steps), model, estimator, goal="...", total_steps=steps,
                       fidelity_every=50, backend="keep")                  # backend="claude", max_cost_usd=15 for the agent
for x, y in batches:
    loss = loop.step(x, y)                                                   # estimate -> grads -> updater -> parameters
    loop.opt.zero_grad(set_to_none=True)
loop.close()
```

Without OPAL: `estimator.estimate(x, y)` fills `.grad` on every parameter and returns the batch's mean loss;
`relay_opal.check(estimator, x, y)` returns the per-site cosine against autograd and seconds per site type. Your own
model needs a `RelaySpec` (`relay_opal/relay/spec.py`): module names for Relay's sites and the model's own forward,
rotary tables and attention kernel. `relay_opal/models/noise_gpt.py` is the reference.

`backend="keep"` is bit-identical to Relay plus the plain optimizer (`tests/test_opal.py`). The agent's `estimator.py`
defines `settings(update, total_updates) -> dict`; the loop reads it before every estimate, validates it, and reverts
settings that fail validation or whose estimate raises to the last settings that passed their trial window, telling the
agent at its next decision. The prompt the agent sees is OPAL's neutral prompt plus an estimator section
(`relay_opal/opal/prompt.md`).

## PTX mode: the agent writes kernels, nothing stops it

`relay_opal.wrap(..., mode="ptx", checkpoint_every=50)` starts the agent from programs that already carry PTX: an
AdamW step kernel in `updater.py` and Relay's hidden-unit step in `estimator.py` (`relay_opal.ptx_seeds`, both
bit-identical to their eager twins), loaded through `relay_opal.ptx.PtxModule`, a ctypes loader on the CUDA driver
API that returns the ptxas text when assembly fails. `estimator.py` may define `KERNELS`, replacements for any step
of `relay_opal.relay.kernels`. There is no parity check, sandbox or fallback between a kernel the agent writes and
the parameters it runs on: a wrong index corrupts the weights, an illegal address kills the process. Consequences are
reported, not prevented: every check shows the agent NaN and inf counts in parameters, gradients and optimizer state,
the weight-norm drift, seconds per site type, the per-site cosine to autograd, and the exact ptxas or driver text of
a failed load; the loop checkpoints every N updates, the agent can call `relay_opal.ptx.request_restore(update)` from
its programs, and a dead process is restarted from the last checkpoint with the previous program and the crash text
in the agent's next message. The fourth Track 1 arm runs this.

## Results

Noise Track 1, the whole epoch (610 updates of 8 sequences), one GPU on Modal, seed 0, one run per arm. The
commands, hardware, time and cost per arm are in `examples/README.md`.

| arm | val loss | seconds per update | train time | fidelity (block matrices) | agent |
|---|---|---|---|---|---|
| backprop, same optimizer | 3.7079 | 0.058 | 35 s | 1 | none |
| relay (OPAL `keep`) | 3.7036 | 1.54 | 954 s | 0.980 at init, 0.998 from update 50 | none |
| relay + opal (Python programs) | 3.7051 | 1.70 between decisions, 2.75 with the waits | 1677 s | 0.999 / 0.998 | 12 decisions, 6 applied, USD 2.64 |
| relay + opal + ptx | PTX_ROW |

The Relay PR reports 3.7044 for its logged seed-0 run and 3.7081 for backprop on 8xH100; one GPU reproduces both to
the third decimal. The agent of the Python arm read the fidelity values, judged the estimator sound ("a fidelity
check below 0.95, or a loss spike, would make me revisit the estimator") and never edited `estimator.py`: its six
applied changes rewrote the analysis, took over the learning-rate schedule unchanged, halved and then cut to 0.3x the
rate of the 16 mixing scalars, and started the warmdown earlier. These are single-seed numbers; the Relay PR's own
spread (3.7107 ± 0.0056 over five seeds for Relay, 3.7122 ± 0.0089 over three for backprop) is larger than the gaps
between the arms.

## Limits

- Relay is an estimator for pre-norm transformer stacks of this family: rms norm without affine parameters, rotary
  attention with normed queries and keys, optional value embeddings with sigmoid gates, relu² MLP, a soft-capped
  linear head, learned mixing scalars at each block input. Other architectures need their own sites, which may not
  have closed-form steps.
- The closed-form steps need the clean forward's cached inputs and outputs at every site, so memory scales with the
  batch; the reference runs one or two sequences per estimate call and averages.
- The maintainers of noise closed the Relay PR: measured block by block against already-measured credit, it is
  analytic gradient in all but name. relay-opal makes no claim otherwise; it packages the estimator.
- Short runs give the agent little to react to: the Track 1 epoch is 610 updates, and OPAL's first decision alone can
  take minutes.
- `check` runs autograd (float32, on a copy of the model) to produce the reference; it is analysis, never training.

## Inspiration

- [Dust](https://github.com/qlabs-eng/dust) (Q Labs): the fixed model, data and score of the speedrun; the
  population-based estimator that Boost, Cut and Relay replaced.
- [Noise](https://github.com/qlabs-eng/noise): the speedrun for pretraining without backpropagation, its rules and
  its fixed Track 1 GPT, which this package ships as the reference adapter.
- [PR #11 Boost → Cut](https://github.com/qlabs-eng/noise/pull/11) (@spicylemonade): the base script, the
  readout probe, the coordinate head and the per-token full-cycle measurement design.
- [PR #15](https://github.com/qlabs-eng/noise/pull/15) and [PR #17 CutBI](https://github.com/qlabs-eng/noise/pull/17)
  (@Euraba): hidden-unit reassociation through the output column, and backward induction over depth with the
  attention step Relay builds on.
- [PR #20 Relay](https://github.com/qlabs-eng/noise/pull/20) (@erodola): the estimator itself, taken as the
  semantics this package keeps.
- [OPAL](https://github.com/xingjian-bai/opal) (Xingjian Bai and co-authors): the agent-in-the-loop optimizer and
  its program model, which `estimator.py` rides on unchanged.
- [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) (Martin and Mahoney): per-layer heavy-tailed
  self-regularisation metrics (alpha), planned as a per-layer signal for the agent's analysis program.
- [TempBalance](https://github.com/YefanZhou/TempBalance) ([arXiv:2312.00359](https://arxiv.org/abs/2312.00359),
  Zhou et al., NeurIPS 2023): per-layer learning rates set from alpha, the starting rule planned for the agent's
  updater once alpha is measured.
- [Cockpit](https://github.com/f-dangel/cockpit) (Schneider, Balles and Hennig): the list of per-layer training
  instruments the planned analysis program follows.
- [PyHessian](https://github.com/amirgholami/PyHessian) (Yao et al.): top Hessian eigenvalue and trace, the
  sharpness and edge-of-stability signal planned for the analysis program.

## Roadmap

Planned, not yet built: an analysis program for the agent with WeightWatcher's per-layer alpha, Cockpit-style
per-layer instruments and PyHessian-style sharpness, so that per-layer learning rates (TempBalance as the starting
rule) and the estimator's knobs can follow measured signals rather than the loss alone.

## Credits

- Relay: Emanuele Rodolà (@erodola), qlabs-eng/noise PR #20 and the release
  [relay-track1-v1](https://github.com/erodola/noise/releases/tag/relay-track1-v1) with `check_relay.py`.
- CutBI and hidden-unit reassociation: @Euraba, qlabs-eng/noise PRs #17 and #15.
- Boost → Cut, the base script: @spicylemonade, qlabs-eng/noise PR #11.
- Noise: Serdar Gülbahar, Bishwas Mandal, Akshay Vegesna and Samip Dahal (Q Labs); Dust: Q Labs.
- OPAL: Xingjian Bai, Jia-Hao Liang, Jayden Teoh and Vincent Sitzmann.

## Licence

MIT. Derived code keeps its upstream notices in `THIRD_PARTY_NOTICES.md`. See `CONTRIBUTING.md` and `CHANGELOG.md`.
