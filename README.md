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
| relay + opal + ptx (agent may write PTX, unprotected) | 3.7050 | 1.53 between decisions, 3.07 with the waits | 1872 s | 0.999 / 0.998 | 13 decisions, 7 applied, USD 2.87 |

The Relay PR reports 3.7044 for its logged seed-0 run and 3.7081 for backprop on 8xH100; one GPU reproduces both to
the third decimal. The agent of the Python arm read the fidelity values, judged the estimator sound ("a fidelity
check below 0.95, or a loss spike, would make me revisit the estimator") and never edited `estimator.py`: its six
applied changes rewrote the analysis, took over the learning-rate schedule unchanged, halved and then cut to 0.3x the
rate of the 16 mixing scalars, and started the warmdown earlier. These are single-seed numbers; the Relay PR's own
spread (3.7107 ± 0.0056 over five seeds for Relay, 3.7122 ± 0.0089 over three for backprop) is larger than the gaps
between the arms.

The path ledger of the PTX arm, the point of that experiment: the agent was told it could rewrite any of its three
programs in Python, Triton or PTX, that nothing would check a kernel before it ran on the weights, and that the run
report was how it would find out. Over 13 decisions it wrote 7 program versions, all Python edits of `updater.py`
and `analysis.py` (per-group learning-rate multipliers for q, k, the mixing scalars and the value-embedding gate,
then the warmdown's start and shape), kept the seed PTX AdamW kernel running under every version and never touched
`estimator.py`, so the seed hidden-unit kernel ran unchanged too. Programs written in PTX or Triton: 0. Failed to
assemble: 0. Ran and corrupted the weights: 0 (NaN and inf counts were 0 in all 61 consequence reports, the cosine
never fell below 0.997 on the block matrices). Reverted by itself, restored from a checkpoint, or crashed: 0. Kept:
all 7. Seconds per update of the estimator and of the update rule were the same at the end as before the agent's
first decision (1.49 s and 0.036 s), because it changed neither. Verdict: given complete control and no protection,
this agent took no path that could have corrupted anything; it reached the same loss as the Python arm by learning-
rate edits alone. Whether it learns from a corruption is untested, because it caused none. It also said at its first
decision that it could not see the Anvil optimizer's source or the warmdown's length (OPAL shows a `LambdaLR`
function's source but not its closure values), which is a fact about what it was shown, to state in `task_facts`
next time.

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

Reading list collected on 2026-10-22 while scoping forward-only credit assignment, agent-in-the-loop optimizers and
per-layer training signals. Each entry says what it is, what was measured if anything, and what it implies for
relay-opal.

### Credit assignment without backpropagation

| Work | What it is | Implication |
| --- | --- | --- |
| [Dust](https://github.com/qlabs-eng/dust) ([paper](https://qlabs.sh/research/dust)), Q Labs, Oct 2026 | Forward-only training by antithetic random perturbations. | A detector class-head pilot reached gradient cosine 0.998 but ran 1.3 to 1.75x slower than backprop with no loss or memory win. A reproduction of Dust's own 1M-token config ties backprop SGD only at population 1024 and 26x the time. Random search alone is not the lever. |
| [noise](https://github.com/qlabs-eng/noise), the no-backprop pretraining speedrun | Track 1 leaderboard of forward-only methods on a small GPT. | Where the structured estimators below were developed and measured against each other. Its fixed Track 1 GPT is the reference adapter this package ships. |
| [Boost -> Cut, PR #11](https://github.com/qlabs-eng/noise/pull/11), @spicylemonade | Two-phase forward-only record, val 3.945. | The base script Relay builds on; also the source of the backprop baseline and test split. |
| [CutBI, PR #17](https://github.com/qlabs-eng/noise/pull/17) and [hidden-unit reassociation, PR #15](https://github.com/qlabs-eng/noise/pull/15), @Euraba | Backward induction over depth for attention; reading every MLP hidden unit through its output column. | The structural ideas that moved the record from 3.95 to 3.73. |
| [Relay, PR #20](https://github.com/qlabs-eng/noise/pull/20), Emanuele Rodolà ([release](https://github.com/erodola/noise/releases/tag/relay-track1-v1)) | Backward induction applied to every path: nonlinear maps measured by sign-flipped Hadamard cycles, linear steps in closed form. Val 3.704 vs backprop 3.708 with the same optimizer, 149 s vs 38 s on 8xH100, gradient cosine 0.977 at init and 0.998 at the end. Closed by the maintainers as analytic gradient in all but name. | The estimator this package packages. The strategic use is training inside an inference-only runtime, where autograd is not available. |

### Agent in the optimizer loop

| Work | What it is | Implication |
| --- | --- | --- |
| [OPAL](https://github.com/xingjian-bai/opal) ([blog](https://xingjianbai.com/reactive-optimizers/)), Xingjian Bai | `opal.wrap` turns the optimizer into two programs, an updater and an analysis, that a coding agent rewrites at decision points during the run. Wins on SiT, RAE, LightningDiT and modded-nanogpt; not always. | Sits after any gradient source, so it composes with Relay. relay-opal adds a third program, the estimator, with Relay's knobs. |

### Per-layer signals an agent could act on

| Work | What it is | Implication |
| --- | --- | --- |
| [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher), Martin and Mahoney | Heavy-tailed self-regularisation: the power-law exponent alpha and stable rank of each weight matrix's spectrum, read from the weights alone. | Gradient-free, so it is the one signal a forward-only estimator cannot distort. Read deltas and outliers between layers, not the absolute 2 to 6 band, on short runs; split attention per head; exclude 1x1 convolutions and output heads. |
| [TempBalance](https://arxiv.org/abs/2312.00359) ([code](https://github.com/YefanZhou/TempBalance)), Zhou, Pang, Liu, Martin, Mahoney, Yang, NeurIPS 2023 | Layer-wise learning rates set from per-layer alpha: lower for small alpha (hot, converged), higher for large alpha (cold, under-trained), total budget unchanged. Temperature here is the SGD noise scale, learning rate over batch size, times one over one minus momentum. | The starting rule for an agent that sets per-group learning rates from WeightWatcher. Shown with SGD on ResNets; under Adam the temperature is per coordinate and the clean picture weakens. |
| [Cockpit](https://github.com/f-dangel/cockpit), Schneider, Balles and Hennig | Training-time instruments per layer: gradient norm, update-to-parameter ratio, gradient signal-to-noise, cosine between consecutive gradients, loss curvature along the step. Built on BackPACK. | The right list of instruments, the wrong dependency: BackPACK hooks the backward pass, covers few layer types and no attention, and is quiet. Reimplement the cheap quantities from the gradients the optimizer already holds. Under Relay these inherit the estimator's error. |
| [PyHessian](https://github.com/amirgholami/PyHessian), Yao, Gholami et al. | Top Hessian eigenvalue, trace and spectral density by Hessian-vector products. | Sharpness times learning rate near 2 is the edge of stability, the most direct learning-rate ceiling there is. A few extra backward passes per decision point; needs double backprop, so backprop arms only. |
| [Deepchecks](https://github.com/deepchecks/deepchecks), [Captum](https://github.com/pytorch/captum) | Offline data and model validation; attribution. | Not optimizer signals. Listed so nobody re-evaluates them for this purpose. |

### How these fit together

The estimator produces gradients (backprop, or Relay). The optimizer turns them into updates (fixed, or
OPAL-rewritten). The per-layer signals tell the agent what the updates did: WeightWatcher for which layers to treat
differently, the Cockpit-style instruments for whether a layer's step is sized right, sharpness for the global
learning-rate ceiling.

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
