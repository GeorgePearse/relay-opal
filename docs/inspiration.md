# Training dynamics: inspiration list

Reading list collected on 2026-10-22 while scoping forward-only credit assignment, agent-in-the-loop optimizers and
per-layer training signals. Each entry says what it is, what was measured if anything, and what it implies for
relay-opal.

## Credit assignment without backpropagation

| Work | What it is | Implication |
| --- | --- | --- |
| [Dust](https://github.com/qlabs-eng/dust) ([paper](https://qlabs.sh/research/dust)), Q Labs, Oct 2026 | Forward-only training by antithetic random perturbations. | A detector class-head pilot reached gradient cosine 0.998 but ran 1.3 to 1.75x slower than backprop with no loss or memory win. A reproduction of Dust's own 1M-token config ties backprop SGD only at population 1024 and 26x the time. Random search alone is not the lever. |
| [noise](https://github.com/qlabs-eng/noise), the no-backprop pretraining speedrun | Track 1 leaderboard of forward-only methods on a small GPT. | Where the structured estimators below were developed and measured against each other. Its fixed Track 1 GPT is the reference adapter this package ships. |
| [Boost -> Cut, PR #11](https://github.com/qlabs-eng/noise/pull/11), @spicylemonade | Two-phase forward-only record, val 3.945. | The base script Relay builds on; also the source of the backprop baseline and test split. |
| [CutBI, PR #17](https://github.com/qlabs-eng/noise/pull/17) and [hidden-unit reassociation, PR #15](https://github.com/qlabs-eng/noise/pull/15), @Euraba | Backward induction over depth for attention; reading every MLP hidden unit through its output column. | The structural ideas that moved the record from 3.95 to 3.73. |
| [Relay, PR #20](https://github.com/qlabs-eng/noise/pull/20), Emanuele Rodolà ([release](https://github.com/erodola/noise/releases/tag/relay-track1-v1)) | Backward induction applied to every path: nonlinear maps measured by sign-flipped Hadamard cycles, linear steps in closed form. Val 3.704 vs backprop 3.708 with the same optimizer, 149 s vs 38 s on 8xH100, gradient cosine 0.977 at init and 0.998 at the end. Closed by the maintainers as analytic gradient in all but name. | The estimator this package packages. The strategic use is training inside an inference-only runtime, where autograd is not available. |

## Agent in the optimizer loop

| Work | What it is | Implication |
| --- | --- | --- |
| [OPAL](https://github.com/xingjian-bai/opal) ([blog](https://xingjianbai.com/reactive-optimizers/)), Xingjian Bai | `opal.wrap` turns the optimizer into two programs, an updater and an analysis, that a coding agent rewrites at decision points during the run. Wins on SiT, RAE, LightningDiT and modded-nanogpt; not always. | Sits after any gradient source, so it composes with Relay. relay-opal adds a third program, the estimator, with Relay's knobs. |

## Per-layer signals an agent could act on

| Work | What it is | Implication |
| --- | --- | --- |
| [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher), Martin and Mahoney | Heavy-tailed self-regularisation: the power-law exponent alpha and stable rank of each weight matrix's spectrum, read from the weights alone. | Gradient-free, so it is the one signal a forward-only estimator cannot distort. Read deltas and outliers between layers, not the absolute 2 to 6 band, on short runs; split attention per head; exclude 1x1 convolutions and output heads. |
| [TempBalance](https://arxiv.org/abs/2312.00359) ([code](https://github.com/YefanZhou/TempBalance)), Zhou, Pang, Liu, Martin, Mahoney, Yang, NeurIPS 2023 | Layer-wise learning rates set from per-layer alpha: lower for small alpha (hot, converged), higher for large alpha (cold, under-trained), total budget unchanged. Temperature here is the SGD noise scale, learning rate over batch size, times one over one minus momentum. | The starting rule for an agent that sets per-group learning rates from WeightWatcher. Shown with SGD on ResNets; under Adam the temperature is per coordinate and the clean picture weakens. |
| [Cockpit](https://github.com/f-dangel/cockpit), Schneider, Balles and Hennig | Training-time instruments per layer: gradient norm, update-to-parameter ratio, gradient signal-to-noise, cosine between consecutive gradients, loss curvature along the step. Built on BackPACK. | The right list of instruments, the wrong dependency: BackPACK hooks the backward pass, covers few layer types and no attention, and is quiet. Reimplement the cheap quantities from the gradients the optimizer already holds. Under Relay these inherit the estimator's error. |
| [PyHessian](https://github.com/amirgholami/PyHessian), Yao, Gholami et al. | Top Hessian eigenvalue, trace and spectral density by Hessian-vector products. | Sharpness times learning rate near 2 is the edge of stability, the most direct learning-rate ceiling there is. A few extra backward passes per decision point; needs double backprop, so backprop arms only. |
| [Deepchecks](https://github.com/deepchecks/deepchecks), [Captum](https://github.com/pytorch/captum) | Offline data and model validation; attribution. | Not optimizer signals. Listed so nobody re-evaluates them for this purpose. |

## How these fit together

The estimator produces gradients (backprop, or Relay). The optimizer turns them into updates (fixed, or
OPAL-rewritten). The per-layer signals tell the agent what the updates did: WeightWatcher for which layers to treat
differently, the Cockpit-style instruments for whether a layer's step is sized right, sharpness for the global
learning-rate ceiling.
