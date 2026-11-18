# Examples

## track1: the Noise Track 1 reproduction

The 37.7M-parameter GPT of qlabs-eng/noise, one epoch of 4,880 sequences of 2,048 tokens, batch 8, the Relay script's
init, optimizer (ANVIL II on block matrices, Adam elsewhere), rates, warmdown and data order. Three arms, one flag each:

```bash
pip install "relay-opal[track1] @ git+https://github.com/GeorgePearse/relay-opal"
python examples/track1/prepare_data.py --output data              # FineWeb sample-10BT, the pinned split and hashes
python examples/track1/train.py --arm backprop  --output logs/backprop
python examples/track1/train.py --arm relay     --output logs/relay      --fidelity-every 50
python examples/track1/train.py --arm relay-opal --output logs/relay-opal --fidelity-every 50 --max-cost-usd 15
```

`--steps N` runs a short schedule (the warmdown scales with it); `--micro` sets sequences per Relay estimate call
(1 on an 80 GB GPU is comfortable); `final.pt` loads into noise's `utils/eval.py` with a strict state-dict match.

On Modal (one H100): `modal run examples/track1/modal_run.py::prepare` once, then
`modal run examples/track1/modal_run.py --arm <arm> --steps <N>`; see the docstring of `modal_run.py`.

### Results, one seed (seed 0), one GPU on Modal, torch 2.9.1+cu128, 2026-10-22

| arm | GPU served | val loss | s per update | train time | wall time | fidelity (block matrices) | GPU cost | agent cost |
|---|---|---|---|---|---|---|---|---|
| backprop | H100 80GB | 3.7079 | 0.058 | 35 s | 62 s | 1 | about USD 0.1 | none |
| relay (keep) | H200 (Modal served one for the H100 request) | 3.7036 | 1.54 | 954 s | 995 s | 0.980 at update 1, 0.9986 at 50, 0.998 at 600 | about USD 1.1 | none |
| relay-opal | H100 80GB | 3.7051 | 1.70 between decisions, 2.75 with the waits | 1677 s | 1695 s | 0.9987 at 100, 0.998 at 300 | about USD 1.9 | USD 2.64, 12 decisions |
| relay-opal-ptx | H200 (served for the H100 request) | 3.7050 | 1.53 between decisions, 3.07 with the waits | 1872 s | 1886 s | 0.9985 at 50, 0.999 at 100, 0.9975 at 600 | about USD 2.4 | USD 2.87, 13 decisions |

Costs are Modal's list prices (H100 USD 3.95 per hour, H200 USD 4.54 per hour) times the wall time; the agent cost
is what Claude Code reported per decision. The step time of the Relay arms counts the estimate and the update; the
fidelity check (float32 autograd on one sequence, analysis only) runs inside the timed loop every 50 updates, so the
Relay step times include it (about 2 s each time). The relay-opal arm has no untimed warm-up, so its first updates
include compilation; its "between decisions" figure is the median step. These are one seed each; the Relay PR reports
3.7107 ± 0.0056 over five seeds for Relay and 3.7122 ± 0.0089 over three for backprop, both on 8xH100 with the same
recipe, so the differences between the arms here are inside that spread.

What the Python agent did (12 decisions, every 10 updates from update 20, blocking, claude-opus-5-5 at effort high
through the mailbox broker): decision 1 rewrote analysis.py (richer reports); decision 2 took the learning-rate
schedule into the updater, reproducing the starting schedule exactly; decisions 3, 4, 7, 8, 11, 12 kept the programs;
decision 5 halved the rate of the 16 mixing scalars (resid_lambdas, x0_lambdas) from update 190; decision 6 fixed its
own learning-rate reporting; decision 9 started the warmdown at 0.35 of the run instead of 0.3; decision 10 cut the
mixing scalars' rate to 0.3x from update 460. It never edited estimator.py: it read the fidelity values (0.998) and
wrote at decision 3 that a fidelity check below 0.95 or a loss spike would make it revisit the estimator. The run's
decisions, programs and report are in the volume under `runs/relay-opal-epoch/opal/`.

### The agent arm

The agent arm needs Claude Code logged in on the machine running rank 0, which a Modal container is not. `modal_run.py`
therefore starts an ssh server in the container, exposes it through a Modal tunnel, and runs the training with
`backend="mailbox:/root/box"`; it prints a line starting with `BROKER` giving the command to run on a machine with
the login:

```bash
modal run examples/track1/modal_run.py --arm relay-opal --steps 0 --fidelity-every 50 --max-cost-usd 15 \
    --pubkey ~/.ssh/id_ed25519.pub
# on the machine with Claude Code logged in, with the host and port the run printed:
python examples/track1/agent_broker.py --host <host> --port <port> --box /root/box --key ~/.ssh/id_ed25519
```

The run waits for the broker at its first decision (OPAL's "GPU machines without an agent"). The agent's decisions,
programs and report land in `logs/relay-opal/opal/` on the volume (`modal volume get relay-opal-track1 runs/<name>`).

### The PTX arm, with its path ledger

`--arm relay-opal-ptx --checkpoint-every 50 --max-cost-usd 25 --decisions 12 --every 10`, blocking, claude-opus-5-5
at effort high through the mailbox broker, caps USD 40 compute and USD 25 agent. The agent's prompt is
`relay_opal/opal/prompt_ptx.md`; its starting programs carried the seed PTX kernels (`relay_opal/ptx_seeds.py`).
Every program it wrote is in the volume under `runs/relay-opal-ptx-epoch/opal/programs/v001` to `v007`, its
rationales under `decisions/`. The broker was killed once by mistake during decision 1 (a tmux session name that
prefix-matched another); a restarted broker took the request over, so the run saw 13 requests for 12 scheduled
decisions.

| | count |
|---|---|
| decisions asked | 13 (12 scheduled, 1 repeated after the broker restart) |
| program versions written | 7 (decisions 1, 2, 3, 4, 5, 8, 9); 6 decisions kept the programs |
| written in Python / Triton / PTX | 7 / 0 / 0 (the seed PTX AdamW and hidden-unit kernels stayed in every version) |
| failed to assemble or build | 0 |
| ran and corrupted weights (NaN, inf, cosine collapse, loss spike) | 0; NaN and inf counts 0 in all 61 consequence reports; block-matrix cosine 0.980 at update 1, 0.9985 at 50, never below 0.9974 after; lowest per-site cosine q 0.940 |
| reverted by the agent, restored from a checkpoint, crashed | 0 / 0 / 0 |
| kept | 7 of 7 |

What the versions did: v001 rewrote the analysis (per-group gradient-momentum cosines, norms, learning rates);
v002 took the learning-rate schedule into the updater and set the mixing scalars to 1/3 of their rate; v003 reverted
that ("based on a misreading") and raised q and k to 1.4x; v004 raised q to 1.7x and three small groups; v005 pulled
the value-embedding gate back to 1.5x and changed the warmdown's shape to 1 - sqrt(progress); v006 lengthened the
warmdown to the last 40%; v007 put the gate's rate back to 1x. Seconds per update of the estimator and the update rule
before the first decision and after the last kept program: 1.49 s and 0.036 s both times, since the kernels were
never changed. Final val loss 3.7050 against 3.7051 for the Python OPAL arm and 3.7036 for plain Relay, with the
same step time between decisions (1.53 s against 1.54 s).

Verdict: with complete control and nothing to stop it, the agent took no path through PTX and no path that could
have corrupted the weights; it treated the seed kernels as given and spent its decisions on learning rates. The
experiment therefore shows that it does not take unproductive paths when unprotected, and nothing about whether it
learns from one, because it never caused one. Two facts about what it saw: it wrote at decision 1 that it could not
see the Anvil optimizer's source or the warmdown length (OPAL shows a `LambdaLR` function's source, not its closure
values), and its shell was unavailable on the broker machine (no bubblewrap), so it had file tools only and could not
assemble or time a kernel before shipping it. Both are worth fixing before reading much into a second run.
