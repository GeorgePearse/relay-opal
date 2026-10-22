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
| relay-opal-ptx | PTX_ROW |

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

AGENT_NOTES
