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

### Results, one seed (seed 0), one H100 80GB on Modal, torch 2.9.1+cu128

RESULTS_TABLE

The step time of the Relay arms counts the estimate and the update; the fidelity check (float32 autograd on one
sequence, analysis only) is run inside the timed loop every 50 updates, so the Relay step times include it.
These are one seed each; the Relay PR reports 3.7107 ± 0.0056 over five seeds for Relay and 3.7122 ± 0.0089 over
three for backprop, both on 8xH100 with the same recipe.

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
