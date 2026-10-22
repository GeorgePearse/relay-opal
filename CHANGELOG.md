# Changelog

## Unreleased

- `relay_opal.ptx`: a ctypes loader for hand-written PTX through the CUDA driver API (`PtxModule`, assembly errors
  carry the ptxas text, `launch` on torch tensors, `request_restore`); `relay_opal.ptx_seeds`: an AdamW step and
  Relay's hidden-unit step in PTX, bit-identical to their eager twins (tested on an H100).
- `RelayEstimator.configure(settings, kernels=...)`: any step of `relay_opal.relay.kernels` can be replaced by the
  agent's function (Python, Triton or PTX); `estimator.py` may define `KERNELS`.
- `relay_opal.wrap(..., mode="ptx", checkpoint_every=N)`: the agent writes PTX with no parity gate, sandbox or
  fallback; every check reports NaN and inf counts, weight-norm drift and seconds per site type; checkpoints back a
  `request_restore(update)` tool and the restart after a crash (crash text handed to the agent as a note).
- `examples/track1`: the Noise Track 1 reproduction with four arms (backprop, relay, relay-opal, relay-opal-ptx), a
  Modal launcher for one H100 with an ssh tunnel for the mailbox broker, and the one-seed results.
- `relay_opal.wrap(optimizer, model, estimator, goal=..., total_steps=..., fidelity_every=..., **opal_kwargs)`: the
  hybrid. OPAL versions, swaps and reverts the modules its updater imports, so the agent gets a third program,
  `estimator.py`, whose `settings(update, total_updates)` the loop reads before every estimate; the fidelity report
  reaches the agent through `opt.log(show_agent=True)`; `backend="keep"` is bit-identical to Relay + the plain
  optimizer (tested). Invalid or failing settings revert to the last that passed their trial window.
- `relay_opal.relay`: the Relay estimator (noise PR #20) as a library: `RelayEstimator(spec, settings, seed)` with
  `.estimate(x, y)` filling `.grad` and no autograd on the path; `RelaySettings` with routes, cycles, measured or
  closed-form head and norm steps, exact or sampled keys, closed-form or sampled values, all-unit or top-k hidden
  credit, bf16 or fp32 MLP cycles, and frozen site types; `RelaySpec`, the adapter protocol.
- `relay_opal.models`: the Noise Track 1 GPT (same state dict as `utils/model.py`, dense attention on CPU), its
  adapter `NoiseGPTSpec`, the Relay script's init and its optimizer (ANVIL II + Adam).
- `relay_opal.fidelity.check`: per-site cosine against float32 autograd, with seconds per site type.
- Tests on CPU: fidelity of every variant, closed form equals measured, no autograd entry point fires.
- Skeleton: package layout, CI on CPU, licence, README stub.
