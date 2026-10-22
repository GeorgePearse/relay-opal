# Changelog

## Unreleased

- `relay_opal.relay`: the Relay estimator (noise PR #20) as a library: `RelayEstimator(spec, settings, seed)` with
  `.estimate(x, y)` filling `.grad` and no autograd on the path; `RelaySettings` with routes, cycles, measured or
  closed-form head and norm steps, exact or sampled keys, closed-form or sampled values, all-unit or top-k hidden
  credit, bf16 or fp32 MLP cycles, and frozen site types; `RelaySpec`, the adapter protocol.
- `relay_opal.models`: the Noise Track 1 GPT (same state dict as `utils/model.py`, dense attention on CPU), its
  adapter `NoiseGPTSpec`, the Relay script's init and its optimizer (ANVIL II + Adam).
- `relay_opal.fidelity.check`: per-site cosine against float32 autograd, with seconds per site type.
- Tests on CPU: fidelity of every variant, closed form equals measured, no autograd entry point fires.
- Skeleton: package layout, CI on CPU, licence, README stub.
