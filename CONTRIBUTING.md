# Contributing

Issues and pull requests are welcome.

- Install: `uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[dev]"`.
- Check: `uv run ruff check . && uv run ruff format --check . && uv run pytest`. CI runs the same on CPU.
- The estimator's semantics follow the Relay script in qlabs-eng/noise PR #20. A change to the numbers a site
  produces needs a fidelity test (`tests/test_fidelity.py`) that still passes, and a note in CHANGELOG.md.
- Do not vendor OPAL. Everything OPAL-side goes through `opal.wrap` arguments and its documented extension points.
- Keep narrative in README.md and docs, not in comments.
