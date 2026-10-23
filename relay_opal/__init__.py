"""Relay forward-only credit assignment, with OPAL's agent rewriting the estimator and the optimizer mid-run."""

from .fidelity import check
from .opal import RelayOpal, analysis_program, wrap
from .relay import RelayEstimator, RelaySettings, RelaySpec

__version__ = "0.1.0"
__all__ = [
    "RelayEstimator",
    "RelayOpal",
    "RelaySettings",
    "RelaySpec",
    "analysis_program",
    "check",
    "wrap",
    "__version__",
]
