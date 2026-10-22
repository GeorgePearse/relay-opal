"""The estimator's knobs: what the OPAL agent's estimator.py returns, validated before every update."""

from dataclasses import asdict, dataclass, field, fields

SITE_TYPES = ("wte", "lm_head", "q", "k", "v", "gate", "value_embeds", "attn_proj", "fc", "mlp_proj", "lambdas")
CHOICES = dict(
    keys=("exact", "sampled"),
    values=("closed_form", "sampled"),
    hidden=("all", "topk"),
    head=("closed_form", "measured"),
    norm=("closed_form", "measured"),
    mlp_dtype=("bf16", "fp32"),
)


@dataclass
class RelaySettings:
    routes: int = 512
    cycles: int = 1
    tokens: int = 256
    chain_sigma: float = 0.01
    fc_sigma: float = 0.05
    key_sigma: float = 0.003
    probe_sigma: float = 0.005
    head_sigma: float = 0.1
    query_draws: int = 64
    key_draws: int = 320
    value_draws: int = 512
    gate_draws: int = 64
    keys: str = "exact"
    values: str = "closed_form"
    hidden: str = "all"
    hidden_topk: int = 256
    head: str = "closed_form"
    norm: str = "closed_form"
    mlp_dtype: str = "bf16"
    freeze: tuple = field(default_factory=tuple)
    key_chunk: int = 8
    key_block: int = 512
    probe_chunk: int = 2**27

    @classmethod
    def from_dict(cls, values):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown estimator settings: {unknown}")
        out = cls(**values)
        out.freeze = tuple(out.freeze)
        return out

    def to_dict(self):
        d = asdict(self)
        d["freeze"] = list(self.freeze)
        return d

    def validate(self, hidden=None):
        for name, choices in CHOICES.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}, not {getattr(self, name)!r}")
        for name in ("cycles", "tokens", "query_draws", "key_draws", "value_draws", "gate_draws", "key_chunk"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{name} must be a positive int, not {v!r}")
        for name in ("chain_sigma", "fc_sigma", "key_sigma", "probe_sigma", "head_sigma"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not v > 0:
                raise ValueError(f"{name} must be positive, not {v!r}")
        if not isinstance(self.routes, int) or self.routes < 0:
            raise ValueError(f"routes must be a non-negative int (0: every unit), not {self.routes!r}")
        if hidden is not None and self.routes > hidden:
            raise ValueError(f"routes={self.routes} exceeds the MLP width {hidden}")
        if hidden is not None and self.hidden == "topk" and not (0 < self.hidden_topk <= hidden):
            raise ValueError(f"hidden_topk={self.hidden_topk} must be in 1..{hidden}")
        if self.hidden == "topk" and self.hidden_topk & (self.hidden_topk - 1):
            raise ValueError("hidden_topk must be a power of two (a Hadamard cycle)")
        bad = sorted(set(self.freeze) - set(SITE_TYPES))
        if bad:
            raise ValueError(f"freeze names unknown site types {bad}; known: {SITE_TYPES}")
        return self
