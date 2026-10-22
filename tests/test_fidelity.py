import pytest
import torch

from relay_opal import RelayEstimator, RelaySettings, check
from relay_opal.fidelity import reference_gradients
from relay_opal.relay.spec import site_types

TINY_SETTINGS = dict(routes=64, tokens=32)


def settings(**kw):
    return RelaySettings(**{**TINY_SETTINGS, **kw})


def test_default_settings_track_autograd(tiny):
    spec, x, y = tiny
    report = check(RelayEstimator(spec, settings(), seed=0), x, y)
    cos = report["cosine"]
    assert cos["block_matrices"] > 0.95
    for kind in ("q", "k", "v", "attn_proj", "gate", "fc", "mlp_proj", "wte", "value_embeds", "lambdas"):
        assert cos[kind] > 0.9, (kind, cos)
    assert cos["lm_head"] > 0.999
    assert set(report["seconds"]) >= {"forward", "top", "mid", "q", "k", "v", "input", "hidden", "wte"}


@pytest.mark.parametrize(
    "variant",
    [
        dict(head="measured"),
        dict(norm="measured"),
        dict(keys="sampled", key_draws=64),
        dict(values="sampled", value_draws=64, gate_draws=64),
        dict(hidden="topk", hidden_topk=64),
        dict(routes=0),
        dict(mlp_dtype="fp32"),
        dict(cycles=2),
    ],
)
def test_variants_track_autograd(tiny, variant):
    spec, x, y = tiny
    cos = check(RelayEstimator(spec, settings(**variant), seed=0), x, y)["cosine"]
    assert cos["block_matrices"] > 0.9, (variant, cos)
    assert cos["all"] > 0.95, (variant, cos)


def test_closed_form_norm_equals_measured_norm(tiny):
    """The closed-form norm difference is the forward evaluation written out: same draws, same numbers."""
    spec, x, y = tiny
    base = dict(mlp_dtype="fp32")  # bf16 products would turn float32 rounding into bf16 rounding flips
    a = RelayEstimator(spec, settings(**base), seed=0).gradients(x, y)
    b = RelayEstimator(spec, settings(**base, norm="measured"), seed=0).gradients(x, y)
    for name in a:
        scale = a[name].abs().max().item()
        assert torch.allclose(a[name], b[name], rtol=1e-3, atol=1e-4 * scale + 1e-9), name


@pytest.mark.parametrize("variant", [dict(head="measured"), dict(mlp_dtype="fp32")])
def test_rounding_variants_change_only_rounding(tiny, variant):
    """The measured head subtracts two nearby float32 losses (the closed form avoids that cancellation); bf16 MLP
    products round. Same draws, same direction."""
    spec, x, y = tiny
    a = RelayEstimator(spec, settings(), seed=0).gradients(x, y)
    b = RelayEstimator(spec, settings(**variant), seed=0).gradients(x, y)
    for name in a:
        if a[name].numel() > 2:
            cos = torch.nn.functional.cosine_similarity(a[name].flatten(), b[name].flatten(), dim=0)
            assert cos > 0.99, (variant, name)


def test_frozen_sites_get_zero_gradient(tiny):
    spec, x, y = tiny
    est = RelayEstimator(spec, settings(freeze=("k", "wte", "lambdas")), seed=0)
    grads = est.gradients(x, y)
    for name, kind in site_types(spec.sites).items():
        if kind in ("k", "wte", "lambdas"):
            assert grads[name].abs().sum() == 0, name
        else:
            assert grads[name].abs().sum() > 0, name


def test_estimate_fills_grads_and_accumulates(tiny):
    spec, x, y = tiny
    est = RelayEstimator(spec, settings(), seed=0)
    loss = est.estimate(x, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    first = {n: p.grad.clone() for n, p in spec.model.named_parameters()}
    RelayEstimator(spec, settings(), seed=0).estimate(x, y, accumulate=True, scale=1.0)  # the same draws again
    for n, p in spec.model.named_parameters():
        assert torch.allclose(p.grad, 2 * first[n], rtol=1e-4, atol=1e-7), n
    reference = reference_gradients(spec.model, x, y)
    valid = y.numel()
    for n in ("transformer.h.0.mlp.c_fc.weight", "lm_head.weight"):
        cos = torch.nn.functional.cosine_similarity(first[n].flatten(), (reference[n] / valid).flatten(), dim=0)
        assert cos > 0.95
        assert abs(first[n].norm() / (reference[n] / valid).norm() - 1) < 0.2


def test_seed_sets_the_draws(tiny):
    spec, x, y = tiny
    a = RelayEstimator(spec, settings(), seed=0).gradients(x, y)
    b = RelayEstimator(spec, settings(), seed=0).gradients(x, y)
    c = RelayEstimator(spec, settings(), seed=1).gradients(x, y)
    name = "transformer.h.1.attn.c_q.weight"
    assert torch.equal(a[name], b[name])
    assert not torch.equal(a[name], c[name])
