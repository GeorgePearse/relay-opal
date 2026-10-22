"""The PTX loader and the seed kernels, on a CUDA device (skipped elsewhere; examples/track1/modal_run.py::gpu_tests
runs them on an H100)."""

import pytest
import torch

from relay_opal import RelayEstimator, RelaySettings
from relay_opal.models import GPTConfig, NoiseGPTSpec, make_model
from relay_opal.relay import kernels as K

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@cuda
def test_broken_ptx_reports_ptxas_text():
    from relay_opal.ptx import PtxError, PtxModule

    with pytest.raises(PtxError) as info:
        PtxModule(".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry broken() { add.f32 %f1, %f1; }")
    assert "ptxas" in str(info.value)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adamw_seed_matches_its_reference_bitwise(dtype):
    from relay_opal.ptx_seeds import adamw, adamw_reference

    g = torch.Generator(device="cuda").manual_seed(0)
    n = 1_000_003
    p0 = torch.randn(n, device="cuda", generator=g).to(dtype)
    grad = (torch.randn(n, device="cuda", generator=g) * 0.1).to(dtype)
    m0, v0 = torch.randn(n, device="cuda", generator=g) * 0.01, torch.rand(n, device="cuda", generator=g) * 1e-4
    a, b = p0.clone(), p0.clone()
    ma, mb, va, vb = m0.clone(), m0.clone(), v0.clone(), v0.clone()
    for t in (1, 2, 7):
        adamw(a, grad, ma, va, lr=3e-3, betas=(0.7, 0.95), eps=1e-8, weight_decay=0.1, t=t)
        adamw_reference(b, grad, mb, vb, lr=3e-3, betas=(0.7, 0.95), eps=1e-8, weight_decay=0.1, t=t)
        torch.cuda.synchronize()
        for name, u, w in (("p", a, b), ("m", ma, mb), ("v", va, vb)):
            bad = (u != w).nonzero().flatten()
            detail = ""
            if len(bad):
                i = int(bad[0])
                detail = f"{len(bad)} mismatches; first at {i}: kernel {u[i].item()!r} vs reference {w[i].item()!r}"
            assert len(bad) == 0, f"t={t} {name}: {detail}"
    assert not torch.equal(a, p0)


@cuda
def test_hidden_all_seed_matches_its_reference_bitwise():
    from relay_opal.ptx_seeds import hidden_all, hidden_all_reference

    g = torch.Generator(device="cuda").manual_seed(0)
    hpre = torch.randn(4096, 2048, device="cuda", generator=g)
    steps = 0.05 * hpre.square().mean(-1, keepdim=True).sqrt()
    units = torch.randn(4096, 2048, device="cuda", generator=g)
    out = hidden_all(hpre, steps, units)
    torch.cuda.synchronize()
    assert torch.equal(out, hidden_all_reference(hpre, steps, units))
    eager = K.hidden_all.__wrapped__ if hasattr(K.hidden_all, "__wrapped__") else K.hidden_all
    cos = torch.nn.functional.cosine_similarity(out.flatten(), eager(hpre, steps, units).flatten(), dim=0)
    assert cos > 0.9999


@cuda
def test_estimator_takes_a_ptx_kernel():
    from relay_opal.ptx_seeds import hidden_all

    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2, n_embd=128)
    model = make_model(torch.device("cuda", 0), 0, cfg)
    spec = NoiseGPTSpec(model)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 256, (2, 128), generator=g).cuda()
    y = torch.randint(0, 256, (2, 128), generator=g).cuda()
    settings = RelaySettings(routes=128, tokens=64)
    plain = RelayEstimator(spec, settings, seed=0).gradients(x, y)
    est = RelayEstimator(spec, settings, seed=0).configure(settings, kernels={"hidden_all": hidden_all})
    ptx = est.gradients(x, y)
    for name in ("transformer.h.0.mlp.c_fc.weight", "transformer.h.1.mlp.c_fc.weight"):
        cos = torch.nn.functional.cosine_similarity(plain[name].flatten(), ptx[name].flatten(), dim=0)
        assert cos > 0.999, name
    with pytest.raises(ValueError, match="unknown kernel"):
        est.configure(settings, kernels={"nope": hidden_all})
