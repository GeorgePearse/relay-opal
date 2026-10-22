"""Seed PTX kernels: the first PTX the agent sees, known to assemble and to match their eager Python twins bit for bit.

- adamw: one AdamW step, one thread per element, grid-stride; float32 or bfloat16 parameters and gradients,
  float32 moments. Its twin is adamw_reference below (plain torch ops in the same order; it is not torch.optim.AdamW,
  whose fused kernels round differently).
- hidden_all: Relay's MLP hidden-unit step (every unit by its own central difference through relu^2, read through
  its output column). Its twin is relay_opal.relay.kernels.hidden_all run eagerly.

Every float op carries .rn so ptxas cannot contract a multiply and an add into an fma the eager twin does not do.
"""

import ctypes

import numpy as np
import torch

from .ptx import PtxModule, grid_for

HEADER = """
.version 8.0
.target sm_90
.address_size 64
"""

ADAMW_PTX = (
    HEADER
    + """
// p, g, m, v: pointers; decay = 1 - lr * wd; omb1 = 1 - beta1; omb2 = 1 - beta2; bc2s = 1 / sqrt(1 - beta2^t)
// (a multiply by the reciprocal, as torch divides a tensor by a scalar);
// step = lr / (1 - beta1^t); n = element count. Grid-stride loop, one element per iteration.
.visible .entry adamw_f32(
    .param .u64 p_ptr, .param .u64 g_ptr, .param .u64 m_ptr, .param .u64 v_ptr,
    .param .f32 decay, .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,
    .param .f32 eps, .param .f32 bc2s, .param .f32 step, .param .u64 n)
{
    .reg .b64 %rd<16>;
    .reg .b32 %r<8>;
    .reg .f32 %f<24>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [p_ptr];
    ld.param.u64 %rd2, [g_ptr];
    ld.param.u64 %rd3, [m_ptr];
    ld.param.u64 %rd4, [v_ptr];
    ld.param.u64 %rd5, [n];
    ld.param.f32 %f1, [decay];
    ld.param.f32 %f2, [b1];
    ld.param.f32 %f3, [omb1];
    ld.param.f32 %f4, [b2];
    ld.param.f32 %f5, [omb2];
    ld.param.f32 %f6, [eps];
    ld.param.f32 %f7, [bc2s];
    ld.param.f32 %f8, [step];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r5, %r2;
    cvt.u64.u32 %rd6, %r4;
    cvt.u64.u32 %rd7, %r6;
LOOP_F32:
    setp.ge.u64 %p1, %rd6, %rd5;
    @%p1 bra DONE_F32;
    shl.b64 %rd8, %rd6, 2;
    add.s64 %rd9, %rd1, %rd8;
    add.s64 %rd10, %rd2, %rd8;
    add.s64 %rd11, %rd3, %rd8;
    add.s64 %rd12, %rd4, %rd8;
    ld.global.f32 %f10, [%rd9];
    ld.global.f32 %f11, [%rd10];
    ld.global.f32 %f12, [%rd11];
    ld.global.f32 %f13, [%rd12];
    mul.rn.f32 %f10, %f10, %f1;
    mul.rn.f32 %f12, %f12, %f2;
    mul.rn.f32 %f14, %f11, %f3;
    add.rn.f32 %f12, %f12, %f14;
    mul.rn.f32 %f13, %f13, %f4;
    mul.rn.f32 %f15, %f11, %f11;
    mul.rn.f32 %f15, %f15, %f5;
    add.rn.f32 %f13, %f13, %f15;
    sqrt.rn.f32 %f16, %f13;
    mul.rn.f32 %f16, %f16, %f7;
    add.rn.f32 %f16, %f16, %f6;
    div.rn.f32 %f17, %f12, %f16;
    mul.rn.f32 %f17, %f17, %f8;
    sub.rn.f32 %f10, %f10, %f17;
    st.global.f32 [%rd9], %f10;
    st.global.f32 [%rd11], %f12;
    st.global.f32 [%rd12], %f13;
    add.s64 %rd6, %rd6, %rd7;
    bra LOOP_F32;
DONE_F32:
    ret;
}

.visible .entry adamw_bf16(
    .param .u64 p_ptr, .param .u64 g_ptr, .param .u64 m_ptr, .param .u64 v_ptr,
    .param .f32 decay, .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,
    .param .f32 eps, .param .f32 bc2s, .param .f32 step, .param .u64 n)
{
    .reg .b64 %rd<16>;
    .reg .b32 %r<8>;
    .reg .b16 %rs<4>;
    .reg .f32 %f<24>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [p_ptr];
    ld.param.u64 %rd2, [g_ptr];
    ld.param.u64 %rd3, [m_ptr];
    ld.param.u64 %rd4, [v_ptr];
    ld.param.u64 %rd5, [n];
    ld.param.f32 %f1, [decay];
    ld.param.f32 %f2, [b1];
    ld.param.f32 %f3, [omb1];
    ld.param.f32 %f4, [b2];
    ld.param.f32 %f5, [omb2];
    ld.param.f32 %f6, [eps];
    ld.param.f32 %f7, [bc2s];
    ld.param.f32 %f8, [step];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r5, %r2;
    cvt.u64.u32 %rd6, %r4;
    cvt.u64.u32 %rd7, %r6;
LOOP_BF16:
    setp.ge.u64 %p1, %rd6, %rd5;
    @%p1 bra DONE_BF16;
    shl.b64 %rd8, %rd6, 1;
    shl.b64 %rd13, %rd6, 2;
    add.s64 %rd9, %rd1, %rd8;
    add.s64 %rd10, %rd2, %rd8;
    add.s64 %rd11, %rd3, %rd13;
    add.s64 %rd12, %rd4, %rd13;
    ld.global.b16 %rs1, [%rd9];
    ld.global.b16 %rs2, [%rd10];
    cvt.f32.bf16 %f10, %rs1;
    cvt.f32.bf16 %f11, %rs2;
    ld.global.f32 %f12, [%rd11];
    ld.global.f32 %f13, [%rd12];
    mul.rn.f32 %f10, %f10, %f1;
    mul.rn.f32 %f12, %f12, %f2;
    mul.rn.f32 %f14, %f11, %f3;
    add.rn.f32 %f12, %f12, %f14;
    mul.rn.f32 %f13, %f13, %f4;
    mul.rn.f32 %f15, %f11, %f11;
    mul.rn.f32 %f15, %f15, %f5;
    add.rn.f32 %f13, %f13, %f15;
    sqrt.rn.f32 %f16, %f13;
    mul.rn.f32 %f16, %f16, %f7;
    add.rn.f32 %f16, %f16, %f6;
    div.rn.f32 %f17, %f12, %f16;
    mul.rn.f32 %f17, %f17, %f8;
    sub.rn.f32 %f10, %f10, %f17;
    cvt.rn.bf16.f32 %rs3, %f10;
    st.global.b16 [%rd9], %rs3;
    st.global.f32 [%rd11], %f12;
    st.global.f32 [%rd12], %f13;
    add.s64 %rd6, %rd6, %rd7;
    bra LOOP_BF16;
DONE_BF16:
    ret;
}
"""
)

HIDDEN_ALL_PTX = (
    HEADER
    + """
// out[i] = (relu(h[i] + s[row])^2 - relu(h[i] - s[row])^2) / (2 s[row]) * g[i], row = i / width.
.visible .entry hidden_all(
    .param .u64 h_ptr, .param .u64 s_ptr, .param .u64 g_ptr, .param .u64 o_ptr, .param .u64 width, .param .u64 n)
{
    .reg .b64 %rd<20>;
    .reg .b32 %r<8>;
    .reg .f32 %f<20>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [h_ptr];
    ld.param.u64 %rd2, [s_ptr];
    ld.param.u64 %rd3, [g_ptr];
    ld.param.u64 %rd4, [o_ptr];
    ld.param.u64 %rd5, [width];
    ld.param.u64 %rd6, [n];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r5, %r2;
    cvt.u64.u32 %rd7, %r4;
    cvt.u64.u32 %rd8, %r6;
LOOP_H:
    setp.ge.u64 %p1, %rd7, %rd6;
    @%p1 bra DONE_H;
    div.u64 %rd9, %rd7, %rd5;
    shl.b64 %rd10, %rd7, 2;
    shl.b64 %rd11, %rd9, 2;
    add.s64 %rd12, %rd1, %rd10;
    add.s64 %rd13, %rd2, %rd11;
    add.s64 %rd14, %rd3, %rd10;
    add.s64 %rd15, %rd4, %rd10;
    ld.global.f32 %f1, [%rd12];
    ld.global.f32 %f2, [%rd13];
    ld.global.f32 %f3, [%rd14];
    add.rn.f32 %f4, %f1, %f2;
    max.f32 %f4, %f4, 0f00000000;
    mul.rn.f32 %f4, %f4, %f4;
    sub.rn.f32 %f5, %f1, %f2;
    max.f32 %f5, %f5, 0f00000000;
    mul.rn.f32 %f5, %f5, %f5;
    sub.rn.f32 %f6, %f4, %f5;
    mul.rn.f32 %f7, %f2, 0f40000000;
    div.rn.f32 %f6, %f6, %f7;
    mul.rn.f32 %f6, %f6, %f3;
    st.global.f32 [%rd15], %f6;
    add.s64 %rd7, %rd7, %rd8;
    bra LOOP_H;
DONE_H:
    ret;
}
"""
)


def _f32(x):
    return float(np.float32(x))


def adamw_scalars(lr, betas, eps, weight_decay, t):
    """The per-step scalars, each rounded to float32 once, shared by the kernel and its reference."""
    b1, b2 = betas
    return dict(
        decay=_f32(1 - lr * weight_decay),
        b1=_f32(b1),
        omb1=_f32(1 - b1),
        b2=_f32(b2),
        omb2=_f32(1 - b2),
        eps=_f32(eps),
        bc2s=float(np.float32(1) / np.float32((1 - b2**t) ** 0.5)),
        step=_f32(lr / (1 - b1**t)),
    )


def adamw(p, g, m, v, lr, betas, eps, weight_decay, t, block=256):
    """One AdamW step in place on p (float32 or bfloat16, g in the same dtype; m, v float32), t counted from 1."""
    s = adamw_scalars(lr, betas, eps, weight_decay, t)
    module = PtxModule(ADAMW_PTX)
    name = "adamw_f32" if p.dtype == torch.float32 else "adamw_bf16"
    args = [p, g, m, v] + [s[k] for k in ("decay", "b1", "omb1", "b2", "omb2", "eps", "bc2s", "step")]
    module.launch(name, grid_for(p.numel(), block), block, args + [ctypes.c_uint64(p.numel())])


def adamw_reference(p, g, m, v, lr, betas, eps, weight_decay, t):
    """adamw's eager twin: the same operations in the same order, each rounded to float32."""
    s = adamw_scalars(lr, betas, eps, weight_decay, t)
    pw = p.float() if p.dtype != torch.float32 else p
    gw = g.float()
    pw = pw * s["decay"]
    m.mul_(s["b1"]).add_(gw * s["omb1"])
    v.mul_(s["b2"]).add_((gw * gw) * s["omb2"])
    denom = v.sqrt().mul(s["bc2s"]).add(s["eps"])
    pw = pw - (m / denom) * s["step"]
    p.copy_(pw.to(p.dtype))


def hidden_all(hpre, steps, g_units, block=256):
    """Relay's hidden step as a PTX kernel: hpre, g_units (n, H) float32 contiguous, steps (n, 1) float32."""
    hpre, g_units, steps = hpre.contiguous(), g_units.contiguous(), steps.contiguous()
    out = torch.empty_like(hpre)
    n, width = hpre.numel(), hpre.shape[-1]
    PtxModule(HIDDEN_ALL_PTX).launch(
        "hidden_all", grid_for(n, block), block, [hpre, steps, g_units, out, ctypes.c_uint64(width), ctypes.c_uint64(n)]
    )
    return out


def hidden_all_reference(hpre, steps, g_units):
    up, down = torch.relu(hpre + steps).square(), torch.relu(hpre - steps).square()
    return (up - down) / (2 * steps) * g_units
