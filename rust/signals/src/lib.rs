//! Per-layer training signals, tensor-agnostic: flat `f32`/`f64` slices in, plain numbers and rows out.
//!
//! - [`spectrum`]: WeightWatcher's empirical spectral density, power-law alpha and the metrics that follow.
//! - [`tempbalance`]: TempBalance's per-layer learning rates from per-layer alpha.
//! - [`instruments`]: Cockpit's cheap per-step instruments from the gradients the optimizer already holds.
//! - [`hessian`]: PyHessian's power iteration, Hutchinson trace and stochastic Lanczos quadrature over a
//!   Hessian-vector-product callback.

pub mod hessian;
pub mod instruments;
pub mod spectrum;
pub mod tempbalance;
