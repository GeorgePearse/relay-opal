//! Cockpit's cheap instruments (Schneider, Balles and Hennig), per parameter group and per step, from what the
//! optimizer already holds: no per-sample gradients, no backward hooks.
//!
//! For step t the inputs are the gradient `g_t` and parameters `theta_t` before the update, the parameters after
//! it (so the update actually applied is `delta_t = theta_{t+1} - theta_t`, whatever the optimizer did), the loss
//! `loss_t`, and from the next step `g_{t+1}` and `loss_{t+1}` (on the next batch, as in Cockpit). The row for
//! step t is therefore complete when step t+1's gradient arrives.
//!
//! - `grad_norm`: `||g_t||`.
//! - `update_ratio`: `||delta_t|| / ||theta_t||`.
//! - `grad_cosine`: cosine between `g_t` and `g_{t+1}`.
//! - `snr`: the step-wise signal-to-noise `||mean(g)||^2 / mean(||g - mean(g)||^2)` over the window of the last
//!   `k` step gradients ending at `g_t`. Cockpit's quantity uses per-sample gradients within one batch; this one
//!   uses consecutive steps, so it mixes curvature with noise and is named step-wise for that reason.
//! - `alpha`: Cockpit's local step size. Fit a quadratic `f(x) = c + b x + a x^2` along the step direction
//!   `d = delta / ||delta||` by least squares through `f(0) = loss_t`, `f(T) = loss_{t+1}`, `f'(0) = g_t . d`,
//!   `f'(T) = g_{t+1} . d` with `T = ||delta||`, then `alpha = 2 T / (-b / a) - 1` (Cockpit's `_get_alpha`): -1
//!   means the step stayed at its start, 0 landed on the minimum of the fit, +1 overshot to the loss it started
//!   from; a non-convex fit (`a <= 0`) reports -1 like Cockpit. Cockpit weights the fit by per-sample variances,
//!   which are not available here, so the fit is unweighted.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// One row of the instruments table.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepInstruments {
    pub grad_norm: f64,
    pub update_ratio: f64,
    pub grad_cosine: f64,
    /// `None` until the window holds at least two gradients.
    pub snr: Option<f64>,
    /// `None` when the step has zero length.
    pub alpha: Option<f64>,
    pub loss: f64,
    pub loss_next: f64,
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

/// Cosine between two gradients; 0 when either is zero.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot(a, b) / (na * nb)
    }
}

/// Cockpit's local step size from the two losses and the two directional derivatives along the step, with
/// `g_dot_delta` and `g_next_dot_delta` the gradients dotted with the full (unnormalised) update and `step_len` its
/// norm. `None` for a zero-length step.
pub fn local_step_alpha(
    loss: f64,
    loss_next: f64,
    g_dot_delta: f64,
    g_next_dot_delta: f64,
    step_len: f64,
) -> Option<f64> {
    if step_len <= 0.0 || !step_len.is_finite() {
        return None;
    }
    let t = step_len;
    let (df0, df1) = (g_dot_delta / t, g_next_dot_delta / t);
    // Rows of the design matrix for [c, b, a]: f(0), f(t), f'(0), f'(t).
    let rows = [[1.0, 0.0, 0.0], [1.0, t, t * t], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0 * t]];
    let y = [loss, loss_next, df0, df1];
    let mut ata = [[0.0f64; 3]; 3];
    let mut aty = [0.0f64; 3];
    for (r, &yi) in rows.iter().zip(&y) {
        for i in 0..3 {
            aty[i] += r[i] * yi;
            for j in 0..3 {
                ata[i][j] += r[i] * r[j];
            }
        }
    }
    let [_c, b, a] = solve3(ata, aty)?;
    if a <= 0.0 {
        return Some(-1.0);
    }
    let alpha_bar = -b / a;
    Some(2.0 * t / alpha_bar - 1.0)
}

fn solve3(mut m: [[f64; 3]; 3], mut v: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[pivot][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, pivot);
        v.swap(col, pivot);
        for row in 0..3 {
            if row != col {
                let f = m[row][col] / m[col][col];
                let pivot_row = m[col];
                for (k, x) in m[row].iter_mut().enumerate().skip(col) {
                    *x -= f * pivot_row[k];
                }
                v[row] -= f * v[col];
            }
        }
    }
    Some([v[0] / m[0][0], v[1] / m[1][1], v[2] / m[2][2]])
}

/// `||mean(g)||^2 / mean(||g - mean(g)||^2)` over a set of gradients; `None` below two gradients or with no noise.
pub fn step_snr<'a>(grads: impl Iterator<Item = &'a [f32]> + Clone) -> Option<f64> {
    let n = grads.clone().count();
    if n < 2 {
        return None;
    }
    let dim = grads.clone().next()?.len();
    let mut mean = vec![0.0f64; dim];
    for g in grads.clone() {
        for (m, &x) in mean.iter_mut().zip(g) {
            *m += x as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let signal: f64 = mean.iter().map(|m| m * m).sum();
    let noise: f64 =
        grads.map(|g| g.iter().zip(&mean).map(|(&x, m)| (x as f64 - m).powi(2)).sum::<f64>()).sum::<f64>() / n as f64;
    if noise > 0.0 {
        Some(signal / noise)
    } else {
        None
    }
}

#[derive(Default)]
struct Group {
    grads: VecDeque<Vec<f32>>,
    theta: Option<Vec<f32>>,
    loss: Option<f64>,
    pending: Option<Pending>,
}

struct Pending {
    delta: Vec<f32>,
    step_len: f64,
    update_ratio: f64,
    g_dot_delta: f64,
}

/// Per-group state: the last `k` step gradients, the parameters before the current step and the update applied,
/// so the caller passes a view per step and keeps no history.
pub struct Window {
    k: usize,
    groups: Vec<Group>,
}

impl Window {
    pub fn new(k: usize, n_groups: usize) -> Self {
        Window { k: k.max(1), groups: (0..n_groups).map(|_| Group::default()).collect() }
    }

    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// The gradient, parameters and loss of group `group` before its update. Returns the completed row of the
    /// previous step when there was one.
    pub fn before_step(&mut self, group: usize, grad: &[f32], theta: &[f32], loss: f64) -> Option<StepInstruments> {
        let k = self.k;
        let g = &mut self.groups[group];
        let row = match (g.pending.take(), g.grads.back(), g.loss) {
            (Some(p), Some(prev), Some(prev_loss)) => Some(StepInstruments {
                grad_norm: norm(prev),
                update_ratio: p.update_ratio,
                grad_cosine: cosine(prev, grad),
                snr: step_snr(g.grads.iter().map(|v| v.as_slice())),
                alpha: local_step_alpha(prev_loss, loss, p.g_dot_delta, dot(grad, &p.delta), p.step_len),
                loss: prev_loss,
                loss_next: loss,
            }),
            _ => None,
        };
        if g.grads.len() == k {
            g.grads.pop_front();
        }
        g.grads.push_back(grad.to_vec());
        g.theta = Some(theta.to_vec());
        g.loss = Some(loss);
        row
    }

    /// The parameters of group `group` after its update.
    pub fn after_step(&mut self, group: usize, theta_next: &[f32]) {
        let g = &mut self.groups[group];
        let (Some(theta), Some(grad)) = (g.theta.take(), g.grads.back()) else { return };
        let delta: Vec<f32> = theta_next.iter().zip(&theta).map(|(&a, &b)| a - b).collect();
        let step_len = norm(&delta);
        let theta_norm = norm(&theta);
        g.pending = Some(Pending {
            g_dot_delta: dot(grad, &delta),
            update_ratio: if theta_norm > 0.0 { step_len / theta_norm } else { f64::NAN },
            step_len,
            delta,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_alpha_is_exact() {
        // f(x) = (x - 1)^2 in 1-D from x = 0: the minimum is a step of 1, the "other side" a step of 2.
        let f = |x: f64| (x - 1.0).powi(2);
        let df = |x: f64| 2.0 * (x - 1.0);
        for (step, want) in [(1.0, 0.0), (2.0, 1.0), (0.5, -0.5), (3.0, 2.0)] {
            let a = local_step_alpha(f(0.0), f(step), df(0.0) * step, df(step) * step, step).unwrap();
            assert!((a - want).abs() < 1e-9, "step {step}: {a} vs {want}");
        }
    }

    #[test]
    fn concave_fit_reports_minus_one() {
        let f = |x: f64| -(x * x);
        let df = |x: f64| -2.0 * x;
        assert_eq!(local_step_alpha(f(0.0), f(1.0), df(0.0), df(1.0), 1.0), Some(-1.0));
    }

    #[test]
    fn snr_closed_form() {
        let g1 = [1.0f32, 0.0];
        let g2 = [3.0f32, 0.0];
        // mean = 2, signal = 4, noise = mean of (1, 1) = 1
        let s = step_snr([g1.as_slice(), g2.as_slice()].into_iter()).unwrap();
        assert!((s - 4.0).abs() < 1e-12);
        assert_eq!(step_snr([g1.as_slice()].into_iter()), None);
    }

    #[test]
    fn window_rows_in_one_dimension() {
        // f(x) = (x - 1)^2, SGD with lr 0.5 from x = 0 lands on the minimum: alpha = 0, cosine between
        // g_0 = -2 and g_1 = 0 is 0, update ratio is undefined at theta = 0 (NaN) and then 0.
        let mut w = Window::new(4, 1);
        assert!(w.before_step(0, &[-2.0], &[0.0], 1.0).is_none());
        w.after_step(0, &[1.0]);
        let row = w.before_step(0, &[0.0], &[1.0], 0.0).unwrap();
        assert_eq!(row.grad_norm, 2.0);
        assert!(row.update_ratio.is_nan());
        assert_eq!(row.grad_cosine, 0.0);
        assert!((row.alpha.unwrap()).abs() < 1e-9);
        assert_eq!(row.snr, None);
        assert_eq!((row.loss, row.loss_next), (1.0, 0.0));
    }
}
