//! Weight spectra after WeightWatcher (Martin and Mahoney): the empirical spectral density (ESD) of a weight
//! matrix is the set of eigenvalues of W^T W, the squared singular values, and the heavy-tail exponent `alpha` is
//! the continuous power-law MLE of Clauset, Shalizi and Newman over the tail above `xmin`, with `xmin` chosen by the
//! smallest Kolmogorov–Smirnov distance over every eigenvalue except the largest. This module reproduces
//! WeightWatcher 0.7.7's own fit (`WW_powerlaw.WWFit`, its default `pl_package="ww"`, `xmax=None`): eigenvalues
//! below 1e-5 are dropped before the fit, fewer than 20 remaining eigenvalues switch to the bias-corrected small-N
//! fit with its KS-minus-0.868/sqrt(n) objective, and the metric formulas (`alpha_weighted = alpha *
//! log10(lambda_max)`, `stable_rank = sum / max`, `entropy` normalised by log(rank)) follow `apply_norm_metrics` and
//! `RMT_Util` so that a layer analysed by both yields the same numbers.
//!
//! Matrix extraction is the caller's job (torch on the Python side). The rules this crate expects, which are
//! WeightWatcher's defaults: a Linear layer is the matrix itself; a Conv2d weight of shape (out, in, k, k) gives k*k
//! separate out x in matrices whose eigenvalues are pooled into one ESD per layer (WeightWatcher `pool=True`),
//! with `n` the larger and `m` the smaller of (out, in); 1x1 convolutions and output heads are excluded by default;
//! fused attention projections (in_proj, or q, k, v) are split one matrix per head, each fitted alone and the layer
//! reported as the mean over heads. Layers whose smaller dimension is below 8 have no fit (WeightWatcher's
//! `min_evals`).

use faer::MatRef;
use serde::{Deserialize, Serialize};

/// WeightWatcher's threshold below which an eigenvalue is noise and excluded from the fit (`EVALS_THRESH`).
pub const EVALS_THRESH: f64 = 1e-5;
/// Below this many eigenvalues WeightWatcher 0.7.7 uses its bias-corrected small-N fit.
pub const SMALL_N_CUTOFF: usize = 20;
/// WeightWatcher's `EPSILON`, added inside the entropy to keep the logarithms finite.
pub const EPSILON: f64 = 1e-10;
/// Fewer eigenvalues than this and WeightWatcher skips the layer (`DEFAULT_MIN_EVALS`).
pub const MIN_EVALS: usize = 8;

/// A continuous power-law fit of an ESD tail.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerLaw {
    /// The tail exponent, `1 + n / sum(ln(x / xmin))` over the `n_tail` eigenvalues at or above `xmin`.
    pub alpha: f64,
    /// The smallest eigenvalue in the fitted tail.
    pub xmin: f64,
    /// Kolmogorov–Smirnov distance between the tail's empirical CDF and the fitted CDF.
    pub ks_distance: f64,
    /// WeightWatcher's `sigma`: `(alpha - 1) / sqrt(n_tail)`.
    pub sigma: f64,
    /// Eigenvalues in the fitted tail.
    pub n_tail: usize,
}

/// Everything WeightWatcher reports per layer that follows from the ESD alone.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerSpectrum {
    pub alpha: f64,
    /// `alpha * log10(lambda_max)`.
    pub alpha_weighted: f64,
    pub xmin: f64,
    /// The KS distance of the chosen fit (WeightWatcher's `D`).
    pub ks: f64,
    pub sigma: f64,
    pub lambda_max: f64,
    /// `sum(esd) / lambda_max`.
    pub stable_rank: f64,
    /// `log10(lambda_max)`.
    pub log_spectral_norm: f64,
    /// `log10(sum(esd))`, the log of the squared Frobenius norm (WeightWatcher's `log_norm`).
    pub log_frobenius_norm: f64,
    /// `log10(sum(esd ^ alpha))`.
    pub log_alpha_norm: f64,
    /// `-sum(p ln p) / ln(rank)` with `p = esd / sum(esd)`.
    pub entropy: f64,
    /// Eigenvalues in the ESD (pooled over conv slices).
    pub num_evals: usize,
    /// Eigenvalues in the fitted tail (WeightWatcher's `num_pl_spikes`).
    pub num_pl_spikes: usize,
    /// Singular values at or below `max * n * eps`, summed over the slices.
    pub rank_loss: usize,
    /// Eigenvalues below 1e-6 (WeightWatcher's `weak_rank_loss`).
    pub weak_rank_loss: usize,
}

/// Squared singular values of a row-major `rows x cols` matrix, ascending. The SVD runs on the CPU in f64.
pub fn esd(matrix: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(matrix.len(), rows * cols, "matrix has {} entries, not {rows}x{cols}", matrix.len());
    let data: Vec<f64> = matrix.iter().map(|&x| x as f64).collect();
    let m = MatRef::from_row_major_slice(&data, rows, cols);
    let svals = m.singular_values().expect("SVD did not converge");
    esd_from_svals(&svals)
}

/// Squared singular values, ascending, for callers that computed the SVD elsewhere (on the GPU, say).
pub fn esd_from_svals(svals: &[f64]) -> Vec<f64> {
    let mut e: Vec<f64> = svals.iter().map(|s| s * s).collect();
    e.sort_by(|a, b| a.total_cmp(b));
    e
}

fn sorted(esd: &[f64]) -> Vec<f64> {
    let mut e = esd.to_vec();
    e.sort_by(|a, b| a.total_cmp(b));
    e
}

/// WeightWatcher's power-law fit of an ESD: drops eigenvalues at or below [`EVALS_THRESH`], then picks the xmin
/// (every value but the largest) with the smallest KS distance, or the small-N objective below [`SMALL_N_CUTOFF`].
/// `None` when fewer than two eigenvalues remain or the fit degenerates.
pub fn power_law_fit(esd: &[f64]) -> Option<PowerLaw> {
    let data: Vec<f64> = sorted(esd).into_iter().filter(|&x| x > EVALS_THRESH).collect();
    if data.len() < 2 {
        return None;
    }
    if data.len() < SMALL_N_CUTOFF {
        fit_small_n(&data)
    } else {
        fit_standard(&data)
    }
}

fn ks_distance(tail: &[f64], xmin: f64, alpha: f64) -> f64 {
    let n = tail.len() as f64;
    tail.iter().enumerate().map(|(j, &x)| (1.0 - (x / xmin).powf(1.0 - alpha) - j as f64 / n).abs()).fold(0.0, f64::max)
}

fn fit_standard(data: &[f64]) -> Option<PowerLaw> {
    let total = data.len();
    let logs: Vec<f64> = data.iter().map(|x| x.ln()).collect();
    let mut suffix = vec![0.0; total + 1];
    for i in (0..total).rev() {
        suffix[i] = suffix[i + 1] + logs[i];
    }
    let mut best: Option<(f64, usize, f64)> = None;
    for i in 0..total - 1 {
        let n = (total - i) as f64;
        let alpha = 1.0 + n / (suffix[i] - n * logs[i]);
        let d = if alpha > 1.0 { ks_distance(&data[i..], data[i], alpha) } else { 1.0 };
        if best.is_none_or(|(bd, _, _)| d < bd) {
            best = Some((d, i, alpha));
        }
    }
    let (d, i, alpha) = best?;
    let n_tail = total - i;
    Some(PowerLaw { alpha, xmin: data[i], ks_distance: d, sigma: (alpha - 1.0) / (n_tail as f64).sqrt(), n_tail })
}

fn fit_small_n(data: &[f64]) -> Option<PowerLaw> {
    const K_MIN: usize = 8;
    let total = data.len();
    let logs: Vec<f64> = data.iter().map(|x| x.ln()).collect();
    let mut best: Option<(f64, usize, f64, f64)> = None;
    for i in 0..total - 1 {
        let n_int = total - i;
        if n_int < K_MIN {
            continue;
        }
        let n = n_int as f64;
        let s = logs[i..].iter().sum::<f64>() - n * logs[i];
        if s <= 1e-12 {
            continue;
        }
        let alpha = 1.0 + (n - 1.0) / s;
        if alpha <= 1.0 {
            continue;
        }
        let d = ks_distance(&data[i..], data[i], alpha);
        let j = d - 0.868 / n.sqrt();
        if best.is_none_or(|(bj, _, _, _)| j < bj) {
            best = Some((j, i, alpha, d));
        }
    }
    let (i, alpha, d) = match best {
        Some((_, i, alpha, d)) => (i, alpha, d),
        None => {
            let n = total as f64;
            let s = logs.iter().sum::<f64>() - n * logs[0];
            if s <= 1e-12 {
                return Some(PowerLaw { alpha: 1.0, xmin: data[0], ks_distance: 1.0, sigma: 0.0, n_tail: total });
            }
            let alpha = 1.0 + (n - 1.0) / s;
            (0, alpha, ks_distance(data, data[0], alpha))
        }
    };
    let n_tail = total - i;
    Some(PowerLaw { alpha, xmin: data[i], ks_distance: d, sigma: (alpha - 1.0) / (n_tail as f64).sqrt(), n_tail })
}

/// TempBalance's Hill estimator: `xmin` fixed at the eigenvalue at index `len / xmin_pos` of the ascending ESD
/// (`xmin_pos = 2` is the median), alpha the CSN MLE above it. No xmin search, so it is stable between steps.
pub fn alpha_hill(esd: &[f64], xmin_pos: usize) -> Option<PowerLaw> {
    let data: Vec<f64> = sorted(esd).into_iter().filter(|&x| x > EVALS_THRESH).collect();
    let total = data.len();
    if total < 2 || xmin_pos == 0 {
        return None;
    }
    let i = (total / xmin_pos).min(total - 1);
    let n = (total - i) as f64;
    let s: f64 = data[i..].iter().map(|x| x.ln()).sum::<f64>() - n * data[i].ln();
    if s <= 0.0 {
        return None;
    }
    let alpha = 1.0 + n / s;
    let d = ks_distance(&data[i..], data[i], alpha);
    let n_tail = total - i;
    Some(PowerLaw { alpha, xmin: data[i], ks_distance: d, sigma: (alpha - 1.0) / (n_tail as f64).sqrt(), n_tail })
}

/// `numpy.linalg.matrix_rank` from the singular values: count above `max(s) * n * eps(f64)`.
pub fn matrix_rank(svals: &[f64], n: usize) -> usize {
    let max = svals.iter().cloned().fold(0.0, f64::max);
    let tol = max * n as f64 * f64::EPSILON;
    svals.iter().filter(|&&s| s > tol).count()
}

/// WeightWatcher's per-layer metrics from a pooled ESD. `n` is the larger matrix dimension (used by the rank
/// tolerance) and `rank_loss` the sum of per-slice rank losses the caller computed with [`matrix_rank`] (pass
/// `None` to compute it on the pooled singular values as a single slice). `None` when the ESD is empty or the fit
/// fails.
pub fn layer_metrics(esd: &[f64], n: usize, rank_loss: Option<usize>) -> Option<LayerSpectrum> {
    let e = sorted(esd);
    if e.is_empty() {
        return None;
    }
    let fit = power_law_fit(&e)?;
    Some(metrics_with_fit(&e, n, rank_loss, fit))
}

/// [`layer_metrics`] with the fit already chosen (the Hill estimator, say).
pub fn metrics_with_fit(esd: &[f64], n: usize, rank_loss: Option<usize>, fit: PowerLaw) -> LayerSpectrum {
    let e = sorted(esd);
    let lambda_max = *e.last().unwrap();
    let norm: f64 = e.iter().sum();
    let svals: Vec<f64> = e.iter().map(|x| x.sqrt()).collect();
    let rank = matrix_rank(&svals, n);
    let entropy = -e.iter().map(|x| x / norm + EPSILON).map(|p| p * p.ln()).sum::<f64>() / (rank as f64 + EPSILON).ln();
    LayerSpectrum {
        alpha: fit.alpha,
        alpha_weighted: fit.alpha * lambda_max.log10(),
        xmin: fit.xmin,
        ks: fit.ks_distance,
        sigma: fit.sigma,
        lambda_max,
        stable_rank: norm / lambda_max,
        log_spectral_norm: lambda_max.log10(),
        log_frobenius_norm: norm.log10(),
        log_alpha_norm: e.iter().map(|x| x.powf(fit.alpha)).sum::<f64>().log10(),
        entropy,
        num_evals: e.len(),
        num_pl_spikes: fit.n_tail,
        rank_loss: rank_loss.unwrap_or(svals.len() - rank),
        weak_rank_loss: e.iter().filter(|&&x| x < 1e-6).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esd_matches_svals_squared() {
        let m = [3.0f32, 0.0, 0.0, 4.0, 0.0, 0.0];
        let e = esd(&m, 3, 2);
        assert!((e[0] - 9.0).abs() < 1e-9 && (e[1] - 16.0).abs() < 1e-9, "{e:?}");
    }

    #[test]
    fn pareto_tail_recovers_alpha() {
        let alpha = 3.0;
        let n = 4000;
        let data: Vec<f64> = (0..n).map(|i| ((i as f64 + 0.5) / n as f64).powf(-1.0 / (alpha - 1.0))).collect();
        let fit = power_law_fit(&data).unwrap();
        assert!((fit.alpha - alpha).abs() < 0.05, "{fit:?}");
        assert!(fit.ks_distance < 0.02, "{fit:?}");
    }

    #[test]
    fn hill_uses_the_median() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let fit = alpha_hill(&data, 2).unwrap();
        assert_eq!(fit.xmin, 51.0);
        assert_eq!(fit.n_tail, 50);
    }

    #[test]
    fn small_n_path_runs() {
        let data: Vec<f64> = (1..=12).map(|i| (i as f64).powi(2)).collect();
        let fit = power_law_fit(&data).unwrap();
        assert!(fit.alpha > 1.0 && fit.n_tail >= 8, "{fit:?}");
    }

    #[test]
    fn metrics_shape() {
        let data: Vec<f64> = (1..=64).map(|i| (i as f64).powf(1.5)).collect();
        let m = layer_metrics(&data, 64, None).unwrap();
        assert_eq!(m.num_evals, 64);
        assert!((m.lambda_max - 512.0).abs() < 1e-9);
        assert!((m.alpha_weighted - m.alpha * 512f64.log10()).abs() < 1e-12);
        assert!(m.entropy > 0.0 && m.entropy <= 1.0 + 1e-9);
        assert_eq!(m.rank_loss, 0);
    }
}
