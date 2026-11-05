//! PyHessian's Krylov drivers (Yao, Gholami, Keutzer and Mahoney) with the Hessian-vector product supplied as a
//! callback, so autograd stays wherever the model lives. The iterations copy PyHessian step for step, including
//! its normalisation `v / (||v|| + 1e-6)`, its Gram–Schmidt deflation and its stopping rules, so the numbers agree
//! with `pyhessian.hessian` to the randomness of the probes.
//!
//! - [`top_eigenvalues`]: power iteration with deflation, `top_n` eigenvalues, PyHessian's `eigenvalues()`.
//! - [`lanczos_eigenvalues`]: the same by Lanczos with full reorthogonalisation; fewer products for the same
//!   accuracy.
//! - [`trace`]: Hutchinson with Rademacher probes, PyHessian's `trace()` stopping rule.
//! - [`slq`] and [`density_on_grid`]: stochastic Lanczos quadrature, PyHessian's `density()` and
//!   `density_generate()`.
//! - [`edge_of_stability`]: `lambda_max * lr / 2`, 1 at the edge.

use faer::{Mat, Side};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Hessian-vector products as a callback: `v -> H v`, both of length `dim`.
pub trait Hvp {
    fn apply(&mut self, v: &[f32]) -> Vec<f32>;
}

impl<F: FnMut(&[f32]) -> Vec<f32>> Hvp for F {
    fn apply(&mut self, v: &[f32]) -> Vec<f32> {
        self(v)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// PyHessian's `normalization`: divides by the norm plus 1e-6.
fn normalize(v: &mut [f32]) {
    let s = dot(v, v).sqrt() + 1e-6;
    for x in v.iter_mut() {
        *x = (*x as f64 / s) as f32;
    }
}

fn orthonormal(w: &mut [f32], basis: &[Vec<f32>]) {
    for v in basis {
        let c = dot(w, v);
        for (wi, &vi) in w.iter_mut().zip(v) {
            *wi = (*wi as f64 - c * vi as f64) as f32;
        }
    }
    normalize(w);
}

fn gaussian(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            let u1 = rng.random::<f64>().max(1e-300);
            let u2: f64 = rng.random();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

fn rademacher(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 }).collect()
}

/// Power iteration with Gram–Schmidt deflation, PyHessian's `eigenvalues(maxIter, tol, top_n)`: each eigenvalue
/// stops when its relative change falls below `tol` or after `max_iter` products. Returns the eigenvalues, largest
/// first, and their vectors.
pub fn top_eigenvalues(
    hvp: &mut impl Hvp,
    dim: usize,
    top_n: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> (Vec<f64>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(top_n);
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(top_n);
    while values.len() < top_n {
        let mut v = gaussian(&mut rng, dim);
        normalize(&mut v);
        let mut eigenvalue: Option<f64> = None;
        for _ in 0..max_iter {
            orthonormal(&mut v, &vectors);
            let hv = hvp.apply(&v);
            let tmp = dot(&hv, &v);
            v = hv;
            normalize(&mut v);
            match eigenvalue {
                None => eigenvalue = Some(tmp),
                Some(prev) if (prev - tmp).abs() / (prev.abs() + 1e-6) < tol => break,
                Some(_) => eigenvalue = Some(tmp),
            }
        }
        values.push(eigenvalue.unwrap_or(f64::NAN));
        vectors.push(v);
    }
    (values, vectors)
}

/// Hutchinson's trace estimate with Rademacher probes, PyHessian's `trace(maxIter, tol)`: stops when the running
/// mean changes by less than `tol` relative. Returns every probe's `v . H v`; the estimate is their mean. One
/// deliberate difference: PyHessian divides the change by the signed running mean, so a negative first probe makes
/// the ratio negative and stops it after two probes; this divides by its absolute value. `tol = 0` runs every probe.
pub fn trace(hvp: &mut impl Hvp, dim: usize, max_probes: usize, tol: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vhv = Vec::new();
    let mut running = 0.0;
    for _ in 0..max_probes {
        let v = rademacher(&mut rng, dim);
        let hv = hvp.apply(&v);
        vhv.push(dot(&hv, &v));
        let mean = vhv.iter().sum::<f64>() / vhv.len() as f64;
        if (mean - running).abs() / (running.abs() + 1e-6) < tol {
            return vhv;
        }
        running = mean;
    }
    vhv
}

/// One Lanczos run of `iters` steps from a normalised start vector with full reorthogonalisation, exactly
/// PyHessian's loop: returns the tridiagonal (alphas, betas), with `betas.len() == alphas.len() - 1`.
fn lanczos(hvp: &mut impl Hvp, start: Vec<f32>, iters: usize, rng: &mut StdRng) -> (Vec<f64>, Vec<f64>) {
    let dim = start.len();
    let mut basis = vec![start];
    let mut alphas = Vec::with_capacity(iters);
    let mut betas = Vec::with_capacity(iters);
    let mut w: Vec<f32> = Vec::new();
    for i in 0..iters {
        if i == 0 {
            let v = &basis[0];
            let w_prime = hvp.apply(v);
            let alpha = dot(&w_prime, v);
            alphas.push(alpha);
            w = w_prime.iter().zip(v).map(|(&a, &b)| (a as f64 - alpha * b as f64) as f32).collect();
        } else {
            let beta = dot(&w, &w).sqrt();
            betas.push(beta);
            let mut v = if beta != 0.0 { w.clone() } else { gaussian(rng, dim) };
            orthonormal(&mut v, &basis);
            basis.push(v);
            let v = basis.last().unwrap();
            let w_prime = hvp.apply(v);
            let alpha = dot(&w_prime, v);
            alphas.push(alpha);
            let prev = &basis[basis.len() - 2];
            w = w_prime
                .iter()
                .zip(v)
                .zip(prev)
                .map(|((&a, &b), &c)| (a as f64 - alpha * b as f64 - beta * c as f64) as f32)
                .collect();
        }
    }
    (alphas, betas)
}

/// Eigenvalues (ascending) and first-row-squared weights of the symmetric tridiagonal matrix.
fn tridiagonal_eigen(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let m = alphas.len();
    let t = Mat::<f64>::from_fn(m, m, |i, j| {
        if i == j {
            alphas[i]
        } else if i + 1 == j || j + 1 == i {
            betas[i.min(j)]
        } else {
            0.0
        }
    });
    let eig = t.self_adjoint_eigen(Side::Lower).expect("tridiagonal eigendecomposition failed");
    let values: Vec<f64> = (0..m).map(|i| eig.S().column_vector()[i]).collect();
    let weights: Vec<f64> = (0..m).map(|j| eig.U()[(0, j)].powi(2)).collect();
    (values, weights)
}

/// Top eigenvalues by Lanczos: `iters` products from a Gaussian start, the `top_n` largest Ritz values, largest
/// first.
pub fn lanczos_eigenvalues(hvp: &mut impl Hvp, dim: usize, top_n: usize, iters: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut start = gaussian(&mut rng, dim);
    normalize(&mut start);
    let (alphas, betas) = lanczos(hvp, start, iters.min(dim).max(1), &mut rng);
    let (mut values, _) = tridiagonal_eigen(&alphas, &betas);
    values.reverse();
    values.truncate(top_n);
    values
}

/// Stochastic Lanczos quadrature, PyHessian's `density(iter, n_v)`: one Rademacher probe per run, `iters` Lanczos
/// steps, and per run the Ritz values with the squared first components of their vectors as weights.
pub fn slq(hvp: &mut impl Hvp, dim: usize, n_probes: usize, iters: usize, seed: u64) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_probes)
        .map(|_| {
            let mut v = rademacher(&mut rng, dim);
            normalize(&mut v);
            let (alphas, betas) = lanczos(hvp, v, iters.min(dim).max(1), &mut rng);
            tridiagonal_eigen(&alphas, &betas)
        })
        .collect()
}

/// PyHessian's `density_generate`: the Gaussian-smoothed density on `num_bins` points from `min - overhead` to
/// `max + overhead`, width `sigma_squared * max(1, range)`, averaged over the runs and normalised to integrate to
/// one. Returns `(density, grid)`.
pub fn density_on_grid(
    runs: &[(Vec<f64>, Vec<f64>)],
    num_bins: usize,
    sigma_squared: f64,
    overhead: f64,
) -> (Vec<f64>, Vec<f64>) {
    let all = runs.iter().flat_map(|(e, _)| e.iter().cloned());
    let lambda_max = all.clone().fold(f64::NEG_INFINITY, f64::max) + overhead;
    let lambda_min = all.fold(f64::INFINITY, f64::min) - overhead;
    let grid: Vec<f64> = (0..num_bins)
        .map(|i| lambda_min + (lambda_max - lambda_min) * i as f64 / (num_bins.max(2) - 1) as f64)
        .collect();
    let sigma = sigma_squared * (lambda_max - lambda_min).max(1.0);
    let mut density = vec![0.0; num_bins];
    for (eigs, weights) in runs {
        for (x, d) in grid.iter().zip(density.iter_mut()) {
            let mut acc = 0.0;
            for (&e, &w) in eigs.iter().zip(weights) {
                acc += w * (-(e - x).powi(2) / (2.0 * sigma)).exp() / (2.0 * std::f64::consts::PI * sigma).sqrt();
            }
            *d += acc / runs.len() as f64;
        }
    }
    let step = if num_bins > 1 { grid[1] - grid[0] } else { 1.0 };
    let total: f64 = density.iter().sum::<f64>() * step;
    if total > 0.0 {
        for d in &mut density {
            *d /= total;
        }
    }
    (density, grid)
}

/// `lambda_max * lr / 2`: the edge of stability is 1 (gradient descent diverges along the top eigenvector above it).
pub fn edge_of_stability(lambda_max: f64, lr: f64) -> f64 {
    lambda_max * lr / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(d: &[f64]) -> impl FnMut(&[f32]) -> Vec<f32> + '_ {
        move |v: &[f32]| v.iter().zip(d).map(|(&x, &k)| (x as f64 * k) as f32).collect()
    }

    const D: [f64; 6] = [5.0, -3.0, 2.0, 1.0, 0.5, 0.1];

    #[test]
    fn power_iteration_finds_the_top() {
        let mut h = diag(&D);
        let (vals, vecs) = top_eigenvalues(&mut h, 6, 2, 200, 1e-8, 0);
        assert!((vals[0] - 5.0).abs() < 1e-3, "{vals:?}");
        assert!((vals[1] + 3.0).abs() < 1e-2, "{vals:?}");
        assert!(vecs[0][0].abs() > 0.99);
    }

    #[test]
    fn lanczos_is_exact_at_full_dimension() {
        let mut h = diag(&D);
        let vals = lanczos_eigenvalues(&mut h, 6, 6, 6, 1);
        let mut want = D.to_vec();
        want.sort_by(|a, b| b.total_cmp(a));
        for (v, w) in vals.iter().zip(&want) {
            assert!((v - w).abs() < 1e-4, "{vals:?}");
        }
    }

    #[test]
    fn hutchinson_is_exact_for_diagonal() {
        let mut h = diag(&D);
        let vhv = trace(&mut h, 6, 5, 0.0, 3);
        for x in &vhv {
            assert!((x - D.iter().sum::<f64>()).abs() < 1e-4, "{vhv:?}");
        }
    }

    #[test]
    fn slq_weights_sum_to_one_and_density_integrates() {
        let mut h = diag(&D);
        let runs = slq(&mut h, 6, 2, 6, 4);
        for (e, w) in &runs {
            assert_eq!(e.len(), 6);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6, "{w:?}");
        }
        let (density, grid) = density_on_grid(&runs, 2000, 1e-5, 0.01);
        let integral: f64 = density.iter().sum::<f64>() * (grid[1] - grid[0]);
        assert!((integral - 1.0).abs() < 1e-9);
    }

    #[test]
    fn edge() {
        assert_eq!(edge_of_stability(4.0, 0.5), 1.0);
    }
}
