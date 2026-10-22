//! `relay_opal._signals`: the Rust signals behind a thin numpy surface. Arrays in are 1-D float32 (gradients,
//! parameters, matrices, Hessian vectors) or float64 (eigenvalues); results are Python dicts, lists and 1-D numpy
//! arrays that `relay_opal.signals` turns into polars frames.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use relay_opal_signals::{hessian, instruments, spectrum, tempbalance};

type F64Array<'py> = Bound<'py, PyArray1<f64>>;
type F32Array<'py> = Bound<'py, PyArray1<f32>>;

fn power_law_dict<'py>(py: Python<'py>, fit: &spectrum::PowerLaw) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("alpha", fit.alpha)?;
    d.set_item("xmin", fit.xmin)?;
    d.set_item("ks_distance", fit.ks_distance)?;
    d.set_item("sigma", fit.sigma)?;
    d.set_item("n_tail", fit.n_tail)?;
    Ok(d)
}

fn spectrum_dict<'py>(py: Python<'py>, m: &spectrum::LayerSpectrum) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("alpha", m.alpha)?;
    d.set_item("alpha_weighted", m.alpha_weighted)?;
    d.set_item("xmin", m.xmin)?;
    d.set_item("ks", m.ks)?;
    d.set_item("sigma", m.sigma)?;
    d.set_item("lambda_max", m.lambda_max)?;
    d.set_item("stable_rank", m.stable_rank)?;
    d.set_item("log_spectral_norm", m.log_spectral_norm)?;
    d.set_item("log_frobenius_norm", m.log_frobenius_norm)?;
    d.set_item("log_alpha_norm", m.log_alpha_norm)?;
    d.set_item("entropy", m.entropy)?;
    d.set_item("num_evals", m.num_evals)?;
    d.set_item("num_pl_spikes", m.num_pl_spikes)?;
    d.set_item("rank_loss", m.rank_loss)?;
    d.set_item("weak_rank_loss", m.weak_rank_loss)?;
    Ok(d)
}

/// Eigenvalues of W^T W (ascending) of a row-major `rows x cols` float32 matrix, by SVD on the CPU.
#[pyfunction]
fn esd<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray1<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = matrix.as_slice()?;
    if m.len() != rows * cols {
        return Err(PyValueError::new_err(format!("matrix has {} entries, not {rows}x{cols}", m.len())));
    }
    Ok(PyArray1::from_vec(py, spectrum::esd(m, rows, cols)))
}

/// Squared singular values, ascending.
#[pyfunction]
fn esd_from_svals<'py>(py: Python<'py>, svals: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec(py, spectrum::esd_from_svals(svals.as_slice()?)))
}

/// WeightWatcher's power-law fit of an ESD; None when it fails.
#[pyfunction]
fn power_law_fit<'py>(py: Python<'py>, esd: PyReadonlyArray1<f64>) -> PyResult<Option<Bound<'py, PyDict>>> {
    spectrum::power_law_fit(esd.as_slice()?).map(|f| power_law_dict(py, &f)).transpose()
}

/// TempBalance's Hill estimator with xmin at index len / xmin_pos.
#[pyfunction]
#[pyo3(signature = (esd, xmin_pos=2))]
fn alpha_hill<'py>(
    py: Python<'py>,
    esd: PyReadonlyArray1<f64>,
    xmin_pos: usize,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    spectrum::alpha_hill(esd.as_slice()?, xmin_pos).map(|f| power_law_dict(py, &f)).transpose()
}

/// Every per-layer metric from a pooled ESD; `n` is the larger matrix dimension; `rank_loss` the caller's per-slice
/// sum, or None to compute it on the pooled values. `hill_xmin_pos` switches the fit to the Hill estimator.
#[pyfunction]
#[pyo3(signature = (esd, n, rank_loss=None, hill_xmin_pos=None))]
fn layer_metrics<'py>(
    py: Python<'py>,
    esd: PyReadonlyArray1<f64>,
    n: usize,
    rank_loss: Option<usize>,
    hill_xmin_pos: Option<usize>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let e = esd.as_slice()?;
    let m = match hill_xmin_pos {
        Some(pos) => spectrum::alpha_hill(e, pos).map(|fit| spectrum::metrics_with_fit(e, n, rank_loss, fit)),
        None => spectrum::layer_metrics(e, n, rank_loss),
    };
    m.map(|m| spectrum_dict(py, &m)).transpose()
}

/// numpy.linalg.matrix_rank from singular values with the larger dimension `n`.
#[pyfunction]
fn matrix_rank(svals: PyReadonlyArray1<f64>, n: usize) -> PyResult<usize> {
    Ok(spectrum::matrix_rank(svals.as_slice()?, n))
}

/// One learning rate per layer from its alpha. rule: linear_map | sqrt | log2 | step.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (alphas, base_lr, rule="linear_map", lr_min_ratio=0.5, lr_max_ratio=1.5, exclude=None, rescale=true))]
fn layer_lrs<'py>(
    py: Python<'py>,
    alphas: PyReadonlyArray1<f64>,
    base_lr: f64,
    rule: &str,
    lr_min_ratio: f64,
    lr_max_ratio: f64,
    exclude: Option<Vec<bool>>,
    rescale: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rule = tempbalance::Rule::parse(rule)
        .ok_or_else(|| PyValueError::new_err(format!("unknown rule {rule:?}: linear_map, sqrt, log2 or step")))?;
    let a = alphas.as_slice()?;
    let exclude = exclude.unwrap_or_else(|| vec![false; a.len()]);
    Ok(PyArray1::from_vec(py, tempbalance::layer_lrs(a, base_lr, rule, lr_min_ratio, lr_max_ratio, &exclude, rescale)))
}

/// The SGD noise scale lr / batch_size / (1 - momentum).
#[pyfunction]
fn temperature(lr: f64, batch_size: f64, momentum: f64) -> f64 {
    tempbalance::temperature(lr, batch_size, momentum)
}

/// Cockpit's local step size from the two losses, the two gradient-update dots and the update's norm.
#[pyfunction]
fn local_step_alpha(loss: f64, loss_next: f64, g_dot_delta: f64, g_next_dot_delta: f64, step_len: f64) -> Option<f64> {
    instruments::local_step_alpha(loss, loss_next, g_dot_delta, g_next_dot_delta, step_len)
}

/// The per-group window of recent gradients and the pending update; one per Instruments.
#[pyclass]
struct Window {
    inner: instruments::Window,
}

#[pymethods]
impl Window {
    #[new]
    fn new(k: usize, n_groups: usize) -> Self {
        Window { inner: instruments::Window::new(k, n_groups) }
    }

    #[getter]
    fn n_groups(&self) -> usize {
        self.inner.n_groups()
    }

    /// Returns the completed row of the previous step (a dict) or None.
    fn before_step<'py>(
        &mut self,
        py: Python<'py>,
        group: usize,
        grad: PyReadonlyArray1<f32>,
        theta: PyReadonlyArray1<f32>,
        loss: f64,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        if group >= self.inner.n_groups() {
            return Err(PyValueError::new_err(format!("group {group} of {}", self.inner.n_groups())));
        }
        let row = self.inner.before_step(group, grad.as_slice()?, theta.as_slice()?, loss);
        row.map(|r| {
            let d = PyDict::new(py);
            d.set_item("grad_norm", r.grad_norm)?;
            d.set_item("update_ratio", r.update_ratio)?;
            d.set_item("grad_cosine", r.grad_cosine)?;
            d.set_item("snr", r.snr)?;
            d.set_item("alpha", r.alpha)?;
            d.set_item("loss", r.loss)?;
            d.set_item("loss_next", r.loss_next)?;
            Ok(d)
        })
        .transpose()
    }

    fn after_step(&mut self, group: usize, theta_next: PyReadonlyArray1<f32>) -> PyResult<()> {
        if group >= self.inner.n_groups() {
            return Err(PyValueError::new_err(format!("group {group} of {}", self.inner.n_groups())));
        }
        self.inner.after_step(group, theta_next.as_slice()?);
        Ok(())
    }
}

struct PyHvp<'py> {
    py: Python<'py>,
    f: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

impl hessian::Hvp for PyHvp<'_> {
    fn apply(&mut self, v: &[f32]) -> Vec<f32> {
        if self.error.is_some() {
            return vec![0.0; v.len()];
        }
        let out = (|| -> PyResult<Vec<f32>> {
            let arr = PyArray1::from_slice(self.py, v);
            let res = self.f.call1((arr,))?;
            let hv: PyReadonlyArray1<f32> = res.extract()?;
            let s = hv.as_slice()?;
            if s.len() != v.len() {
                return Err(PyValueError::new_err(format!(
                    "hvp returned {} values for a vector of {}",
                    s.len(),
                    v.len()
                )));
            }
            Ok(s.to_vec())
        })();
        match out {
            Ok(hv) => hv,
            Err(e) => {
                self.error = Some(e);
                vec![0.0; v.len()]
            }
        }
    }
}

fn raise_if_failed(h: PyHvp<'_>) -> PyResult<()> {
    match h.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// PyHessian's power iteration with deflation: (eigenvalues largest first, eigenvectors as a list of arrays).
#[pyfunction]
#[pyo3(signature = (hvp, dim, top_n=1, max_iter=100, tol=1e-3, seed=0))]
fn top_eigenvalues<'py>(
    py: Python<'py>,
    hvp: Bound<'py, PyAny>,
    dim: usize,
    top_n: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<(Vec<f64>, Vec<F32Array<'py>>)> {
    let mut h = PyHvp { py, f: hvp, error: None };
    let (values, vectors) = hessian::top_eigenvalues(&mut h, dim, top_n, max_iter, tol, seed);
    raise_if_failed(h)?;
    Ok((values, vectors.into_iter().map(|v| PyArray1::from_vec(py, v)).collect()))
}

/// Top eigenvalues by Lanczos with full reorthogonalisation, largest first.
#[pyfunction]
#[pyo3(signature = (hvp, dim, top_n=1, iters=20, seed=0))]
fn lanczos_eigenvalues(hvp: Bound<'_, PyAny>, dim: usize, top_n: usize, iters: usize, seed: u64) -> PyResult<Vec<f64>> {
    let py = hvp.py();
    let mut h = PyHvp { py, f: hvp, error: None };
    let values = hessian::lanczos_eigenvalues(&mut h, dim, top_n, iters, seed);
    raise_if_failed(h)?;
    Ok(values)
}

/// Hutchinson's v.Hv per Rademacher probe, PyHessian's stopping rule; the trace estimate is the mean.
#[pyfunction]
#[pyo3(signature = (hvp, dim, max_probes=100, tol=1e-3, seed=0))]
fn trace(hvp: Bound<'_, PyAny>, dim: usize, max_probes: usize, tol: f64, seed: u64) -> PyResult<Vec<f64>> {
    let py = hvp.py();
    let mut h = PyHvp { py, f: hvp, error: None };
    let values = hessian::trace(&mut h, dim, max_probes, tol, seed);
    raise_if_failed(h)?;
    Ok(values)
}

/// Stochastic Lanczos quadrature: per probe (eigenvalues ascending, weights), PyHessian's density().
#[pyfunction]
#[pyo3(signature = (hvp, dim, n_probes=1, iters=100, seed=0))]
fn slq(
    hvp: Bound<'_, PyAny>,
    dim: usize,
    n_probes: usize,
    iters: usize,
    seed: u64,
) -> PyResult<Vec<(Vec<f64>, Vec<f64>)>> {
    let py = hvp.py();
    let mut h = PyHvp { py, f: hvp, error: None };
    let runs = hessian::slq(&mut h, dim, n_probes, iters, seed);
    raise_if_failed(h)?;
    Ok(runs)
}

/// PyHessian's density_generate: (density, grid) from slq() runs.
#[pyfunction]
#[pyo3(signature = (runs, num_bins=10000, sigma_squared=1e-5, overhead=0.01))]
fn density_on_grid<'py>(
    py: Python<'py>,
    runs: Vec<(Vec<f64>, Vec<f64>)>,
    num_bins: usize,
    sigma_squared: f64,
    overhead: f64,
) -> PyResult<(F64Array<'py>, F64Array<'py>)> {
    if runs.is_empty() || runs.iter().any(|(e, _)| e.is_empty()) {
        return Err(PyRuntimeError::new_err("density_on_grid needs at least one non-empty run"));
    }
    let (density, grid) = hessian::density_on_grid(&runs, num_bins, sigma_squared, overhead);
    Ok((PyArray1::from_vec(py, density), PyArray1::from_vec(py, grid)))
}

/// lambda_max * lr / 2.
#[pyfunction]
fn edge_of_stability(lambda_max: f64, lr: f64) -> f64 {
    hessian::edge_of_stability(lambda_max, lr)
}

#[pymodule]
fn _signals(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(esd, m)?)?;
    m.add_function(wrap_pyfunction!(esd_from_svals, m)?)?;
    m.add_function(wrap_pyfunction!(power_law_fit, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_hill, m)?)?;
    m.add_function(wrap_pyfunction!(layer_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_rank, m)?)?;
    m.add_function(wrap_pyfunction!(layer_lrs, m)?)?;
    m.add_function(wrap_pyfunction!(temperature, m)?)?;
    m.add_function(wrap_pyfunction!(local_step_alpha, m)?)?;
    m.add_class::<Window>()?;
    m.add_function(wrap_pyfunction!(top_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(lanczos_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(slq, m)?)?;
    m.add_function(wrap_pyfunction!(density_on_grid, m)?)?;
    m.add_function(wrap_pyfunction!(edge_of_stability, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
