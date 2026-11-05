//! The same fixture the Python test writes from WeightWatcher 0.7.7's analyze(): 20 heavy-tailed ESDs with the
//! numbers WeightWatcher reported for them. The Rust fit must reproduce them without Python in the loop.

use relay_opal_signals::spectrum::layer_metrics;
use serde::Deserialize;

#[derive(Deserialize)]
struct Layer {
    name: String,
    n: usize,
    esd: Vec<f64>,
    alpha: f64,
    alpha_weighted: f64,
    stable_rank: f64,
    log_frobenius_norm: f64,
    xmin: f64,
    ks: f64,
    num_evals: f64,
    entropy: f64,
}

#[test]
fn reproduces_weightwatcher_on_the_fixture() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/signals_weightwatcher.json");
    let layers: Vec<Layer> = serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).unwrap();
    assert_eq!(layers.len(), 20);
    for l in &layers {
        let m = layer_metrics(&l.esd, l.n, None).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1.0);
        assert!(close(m.alpha, l.alpha), "{}: alpha {} vs {}", l.name, m.alpha, l.alpha);
        assert!(close(m.xmin, l.xmin), "{}: xmin {} vs {}", l.name, m.xmin, l.xmin);
        assert!(close(m.ks, l.ks), "{}: ks {} vs {}", l.name, m.ks, l.ks);
        assert!(close(m.alpha_weighted, l.alpha_weighted), "{}: alpha_weighted", l.name);
        assert!(close(m.stable_rank, l.stable_rank), "{}: stable_rank", l.name);
        assert!(close(m.log_frobenius_norm, l.log_frobenius_norm), "{}: log_norm", l.name);
        assert!(close(m.entropy, l.entropy), "{}: entropy {} vs {}", l.name, m.entropy, l.entropy);
        assert_eq!(m.num_evals as f64, l.num_evals, "{}", l.name);
    }
}
