//! The fixture the Python TempBalance test writes: ResNet-shaped layer alphas, the base rate and the rates the
//! Python API set (linear map, first and last layer excluded, mean preserved). Rust reproduces them alone.

use relay_opal_signals::tempbalance::{layer_lrs, Rule};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    alphas: BTreeMap<String, f64>,
    base_lr: f64,
    lr: BTreeMap<String, f64>,
}

#[test]
fn reproduces_the_python_rates() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/signals_tempbalance.json");
    let f: Fixture = serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).unwrap();
    let names = [
        "conv1",
        "layer1_0_conv1",
        "layer1_0_conv2",
        "layer2_0_conv1",
        "layer2_0_conv2",
        "layer3_0_conv1",
        "layer3_0_conv2",
        "layer4_0_conv1",
        "layer4_0_conv2",
        "fc",
    ];
    let alphas: Vec<f64> = names.iter().map(|n| f.alphas[*n]).collect();
    let exclude: Vec<bool> = names.iter().map(|n| *n == "conv1" || *n == "fc").collect();
    let lrs = layer_lrs(&alphas, f.base_lr, Rule::LinearMap, 0.5, 1.5, &exclude, true);
    for (n, lr) in names.iter().zip(lrs) {
        assert!((lr - f.lr[*n]).abs() < 1e-12, "{n}: {lr} vs {}", f.lr[*n]);
    }
}
