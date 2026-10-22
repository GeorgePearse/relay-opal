//! TempBalance (Zhou, Pang, Liu, Martin, Mahoney, Yang, NeurIPS 2023): per-layer learning rates from per-layer
//! alpha. A small alpha is a hot, heavy-tailed, well-trained layer and gets a lower rate; a large alpha is a cold,
//! under-trained layer and gets a higher one. The paper defines one assignment, its equation (2), the linear map
//! from `[alpha_min, alpha_max]` onto `[s1 * lr, s2 * lr]`; the reference code adds `sqrt`, `log2` and `step`. The
//! paper has no sigmoid rule, so none is offered. The reference code's linear map does not rescale, so the budget
//! moves with the alphas; [`layer_lrs`] rescales by default so the mean rate over the balanced layers is the base
//! rate (the sum is `n * base_lr`), which is what "total budget unchanged" means here, and the `sqrt` and `log2`
//! rules are budget-preserving already. Excluded layers (the reference excludes the first and the last) keep the
//! base rate and do not count towards the budget.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rule {
    /// `lr_min + (lr_max - lr_min) * (alpha - alpha_min) / (alpha_max - alpha_min)`, the paper's equation (2).
    LinearMap,
    /// `base_lr * n * sqrt(alpha) / sum(sqrt(alpha))`.
    Sqrt,
    /// `base_lr * n * log2(alpha) / sum(log2(alpha))`.
    Log2,
    /// Ranks: the i-th smallest alpha gets `base_lr * (lr_min_ratio + (lr_max_ratio - lr_min_ratio) * i / n)`.
    Step,
}

impl Rule {
    pub fn parse(name: &str) -> Option<Rule> {
        match name {
            "linear_map" | "tb_linear_map" => Some(Rule::LinearMap),
            "sqrt" | "tb_sqrt" => Some(Rule::Sqrt),
            "log2" | "tb_log2" => Some(Rule::Log2),
            "step" | "tb_step" => Some(Rule::Step),
            _ => None,
        }
    }
}

/// One learning rate per layer. `exclude[i]` keeps layer i at `base_lr` and leaves it out of the balancing; a
/// non-finite alpha excludes the layer too. `rescale` makes the mean over the balanced layers equal `base_lr`.
pub fn layer_lrs(
    alphas: &[f64],
    base_lr: f64,
    rule: Rule,
    lr_min_ratio: f64,
    lr_max_ratio: f64,
    exclude: &[bool],
    rescale: bool,
) -> Vec<f64> {
    let mut out = vec![base_lr; alphas.len()];
    let idx: Vec<usize> =
        (0..alphas.len()).filter(|&i| !exclude.get(i).copied().unwrap_or(false) && alphas[i].is_finite()).collect();
    if idx.is_empty() {
        return out;
    }
    let a: Vec<f64> = idx.iter().map(|&i| alphas[i]).collect();
    let n = a.len() as f64;
    let (lo, hi) = (lr_min_ratio * base_lr, lr_max_ratio * base_lr);
    let mut lrs: Vec<f64> = match rule {
        Rule::LinearMap => {
            let amin = a.iter().cloned().fold(f64::INFINITY, f64::min);
            let amax = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if amax > amin {
                a.iter().map(|x| lo + (hi - lo) * (x - amin) / (amax - amin)).collect()
            } else {
                vec![base_lr; a.len()]
            }
        }
        Rule::Sqrt => {
            let s: f64 = a.iter().map(|x| x.sqrt()).sum();
            a.iter().map(|x| base_lr * n * x.sqrt() / s).collect()
        }
        Rule::Log2 => {
            let s: f64 = a.iter().map(|x| x.log2()).sum();
            a.iter().map(|x| base_lr * n * x.log2() / s).collect()
        }
        Rule::Step => {
            let mut order: Vec<usize> = (0..a.len()).collect();
            order.sort_by(|&i, &j| a[i].total_cmp(&a[j]));
            let mut lrs = vec![base_lr; a.len()];
            for (rank, &i) in order.iter().enumerate() {
                lrs[i] = base_lr * (lr_min_ratio + (lr_max_ratio - lr_min_ratio) * rank as f64 / n);
            }
            lrs
        }
    };
    if rescale {
        let mean = lrs.iter().sum::<f64>() / n;
        if mean > 0.0 {
            for lr in &mut lrs {
                *lr *= base_lr / mean;
            }
        }
    }
    for (&i, lr) in idx.iter().zip(lrs) {
        out[i] = lr;
    }
    out
}

/// The paper's temperature, the SGD noise scale: `lr / batch_size / (1 - momentum)`.
pub fn temperature(lr: f64, batch_size: f64, momentum: f64) -> f64 {
    lr / batch_size / (1.0 - momentum)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHAS: [f64; 5] = [2.0, 6.0, 3.0, 4.5, 2.5];

    #[test]
    fn linear_map_orders_and_preserves_budget() {
        let lrs = layer_lrs(&ALPHAS, 0.1, Rule::LinearMap, 0.5, 1.5, &[false; 5], true);
        assert!(lrs[0] < lrs[4] && lrs[4] < lrs[2] && lrs[2] < lrs[3] && lrs[3] < lrs[1], "{lrs:?}");
        assert!((lrs.iter().sum::<f64>() / 5.0 - 0.1).abs() < 1e-12);
    }

    #[test]
    fn unscaled_linear_map_is_the_paper_formula() {
        let lrs = layer_lrs(&ALPHAS, 0.1, Rule::LinearMap, 0.5, 1.5, &[false; 5], false);
        assert!((lrs[0] - 0.05).abs() < 1e-12 && (lrs[1] - 0.15).abs() < 1e-12, "{lrs:?}");
        assert!((lrs[2] - (0.05 + 0.1 * 0.25)).abs() < 1e-12);
    }

    #[test]
    fn excluded_keep_base() {
        let lrs = layer_lrs(&ALPHAS, 0.1, Rule::Sqrt, 0.5, 1.5, &[true, false, false, false, true], true);
        assert_eq!(lrs[0], 0.1);
        assert_eq!(lrs[4], 0.1);
        assert!((lrs[1..4].iter().sum::<f64>() / 3.0 - 0.1).abs() < 1e-12);
    }

    #[test]
    fn step_ranks() {
        let lrs = layer_lrs(&ALPHAS, 1.0, Rule::Step, 0.5, 1.5, &[false; 5], false);
        assert_eq!(lrs, vec![0.5, 1.3, 0.9, 1.1, 0.7]);
    }

    #[test]
    fn temperature_formula() {
        assert!((temperature(0.1, 128.0, 0.9) - 0.1 / 128.0 / 0.1).abs() < 1e-15);
    }
}
