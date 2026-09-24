//! `sanity_gpu` against `sanity`, every variance rule. Needs a wgpu adapter.
//!
//! The device is `f32`, so parity is a tolerance, not equality. The gates sit
//! at the resolution of the method itself: the variance grid is resolved to 1%
//! of an error bar (`DEFAULT_VARIANCE_BINS`), so a log fold change within that
//! of the CPU's is as right as the CPU's.

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use sanity_sc_rs::config::{SanityParams, VarianceRule};
use sanity_sc_rs::gpu::sanity_gpu;
use sanity_sc_rs::sanity;
use sanity_sc_rs::simulate::{Simulation, SimulationParams, simulate};

/// Largest log fold change error, in units of the CPU's error bar.
const FOLD_GATE: f64 = 1e-2;

/// Largest relative error bar error.
const ERROR_GATE: f64 = 1e-2;

/// Largest error in `m`, in units of the CPU's error bar on `m`.
const MEAN_GATE: f64 = 5e-2;

/// Largest relative error in the reported variance.
const VARIANCE_GATE: f64 = 1e-2;

/// A simulated matrix with a wide spread of expression, from genes with a few
/// dozen counts to genes holding a large share of the reads.
///
/// Large enough to hold near ties: without the `MaxPosterior` tie-break, seven
/// of its genes land on a different bin than the CPU's.
fn fixture() -> Simulation {
    simulate(Some(SimulationParams {
        n_genes: 400,
        n_cells: 5000,
        library_size: 500.0,
        seed: 11,
        ..Default::default()
    }))
    .expect("simulates")
}

/// Run both paths under one rule and assert every gate.
fn assert_parity(rule: VarianceRule) {
    let sim = fixture();
    let n_cells = sim.cell_totals.len();
    let params = SanityParams {
        variance_rule: rule,
        ..SanityParams::default()
    };
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let cpu = sanity::<f64>(&sim.counts, &sim.cell_totals, Some(params)).expect("CPU runs");
    let gpu = sanity_gpu::<f64, WgpuRuntime>(&sim.counts, &sim.cell_totals, Some(params), &client)
        .expect("GPU runs");

    // A launch that silently did no work leaves zeros, which no error bar is.
    assert!(
        gpu.error_bars.iter().all(|&e| e > 0.0),
        "{rule:?}: the GPU left error bars at zero or NaN"
    );
    assert_eq!(gpu.genes, cpu.genes);

    for g in 0..cpu.n_genes {
        for c in g * n_cells..(g + 1) * n_cells {
            let e = cpu.error_bars[c];
            let fold = (gpu.log_fold_changes[c] - cpu.log_fold_changes[c]).abs() / e;
            let error = (gpu.error_bars[c] - e).abs() / e;
            assert!(
                fold <= FOLD_GATE,
                "{rule:?}: gene {g}, fold change off by {fold:e} error bars"
            );
            assert!(
                error <= ERROR_GATE,
                "{rule:?}: gene {g}, error bar off by {error:e}"
            );
        }
        let mean = (gpu.mean_log_quotient[g] - cpu.mean_log_quotient[g]).abs()
            / cpu.mean_log_quotient_error[g];
        let variance = (gpu.variance[g] - cpu.variance[g]).abs() / cpu.variance[g];
        assert!(
            mean <= MEAN_GATE,
            "{rule:?}: gene {g}, m off by {mean:e} error bars"
        );
        assert!(
            variance <= VARIANCE_GATE,
            "{rule:?}: gene {g}, variance off by {variance:e}"
        );
    }
}

#[test]
fn test_gpu_marginalise_matches_cpu() {
    assert_parity(VarianceRule::Marginalise);
}

#[test]
fn test_gpu_posterior_mean_matches_cpu() {
    assert_parity(VarianceRule::PosteriorMean);
}

#[test]
fn test_gpu_max_posterior_matches_cpu() {
    assert_parity(VarianceRule::MaxPosterior);
}

#[test]
fn test_gpu_fixed_matches_cpu() {
    assert_parity(VarianceRule::Fixed(1.0));
}

#[test]
fn test_gpu_rejects_what_the_cpu_rejects() {
    let sim = fixture();
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let short = &sim.cell_totals[1..];
    assert!(sanity_gpu::<f32, WgpuRuntime>(&sim.counts, short, None, &client).is_err());
    let bad = SanityParams {
        variance_rule: VarianceRule::Fixed(-1.0),
        ..SanityParams::default()
    };
    assert!(
        sanity_gpu::<f32, WgpuRuntime>(&sim.counts, &sim.cell_totals, Some(bad), &client).is_err()
    );
}
