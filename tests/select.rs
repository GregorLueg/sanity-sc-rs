//! `sanity_select` against a post hoc filter of `sanity`.

use sanity_rs::simulate::{Simulation, SimulationParams, simulate};
use sanity_rs::{GeneView, SanityOutput, sanity, sanity_select};

/// Bonsai's S6 score, `mean_c(d_c^2 / e_c^2)`, on one row.
fn signal_to_noise(d: &[f64], e: &[f64]) -> f64 {
    d.iter().zip(e).map(|(d, e)| d * d / (e * e)).sum::<f64>() / d.len() as f64
}

/// A small simulated matrix.
fn fixture() -> Simulation {
    simulate(Some(SimulationParams {
        n_genes: 200,
        n_cells: 300,
        library_size: 5000.0,
        seed: 7,
        ..Default::default()
    }))
    .expect("simulates")
}

/// Rows of `full` at `genes`, gene-major.
fn gather(full: &[f32], genes: &[usize], n_cells: usize) -> Vec<f32> {
    genes
        .iter()
        .flat_map(|&g| full[g * n_cells..(g + 1) * n_cells].iter().copied())
        .collect()
}

#[test]
fn test_select_all_matches_sanity() {
    let sim = fixture();
    let full: SanityOutput<f32> = sanity(&sim.counts, &sim.cell_totals, None).expect("runs");
    let all: SanityOutput<f32> =
        sanity_select(&sim.counts, &sim.cell_totals, None, |_| true).expect("runs");
    assert_eq!(all.genes, full.genes);
    assert_eq!(all.n_genes, full.n_genes);
    assert_eq!(all.log_fold_changes, full.log_fold_changes);
    assert_eq!(all.error_bars, full.error_bars);
    assert_eq!(all.mean_log_quotient, full.mean_log_quotient);
    assert_eq!(all.mean_log_quotient_error, full.mean_log_quotient_error);
    assert_eq!(all.variance, full.variance);
}

#[test]
fn test_select_signal_to_noise_matches_post_hoc_filter() {
    let sim = fixture();
    let n_cells = sim.cell_totals.len();
    let full: SanityOutput<f64> = sanity(&sim.counts, &sim.cell_totals, None).expect("runs");
    let expected: Vec<usize> = (0..full.n_genes)
        .filter(|&g| {
            let row = g * n_cells..(g + 1) * n_cells;
            signal_to_noise(&full.log_fold_changes[row.clone()], &full.error_bars[row]) >= 1.0
        })
        .collect();
    assert!(
        !expected.is_empty() && expected.len() < full.n_genes,
        "the fixture should keep some genes and drop others, kept {}",
        expected.len()
    );

    let kept: SanityOutput<f32> =
        sanity_select(&sim.counts, &sim.cell_totals, None, |g: GeneView| {
            signal_to_noise(g.log_fold_changes, g.error_bars) >= 1.0
        })
        .expect("runs");
    let full32: SanityOutput<f32> = sanity(&sim.counts, &sim.cell_totals, None).expect("runs");

    assert_eq!(kept.genes, expected);
    assert_eq!(kept.n_genes, expected.len());
    assert_eq!(
        kept.log_fold_changes,
        gather(&full32.log_fold_changes, &expected, n_cells)
    );
    assert_eq!(
        kept.error_bars,
        gather(&full32.error_bars, &expected, n_cells)
    );
    let variance: Vec<f32> = expected.iter().map(|&g| full32.variance[g]).collect();
    assert_eq!(kept.variance, variance);
}

#[test]
fn test_select_none_is_empty() {
    let sim = fixture();
    let none: SanityOutput<f64> =
        sanity_select(&sim.counts, &sim.cell_totals, None, |_| false).expect("runs");
    assert_eq!(none.n_genes, 0);
    assert!(none.genes.is_empty());
    assert!(none.log_fold_changes.is_empty());
    assert!(none.log_transcription_quotients().is_empty());
}
