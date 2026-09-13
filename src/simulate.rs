//! Ground-truth simulator.
//!
//! SPEC section 10, SI S1.2. Generates UMI counts from the model's own
//! generative process, so that both this crate and any reference binary can be
//! measured against a truth neither of them produced.
//!
//! Two recipes. The independent one draws every cell's log fold change from the
//! prior. The branched one walks the log fold changes along a tree, which
//! correlates cells and is the harder case for anything downstream.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, LogNormal, Normal, Poisson};

use crate::errors::SanityErrors;
use crate::input::CountMatrix;

/// Mean of the exponential distribution the per-gene variances are drawn from.
///
/// SI S1.2 reports that Sanity's estimated variances are roughly exponential
/// with a mean near two in the reference dataset. That is a statement about
/// real data in the paper's text, not a constant lifted from their code.
pub const DEFAULT_VARIANCE_MEAN: f64 = 2.0;

/// Standard deviation, in the logarithm, of the simulated library sizes.
///
/// Droplet library sizes are heavily right skewed and a log-normal with this
/// width reproduces the usual order-of-magnitude spread between the smallest and
/// largest cells in a run.
pub const DEFAULT_LIBRARY_LOG_SD: f64 = 0.5;

/// Standard deviation, in the logarithm, of the simulated mean quotients.
///
/// Wide enough that the simulated genes span the range from a few UMIs per
/// thousand cells to a few per cell, which is where the interesting behaviour
/// of the method lives.
pub const DEFAULT_QUOTIENT_LOG_SD: f64 = 2.0;

/// Cells per branch in the branched recipe.
///
/// SI S1.2 restarts the walk from a random existing cell at this interval.
pub const DEFAULT_BRANCH_LENGTH: usize = 13;

/// How the per-cell log fold changes are drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExpressionPattern {
    /// Every cell drawn independently from the prior. No structure between
    /// cells.
    Independent,
    /// A branched random walk over cells, rescaled so each gene still has its
    /// assigned variance.
    Branched {
        /// Cells per branch before the walk restarts from a random earlier cell.
        branch_length: usize,
    },
}

/// Parameters for a simulated dataset.
#[derive(Clone, Copy, Debug)]
pub struct SimulationParams {
    /// Number of genes.
    pub n_genes: usize,
    /// Number of cells.
    pub n_cells: usize,
    /// Median library size. Sizes are log-normal about this.
    pub library_size: f64,
    /// Standard deviation of the log library size.
    pub library_log_sd: f64,
    /// Mean of the exponential the per-gene variances are drawn from.
    pub variance_mean: f64,
    /// Standard deviation of the log mean quotients before normalisation.
    pub quotient_log_sd: f64,
    /// How the log fold changes are drawn.
    pub pattern: ExpressionPattern,
    /// Seed. The same seed gives the same dataset.
    pub seed: u64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            n_genes: 1000,
            n_cells: 500,
            library_size: 5000.0,
            library_log_sd: DEFAULT_LIBRARY_LOG_SD,
            variance_mean: DEFAULT_VARIANCE_MEAN,
            quotient_log_sd: DEFAULT_QUOTIENT_LOG_SD,
            pattern: ExpressionPattern::Independent,
            seed: 0,
        }
    }
}

/// A simulated dataset and the truth behind it.
#[derive(Clone, Debug)]
pub struct Simulation {
    /// The sampled UMI counts, gene-major sparse.
    pub counts: CountMatrix,
    /// Total UMI count of every cell, over the simulated genes.
    pub cell_totals: Vec<f64>,
    /// True mean log transcription quotient of each gene, before the `-v/2`
    /// shift of SPEC section 10, step 3.
    ///
    /// This is the *arithmetic* mean quotient: `exp` of it is `<alpha_gc>`. The
    /// method estimates the *geometric* one, so
    /// [`crate::SanityOutput::mean_log_quotient`] compares against
    /// `mean_log_quotient - variance / 2`, not against this directly.
    pub mean_log_quotient: Vec<f64>,
    /// True variance in log fold change of each gene.
    pub variance: Vec<f64>,
    /// True log fold changes, gene-major, `n_genes * n_cells`.
    pub log_fold_changes: Vec<f64>,
}

/// Simulate a dataset from the model's generative process.
///
/// ### Params
///
/// * `params` - Simulation parameters, or [`SimulationParams::default`].
///
/// ### Returns
///
/// The counts, the library sizes, and the truth the counts were drawn from.
pub fn simulate(params: Option<SimulationParams>) -> Result<Simulation, SanityErrors> {
    let params = params.unwrap_or_default();
    let mut rng = StdRng::seed_from_u64(params.seed);

    // Mean quotients, normalised so that sum_g exp(mean_log_quotient) = 1.
    let quotient = LogNormal::new(0.0, params.quotient_log_sd)
        .expect("the log-normal parameters are finite and the width is positive");
    let raw: Vec<f64> = (0..params.n_genes).map(|_| quotient.sample(&mut rng)).collect();
    let raw_sum: f64 = raw.iter().sum();
    let mean_log_quotient: Vec<f64> = raw.iter().map(|x| (x / raw_sum).ln()).collect();

    let exponential = Exp::new(1.0 / params.variance_mean)
        .expect("the exponential rate is positive by construction");
    let variance: Vec<f64> = (0..params.n_genes)
        .map(|_| exponential.sample(&mut rng))
        .collect();

    let library = LogNormal::new(params.library_size.ln(), params.library_log_sd)
        .expect("the log-normal parameters are finite and the width is positive");
    let library_sizes: Vec<f64> = (0..params.n_cells)
        .map(|_| library.sample(&mut rng).round().max(1.0))
        .collect();

    let log_fold_changes = draw_log_fold_changes(&params, &variance, &mut rng);

    // Poisson sampling, gene by gene, into a sparse column.
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let mut indptr = Vec::with_capacity(params.n_genes + 1);
    indptr.push(0usize);
    let mut cell_totals = vec![0.0f64; params.n_cells];

    for g in 0..params.n_genes {
        let shift = mean_log_quotient[g] - 0.5 * variance[g];
        let row = &log_fold_changes[g * params.n_cells..(g + 1) * params.n_cells];
        for (c, &d) in row.iter().enumerate() {
            let lambda = library_sizes[c] * (shift + d).exp();
            if lambda <= 0.0 || !lambda.is_finite() {
                continue;
            }
            let k = Poisson::new(lambda)
                .expect("the Poisson rate is finite and positive here")
                .sample(&mut rng)
                .round();
            if k > 0.0 {
                indices.push(c as u32);
                values.push(k as u32);
                cell_totals[c] += k;
            }
        }
        indptr.push(indices.len());
    }

    // A cell that caught nothing cannot be conditioned on; give it the floor of
    // one UMI rather than dropping it and renumbering everything downstream.
    for total in cell_totals.iter_mut() {
        if *total <= 0.0 {
            *total = 1.0;
        }
    }

    Ok(Simulation {
        counts: CountMatrix::new(indices, values, indptr, params.n_cells)?,
        cell_totals,
        mean_log_quotient,
        variance,
        log_fold_changes,
    })
}

/// Draw the log fold changes under the requested pattern.
///
/// ### Params
///
/// * `params` - Simulation parameters.
/// * `variance` - Per-gene variance.
/// * `rng` - The seeded generator.
///
/// ### Returns
///
/// A gene-major `n_genes * n_cells` matrix of log fold changes.
fn draw_log_fold_changes(
    params: &SimulationParams,
    variance: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let n = params.n_genes * params.n_cells;
    let mut out = vec![0.0f64; n];
    let unit = Normal::new(0.0, 1.0).expect("the unit normal is well formed");

    match params.pattern {
        ExpressionPattern::Independent => {
            for g in 0..params.n_genes {
                let sd = variance[g].sqrt();
                let row = &mut out[g * params.n_cells..(g + 1) * params.n_cells];
                for x in row.iter_mut() {
                    *x = sd * unit.sample(rng);
                }
            }
        }
        ExpressionPattern::Branched { branch_length } => {
            // One tree shared by all genes: the parent assignment is a property
            // of the cells, not of any gene.
            let branch_length = branch_length.max(1);
            let mut parent = vec![0usize; params.n_cells];
            for (c, slot) in parent.iter_mut().enumerate().skip(1) {
                *slot = if c % branch_length == 0 {
                    rng.random_range(0..c)
                } else {
                    c - 1
                };
            }

            for g in 0..params.n_genes {
                let row = &mut out[g * params.n_cells..(g + 1) * params.n_cells];
                for c in 0..params.n_cells {
                    row[c] = if c == 0 {
                        unit.sample(rng)
                    } else {
                        row[parent[c]] + unit.sample(rng)
                    };
                }

                // Rescale so the gene's realised variance is the assigned one.
                let mean = row.iter().sum::<f64>() / params.n_cells as f64;
                let realised =
                    row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / params.n_cells as f64;
                let scale = if realised > 0.0 {
                    (variance[g] / realised).sqrt()
                } else {
                    0.0
                };
                for x in row.iter_mut() {
                    *x = (*x - mean) * scale;
                }
            }
        }
    }

    out
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simulation_is_reproducible() {
        let params = SimulationParams {
            n_genes: 40,
            n_cells: 60,
            seed: 7,
            ..Default::default()
        };
        let a = simulate(Some(params)).expect("simulates");
        let b = simulate(Some(params)).expect("simulates");
        assert_eq!(a.cell_totals, b.cell_totals);
        assert_eq!(a.log_fold_changes, b.log_fold_changes);
        for g in 0..40 {
            assert_eq!(a.counts.gene(g), b.counts.gene(g));
        }
    }

    #[test]
    fn test_mean_quotients_sum_to_one() {
        let sim = simulate(Some(SimulationParams {
            n_genes: 500,
            n_cells: 10,
            seed: 3,
            ..Default::default()
        }))
        .expect("simulates");
        let total: f64 = sim.mean_log_quotient.iter().map(|m| m.exp()).sum();
        assert_relative_eq!(total, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn test_branched_walk_matches_assigned_variance() {
        let sim = simulate(Some(SimulationParams {
            n_genes: 20,
            n_cells: 400,
            pattern: ExpressionPattern::Branched {
                branch_length: DEFAULT_BRANCH_LENGTH,
            },
            seed: 11,
            ..Default::default()
        }))
        .expect("simulates");

        for g in 0..20 {
            let row = &sim.log_fold_changes[g * 400..(g + 1) * 400];
            let mean = row.iter().sum::<f64>() / 400.0;
            let realised = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / 400.0;
            assert_relative_eq!(realised, sim.variance[g], max_relative = 1e-9);
        }
    }

    #[test]
    fn test_counts_are_sampled_around_their_rate() {
        // Aggregate over genes: the total UMIs in a cell should track its
        // library size, since the quotients sum to one.
        let sim = simulate(Some(SimulationParams {
            n_genes: 2000,
            n_cells: 50,
            library_size: 20000.0,
            library_log_sd: 0.0,
            variance_mean: 1e-6,
            seed: 5,
            ..Default::default()
        }))
        .expect("simulates");

        let mean_total = sim.cell_totals.iter().sum::<f64>() / 50.0;
        assert!(
            (mean_total - 20000.0).abs() < 20000.0 * 0.05,
            "mean total {mean_total} is far from the library size"
        );
    }
}
