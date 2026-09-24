//! The per-gene driver: sweep the variance grid, then aggregate over it.
//!
//! ### Memory
//!
//! SI eq. 42 wants the spread of `d*_c` about its own posterior mean, which
//! naively means holding `d*` and `var(d_c)` for every bin and every cell. It is
//! avoided: the offset `z_b` is one scalar per bin, so the first pass stores
//! `O(B)` state, and the second pass re-derives the per-cell quantities at a
//! known offset with a single sweep and no iteration. Scratch is `O(B + C)`.
//! The point-estimate rules skip the second pass entirely.
//!
//! Running the second pass cell block outermost instead, so that a block stays
//! in L1 across the whole grid, was measured on 2026-09-13 and was 10% slower:
//! the pass is bound by the transcendentals in the Wright omega and error bar
//! solves, not by memory traffic, so the tiling bought nothing and the extra
//! indexing cost.

use super::fractions::{Stationary, SweepState, refresh, solve_stationary};
use super::likelihood::laplace;
use super::variance::cell_variance;
use crate::config::{SanityParams, VarianceGrid, VarianceRule};
use crate::errors::SanityErrors;
use crate::utils::polygamma::{digamma, trigamma};

////////////
// Consts //
////////////

/// Posterior weight below which a bin is dropped from the second pass.
///
/// SI eq. 39 and 42 both weight a bin's contribution by `W_b`, so a bin under
/// this contributes less than this fraction of either estimate, against error
/// bars the method reports at the first or second digit. Every surviving bin
/// costs a full sweep over the cells plus an error bar solve in each of them,
/// which is the whole cost of the rule.
///
/// Measured 2026-09-13 on an M1 Max, 161 bins. On 400 simulated genes over
/// 4000 cells at 7.5% density the rule takes 6.78 s with nothing dropped,
/// 5.19 s at `1e-14`, 5.00 s here and 4.59 s at `1e-6`. Worst drift in the log
/// fold change, in units of the error bar the method reports for it, measured
/// on 200 genes over 2000 cells against an unpruned run: `4e-10` error bars at
/// `1e-12`, `1e-8` here, `1e-4` at `1e-6`. Almost all of the saving is already
/// had at `1e-14`, so this sits well inside the flat part of the curve.
///
/// Must stay strictly positive. It is also what stops the leading run of
/// underflowed bins from reaching the Welford update, where the first one would
/// divide a zero weight by a zero running sum.
const MARGINALISE_MIN_WEIGHT: f64 = 1e-10;

/////////////////
// GeneScratch //
/////////////////

/// Per-thread scratch for one gene, reused across genes.
///
/// Allocated once per Rayon worker through `for_each_init`, never per gene.
#[derive(Clone, Debug)]
pub(crate) struct GeneScratch {
    /// Dense UMI counts for the gene in hand, length `n_cells`.
    counts: Vec<f64>,
    /// `omega(x_c)` at the current variance, length `n_cells`.
    omega: Vec<f64>,
    /// `ln omega(x_c)` at the current variance, length `n_cells`.
    log_omega: Vec<f64>,
    /// Running weighted mean of `d*_c` over bins, length `n_cells`.
    mean_d: Vec<f64>,
    /// Running weighted sum of squared deviations of `d*_c`, length `n_cells`.
    m2_d: Vec<f64>,
    /// Running weighted mean of `var(d_c)` over bins, length `n_cells`.
    mean_var: Vec<f64>,
    /// `ln P(k | v_b)` per bin, length `n_bins`.
    log_lik: Vec<f64>,
    /// `z(v_b)` per bin, length `n_bins`.
    offsets: Vec<f64>,
    /// `S_A(v_b)` per bin, length `n_bins`.
    curvature: Vec<f64>,
    /// Posterior weights `W_b`, length `n_bins`.
    weights: Vec<f64>,
    /// What `omega` and `log_omega` currently hold, for warm starting.
    ///
    /// Cleared at the start of every gene: the arrays survive across genes but
    /// their contents belong to the gene that wrote them.
    state: SweepState,
}

impl GeneScratch {
    /// Allocate scratch for a run.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells.
    /// * `n_bins` - Number of variance bins.
    ///
    /// ### Returns
    ///
    /// Zeroed scratch of `O(n_cells + n_bins)`.
    pub(crate) fn new(n_cells: usize, n_bins: usize) -> Self {
        Self {
            counts: vec![0.0; n_cells],
            omega: vec![0.0; n_cells],
            log_omega: vec![0.0; n_cells],
            mean_d: vec![0.0; n_cells],
            m2_d: vec![0.0; n_cells],
            mean_var: vec![0.0; n_cells],
            log_lik: vec![0.0; n_bins],
            offsets: vec![0.0; n_bins],
            curvature: vec![0.0; n_bins],
            weights: vec![0.0; n_bins],
            state: None,
        }
    }
}

//////////////
// Frontend //
//////////////

/////////////
// Helpers //
/////////////

/// First pass: solve the offset and the marginal likelihood in every bin.
///
/// The grid ascends in `v` and `z` grows with `v`, so each bin warm starts the
/// Newton solve from its predecessor's offset.
///
/// ### Params
///
/// * `s` - `K`, the total UMI count of this gene.
/// * `log_totals` - `ln T_c` for every cell.
/// * `log_total_sum` - `ln(sum_c T_c)`.
/// * `grid` - The variance grid.
/// * `scratch` - Per-thread scratch; `log_lik`, `offsets` and `curvature` are
///   filled.
///
/// ### Returns
///
/// Nothing, or a solver failure.
fn sweep_grid(
    s: f64,
    log_totals: &[f64],
    log_total_sum: f64,
    grid: &VarianceGrid,
    scratch: &mut GeneScratch,
) -> Result<(), SanityErrors> {
    let mut guess = log_total_sum + 0.5 * grid.values[0];
    for (b, &v) in grid.values.iter().enumerate() {
        let point = solve_stationary(
            v,
            s,
            &scratch.counts,
            log_totals,
            guess,
            &mut scratch.state,
            &mut scratch.omega,
            &mut scratch.log_omega,
        )?;
        let fit = laplace(
            &point,
            &scratch.counts,
            log_totals,
            &scratch.omega,
            &scratch.log_omega,
        );
        scratch.log_lik[b] = fit.log_marginal;
        scratch.offsets[b] = point.z;
        scratch.curvature[b] = point.curvature_sum;
        guess = point.z;
    }
    Ok(())
}

/// Normalise the bin log-likelihoods into posterior weights.
///
/// SI eq. 34. The scale prior on `v` is already carried by the grid being
/// uniform in `ln v`, so this is a plain softmax, shifted by the maximum.
///
/// ### Params
///
/// * `log_lik` - `ln P(k | v_b)` per bin.
/// * `weights` - Output, `W_b` per bin, summing to one.
///
/// ### Returns
///
/// Nothing; `weights` is overwritten, or
/// [`SanityErrors::NonFiniteBinLikelihood`] if a bin is unusable. The check is
/// up front because `fold(NEG_INFINITY, f64::max)` skips a NaN, which would
/// then propagate silently into every output for the gene.
fn posterior_weights(log_lik: &[f64], weights: &mut [f64]) -> Result<(), SanityErrors> {
    if let Some((bin, &value)) = log_lik.iter().enumerate().find(|(_, l)| !l.is_finite()) {
        return Err(SanityErrors::NonFiniteBinLikelihood { bin, value });
    }
    let peak = log_lik.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for (w, &l) in weights.iter_mut().zip(log_lik) {
        *w = (l - peak).exp();
        total += *w;
    }
    let scale = if total > 0.0 { 1.0 / total } else { 0.0 };
    for w in weights.iter_mut() {
        *w *= scale;
    }
    Ok(())
}

/// Second pass: integrate the per-cell estimates over the variance posterior.
///
/// SI eq. 39 and 42. `z_b` is already known, so each bin costs one sweep with no
/// iteration. The spread of `d*_c` across bins accumulates by weighted Welford,
/// which is SI eq. 42 rather than the algebraically equivalent SI eq. 41 and so
/// does not lose digits when the log fold change is large.
///
/// ### Params
///
/// * `s` - `K`, the total UMI count of this gene.
/// * `log_totals` - `ln T_c` for every cell.
/// * `grid` - The variance grid.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
/// * `n_cells` - Number of cells.
///
/// ### Returns
///
/// The gene-level summary.
fn marginalise(
    s: f64,
    log_totals: &[f64],
    grid: &VarianceGrid,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
    n_cells: usize,
) -> GeneSummary {
    // The gene-level scalars do not touch the cells, so they come out of the
    // tiled loop entirely and are accumulated once over the grid.
    let mut weight_sum = 0.0;
    let mut mean_offset = 0.0;
    let mut m2_offset = 0.0;
    let mut mean_variance = 0.0;
    for (b, &v) in grid.values.iter().enumerate() {
        let weight = scratch.weights[b];
        if weight < MARGINALISE_MIN_WEIGHT {
            continue;
        }
        weight_sum += weight;
        let share = weight / weight_sum;
        let delta = scratch.offsets[b] - mean_offset;
        mean_offset += share * delta;
        m2_offset += weight * delta * (scratch.offsets[b] - mean_offset);
        mean_variance += share * (v - mean_variance);
    }

    scratch.mean_d[..n_cells].fill(0.0);
    scratch.m2_d[..n_cells].fill(0.0);
    scratch.mean_var[..n_cells].fill(0.0);

    let mut running = 0.0;
    for (b, &v) in grid.values.iter().enumerate() {
        let weight = scratch.weights[b];
        if weight < MARGINALISE_MIN_WEIGHT {
            continue;
        }
        let point = Stationary {
            v,
            s,
            log_vs: (v * s).ln(),
            z: scratch.offsets[b],
            curvature_sum: scratch.curvature[b],
        };
        refresh(
            &point,
            &scratch.counts,
            log_totals,
            &mut scratch.state,
            &mut scratch.omega,
            &mut scratch.log_omega,
        );

        running += weight;
        let share = weight / running;

        #[allow(clippy::needless_range_loop)]
        for c in 0..n_cells {
            let d = point.log_fold_change(scratch.log_omega[c], log_totals[c]);
            let var = cell_variance(&point, scratch.counts[c], d, scratch.omega[c]);

            let delta = d - scratch.mean_d[c];
            scratch.mean_d[c] += share * delta;
            scratch.m2_d[c] += weight * delta * (d - scratch.mean_d[c]);
            scratch.mean_var[c] += share * (var - scratch.mean_var[c]);
        }
    }

    for c in 0..n_cells {
        out_fold_change[c] = scratch.mean_d[c];
        out_error[c] = (scratch.mean_var[c] + scratch.m2_d[c] / weight_sum).sqrt();
    }

    GeneSummary {
        mean_log_quotient: digamma(s) - mean_offset,
        mean_log_quotient_error: (trigamma(s) + m2_offset / weight_sum).sqrt(),
        variance: mean_variance,
    }
}

/// Collapse the variance posterior to a single value and evaluate there.
///
/// SPEC section 7. [`VarianceRule::MaxPosterior`] reuses the offset already
/// stored for that bin; [`VarianceRule::PosteriorMean`] lands between bins and
/// so re-solves, warm started from the offset of the bin nearest `<v>`.
///
/// ### Params
///
/// * `rule` - The collapsing rule.
/// * `s` - `K`, the total UMI count of this gene.
/// * `log_totals` - `ln T_c` for every cell.
/// * `grid` - The variance grid.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
///
/// ### Returns
///
/// The gene-level summary, or a solver failure.
fn collapse(
    rule: VarianceRule,
    s: f64,
    log_totals: &[f64],
    grid: &VarianceGrid,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) -> Result<GeneSummary, SanityErrors> {
    let posterior_mean: f64 = grid
        .values
        .iter()
        .zip(&scratch.weights)
        .map(|(v, w)| v * w)
        .sum();

    let (v, guess) = match rule {
        VarianceRule::MaxPosterior => {
            let best = scratch
                .weights
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |acc, (b, &w)| {
                    if w > acc.1 { (b, w) } else { acc }
                })
                .0;
            (grid.values[best], scratch.offsets[best])
        }
        _ => {
            // The grid ascends, so the first bin at or above `<v>` and the one
            // below it bracket it; take the closer of the two.
            let above = grid.values.partition_point(|&v| v < posterior_mean);
            let nearest = match above {
                0 => 0,
                b if b == grid.values.len() => b - 1,
                b if posterior_mean - grid.values[b - 1] <= grid.values[b] - posterior_mean => {
                    b - 1
                }
                b => b,
            };
            (posterior_mean, scratch.offsets[nearest])
        }
    };

    let point = solve_stationary(
        v,
        s,
        &scratch.counts,
        log_totals,
        guess,
        &mut scratch.state,
        &mut scratch.omega,
        &mut scratch.log_omega,
    )?;
    write_point_estimate(
        &point,
        &scratch.counts,
        log_totals,
        &scratch.omega,
        &scratch.log_omega,
        out_fold_change,
        out_error,
    );

    Ok(GeneSummary {
        mean_log_quotient: digamma(s) - point.z,
        mean_log_quotient_error: trigamma(s).sqrt(),
        variance: posterior_mean,
    })
}

/// Write `d_c` and `e_c` from a single stationary point.
///
/// Used by every rule that does not integrate over the grid, where the error bar
/// is the posterior width at one variance and nothing is added for the spread of
/// `d*_c` across variances.
///
/// ### Params
///
/// * `point` - The stationary point.
/// * `counts` - Dense UMI counts for this gene.
/// * `log_totals` - `ln T_c` for every cell.
/// * `omega` - `omega(x_c)` at the point.
/// * `log_omega` - `ln omega(x_c)` at the point.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
///
/// ### Returns
///
/// Nothing; both outputs are overwritten.
#[allow(clippy::too_many_arguments)]
fn write_point_estimate(
    point: &Stationary,
    counts: &[f64],
    log_totals: &[f64],
    omega: &[f64],
    log_omega: &[f64],
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) {
    for c in 0..counts.len() {
        let d = point.log_fold_change(log_omega[c], log_totals[c]);
        out_fold_change[c] = d;
        out_error[c] = cell_variance(point, counts[c], d, omega[c]).sqrt();
    }
}

/////////////////
// GeneSummary //
/////////////////

/// Everything one gene contributes to the run's output.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GeneSummary {
    /// Posterior mean of the gene's log transcription quotient, `m`.
    pub mean_log_quotient: f64,
    /// Error bar on `m`.
    pub mean_log_quotient_error: f64,
    /// Posterior estimate of the gene's variance in log fold change.
    pub variance: f64,
}

/// Run one gene end to end.
///
/// Writes `d_c` into `out_fold_change` and `e_c` into `out_error`, both of
/// length `n_cells`, and returns the gene-level summary.
///
/// ### Params
///
/// * `indices` - Cell indices of this gene's stored counts.
/// * `values` - The stored counts, aligned with `indices`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `log_total_sum` - `ln(sum_c T_c)`, the seed for the first offset solve.
/// * `grid` - The variance grid.
/// * `params` - Run parameters.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
///
/// ### Returns
///
/// The gene-level summary, or a solver failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gene(
    indices: &[u32],
    values: &[u32],
    log_totals: &[f64],
    log_total_sum: f64,
    grid: &VarianceGrid,
    params: &SanityParams,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) -> Result<GeneSummary, SanityErrors> {
    let n_cells = log_totals.len();

    // Scatter the sparse column. Only the touched entries are cleared again at
    // the end, so this stays O(nnz) rather than O(n_cells) for the reset.
    scratch.state = None;
    let mut total_counts = 0.0;
    for (&i, &k) in indices.iter().zip(values) {
        scratch.counts[i as usize] = k as f64;
        total_counts += k as f64;
    }
    // SPEC section 1: the 1/alpha prior makes the exponent K, not K + 1.
    // `prepare_run` rejects K = 0, where the posterior is improper.
    let s = total_counts;

    let summary = match params.variance_rule {
        VarianceRule::Fixed(v) => {
            let point = solve_stationary(
                v,
                s,
                &scratch.counts,
                log_totals,
                log_total_sum + 0.5 * v,
                &mut scratch.state,
                &mut scratch.omega,
                &mut scratch.log_omega,
            )?;
            write_point_estimate(
                &point,
                &scratch.counts,
                log_totals,
                &scratch.omega,
                &scratch.log_omega,
                out_fold_change,
                out_error,
            );
            GeneSummary {
                mean_log_quotient: digamma(s) - point.z,
                mean_log_quotient_error: trigamma(s).sqrt(),
                variance: v,
            }
        }
        rule => {
            sweep_grid(s, log_totals, log_total_sum, grid, scratch)?;
            posterior_weights(&scratch.log_lik, &mut scratch.weights)?;
            match rule {
                VarianceRule::Marginalise => marginalise(
                    s,
                    log_totals,
                    grid,
                    scratch,
                    out_fold_change,
                    out_error,
                    n_cells,
                ),
                _ => collapse(
                    rule,
                    s,
                    log_totals,
                    grid,
                    scratch,
                    out_fold_change,
                    out_error,
                )?,
            }
        }
    };

    for &i in indices {
        scratch.counts[i as usize] = 0.0;
    }

    Ok(summary)
}
