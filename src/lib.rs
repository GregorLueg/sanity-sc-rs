//! Bayesian inference of gene expression states from single cell UMI counts.
//!
//! A clean-room implementation of the Sanity method. Given raw UMI counts and
//! per-cell totals, this crate returns a posterior log expression level and an
//! error bar for every gene in every cell.
//!
//! Genes are independent under this model, so the whole method is one parallel
//! iteration over genes. A gene's input is its sparse column plus the full
//! vector of per-cell totals, because a cell with zero counts for that gene
//! still contributes through its library size.
//!
//! Inputs are raw UMI counts. Anything already log-normalised is the wrong
//! input: the point of the method is to model the Poisson sampling in the raw
//! counts.
//!
//! # Provenance
//!
//! The reference implementation (`jmbreda/Sanity`) is GPL-3.0 and is **not** a
//! source for this crate. Everything here is written from `docs/SPEC.md`, which
//! restates the mathematics of the paper's Supplementary Information. See
//! `PROVENANCE.md` for the full position.
//!
//! # References
//!
//! Breda, Zavolan, van Nimwegen. *Bayesian inference of gene expression states
//! from single-cell RNA-seq data.* Nature Biotechnology 39(8):1008-1016 (2021).

#![warn(missing_docs)]

pub mod config;
pub mod errors;
pub mod float;
pub mod input;
pub mod simulate;

mod model;
mod utils;

use rayon::prelude::*;

use crate::config::{SanityParams, VarianceGrid, VarianceRule};
use crate::errors::SanityErrors;
use crate::float::{SanityFloat, narrow};
use crate::input::CountMatrix;
use crate::model::gene::{GeneScratch, run_gene};

//////////////////
// SanityOutput //
//////////////////

/// Everything a run produces.
///
/// The two per-cell matrices are gene-major and dense: gene `g`, cell `c` is at
/// `g * n_cells + c`.
#[derive(Clone, Debug)]
pub struct SanityOutput<T: SanityFloat> {
    /// Posterior log fold change of each gene in each cell, `d_c`.
    pub log_fold_changes: Vec<T>,
    /// Error bar on each log fold change, `e_c`.
    pub error_bars: Vec<T>,
    /// Posterior mean log transcription quotient of each gene, `m`.
    ///
    /// The *geometric* mean: the prior centres the log fold changes on zero, so
    /// the arithmetic mean quotient is `exp(m + v / 2)`.
    pub mean_log_quotient: Vec<T>,
    /// Error bar on each `m`.
    pub mean_log_quotient_error: Vec<T>,
    /// Posterior estimate of each gene's variance in log fold change.
    pub variance: Vec<T>,
    /// Number of genes.
    pub n_genes: usize,
    /// Number of cells.
    pub n_cells: usize,
}

impl<T: SanityFloat> SanityOutput<T> {
    /// The normalised expression matrix, `m + d_c`.
    ///
    /// This is the log transcription quotient: the quantity downstream
    /// consumers read as the normalised expression level. Its error bar is
    /// [`SanityOutput::error_bars`]; [`SanityOutput::mean_log_quotient_error`]
    /// is the separate uncertainty on the gene's overall level.
    ///
    /// ### Returns
    ///
    /// A freshly allocated `n_genes * n_cells` gene-major matrix.
    pub fn log_transcription_quotients(&self) -> Vec<T> {
        let mut out = self.log_fold_changes.clone();
        for (g, row) in out.chunks_exact_mut(self.n_cells).enumerate() {
            let mean = self.mean_log_quotient[g];
            for x in row.iter_mut() {
                *x = *x + mean;
            }
        }
        out
    }
}

/// Run Sanity over a count matrix.
///
/// One `par_iter` over genes; there is no other loop. Each Rayon worker holds
/// scratch of `O(n_cells + n_bins)` and reuses it across the genes it is given.
///
/// ### Params
///
/// * `counts` - Raw UMI counts, gene-major sparse.
/// * `cell_totals` - Total UMI count of every cell over *all* genes, not just
///   those present in `counts`. Taken as `f64` because these are integer counts
///   in disguise and are exact there to `2^53`, which `f32` is not.
/// * `params` - Run parameters, or [`SanityParams::default`].
///
/// ### Returns
///
/// The posterior log fold changes, their error bars, and the per-gene means,
/// error bars and variances.
pub fn sanity<T: SanityFloat>(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: Option<SanityParams>,
) -> Result<SanityOutput<T>, SanityErrors> {
    let params = params.unwrap_or_default();
    let n_cells = counts.n_cells();
    let n_genes = counts.n_genes();

    if cell_totals.len() != n_cells {
        return Err(SanityErrors::CellTotalsLengthMismatch {
            n_totals: cell_totals.len(),
            n_cells,
        });
    }
    if let Some((index, total)) = cell_totals
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, t)| !t.is_finite() || t <= 0.0)
    {
        return Err(SanityErrors::NonPositiveCellTotal { index, total });
    }
    if let VarianceRule::Fixed(v) = params.variance_rule
        && (!v.is_finite() || v <= 0.0)
    {
        return Err(SanityErrors::InvalidFixedVariance { variance: v });
    }
    if params.variance_rule.needs_grid()
        && (!params.variance_min.is_finite()
            || params.variance_min <= 0.0
            || !params.variance_max.is_finite()
            || params.variance_max <= params.variance_min
            || params.variance_bins == 0)
    {
        return Err(SanityErrors::InvalidVarianceGrid {
            min: params.variance_min,
            max: params.variance_max,
            bins: params.variance_bins,
        });
    }

    let grid = VarianceGrid::new(
        params.variance_min,
        params.variance_max,
        params.variance_bins,
    );
    let log_totals: Vec<f64> = cell_totals.iter().map(|t| t.ln()).collect();
    let log_total_sum = cell_totals.iter().sum::<f64>().ln();

    let mut log_fold_changes = vec![T::zero(); n_genes * n_cells];
    let mut error_bars = vec![T::zero(); n_genes * n_cells];
    let mut mean_log_quotient = vec![T::zero(); n_genes];
    let mut mean_log_quotient_error = vec![T::zero(); n_genes];
    let mut variance = vec![T::zero(); n_genes];

    let n_bins = grid.len();
    (
        log_fold_changes.par_chunks_mut(n_cells),
        error_bars.par_chunks_mut(n_cells),
        mean_log_quotient.par_iter_mut(),
        mean_log_quotient_error.par_iter_mut(),
        variance.par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .try_for_each_init(
            || {
                (
                    GeneScratch::new(n_cells, n_bins),
                    vec![0.0f64; n_cells],
                    vec![0.0f64; n_cells],
                )
            },
            |(scratch, row_d, row_e), (gene, (out_d, out_e, out_m, out_dm, out_v))| {
                let (indices, values) = counts.gene(gene);
                let summary = run_gene(
                    indices,
                    values,
                    &log_totals,
                    log_total_sum,
                    &grid,
                    &params,
                    scratch,
                    row_d,
                    row_e,
                )?;

                for c in 0..n_cells {
                    out_d[c] = narrow(row_d[c]);
                    out_e[c] = narrow(row_e[c]);
                }
                *out_m = narrow(summary.mean_log_quotient);
                *out_dm = narrow(summary.mean_log_quotient_error);
                *out_v = narrow(summary.variance);

                Ok::<(), SanityErrors>(())
            },
        )?;

    Ok(SanityOutput {
        log_fold_changes,
        error_bars,
        mean_log_quotient,
        mean_log_quotient_error,
        variance,
        n_genes,
        n_cells,
    })
}
