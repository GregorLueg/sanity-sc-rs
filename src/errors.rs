//! The single error enum for this crate, grouped by subsystem.

use thiserror::Error;

/// Everything a caller of `sanity-rs` can be handed back instead of a result.
#[derive(Debug, Error)]
pub enum SanityErrors {
    /////////////
    // Inputs  //
    /////////////
    #[error(
        "A gene's index and count vectors disagree in length: {n_indices} indices against {n_counts} counts."
    )]
    /// The sparse column for a gene is malformed.
    RaggedGene {
        /// Number of cell indices supplied.
        n_indices: usize,
        /// Number of counts supplied.
        n_counts: usize,
    },

    #[error("Cell index {index} is out of bounds for {n_cells} cells.")]
    /// A sparse column names a cell that does not exist.
    CellIndexOutOfBounds {
        /// The offending index.
        index: usize,
        /// Number of cells the run was told about.
        n_cells: usize,
    },

    #[error("Cell {index} has a total count of {total}, which must be strictly positive.")]
    /// A cell with no counts at all cannot be conditioned on.
    NonPositiveCellTotal {
        /// Position of the offending cell.
        index: usize,
        /// The total that was supplied.
        total: f64,
    },

    #[error(
        "The gene offsets are malformed: they run from {first} to {last} over {n_stored} stored counts, and must be ascending from zero to that total."
    )]
    /// The `indptr` of the count matrix is not a valid ascending offset array.
    MalformedIndptr {
        /// First offset seen, or the left edge of the offending pair.
        first: usize,
        /// Last offset seen, or the right edge of the offending pair.
        last: usize,
        /// Number of stored counts the offsets must cover.
        n_stored: usize,
    },

    #[error("{n_totals} per-cell totals were supplied for {n_cells} cells.")]
    /// The vector of library sizes does not match the matrix.
    CellTotalsLengthMismatch {
        /// Number of totals supplied.
        n_totals: usize,
        /// Number of cells the matrix declares.
        n_cells: usize,
    },

    #[error(
        "The variance grid [{min:e}, {max:e}] over {bins} bins is not usable; bounds must be finite, strictly positive and ascending, with at least one bin."
    )]
    /// The requested variance grid cannot be built.
    InvalidVarianceGrid {
        /// Lower bound requested.
        min: f64,
        /// Upper bound requested.
        max: f64,
        /// Bin count requested.
        bins: usize,
    },

    #[error(
        "A fixed variance of {variance:e} was requested; it must be finite and strictly positive."
    )]
    /// `VarianceRule::Fixed` was handed a variance the model cannot use.
    InvalidFixedVariance {
        /// The variance that was supplied.
        variance: f64,
    },

    /////////////
    // Solvers //
    /////////////
    #[error(
        "The per-cell fraction solve did not converge in {iterations} iterations; the residual was {residual:e}."
    )]
    /// The root find for the normalisation constant failed to settle.
    FractionSolveDiverged {
        /// Iterations taken before giving up.
        iterations: usize,
        /// Residual at the point of giving up.
        residual: f64,
    },
}
