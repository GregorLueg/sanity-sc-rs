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

    //////////////
    // Solvers  //
    //////////////
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
