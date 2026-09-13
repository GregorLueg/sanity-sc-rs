//! Input container for the sparse UMI count matrix.
//!
//! Gene-major: genes are the unit of work and a gene's whole column must be
//! contiguous. Counts stay sparse on the way in; the per-gene kernels scatter
//! one column at a time into dense scratch, because a cell with no counts for
//! this gene still contributes through its library size.

use crate::errors::SanityErrors;

/////////////////
// CountMatrix //
/////////////////

/// A sparse UMI count matrix, gene-major.
///
/// Layout is CSC with genes as columns: `indptr[g]..indptr[g + 1]` slices
/// `indices` and `values` for gene `g`.
#[derive(Clone, Debug)]
pub struct CountMatrix {
    /// Cell index of every stored count, grouped by gene.
    indices: Vec<u32>,
    /// The stored UMI counts, aligned with `indices`.
    values: Vec<u32>,
    /// Offsets into `indices` and `values`, length `n_genes + 1`, ascending.
    indptr: Vec<usize>,
    /// Number of cells, which is the dense extent of every column.
    n_cells: usize,
}

impl CountMatrix {
    /// Build and validate a count matrix.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell index of every stored count, grouped by gene.
    /// * `values` - The stored UMI counts, aligned with `indices`.
    /// * `indptr` - Offsets into the two above, length `n_genes + 1`.
    /// * `n_cells` - Number of cells.
    ///
    /// ### Returns
    ///
    /// The matrix, or the first structural fault found.
    pub fn new(
        indices: Vec<u32>,
        values: Vec<u32>,
        indptr: Vec<usize>,
        n_cells: usize,
    ) -> Result<Self, SanityErrors> {
        if indices.len() != values.len() {
            return Err(SanityErrors::RaggedGene {
                n_indices: indices.len(),
                n_counts: values.len(),
            });
        }
        if indptr.is_empty() || indptr[0] != 0 || *indptr.last().unwrap_or(&0) != indices.len() {
            return Err(SanityErrors::MalformedIndptr {
                first: indptr.first().copied().unwrap_or(0),
                last: indptr.last().copied().unwrap_or(0),
                n_stored: indices.len(),
            });
        }
        for window in indptr.windows(2) {
            if window[1] < window[0] {
                return Err(SanityErrors::MalformedIndptr {
                    first: window[0],
                    last: window[1],
                    n_stored: indices.len(),
                });
            }
        }
        if let Some(&bad) = indices.iter().find(|&&i| i as usize >= n_cells) {
            return Err(SanityErrors::CellIndexOutOfBounds {
                index: bad as usize,
                n_cells,
            });
        }

        Ok(Self {
            indices,
            values,
            indptr,
            n_cells,
        })
    }

    /// Number of genes.
    ///
    /// ### Returns
    ///
    /// The gene count.
    #[inline]
    pub fn n_genes(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Number of cells.
    ///
    /// ### Returns
    ///
    /// The cell count.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// The stored column for one gene.
    ///
    /// ### Params
    ///
    /// * `gene` - Gene index, below [`CountMatrix::n_genes`].
    ///
    /// ### Returns
    ///
    /// The cell indices and the counts at those cells.
    #[inline]
    pub fn gene(&self, gene: usize) -> (&[u32], &[u32]) {
        let range = self.indptr[gene]..self.indptr[gene + 1];
        (&self.indices[range.clone()], &self.values[range])
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_slices_genes() {
        let m = CountMatrix::new(
            vec![0, 2, 1, 3, 3],
            vec![5, 1, 2, 7, 4],
            vec![0, 2, 4, 5],
            4,
        )
        .expect("well formed");
        assert_eq!(m.n_genes(), 3);
        assert_eq!(m.n_cells(), 4);
        assert_eq!(m.gene(1), (&[1u32, 3][..], &[2u32, 7][..]));
    }

    #[test]
    fn test_matrix_rejects_out_of_bounds_cell() {
        let err = CountMatrix::new(vec![0, 9], vec![1, 1], vec![0, 2], 4).unwrap_err();
        assert!(matches!(err, SanityErrors::CellIndexOutOfBounds { .. }));
    }

    #[test]
    fn test_matrix_rejects_ragged_column() {
        let err = CountMatrix::new(vec![0, 1], vec![1], vec![0, 2], 4).unwrap_err();
        assert!(matches!(err, SanityErrors::RaggedGene { .. }));
    }

    #[test]
    fn test_matrix_rejects_broken_indptr() {
        let err = CountMatrix::new(vec![0, 1], vec![1, 1], vec![0, 1], 4).unwrap_err();
        assert!(matches!(err, SanityErrors::MalformedIndptr { .. }));
    }
}
