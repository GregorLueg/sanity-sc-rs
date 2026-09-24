# sanity-sc-rs

Bayesian inference of gene expression states from single cell UMI counts. A
clean-room Rust implementation of Sanity (Breda, Zavolan, van Nimwegen, *Nature
Biotechnology* 39(8):1008-1016, 2021, doi
[10.1038/s41587-021-00875-x](https://doi.org/10.1038/s41587-021-00875-x)).

Give it raw UMI counts and per-cell totals. For every gene in every cell, you
get back a posterior log expression level and its error bar. For every gene, you
also get the mean log transcription quotient, its error bar and the variance in
log fold change.

This exists mainly to feed [`bonsai-rs`](https://github.com/GregorLueg/bonsai-rs).
Bonsai marginalises over per-cell, per-feature measurement noise. Without real
error bars it's a different and worse method, and Sanity is where those error
bars come from. The intended chain for scRNA-seq is raw UMIs, then Sanity, then
Bonsai.

Feed it raw counts only. Anything already log-normalised is the wrong input: the
whole point of the method is to model the Poisson sampling in the raw counts.

## Usage

```rust
use sanity_sc_rs::config::{SanityParams, VarianceRule};
use sanity_sc_rs::input::CountMatrix;
use sanity_sc_rs::{sanity, sanity_select};

// Gene-major sparse: cell indices and counts per gene, CSR style.
let counts = CountMatrix::new(indices, values, indptr, n_cells)?;

// Total UMIs per cell over all genes, not just the ones in `counts`.
let out = sanity::<f32>(&counts, &cell_totals, None)?;

// m + d_c, the normalised expression matrix, gene-major.
let expr = out.log_transcription_quotients();
let error_bars = &out.error_bars;

// Cheaper variance rule: collapse to the posterior mean instead of marginalising.
let params = SanityParams {
    variance_rule: VarianceRule::PosteriorMean,
    ..SanityParams::default()
};
let fast = sanity::<f32>(&counts, &cell_totals, Some(params))?;

// Keep only the genes a per-gene predicate accepts. Rejected genes are never
// stored, so resident output scales with the kept set.
let kept = sanity_select::<f32, _>(&counts, &cell_totals, None, |g| g.variance > 0.1)?;
```

Storage is `f32` or `f64`. Every reduction accumulates in `f64` either way.

### Handing over to Bonsai

Bonsai's means are `log_transcription_quotients()` and its standard deviations
are `error_bars`. Watch the layout: Sanity writes gene-major
(`g * n_cells + c`), Bonsai reads row-major `[cell][gene]`, so transpose both
before the call. `sanity_select` with a signal-to-noise predicate drops the
uninformative genes before they're ever stored.

### Variance rules

| rule | what it does |
|---|---|
| `Marginalise` | Integrates over the full posterior on the variance. The default, and the one to quote. |
| `PosteriorMean` | Collapses to the posterior mean variance and re-solves there. Skips the second grid sweep. |
| `MaxPosterior` | Collapses to the most probable bin. Fast, but the answer moves with the grid. |
| `Fixed(v)` | One variance for every gene. No grid at all. |

## Design

Genes are independent under this model, so the method is one Rayon `par_iter`
over genes and nothing else. Each worker holds `O(n_bins + n_cells)` scratch and
reuses it across genes. The input stays sparse; the algorithm is dense in the
cell axis, because a cell with zero counts still contributes through its library
size.

The per-cell stationarity condition is solved as Wright omega rather than
Lambert W. The Lambert form overflows for ordinary inputs; the Wright form
doesn't.

## Licence and provenance

MIT. The reference implementation (`jmbreda/Sanity`) is GPL-3.0 and is not a
source for this crate. Everything here is written from `docs/SPEC.md`, which
restates the mathematics of the paper's Supplementary Information in our own
notation. See `docs/PROVENANCE.md` for the full position and the disclosure.

If you use this, cite the original paper.
