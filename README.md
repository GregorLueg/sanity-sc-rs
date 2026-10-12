[![CI](https://github.com/GregorLueg/sanity-sc-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/sanity-sc-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/sanity-sc-rs.svg)](https://crates.io/crates/sanity-sc-rs)
[![docs.rs](https://img.shields.io/docsrs/sanity-sc-rs)](https://docs.rs/sanity-sc-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
Changes live in the [CHANGELOG](https://github.com/GregorLueg/sanity-sc-rs/blob/main/CHANGELOG.md).

## Installation

```toml
[dependencies]
sanity-sc-rs = "0.4"
```

Want the GPU path? Turn on `gpu` and pull in cubecl and cubecl-utils-rs
yourself to build the client:

```toml
[dependencies]
sanity-sc-rs = { version = "0.4", features = ["gpu"] }
cubecl = { version = "0.11", features = ["wgpu"] }
cubecl-utils-rs = "0.2"
```

## Performance

M1 Max, `Marginalise`, simulated counts at 8 to 9% density. Against the
reference Sanity binary, whole process on 8 threads, text I/O included:

| genes x cells | reference | sanity-sc-rs | speedup |
|---|---|---|---|
| 490 x 2000 | 10.4 s | 0.59 s | 18x |
| 1998 x 20000 | 415 s | 22.3 s | 19x |

The outputs agree to the six decimals the reference writes. A third of our
22.3 s is formatting 40 million numbers as text.

CPU against GPU, wall clock of the call only, CPU on all 10 cores:

| genes x cells | `sanity` | `sanity_gpu` | speedup |
|---|---|---|---|
| 1998 x 20000 | 13.1 s | 0.57 s | 23x |
| 500 x 200000 | 57.8 s | 2.40 s | 24x |

Every variance rule, peak memory and the GPU's error against the CPU are in
`docs/BENCHMARKS.md`.

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

Want more or less detail? Set the verbosity.

```rust
use sanity_sc_rs::config::Verbosity;

// more details
let params = SanityParams {
    verbosity: Verbosity::Detailed,
    ..SanityParams::default()
};

// quiet
let params = SanityParams {
    verbosity: Verbosity::Quiet,
    ..SanityParams::default()
};
```

`Normal` prints a header and a progress line at every tenth of the genes.
`Detailed` adds a per-batch stage split on the GPU. The default is `Quiet`.

### Handing over to Bonsai

Don't wire this up by hand. Turn on bonsai-rs's `sanity` feature and pass the
output straight to `ingest::from_sanity_output`. It transposes to Bonsai's
`[cell][gene]`, undoes Sanity's prior shrinkage with `variance` (S5 in the
Bonsai supplement) and drops the genes where that inversion is ill-conditioned.

```rust
use bonsai_rs::bonsai::bonsai;
use bonsai_rs::ingest::from_sanity_output;

let post = sanity::<f32>(&counts, &cell_totals, None)?;
let lik = from_sanity_output(&post, None)?;
let out = bonsai(&lik.means, &lik.sds, lik.n_cells, lik.features.len(), Some(&lik.variances), None)?;
```

Bonsai reads `log_fold_changes`, not `log_transcription_quotients()`: S5
inverts the zero-mean prior on `d_c`, and adding `m` back breaks it.
`sanity_gpu` output goes in unchanged, and so does `sanity_select` output; its
`genes` carry the input indices through to `lik.features`.

### Variance rules

| rule | what it does |
|---|---|
| `Marginalise` | Integrates over the full posterior on the variance. The default, and the one to quote. |
| `PosteriorMean` | Collapses to the posterior mean variance and re-solves there. Skips the second grid sweep. |
| `MaxPosterior` | Collapses to the most probable bin. Fast, but the answer moves with the grid. |
| `Fixed(v)` | One variance for every gene. No grid at all. |

### GPU

Behind the `gpu` feature: CubeCL 0.11 on wgpu, so Metal, Vulkan or DX12. Same
inference, same output struct.

```rust
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl_utils_rs::prelude::Runtime;
use sanity_sc_rs::gpu::sanity_gpu;

let client = <WgpuRuntime>::client(&WgpuDevice::default());
let out = sanity_gpu::<f32>(&counts, &cell_totals, None, &client)?;
```

On Apple the kernels go through cubecl's WGSL and wgpu's translation to Metal,
not cubecl's native `metal` feature. That costs 20 to 30% on these kernels
under cubecl 0.11, but cargo unifies features across a build, so turning it on
here would flip every crate sharing cubecl onto Metal source. Not worth it.

wgpu has no `f64`, so the device works in `f32`. The per-cell maths is
rearranged so no sum ever cancels, the offset is solved relative to an `f64`
anchor held by the host, and the likelihood over the variance grid is
assembled in `f64` on the host. Against the CPU path the worst log fold change
moves by under 1% of its own error bar, which is as far as the variance grid
itself is resolved. `MaxPosterior` settles near ties between bins on the CPU,
so it matches bin for bin.

## Design

Genes are independent under this model, so the method is one Rayon `par_iter`
over genes and nothing else. Each worker holds `O(n_bins + n_cells)` scratch and
reuses it across genes.

A cell with zero counts still contributes through its library size, so the
maths is dense in the cell axis. The work isn't. An empty cell depends on the
gene only through one scalar offset and on itself only through `ln T_c`, so its
sums come from a Chebyshev table in that offset, built once per run, and its
outputs from a Chebyshev fit in `ln T_c`. Per gene, the root solves run over the
nonzero cells plus 49 nodes. At 8% density that's where the 5x to 8x over the
dense sweep comes from. Genes above 50% density, metacells for instance, keep
the dense sweep.

The per-cell stationarity condition is solved as Wright omega rather than
Lambert W. The Lambert form overflows for ordinary inputs; the Wright form
doesn't.

## Licence and provenance

MIT. The reference implementation (`jmbreda/Sanity`) is GPL-3.0 and is not a
source for this crate. Everything here is written from `docs/SPEC.md`, which
restates the mathematics of the paper's Supplementary Information in our own
notation. See `docs/PROVENANCE.md` for the full position and the disclosure.

If you use this, cite the original paper.
