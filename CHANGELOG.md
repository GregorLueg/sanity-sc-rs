# Changelog

## v0.4.0

### Breaking

* cubecl 0.11 and cubecl-utils-rs 0.2.0. `sanity_gpu` and `sanity_gpu_select`
  lose their runtime generic and take cubecl's `&Client`: call
  `sanity_gpu::<f32>(&counts, &totals, None, &client)`.

### Performance

* cubecl 0.11's WGSL is 20-30% slower than 0.10's on these kernels. It lowers
  every `break` and `return` into a flag variable, and both kernels sit in
  loops that break early. cubecl's `metal` feature dodges it on Apple but is
  not used: cargo unifies features, so it would switch every crate sharing
  cubecl in a build onto Metal source and some of the kernels are actually
  slower with cubecl 0.11.
* The GPU second pass no longer solves every empty cell in every bin. Like the
  CPU, it evaluates a gene's nonzero cells and the 49 nodes of the empty-cell
  fit in `ln T`, then interpolates.
* The GPU first pass dispatches the genes that walk the most cells first, so
  the slowest ones no longer start last.
* The GPU output is copied out in parallel.
* Net against v0.3.0 on an M1 Max: `Marginalise` on the GPU goes from 0.82 s
  to 0.57 s at 1998 genes by 20000 cells and stays flat at 200000 cells (2.36 s
  to 2.40 s). `MaxPosterior` at 200000 cells is slower, 2.23 s to 2.44 s. The
  v0.3.0 run has no load record, so read these as good to 10 to 20%.

### Numerics

* cubecl 0.11 fuses multiply-adds into `fma` and can't be told not to. The
  worst GPU log fold change at 200000 cells moves from `1.9e-3` to `3.1e-3` of
  an error bar against the CPU, still well inside the 1% the grid resolves.

### Docs

* README: badges, an installation section and performance tables against the
  reference binary and between CPU and GPU.

## v0.3.0

### Performance

* The CPU path no longer solves cells with zero counts one by one. An empty
  cell depends on the gene only through one scalar offset, so the first pass
  reads its sums from a table built once per run, and the second pass
  evaluates empty cells at 49 Chebyshev nodes in `ln T` and interpolates.
* Genes above 50% density keep the dense first pass; below that the sparse one
  wins. Metacell-style data is never slower than before.
* The GPU first pass uses the same table for genes up to 50% dense. 1.5x to
  1.7x on `Marginalise` and `PosteriorMean`, 6x on `MaxPosterior`, whose near
  ties are now re-solved sparse on the CPU. The worst log fold change against
  the CPU path goes from `2e-4` to `4e-4` of an error bar at 20000 cells.

### Fixes

* A warm-started Wright omega solve that did not converge within its eight
  iterations returned the unconverged root silently. It now restarts cold.

## v0.2.1

### Features

* The GPU-accelerated form of Sanity also got a filter gene option to only
  keep these genes.

## v0.2.0

### Features

* Progress reporting via `SanityParams::verbosity` and the new `Verbosity`
  enum (`Quiet`, `Normal`, `Detailed`), on `sanity`, `sanity_select` and
  `sanity_gpu`. `Normal` prints a header and a line at every tenth of the
  genes; `Detailed` adds a per-batch stage split on the GPU. Default is
  `Quiet`.
* Breaking: `SanityParams::new` takes a trailing `verbosity`. Struct literals
  that end in `..SanityParams::default()` compile unchanged.

### Docs

* README: the hand-over to Bonsai goes through bonsai-rs's
  `ingest::from_sanity_output`, which reads `log_fold_changes`, not the log
  transcription quotients.

## v0.1.0

### Features

* GPU-accelerated Sanity added via the wgpu/cubecl framework, behind the `gpu`
  feature. `sanity_gpu` takes and returns what `sanity` does and supports every
  variance rule. The device works in `f32`; the maths is rearranged so no sum
  cancels, the offset is solved against an `f64` anchor and the likelihood over
  the variance grid is assembled in `f64` on the host.
* `MaxPosterior` on the GPU settles near ties between bins in `f64` on the CPU,
  so it lands on the same bin as the CPU path.
* `gpu-tests` feature with CPU parity tests for every rule, and a GPU lane in
  CI.

## v0.0.1

### Features

* Clean-room Rust implementation of Sanity from `docs/SPEC.md`: `sanity`,
  `sanity_select`, the four variance rules and the simulator.
