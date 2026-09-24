# Changelog

## v0.1.0

* GPU-accelerated Sanity added via the wgpu/cubecl framework, behind the `gpu`
  feature. `sanity_gpu` takes and returns what `sanity` does and supports every
  variance rule. The device works in `f32`; the maths is rearranged so no sum
  cancels, the offset is solved against an `f64` anchor and the likelihood over
  the variance grid is assembled in `f64` on the host. On an M1 Max,
  `Marginalise` runs 78x faster at 2000 genes by 20000 cells and 60x at 500 by
  200000, with the worst log fold change under 1% of its error bar from the
  CPU path's.
* `MaxPosterior` on the GPU settles near ties between bins in `f64` on the CPU,
  so it lands on the same bin as the CPU path.
* `gpu-tests` feature with CPU parity tests for every rule, and a GPU lane in
  CI.

## v0.0.1

* Clean-room Rust implementation of Sanity from `docs/SPEC.md`: `sanity`,
  `sanity_select`, the four variance rules and the simulator.
