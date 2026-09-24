//! The CubeCL path. Feature-gated on `gpu`, CPU-only build otherwise.
//!
//! Everything on the device is `f32`, because wgpu exposes no `f64` on any
//! backend. That is a departure from the crate's numeric policy, and the reason
//! this path is held to its own measured tolerance rather than to the CPU's.
//!
//! ### Mapping
//!
//! The first pass is [`kernels::sweep_grid_gpu`]: one plane per gene, running
//! the whole variance grid, with each cell sweep reduced across the plane so
//! every lane agrees on the offset solve without a barrier. The host then
//! assembles each bin's log marginal likelihood in `f64` and forms the weights.
//! The second pass is [`kernels::marginalise_gpu`]: one thread per gene and
//! cell, integrating over the kept bins with the offsets the first pass left
//! on the device.
//!
//! ### Precision
//!
//! Compensated summation is no defence on this backend: wgpu compiles Metal
//! shaders with fast-math on, which folds a `two_sum`. So the arithmetic is
//! arranged so that nothing cancels, by three exact rewrites:
//!
//! * **The offset term leaves the likelihood.** With `d_c = t_c - ln T_c -
//!   ln(v s) + z` and `s = sum_c k_c`, `sum_c k_c d_c - s z = sum_c k_c t_c -
//!   sum_c k_c ln T_c - s ln(v s)`. The pair `sum_c k_c d_c` and `s z`, each of
//!   order `1e6` for an expressed gene, never forms.
//! * **Every per-cell term is of order one.** The stationarity condition is
//!   `omega_c = v k_c - d_c`, so the offset residual `sum_c omega_c - v s` is
//!   `-sum_c d_c`, an empty cell's log fold change is `-omega_c`, and the two
//!   large likelihood terms, `k_c ln omega_c` and `ln(1 + omega_c)`, split into
//!   an anchor the host knows exactly plus `ln(1 - d_c / (v k_c))` and
//!   `ln(1 + (1 - d_c) / (v k_c))`. See `kernels::sweep`.
//! * **The host assembles the likelihood.** The device returns only the five
//!   sums; the combination, which does cancel, happens in `f64`.
//!
//! Without the second rewrite, the sums were of order `v s` and `K ln K`. On a
//! simulated gene holding a quarter of the reads (`K = 2.6e6`) that left its log
//! fold changes `1.6e-2` of an error bar from the CPU's, above the 1% to which
//! the grid itself is resolved ([`crate::config::DEFAULT_VARIANCE_BINS`]).

pub mod kernels;

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use rayon::prelude::*;

use crate::config::{SanityParams, VarianceRule};
use crate::errors::SanityErrors;
use crate::float::{SanityFloat, narrow};
use crate::input::CountMatrix;
use crate::model::gene::{
    MARGINALISE_MIN_WEIGHT, collapse_target, marginal_summary, posterior_weights,
};
use crate::utils::polygamma::{digamma, trigamma};
use crate::{SanityOutput, prepare_run};

use self::kernels::{marginalise_gpu, sweep_grid_gpu};

////////////
// Consts //
////////////

/// Widest-plane genes per workgroup in the first pass, one plane each.
///
/// The planes share nothing, so this is purely an occupancy knob. Taken from
/// the edge-rs NEBULA kernel, which measured two 32-lane planes best on an M1
/// Max; not yet measured here. The workgroup is sized for the device's widest
/// plane, so a driver that picks a narrower one fits more genes into it.
const PLANES_PER_CUBE: u32 = 2;

/// Preferred workgroup width of the second pass, which has no reduction and
/// needs no particular width.
const MARGINALISE_WORKGROUP: u32 = 256;

/// Ceiling on the largest buffer of one batch, the second pass's output.
///
/// Batching bounds device memory; the device's own per-binding limit applies on
/// top. At 20000 cells, 512 MiB is 3355 genes per batch.
const GPU_BATCH_BYTES: u64 = 512 << 20;

/////////////
// Staging //
/////////////

/// One batch's resident inputs.
struct Batch<R: Runtime> {
    /// Dense counts, `[gene * n_cells + c]`.
    counts: GpuTensor<R, f32>,
    /// `K` per gene.
    totals: Vec<f64>,
    /// Cells with a non-zero count, per gene.
    n_expressed: Vec<usize>,
    /// Input index of the first gene.
    first: usize,
    /// Genes in the batch.
    n_genes: usize,
}

/// The first pass's output, on the device and read back.
struct SweepResult<R: Runtime> {
    /// The raw output, left on the device for the second pass.
    device: GpuTensor<R, f32>,
    /// Per-gene, per-bin parameters the pass ran with, `v` then `ln v`.
    bins: GpuTensor<R, f32>,
    /// Per-gene scalars the pass ran with.
    scalars: GpuTensor<R, f32>,
    /// The output read back, `[(r * n_genes + gene) * n_bins + b]`.
    host: Vec<f32>,
}

/// Upload one batch of genes as dense rows.
///
/// ### Params
///
/// * `counts` - The count matrix.
/// * `first` - Input index of the first gene.
/// * `n_genes` - Genes in the batch.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The batch, or a device-limit error.
fn stage_batch<R: Runtime>(
    counts: &CountMatrix,
    first: usize,
    n_genes: usize,
    client: &ComputeClient<R>,
) -> Result<Batch<R>, SanityErrors> {
    let n_cells = counts.n_cells();
    let mut dense = vec![0.0f32; n_genes * n_cells];
    dense
        .par_chunks_mut(n_cells)
        .enumerate()
        .for_each(|(g, row)| {
            let (indices, values) = counts.gene(first + g);
            for (&i, &k) in indices.iter().zip(values) {
                row[i as usize] = k as f32;
            }
        });
    let (totals, n_expressed) = (first..first + n_genes)
        .map(|g| {
            let values = counts.gene(g).1;
            (
                values.iter().map(|&x| x as f64).sum::<f64>(),
                values.iter().filter(|&&x| x > 0).count(),
            )
        })
        .unzip();
    Ok(Batch {
        counts: GpuTensor::from_slice(&dense, vec![n_genes * n_cells], client)?,
        totals,
        n_expressed,
        first,
        n_genes,
    })
}

/// Run the first pass over a batch.
///
/// ### Params
///
/// * `batch` - The resident batch.
/// * `log_totals` - `ln T_c` on the device.
/// * `n_cells` - Cells.
/// * `bin_v` - `v` per gene and bin, `[gene * n_bins + b]`.
/// * `guess` - First bin's offset guess per gene.
/// * `cold_start` - Whether the guess is cold.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The pass's output, or [`SanityErrors::GpuOffsetSolveDiverged`] for the first
/// gene whose solve failed.
fn run_sweep<R: Runtime>(
    batch: &Batch<R>,
    log_totals: &GpuTensor<R, f32>,
    n_cells: usize,
    bin_v: &[f64],
    guess: &[f64],
    cold_start: bool,
    client: &ComputeClient<R>,
) -> Result<SweepResult<R>, SanityErrors> {
    let n_genes = batch.n_genes;
    let n_bins = bin_v.len() / n_genes;
    let limits = GpuLimits::from_client(client);

    let cube_width = PLANES_PER_CUBE * limits.plane_size_max;
    if limits.plane_size_max == 0 || cube_width > limits.max_units_per_cube {
        return Err(SanityErrors::GpuUnsupported {
            reason: format!(
                "the first pass needs plane operations; this device reports planes of {} to {} lanes and at most {} units per cube",
                limits.plane_size_min, limits.plane_size_max, limits.max_units_per_cube
            ),
        });
    }

    let scalars: Vec<f32> = batch
        .totals
        .iter()
        .map(|&k| k as f32)
        .chain(batch.totals.iter().map(|&k| k.ln() as f32))
        .chain(guess.iter().map(|&z| z as f32))
        .collect();
    let mut bins: Vec<f32> = bin_v.iter().map(|&v| v as f32).collect();
    bins.extend(bin_v.iter().map(|&v| v.ln() as f32));

    let scalars = GpuTensor::<R, f32>::from_slice(&scalars, vec![3 * n_genes], client)?;
    let bins = GpuTensor::<R, f32>::from_slice(&bins, vec![2 * n_genes * n_bins], client)?;
    let out = GpuTensor::<R, f32>::empty(vec![5 * n_genes * n_bins], client)?;
    let status = GpuTensor::<R, u32>::empty(vec![n_genes], client)?;

    let blocks = (n_genes as u32).div_ceil(PLANES_PER_CUBE);
    let (gx, gy) = grid_2d(blocks, &limits)?;
    let count = checked_cube_count("sweep_grid_gpu", gx, gy, 1, &limits)?;
    unsafe {
        sweep_grid_gpu::launch_unchecked::<f32, R>(
            client,
            count,
            CubeDim::new_1d(cube_width),
            batch.counts.into_tensor_arg(),
            log_totals.into_tensor_arg(),
            scalars.into_tensor_arg(),
            bins.into_tensor_arg(),
            out.into_tensor_arg(),
            status.into_tensor_arg(),
            n_genes as u32,
            n_cells as u32,
            n_bins as u32,
            u32::from(cold_start),
        );
    }

    let status = status.read(client)?;
    if let Some((g, &code)) = status.iter().enumerate().find(|(_, c)| **c != 0) {
        return Err(SanityErrors::GpuOffsetSolveDiverged {
            gene: batch.first + g,
            bin: code as usize - 1,
        });
    }
    let host = out.clone().read(client)?;
    Ok(SweepResult {
        device: out,
        bins,
        scalars,
        host,
    })
}

/// Run the second pass over a batch.
///
/// ### Params
///
/// * `batch` - The resident batch.
/// * `log_totals` - `ln T_c` on the device.
/// * `n_cells` - Cells.
/// * `sweep` - The first pass whose bins to integrate over.
/// * `weights` - `W_b` per gene and bin, zero for a dropped bin.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// `d_c` for every gene and cell, then `e_c`, gene-major.
fn run_marginalise<R: Runtime>(
    batch: &Batch<R>,
    log_totals: &GpuTensor<R, f32>,
    n_cells: usize,
    sweep: &SweepResult<R>,
    weights: &[f32],
    client: &ComputeClient<R>,
) -> Result<Vec<f32>, SanityErrors> {
    let n_genes = batch.n_genes;
    let n_bins = weights.len() / n_genes;
    let limits = GpuLimits::from_client(client);

    let weights = GpuTensor::<R, f32>::from_slice(weights, vec![n_genes * n_bins], client)?;
    let out = GpuTensor::<R, f32>::empty(vec![2 * n_genes * n_cells], client)?;

    let width = resolve_workgroup_size(MARGINALISE_WORKGROUP, &limits);
    let blocks_per_gene = (n_cells as u32).div_ceil(width);
    let (gx, gy) = grid_2d(blocks_per_gene * n_genes as u32, &limits)?;
    let count = checked_cube_count("marginalise_gpu", gx, gy, 1, &limits)?;
    unsafe {
        marginalise_gpu::launch_unchecked::<f32, R>(
            client,
            count,
            CubeDim::new_1d(width),
            batch.counts.into_tensor_arg(),
            log_totals.into_tensor_arg(),
            sweep.scalars.into_tensor_arg(),
            sweep.bins.into_tensor_arg(),
            sweep.device.into_tensor_arg(),
            weights.into_tensor_arg(),
            out.into_tensor_arg(),
            n_genes as u32,
            n_cells as u32,
            n_bins as u32,
            blocks_per_gene,
        );
    }
    Ok(out.read(client)?)
}

/// Log marginal likelihood of one bin from the first pass's sums, up to a
/// constant shared by every bin.
///
/// SI eq. 20 and 33, in the rewritten form of the module doc. The dropped
/// constant is `sum_c k_c ln(k_c T_c / s) - 0.5 sum_{k_c > 0} ln k_c`, which the
/// softmax ignores.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `s` - `K`.
/// * `n_cells` - Cells.
/// * `n_expressed` - Cells with a non-zero count.
/// * `curvature` - `S_A`.
/// * `sum_sq` - `sum_c d_c^2`.
/// * `sum_kw` - `sum_{k > 0} k ln(omega / (v k))`.
/// * `sum_log1p` - `sum_{k = 0} ln(1 + omega) + sum_{k > 0} ln((1 + omega) / (v k))`.
///
/// ### Returns
///
/// `ln P(k | v)` minus the shared constant.
#[allow(clippy::too_many_arguments)]
fn log_marginal(
    v: f64,
    s: f64,
    n_cells: f64,
    n_expressed: f64,
    curvature: f64,
    sum_sq: f64,
    sum_kw: f64,
    sum_log1p: f64,
) -> f64 {
    let log_v = v.ln();
    let log_star = -0.5 * n_cells * log_v - 0.5 * sum_sq / v + sum_kw;
    let log_det = curvature.ln() - (v * s).ln() + sum_log1p + n_expressed * log_v - n_cells * log_v;
    log_star - 0.5 * log_det
}

//////////////
// Frontend //
//////////////

/// Run Sanity on the GPU.
///
/// Same inference as [`crate::sanity`], evaluated in `f32` on the device with
/// the likelihood assembled in `f64` on the host. See the module doc for the
/// precision this costs. Genes are processed in batches bounded by
/// [`GPU_BATCH_BYTES`] and the device's per-binding limit.
///
/// ### Params
///
/// * `counts` - Raw UMI counts, gene-major sparse.
/// * `cell_totals` - Total UMI count of every cell over *all* genes.
/// * `params` - Run parameters, or [`SanityParams::default`].
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The same output as [`crate::sanity`]. Values are `f32` precision whatever `T`.
pub fn sanity_gpu<T: SanityFloat, R: Runtime>(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: Option<SanityParams>,
    client: &ComputeClient<R>,
) -> Result<SanityOutput<T>, SanityErrors> {
    let params = params.unwrap_or_default();
    let n_cells = counts.n_cells();
    let n_genes = counts.n_genes();
    let (grid, log_totals, log_total_sum) = prepare_run(counts, cell_totals, &params)?;
    let limits = GpuLimits::from_client(client);

    let row_bytes = 2 * n_cells as u64 * size_of::<f32>() as u64;
    let budget = GPU_BATCH_BYTES.min(limits.max_binding_bytes);
    let batch_genes = ((budget / row_bytes) as usize).clamp(1, n_genes.max(1));

    let log_totals_f32: Vec<f32> = log_totals.iter().map(|&x| x as f32).collect();
    let log_totals_dev = GpuTensor::<R, f32>::from_slice(&log_totals_f32, vec![n_cells], client)?;

    let mut out = SanityOutput {
        log_fold_changes: vec![T::zero(); n_genes * n_cells],
        error_bars: vec![T::zero(); n_genes * n_cells],
        mean_log_quotient: vec![T::zero(); n_genes],
        mean_log_quotient_error: vec![T::zero(); n_genes],
        variance: vec![T::zero(); n_genes],
        genes: (0..n_genes).collect(),
        n_genes,
        n_cells,
    };

    let mut first = 0;
    while first < n_genes {
        let n = batch_genes.min(n_genes - first);
        let batch = stage_batch(counts, first, n, client)?;
        let ks = &batch.totals;

        let (sweep, weights, summaries) = match params.variance_rule {
            VarianceRule::Fixed(v) => {
                let guess = vec![log_total_sum + 0.5 * v; n];
                let sweep = run_sweep(
                    &batch,
                    &log_totals_dev,
                    n_cells,
                    &vec![v; n],
                    &guess,
                    true,
                    client,
                )?;
                let summaries: Vec<(f64, f64, f64)> = (0..n)
                    .map(|g| {
                        (
                            digamma(ks[g]) - sweep.host[g] as f64,
                            trigamma(ks[g]).sqrt(),
                            v,
                        )
                    })
                    .collect();
                (sweep, vec![1.0f32; n], summaries)
            }
            rule => {
                let n_bins = grid.len();
                let bin_v: Vec<f64> = (0..n).flat_map(|_| grid.values.iter().copied()).collect();
                let guess = vec![log_total_sum + 0.5 * grid.values[0]; n];
                let sweep = run_sweep(
                    &batch,
                    &log_totals_dev,
                    n_cells,
                    &bin_v,
                    &guess,
                    true,
                    client,
                )?;

                let stride = n * n_bins;
                let h = &sweep.host;
                let per_gene: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
                    .into_par_iter()
                    .map(|g| {
                        let mut log_lik = vec![0.0; n_bins];
                        let mut offsets = vec![0.0; n_bins];
                        for (b, &v) in grid.values.iter().enumerate() {
                            let i = g * n_bins + b;
                            offsets[b] = h[i] as f64;
                            log_lik[b] = log_marginal(
                                v,
                                ks[g],
                                n_cells as f64,
                                batch.n_expressed[g] as f64,
                                h[stride + i] as f64,
                                h[2 * stride + i] as f64,
                                h[3 * stride + i] as f64,
                                h[4 * stride + i] as f64,
                            );
                        }
                        let mut weights = vec![0.0; n_bins];
                        posterior_weights(&log_lik, &mut weights)?;
                        Ok((weights, offsets))
                    })
                    .collect::<Result<_, SanityErrors>>()?;

                if matches!(rule, VarianceRule::Marginalise) {
                    let summaries = per_gene
                        .iter()
                        .enumerate()
                        .map(|(g, (weights, offsets))| {
                            let summary = marginal_summary(ks[g], &grid.values, weights, offsets).0;
                            (
                                summary.mean_log_quotient,
                                summary.mean_log_quotient_error,
                                summary.variance,
                            )
                        })
                        .collect();
                    let weights: Vec<f32> = per_gene
                        .iter()
                        .flat_map(|(w, _)| w.iter())
                        .map(|&w| {
                            if w < MARGINALISE_MIN_WEIGHT {
                                0.0
                            } else {
                                w as f32
                            }
                        })
                        .collect();
                    (sweep, weights, summaries)
                } else {
                    let targets: Vec<(f64, f64, f64)> = per_gene
                        .iter()
                        .map(|(w, z)| collapse_target(rule, &grid.values, w, z))
                        .collect();
                    let bin_v: Vec<f64> = targets.iter().map(|t| t.0).collect();
                    let guess: Vec<f64> = targets.iter().map(|t| t.1).collect();
                    drop(sweep);
                    let point = run_sweep(
                        &batch,
                        &log_totals_dev,
                        n_cells,
                        &bin_v,
                        &guess,
                        false,
                        client,
                    )?;
                    let summaries = (0..n)
                        .map(|g| {
                            (
                                digamma(ks[g]) - point.host[g] as f64,
                                trigamma(ks[g]).sqrt(),
                                targets[g].2,
                            )
                        })
                        .collect();
                    (point, vec![1.0f32; n], summaries)
                }
            }
        };

        let rows = run_marginalise(&batch, &log_totals_dev, n_cells, &sweep, &weights, client)?;
        let span = first * n_cells..(first + n) * n_cells;
        out.log_fold_changes[span.clone()]
            .par_iter_mut()
            .zip(&rows[..n * n_cells])
            .for_each(|(o, &x)| *o = narrow(x as f64));
        out.error_bars[span]
            .par_iter_mut()
            .zip(&rows[n * n_cells..])
            .for_each(|(o, &x)| *o = narrow(x as f64));
        for (g, (m, dm, v)) in summaries.into_iter().enumerate() {
            out.mean_log_quotient[first + g] = narrow(m);
            out.mean_log_quotient_error[first + g] = narrow(dm);
            out.variance[first + g] = narrow(v);
        }
        first += n;
    }

    Ok(out)
}
