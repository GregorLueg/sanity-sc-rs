//! The storage float trait.
//!
//! Storage is generic; arithmetic is not. Every reduction in this crate
//! accumulates in `f64` regardless of `T`, because the marginal likelihood sums
//! over all cells and the differences that pick one variance bin over its
//! neighbours are small against that sum.

use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

/// Float types this crate will store inputs and outputs in.
///
/// The bound is deliberately thin: nothing in the algorithm is computed at `T`
/// precision, so `T` only needs to convert to and from `f64` and to cross thread
/// boundaries.
pub trait SanityFloat:
    Float + FromPrimitive + ToPrimitive + Send + Sync + Debug + Default + 'static
{
}

impl SanityFloat for f32 {}
impl SanityFloat for f64 {}

/// Narrow an `f64` accumulator back to the storage type.
///
/// ### Params
///
/// * `x` - The value to narrow.
///
/// ### Returns
///
/// `x` as `T`. Infallible for `f32` and `f64`; falls back to zero only if a
/// future implementor makes the conversion partial.
#[inline(always)]
pub(crate) fn narrow<T: SanityFloat>(x: f64) -> T {
    T::from_f64(x).unwrap_or_else(T::zero)
}
