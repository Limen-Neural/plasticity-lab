// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reward-modulated SNN learning/training orchestration layer for the
//! Limen-Neural stack.
//!
//! This crate sits above [`neuromod::SpikingNetwork`], which owns neuron/network
//! dynamics, neuromodulator state, and the foundational classical and
//! reward-modulated STDP primitives. `plasticity-lab` does not reimplement those
//! primitives — it drives them through `neuromod`'s public API: single-step
//! reward modulation via [`PlasticityTrainer::train_step`] and batch sessions via
//! [`PlasticityTrainer::run_session`] (optional per-step telemetry via
//! [`PlasticityTrainer::run_session_with_observer`]). Seeded replay uses
//! [`PlasticityTrainer::train_step_with_rng`] / [`PlasticityTrainer::run_session_with_rng`]
//! to inject a caller RNG into neuromod's stochastic input encoding.
//! Held-out evaluation uses [`PlasticityTrainer::eval_step`] or
//! [`PlasticityTrainer::run_eval`] (and their caller-RNG variants) to advance
//! runtime dynamics through neuromod's frozen stepping without retaining
//! plasticity changes. Evaluation takes explicit modulators but no reward;
//! callers own the train/held-out split.
//!
//! # Features
//!
//! - **default** — core loop only (`neuromod` + serde/thiserror/rand).
//! - **`critic`** — optional dep on `limbic-critic`, plus the `bridge`
//!   adapter that converts critic `limbic_critic::ModulatorVector` into
//!   [`neuromod::NeuroModulators`].
//!
//! `bridge` and `limbic_critic::ModulatorVector` above are plain code spans,
//! not doc links: both only exist with the `critic` feature enabled.
//!
//! # Quick example
//!
//! ```rust
//! use neuromod::SpikingNetwork;
//! use plasticity_lab::{PlasticityTrainer, TrainingConfig, TrainingExample};
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
//! let mut network = SpikingNetwork::with_dimensions(4, 2, 8);
//! for neuron in &mut network.neurons {
//!     // `with_dimensions` intentionally creates blank weights. Seed the
//!     // documented L1 budget equally across input channels before training.
//!     neuron.weights.fill(2.0 / network.num_channels as f32);
//! }
//! let batch = vec![TrainingExample {
//!     stimuli: vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02],
//!     reward: 1.0,
//! }; 8];
//! let mut rng = StdRng::seed_from_u64(0x5EED);
//! let summary = trainer
//!     .run_session_with_rng(&mut network, &batch, &mut rng)
//!     .unwrap();
//! assert!(summary.total_spikes > 0);
//! assert!(summary.weight_drifts.iter().flatten().any(|delta| delta.abs() > 1e-5));
//! ```
//!
//! # Limbic bridge (`critic`)
//!
//! ```rust,ignore
//! use limbic_critic::SimpleCritic;
//! use plasticity_lab::bridge::{apply_modulator_vector, to_neuromodulators};
//!
//! let vector = SimpleCritic::assess(&env);
//! let _ = apply_modulator_vector(&mut network, &stimuli, &vector);
//! // or: network.step(&stimuli, &to_neuromodulators(&vector));
//! ```
//!
//! See the crate README for the ecosystem map, [scope/ownership
//! boundaries](https://github.com/Limen-Neural/plasticity-lab#scope-and-ownership-boundaries)
//! (including the boundary with `neuromod`'s network dynamics and plasticity
//! primitives), and common usage patterns.

pub mod config;
mod evaluation;
pub mod observer;
pub mod trainer;

#[cfg(test)]
mod replay;

#[cfg(feature = "critic")]
pub mod bridge;

pub use config::{RewardMapping, RewardMappingBuilder, RewardMappingError, TrainingConfig};
pub use evaluation::{EvaluationExample, EvaluationSummary};
pub use observer::{TrainingObserver, TrainingStepEvent};
pub use trainer::{
    PlasticityTrainer, SampleInvariant, TrainerError, TrainingExample, TrainingSummary,
};

#[cfg(feature = "critic")]
pub use bridge::{apply_modulator_vector, from_neuromodulators, to_neuromodulators};

/// Deprecated alias for [`PlasticityTrainer`].
///
/// This crate is pre-1.0. The alias is a short-lived migration aid for existing
/// consumers and is **not** part of the documented public API: new code must use
/// [`PlasticityTrainer`] directly. It will be removed in a future release.
#[deprecated(
    note = "renamed to `PlasticityTrainer`; this alias will be removed in a future release"
)]
#[doc(hidden)]
pub use trainer::PlasticityTrainer as SpikenautTrainer;

#[cfg(test)]
mod deprecated_alias_tests {
    #![allow(deprecated)]

    use super::{SpikenautTrainer, TrainingConfig};

    #[test]
    fn spikenaut_trainer_alias_still_constructs() {
        let _trainer = SpikenautTrainer::new(TrainingConfig::default());
    }
}
