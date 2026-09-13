// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic reward-modulated plasticity loops for spiking neural networks.
//!
//! This crate provides a small training loop around [`neuromod::SpikingNetwork`]:
//! single-step reward modulation via [`PlasticityTrainer::train_step`] and batch
//! sessions via [`PlasticityTrainer::run_session`].
//!
//! # Features
//!
//! - **default** — core loop only (`neuromod` + serde/tracing/thiserror/rand).
//! - **`integration`** — optional deps on `limbic-critic` and `axon-encoder`,
//!   plus the [`bridge`] adapter that converts critic
//!   [`limbic_critic::ModulatorVector`] into [`neuromod::NeuroModulators`].
//!
//! # Quick example
//!
//! ```rust,no_run
//! use neuromod::SpikingNetwork;
//! use plasticity_lab::{PlasticityTrainer, TrainingConfig, TrainingExample};
//!
//! let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
//! let mut network = SpikingNetwork::with_dimensions(32, 8, 64);
//! let batch = vec![TrainingExample {
//!     stimuli: vec![0.25; 64],
//!     reward: 0.2,
//! }];
//! let summary = trainer.run_session(&mut network, &batch).unwrap();
//! assert_eq!(summary.steps_processed, 1);
//! ```
//!
//! # Limbic bridge (`integration`)
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
//! See the crate README for ecosystem map, ownership boundaries, and patterns.

pub mod config;
pub mod trainer;

#[cfg(feature = "integration")]
pub mod bridge;

pub use config::TrainingConfig;
pub use trainer::{PlasticityTrainer, TrainerError, TrainingExample, TrainingSummary};

#[cfg(feature = "integration")]
pub use bridge::{apply_modulator_vector, from_neuromodulators, to_neuromodulators};

/// Deprecated alias for [`PlasticityTrainer`].
///
/// This crate is pre-1.0 and consumed via git dependency, so this alias exists only
/// as a short-lived migration aid for existing consumers. It is **not** part of the
/// documented public API: new code must use [`PlasticityTrainer`] directly, and this
/// alias will be removed in a future release.
#[deprecated(
    note = "renamed to `PlasticityTrainer`; this alias will be removed in a future release"
)]
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
