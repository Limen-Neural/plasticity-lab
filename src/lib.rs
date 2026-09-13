// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reward-modulated SNN learning/training orchestration layer for the
//! Limen-Neural stack.
//!
//! This crate sits above [`neuromod::SpikingNetwork`], which owns neuron/network
//! dynamics, neuromodulator state, and the foundational classical and
//! reward-modulated STDP primitives. `plasticity-lab` does not reimplement those
//! primitives — it drives them through `neuromod`'s public API: single-step
//! reward modulation via [`SpikenautTrainer::train_step`] and batch sessions via
//! [`SpikenautTrainer::run_session`].
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
//! use plasticity_lab::{SpikenautTrainer, TrainingConfig, TrainingExample};
//!
//! let mut trainer = SpikenautTrainer::new(TrainingConfig::default());
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
//! See the crate README for the ecosystem map, [scope/ownership
//! boundaries](https://github.com/Limen-Neural/plasticity-lab#scope-and-ownership-boundaries)
//! (including the boundary with `neuromod`'s network dynamics and plasticity
//! primitives), and common usage patterns.

pub mod config;
pub mod trainer;

#[cfg(feature = "integration")]
pub mod bridge;

pub use config::TrainingConfig;
pub use trainer::{SpikenautTrainer, TrainerError, TrainingExample, TrainingSummary};

#[cfg(feature = "integration")]
pub use bridge::{apply_modulator_vector, from_neuromodulators, to_neuromodulators};
