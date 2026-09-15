// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::config::TrainingConfig;
use neuromod::{NeuroModulators, SpikingNetwork, StepError};
use thiserror::Error;

/// Summary metrics collected over a [`PlasticityTrainer::run_session`] call.
///
/// Drifts are relative to network state at the start of the session.
#[derive(Debug, Default, Clone)]
pub struct TrainingSummary {
    /// Number of training examples processed.
    pub steps_processed: usize,
    /// Total spike events across all steps.
    pub total_spikes: u64,
    /// Mean reward over the batch.
    pub avg_reward: f32,
    /// Per-neuron change in firing threshold (final − initial).
    pub threshold_drifts: Vec<f32>,
    /// Per-neuron, per-channel weight change (final − initial).
    pub weight_drifts: Vec<Vec<f32>>,
    /// Spike count per neuron over the session.
    pub per_neuron_spikes: Vec<u64>,
}

/// One input sample for a generic training session: stimuli plus scalar reward.
///
/// Encoding and reward shaping live outside this crate (`axon-encoder`,
/// `limbic-critic`, or application code).
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Flat stimulus vector (length must match the network input size).
    pub stimuli: Vec<f32>,
    /// Scalar reward for this step (positive → dopamine-biased, negative →
    /// norepinephrine-biased / stress-arousal).
    pub reward: f32,
}

/// Batch-admission invariant violated by one [`TrainingExample`].
///
/// Produced by [`PlasticityTrainer::run_session`]'s preflight pass *before* any
/// network, modulator, eligibility, or metric state mutates. Single-step APIs
/// (`train_step`, `train_step_with_modulators`, and `train_step_from_critic`) do
/// not run this check — they stay compatible with their existing `StepError`
/// contracts, including NaN-reward skipping in `train_step`.
///
/// `TrainingExample` has no sample IDs, so ordering is the batch slice order
/// (index `0` is the first example). `TrainingConfig` currently has no invalid
/// encodings (`use_reward_modulation` is a `bool`); there is therefore no
/// config-level rejection variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SampleInvariant {
    /// `stimuli.len()` did not match [`SpikingNetwork::num_channels`].
    #[error("stimulus length mismatch: expected {expected}, got {got}")]
    StimulusLenMismatch { expected: usize, got: usize },
    /// A stimulus component was NaN or ±infinity.
    #[error("non-finite stimulus at channel {channel}")]
    NonFiniteStimulus { channel: usize },
    /// Reward was ±infinity (NaN remains allowed: `train_step` skips modulation
    /// and `run_session` omits it from `avg_reward`).
    #[error("infinite reward")]
    InfiniteReward,
}

/// Errors from batch training sessions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TrainerError {
    /// Underlying network step failed.
    #[error("network step failed: {0:?}")]
    Step(StepError),
    /// `run_session` was called with an empty batch.
    ///
    /// Empty is a batch-level condition (no sample index). The network, trainer
    /// config, and any caller-owned metrics are left untouched.
    #[error("empty training batch")]
    EmptyBatch,
    /// Preflight rejected the batch because sample `index` violated `reason`.
    ///
    /// No earlier sample has been applied; trainer and network state are unchanged.
    #[error("invalid training sample {index}: {reason}")]
    InvalidSample {
        /// Zero-based index into the batch slice.
        index: usize,
        /// Which admission invariant failed.
        reason: SampleInvariant,
    },
}

/// Reward-modulated training loop over a [`SpikingNetwork`].
///
/// Applies scalar rewards to neuromodulators and steps the network. Domain-specific
/// logic (mining, trading, distillation) does not belong here. For critic-shaped
/// vectors under the `critic` feature, use `Self::train_step_from_critic`
/// or `crate::bridge` (plain code spans, not doc links — both only exist
/// with the `critic` feature enabled).
pub struct PlasticityTrainer {
    /// Active training configuration.
    pub config: TrainingConfig,
}

impl PlasticityTrainer {
    /// Creates a trainer with the given configuration.
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Runs one training step with generic stimuli and an externally computed reward.
    ///
    /// When [`TrainingConfig::use_reward_modulation`] is `true` (default), positive
    /// `reward` increases dopamine and decreases norepinephrine; negative reward does
    /// the opposite emphasis. Modulator values are clamped to `[0.0, 1.0]`. When the
    /// flag is `false`, the network steps with its current modulators unchanged.
    ///
    /// Returns indices of neurons that spiked, or a [`StepError`] from neuromod.
    pub fn train_step(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        reward: f32,
    ) -> Result<Vec<usize>, StepError> {
        let mut modulators: NeuroModulators = network.modulators;

        // Skip modulation on NaN: f32::clamp returns NaN unchanged rather than
        // panicking, so a NaN reward would otherwise propagate silently into
        // modulators that poison subsequent STDP / homeostasis updates.
        if self.config.use_reward_modulation && !reward.is_nan() {
            // Positive reward shifts toward dopamine; negative toward norepinephrine
            // (stress/arousal). neuromod replaced the former cortisol field with
            // norepinephrine (see neuromod::NeuroModulators).
            if reward > 0.0 {
                modulators.dopamine = (modulators.dopamine + reward * 0.1).clamp(0.0, 1.0);
                modulators.norepinephrine =
                    (modulators.norepinephrine - reward * 0.05).clamp(0.0, 1.0);
            } else {
                modulators.norepinephrine =
                    (modulators.norepinephrine - reward * 0.2).clamp(0.0, 1.0);
                modulators.dopamine = (modulators.dopamine + reward * 0.1).clamp(0.0, 1.0);
            }
        }

        network.step(stimuli, &modulators)
    }

    /// Steps the network with explicit neuromodulators (e.g. from the limbic bridge).
    ///
    /// Does not apply scalar reward shaping; callers that already ran a critic
    /// should convert via `crate::to_neuromodulators` (`critic` feature; a
    /// plain code span, not a doc link — that item doesn't exist without the
    /// feature) and pass the result here.
    pub fn train_step_with_modulators(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<Vec<usize>, StepError> {
        network.step(stimuli, modulators)
    }

    /// Steps the network with a critic [`limbic_critic::ModulatorVector`].
    ///
    /// Converts via [`crate::bridge::to_neuromodulators`] then steps. Available only
    /// with the `critic` feature.
    #[cfg(feature = "critic")]
    pub fn train_step_from_critic(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        vector: &limbic_critic::ModulatorVector,
    ) -> Result<Vec<usize>, StepError> {
        self.train_step_with_modulators(
            network,
            stimuli,
            &crate::bridge::to_neuromodulators(vector),
        )
    }

    /// Replays a batch of generic training examples and returns aggregated metrics.
    ///
    /// Admission is atomic: every example is validated (dimensions, finite
    /// stimuli, infinite-reward) *before* the first `train_step`. A malformed
    /// sample at index `N` therefore cannot leave samples `0..N` applied.
    /// Examples are then processed in slice order, matching the historical
    /// sequential contract.
    ///
    /// # Errors
    ///
    /// - [`TrainerError::EmptyBatch`] if `data` is empty (no sample index).
    /// - [`TrainerError::InvalidSample`] if any example fails preflight; the
    ///   error names the first failing index and invariant. Network and trainer
    ///   state are unchanged.
    /// - [`TrainerError::Step`] if a network step fails after admission (for
    ///   example a `StepError` that cannot be seen from the example alone).
    pub fn run_session(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        Self::admit_batch(network, data)?;

        let mut summary = TrainingSummary::default();
        let initial_thresholds = network.get_thresholds();
        let initial_weights: Vec<Vec<f32>> =
            network.neurons.iter().map(|n| n.weights.clone()).collect();

        summary.per_neuron_spikes = vec![0; network.neurons.len()];
        let mut total_reward = 0.0;
        let mut valid_reward_count = 0;

        for example in data {
            let spikes = self
                .train_step(network, &example.stimuli, example.reward)
                .map_err(TrainerError::Step)?;
            if !example.reward.is_nan() {
                total_reward += example.reward;
                valid_reward_count += 1;
            }
            summary.steps_processed += 1;

            summary.total_spikes += spikes.len() as u64;
            for &idx in &spikes {
                if idx < summary.per_neuron_spikes.len() {
                    summary.per_neuron_spikes[idx] += 1;
                }
            }
        }

        summary.avg_reward = if valid_reward_count > 0 {
            total_reward / valid_reward_count as f32
        } else {
            0.0
        };

        let final_thresholds = network.get_thresholds();
        for i in 0..network.neurons.len() {
            summary
                .threshold_drifts
                .push(final_thresholds[i] - initial_thresholds[i]);

            let mut w_deltas = Vec::new();
            for (ch, &w) in network.neurons[i].weights.iter().enumerate() {
                w_deltas.push(w - initial_weights[i][ch]);
            }
            summary.weight_drifts.push(w_deltas);
        }

        Ok(summary)
    }

    /// Validates the whole batch without mutating `network` or `self`.
    ///
    /// Fails closed on the first violation so callers can report a single sample
    /// index. Empty batches are a distinct error (no index to name).
    fn admit_batch(network: &SpikingNetwork, data: &[TrainingExample]) -> Result<(), TrainerError> {
        if data.is_empty() {
            return Err(TrainerError::EmptyBatch);
        }

        for (index, example) in data.iter().enumerate() {
            if let Some(reason) = sample_invariant(network, example) {
                return Err(TrainerError::InvalidSample { index, reason });
            }
        }
        Ok(())
    }
}

/// Returns the first violated admission invariant for `example`, if any.
fn sample_invariant(
    network: &SpikingNetwork,
    example: &TrainingExample,
) -> Option<SampleInvariant> {
    if example.stimuli.len() != network.num_channels {
        return Some(SampleInvariant::StimulusLenMismatch {
            expected: network.num_channels,
            got: example.stimuli.len(),
        });
    }
    if let Some(channel) = example.stimuli.iter().position(|x| !x.is_finite()) {
        return Some(SampleInvariant::NonFiniteStimulus { channel });
    }
    if example.reward.is_infinite() {
        return Some(SampleInvariant::InfiniteReward);
    }
    None
}

/// Deprecated alias for [`PlasticityTrainer`], also reachable via the full module path.
///
/// The crate-root alias (`plasticity_lab::SpikenautTrainer`) doesn't cover code that
/// imports via `plasticity_lab::trainer::SpikenautTrainer` directly — this re-export
/// closes that gap so both paths keep working during the migration window.
#[deprecated(
    note = "renamed to `PlasticityTrainer`; this alias will be removed in a future release"
)]
#[doc(hidden)]
pub use self::PlasticityTrainer as SpikenautTrainer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingConfig;

    fn small_network() -> SpikingNetwork {
        SpikingNetwork::with_dimensions(4, 2, 8)
    }

    fn network_snapshot(network: &SpikingNetwork) -> String {
        serde_json::to_string(network).expect("serialize network snapshot")
    }

    /// Non-default weights, traces, step counter, modulators, and EMA so a
    /// missed preflight (which would run sample 0) cannot match by accident.
    fn seed_nonzero_network_state(network: &mut SpikingNetwork) {
        network.global_step = 17;
        network.modulators.dopamine = 0.42;
        network.modulators.norepinephrine = 0.37;
        network.modulators.serotonin = 0.21;
        network.modulators.acetylcholine = 0.18;
        for (i, value) in network.predictive_state.iter_mut().enumerate() {
            *value = 0.05 * (i as f32 + 1.0);
        }
        for (i, t) in network.input_spike_times.iter_mut().enumerate() {
            *t = i as i64;
        }
        for neuron in &mut network.neurons {
            neuron.membrane_potential = 0.01;
            neuron.last_spike_time = 3;
            neuron.weights.fill(0.2);
            for trace in &mut neuron.eligibility {
                trace.value = 0.3;
            }
        }
    }

    fn valid_example() -> TrainingExample {
        TrainingExample {
            stimuli: vec![0.25; 8],
            reward: 0.2,
        }
    }

    #[test]
    fn train_step_with_reward_modulation_succeeds() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let stimuli = vec![0.2; 8];
        let spikes = trainer
            .train_step(&mut network, &stimuli, 0.5)
            .expect("positive reward step");
        let _ = spikes;
        let _ = trainer
            .train_step(&mut network, &stimuli, -0.3)
            .expect("negative reward step");
    }

    #[test]
    fn train_step_without_reward_modulation_succeeds() {
        let config = TrainingConfig {
            use_reward_modulation: false,
        };
        let mut trainer = PlasticityTrainer::new(config);
        let mut network = small_network();
        let stimuli = vec![0.2; 8];
        trainer
            .train_step(&mut network, &stimuli, 0.9)
            .expect("step with modulation disabled");
    }

    #[test]
    fn train_step_skips_nan_reward_modulation() {
        let mut network = small_network();
        network.modulators.dopamine = 0.4;
        network.modulators.norepinephrine = 0.4;
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        trainer
            .train_step(&mut network, &[0.2; 8], f32::NAN)
            .expect("nan reward must not panic");
        assert!((network.modulators.dopamine - 0.4).abs() < 1e-5);
        assert!((network.modulators.norepinephrine - 0.4).abs() < 1e-5);
    }

    #[test]
    fn positive_reward_raises_dopamine_lowers_norepinephrine() {
        let mut network = small_network();
        network.modulators.dopamine = 0.5;
        network.modulators.norepinephrine = 0.5;

        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        trainer
            .train_step(&mut network, &[0.2; 8], 1.0)
            .expect("train_step");

        assert!((network.modulators.dopamine - 0.6).abs() < 1e-5);
        assert!((network.modulators.norepinephrine - 0.45).abs() < 1e-5);
    }

    #[test]
    fn negative_reward_raises_norepinephrine_lowers_dopamine() {
        let mut network = small_network();
        network.modulators.dopamine = 0.5;
        network.modulators.norepinephrine = 0.5;

        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        trainer
            .train_step(&mut network, &[0.2; 8], -1.0)
            .expect("train_step");

        // dopamine += reward * 0.1 → 0.5 - 0.1 = 0.4
        // norepinephrine -= reward * 0.2 → 0.5 - (-0.2) = 0.7
        assert!((network.modulators.dopamine - 0.4).abs() < 1e-5);
        assert!((network.modulators.norepinephrine - 0.7).abs() < 1e-5);
    }

    #[test]
    fn train_step_with_modulators_applies_explicit_state() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let mods = NeuroModulators {
            dopamine: 0.9,
            serotonin: 0.1,
            acetylcholine: 0.5,
            norepinephrine: 0.3,
        };
        trainer
            .train_step_with_modulators(&mut network, &[0.2; 8], &mods)
            .expect("explicit modulators");
        assert!((network.modulators.dopamine - 0.9).abs() < 1e-5);
        assert!((network.modulators.norepinephrine - 0.3).abs() < 1e-5);
    }

    #[test]
    fn run_session_empty_batch_errors() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);
        let config_before = trainer.config;

        let err = trainer
            .run_session(&mut network, &[])
            .expect_err("empty batch");
        assert!(matches!(err, TrainerError::EmptyBatch));
        assert_eq!(network_snapshot(&network), before);
        assert_eq!(trainer.config, config_before);
    }

    #[test]
    fn run_session_reports_steps_and_avg_reward() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.25; 8],
                reward: 0.2,
            },
            TrainingExample {
                stimuli: vec![0.4; 8],
                reward: -0.1,
            },
        ];
        let summary = trainer.run_session(&mut network, &batch).expect("session");
        assert_eq!(summary.steps_processed, 2);
        assert!((summary.avg_reward - 0.05).abs() < 1e-5);
        assert_eq!(summary.threshold_drifts.len(), network.neurons.len());
    }

    // neuromod's `SpikingNetwork::step` only consults its thread-local RNG to
    // decide, per channel, whether to stamp an input spike time — and only when
    // `|stimulus| > 0.01` (see engine.rs). Below that magnitude, step() is a pure
    // function of network state and inputs. There is no seed hook exposed through
    // this crate (or neuromod) to make the above-threshold path reproducible, so
    // these tests establish determinism on the sub-threshold path instead.

    #[test]
    fn train_step_is_deterministic_for_subthreshold_stimuli() {
        let stimuli = vec![0.005; 8];

        let mut trainer_a = PlasticityTrainer::new(TrainingConfig::default());
        let mut network_a = small_network();
        let spikes_a = trainer_a
            .train_step(&mut network_a, &stimuli, 0.4)
            .expect("step a");

        let mut trainer_b = PlasticityTrainer::new(TrainingConfig::default());
        let mut network_b = small_network();
        let spikes_b = trainer_b
            .train_step(&mut network_b, &stimuli, 0.4)
            .expect("step b");

        assert_eq!(spikes_a, spikes_b);
        assert_eq!(network_a.get_thresholds(), network_b.get_thresholds());
        assert_eq!(network_a.modulators.dopamine, network_b.modulators.dopamine);
        assert_eq!(
            network_a.modulators.norepinephrine,
            network_b.modulators.norepinephrine
        );
    }

    #[test]
    fn run_session_is_deterministic_for_subthreshold_stimuli() {
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: 0.3,
            },
            TrainingExample {
                stimuli: vec![-0.008; 8],
                reward: -0.2,
            },
            TrainingExample {
                stimuli: vec![0.0; 8],
                reward: 0.0,
            },
        ];

        let mut trainer_a = PlasticityTrainer::new(TrainingConfig::default());
        let mut network_a = small_network();
        let summary_a = trainer_a
            .run_session(&mut network_a, &batch)
            .expect("session a");

        let mut trainer_b = PlasticityTrainer::new(TrainingConfig::default());
        let mut network_b = small_network();
        let summary_b = trainer_b
            .run_session(&mut network_b, &batch)
            .expect("session b");

        assert_eq!(summary_a.steps_processed, summary_b.steps_processed);
        assert_eq!(summary_a.total_spikes, summary_b.total_spikes);
        assert_eq!(summary_a.per_neuron_spikes, summary_b.per_neuron_spikes);
        assert_eq!(summary_a.threshold_drifts, summary_b.threshold_drifts);
        assert_eq!(summary_a.weight_drifts, summary_b.weight_drifts);
        assert_eq!(summary_a.avg_reward, summary_b.avg_reward);
    }

    #[test]
    fn run_session_summary_shapes_match_network_topology() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.2; 8],
                reward: 0.1,
            },
            TrainingExample {
                stimuli: vec![0.3; 8],
                reward: 0.2,
            },
        ];
        let summary = trainer.run_session(&mut network, &batch).expect("session");

        assert_eq!(summary.per_neuron_spikes.len(), network.neurons.len());
        assert_eq!(summary.threshold_drifts.len(), network.neurons.len());
        assert_eq!(summary.weight_drifts.len(), network.neurons.len());
        for weights in &summary.weight_drifts {
            assert_eq!(weights.len(), network.num_channels);
        }
    }

    #[test]
    fn run_session_total_spikes_matches_sum_of_per_neuron_spikes() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.5; 8],
                reward: 0.5,
            },
            TrainingExample {
                stimuli: vec![0.6; 8],
                reward: -0.4,
            },
            TrainingExample {
                stimuli: vec![0.1; 8],
                reward: 0.0,
            },
        ];
        let summary = trainer.run_session(&mut network, &batch).expect("session");

        let summed: u64 = summary.per_neuron_spikes.iter().sum();
        assert_eq!(summary.total_spikes, summed);
    }

    #[test]
    fn run_session_avg_reward_ignores_nan_but_counts_the_step() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: 0.4,
            },
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: f32::NAN,
            },
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: 0.2,
            },
        ];
        let summary = trainer.run_session(&mut network, &batch).expect("session");

        assert_eq!(summary.steps_processed, 3);
        assert!((summary.avg_reward - 0.3).abs() < 1e-5);
    }

    #[test]
    fn run_session_avg_reward_defaults_to_zero_when_all_rewards_nan() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: f32::NAN,
            },
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: f32::NAN,
            },
        ];
        let summary = trainer.run_session(&mut network, &batch).expect("session");

        assert_eq!(summary.steps_processed, 2);
        assert_eq!(summary.avg_reward, 0.0);
    }

    #[cfg(feature = "critic")]
    #[test]
    fn train_step_from_critic_uses_bridge() {
        use limbic_critic::ModulatorVector;

        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let vector = ModulatorVector {
            dopamine: 0.65,
            serotonin: 0.2,
            acetylcholine: 0.4,
            norepinephrine: 0.15,
        };
        trainer
            .train_step_from_critic(&mut network, &[0.2; 8], &vector)
            .expect("from critic");
        assert!((network.modulators.dopamine - 0.65).abs() < 1e-5);
        assert!((network.modulators.serotonin - 0.2).abs() < 1e-5);
        assert!((network.modulators.acetylcholine - 0.4).abs() < 1e-5);
        assert!((network.modulators.norepinephrine - 0.15).abs() < 1e-5);
    }

    #[test]
    #[allow(deprecated)]
    fn spikenaut_trainer_module_path_alias_still_constructs() {
        let _trainer = super::SpikenautTrainer::new(TrainingConfig::default());
    }

    #[test]
    fn train_step_still_returns_step_error_on_length_mismatch() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let err = trainer
            .train_step(&mut network, &[0.2; 3], 0.1)
            .expect_err("single-step API stays StepError, not batch preflight");
        assert!(matches!(
            err,
            StepError::InputLenMismatch {
                expected: 8,
                got: 3
            }
        ));
    }

    #[test]
    fn batch_preflight_is_atomic_when_final_sample_is_invalid() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);
        let config_before = trainer.config;
        let global_step_before = network.global_step;
        let eligibility_before: Vec<Vec<f32>> = network
            .neurons
            .iter()
            .map(|n| n.eligibility.iter().map(|t| t.value).collect())
            .collect();
        let weights_before: Vec<Vec<f32>> =
            network.neurons.iter().map(|n| n.weights.clone()).collect();

        let batch = vec![
            TrainingExample {
                stimuli: vec![0.4; 8],
                reward: 0.5,
            },
            TrainingExample {
                stimuli: vec![0.6; 8],
                reward: -0.2,
            },
            TrainingExample {
                stimuli: vec![0.3; 3],
                reward: 0.1,
            },
        ];

        let err = trainer
            .run_session(&mut network, &batch)
            .expect_err("late malformed sample must reject the batch");

        assert_eq!(
            err,
            TrainerError::InvalidSample {
                index: 2,
                reason: SampleInvariant::StimulusLenMismatch {
                    expected: 8,
                    got: 3,
                },
            }
        );
        assert_eq!(
            err.to_string(),
            "invalid training sample 2: stimulus length mismatch: expected 8, got 3"
        );

        assert_eq!(network.global_step, global_step_before);
        assert_eq!(
            network
                .neurons
                .iter()
                .map(|n| n.eligibility.iter().map(|t| t.value).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            eligibility_before
        );
        assert_eq!(
            network
                .neurons
                .iter()
                .map(|n| n.weights.clone())
                .collect::<Vec<_>>(),
            weights_before
        );
        assert_eq!(network_snapshot(&network), before);
        assert_eq!(trainer.config, config_before);
    }

    #[test]
    fn batch_preflight_reports_first_invalid_sample() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            valid_example(),
            TrainingExample {
                stimuli: vec![0.2; 5],
                reward: 0.1,
            },
            TrainingExample {
                stimuli: vec![f32::NAN; 8],
                reward: 0.1,
            },
        ];
        let err = trainer
            .run_session(&mut network, &batch)
            .expect_err("first invalid sample wins");
        assert!(matches!(
            err,
            TrainerError::InvalidSample {
                index: 1,
                reason: SampleInvariant::StimulusLenMismatch {
                    expected: 8,
                    got: 5
                },
            }
        ));
    }

    #[test]
    fn batch_preflight_rejects_non_finite_stimulus() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);

        let mut stimuli = vec![0.2; 8];
        stimuli[4] = f32::INFINITY;
        let batch = vec![
            valid_example(),
            TrainingExample {
                stimuli,
                reward: 0.1,
            },
        ];
        let err = trainer
            .run_session(&mut network, &batch)
            .expect_err("non-finite stimulus");
        assert_eq!(
            err,
            TrainerError::InvalidSample {
                index: 1,
                reason: SampleInvariant::NonFiniteStimulus { channel: 4 },
            }
        );
        assert_eq!(network_snapshot(&network), before);
    }

    #[test]
    fn batch_preflight_rejects_infinite_reward() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);

        let batch = vec![
            valid_example(),
            TrainingExample {
                stimuli: vec![0.2; 8],
                reward: f32::NEG_INFINITY,
            },
        ];
        let err = trainer
            .run_session(&mut network, &batch)
            .expect_err("infinite reward");
        assert_eq!(
            err,
            TrainerError::InvalidSample {
                index: 1,
                reason: SampleInvariant::InfiniteReward,
            }
        );
        assert_eq!(network_snapshot(&network), before);
    }

    #[test]
    fn valid_batch_preserves_sample_ordering_and_avg_reward() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: 0.4,
            },
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: -0.1,
            },
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: 0.0,
            },
        ];
        let summary = trainer
            .run_session(&mut network, &batch)
            .expect("valid batch");
        assert_eq!(summary.steps_processed, batch.len());
        assert!((summary.avg_reward - 0.1).abs() < 1e-5);
    }
}
