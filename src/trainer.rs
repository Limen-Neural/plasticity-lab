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

/// Errors from batch training sessions.
#[derive(Debug, Error)]
pub enum TrainerError {
    /// Underlying network step failed.
    #[error("network step failed: {0:?}")]
    Step(StepError),
    /// `run_session` was called with an empty batch.
    #[error("empty training batch")]
    EmptyBatch,
}

/// Reward-modulated training loop over a [`SpikingNetwork`].
///
/// Applies scalar rewards to neuromodulators and steps the network. Domain-specific
/// logic (mining, trading, distillation) does not belong here. For critic-shaped
/// vectors under the `integration` feature, use [`Self::train_step_from_critic`]
/// or [`crate::bridge`].
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

        // Skip modulation on NaN: f32::clamp panics on NaN in debug and yields
        // non-finite modulators that poison subsequent STDP / homeostasis updates.
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
    /// should convert via [`crate::to_neuromodulators`] (integration feature) and
    /// pass the result here.
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
    /// with the `integration` feature.
    #[cfg(feature = "integration")]
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
    /// # Errors
    ///
    /// - [`TrainerError::EmptyBatch`] if `data` is empty.
    /// - [`TrainerError::Step`] if any network step fails.
    pub fn run_session(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        if data.is_empty() {
            return Err(TrainerError::EmptyBatch);
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingConfig;

    fn small_network() -> SpikingNetwork {
        SpikingNetwork::with_dimensions(4, 2, 8)
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
        let err = trainer
            .run_session(&mut network, &[])
            .expect_err("empty batch");
        assert!(matches!(err, TrainerError::EmptyBatch));
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

    #[cfg(feature = "integration")]
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
}
