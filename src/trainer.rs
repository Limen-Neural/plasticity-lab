// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::config::TrainingConfig;
use crate::observer::{TrainingObserver, TrainingStepEvent};
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
    /// A per-step observer returned an error after a successful network step.
    ///
    /// The network update for `step_index` has already been applied. No further
    /// example is processed (abort-before-next-step). `steps_processed` counts
    /// completed network steps, including the one whose observer then failed.
    #[error(
        "training observer failed at step {step_index} after {steps_processed} processed step(s): {cause}"
    )]
    Observer {
        /// 0-based index of the example whose observer call failed.
        step_index: usize,
        /// Number of successful network steps before aborting (includes the
        /// failing observer's step).
        steps_processed: usize,
        /// Display form of the observer's error.
        cause: String,
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
    /// This is the no-observer compatibility path: it does not construct
    /// [`TrainingStepEvent`]s, format or serialize telemetry, or dynamically
    /// dispatch. For per-step callbacks see [`Self::run_session_with_observer`].
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
        self.run_session_generic::<false, _>(network, data, &mut ())
    }

    /// Replays a batch like [`Self::run_session`], notifying `observer` after
    /// each successful network step.
    ///
    /// Exactly one [`TrainingStepEvent`] is delivered per completed step, in
    /// batch order. The event borrows spike indices and neuromodulator state;
    /// it does not expose mutable network access.
    ///
    /// If `observer` returns an error at step `N` (0-based), the session
    /// aborts before stepping example `N + 1`. The error reports `step_index`
    /// and the number of network steps that completed
    /// ([`TrainerError::Observer`]).
    ///
    /// The observer is a generic type parameter (monomorphized, not `dyn`), so
    /// a simple callback has no dynamic dispatch on the hot path.
    ///
    /// # Errors
    ///
    /// - [`TrainerError::EmptyBatch`] if `data` is empty (observer is not called).
    /// - [`TrainerError::Step`] if a network step fails (observer is not called
    ///   for that failed step; earlier steps have already been observed).
    /// - [`TrainerError::Observer`] if `observer` returns an error.
    pub fn run_session_with_observer<O: TrainingObserver>(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
        observer: &mut O,
    ) -> Result<TrainingSummary, TrainerError> {
        self.run_session_generic::<true, O>(network, data, observer)
    }

    fn run_session_generic<const OBSERVE: bool, O: TrainingObserver>(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
        observer: &mut O,
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

        for (step_index, example) in data.iter().enumerate() {
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

            // `OBSERVE` is a const generic: the no-observer `run_session` path
            // monomorphizes with `false` and drops this block entirely (no
            // event construction, formatting, serialization, or dispatch).
            if OBSERVE {
                let event = TrainingStepEvent {
                    step_index,
                    reward: example.reward,
                    modulators: &network.modulators,
                    spike_indices: &spikes,
                    steps_processed: summary.steps_processed,
                    total_spikes: summary.total_spikes,
                };
                observer
                    .on_step(event)
                    .map_err(|cause| TrainerError::Observer {
                        step_index,
                        steps_processed: summary.steps_processed,
                        cause: cause.to_string(),
                    })?;
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
    use crate::observer::{TrainingObserver, TrainingStepEvent};

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

    #[derive(Default)]
    struct RecordingObserver {
        step_indices: Vec<usize>,
        rewards: Vec<f32>,
        spike_counts: Vec<usize>,
        running_totals: Vec<u64>,
        steps_processed: Vec<usize>,
        dopamine: Vec<f32>,
        norepinephrine: Vec<f32>,
    }

    impl TrainingObserver for RecordingObserver {
        type Error = &'static str;

        fn on_step(&mut self, event: TrainingStepEvent<'_>) -> Result<(), Self::Error> {
            self.step_indices.push(event.step_index);
            self.rewards.push(event.reward);
            self.spike_counts.push(event.spike_indices.len());
            self.running_totals.push(event.total_spikes);
            self.steps_processed.push(event.steps_processed);
            self.dopamine.push(event.modulators.dopamine);
            self.norepinephrine.push(event.modulators.norepinephrine);
            Ok(())
        }
    }

    struct FailAt {
        fail_at: usize,
        seen: Vec<usize>,
    }

    impl TrainingObserver for FailAt {
        type Error = &'static str;

        fn on_step(&mut self, event: TrainingStepEvent<'_>) -> Result<(), Self::Error> {
            self.seen.push(event.step_index);
            if event.step_index == self.fail_at {
                Err("injected observer failure")
            } else {
                Ok(())
            }
        }
    }

    fn subthreshold_batch() -> Vec<TrainingExample> {
        vec![
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
        ]
    }

    fn assert_networks_match(a: &SpikingNetwork, b: &SpikingNetwork) {
        assert_eq!(a.get_thresholds(), b.get_thresholds());
        assert_eq!(a.modulators, b.modulators);
        assert_eq!(a.global_step, b.global_step);
        assert_eq!(a.neurons.len(), b.neurons.len());
        for (na, nb) in a.neurons.iter().zip(&b.neurons) {
            assert_eq!(na.weights, nb.weights);
        }
    }

    fn assert_summaries_match(a: &TrainingSummary, b: &TrainingSummary) {
        assert_eq!(a.steps_processed, b.steps_processed);
        assert_eq!(a.total_spikes, b.total_spikes);
        assert_eq!(a.per_neuron_spikes, b.per_neuron_spikes);
        assert_eq!(a.threshold_drifts, b.threshold_drifts);
        assert_eq!(a.weight_drifts, b.weight_drifts);
        assert_eq!(a.avg_reward, b.avg_reward);
    }

    #[test]
    fn run_session_with_observer_event_count_and_order_match_summary() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = subthreshold_batch();
        let mut observer = RecordingObserver::default();

        let summary = trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect("session");

        assert_eq!(summary.steps_processed, batch.len());
        assert_eq!(observer.step_indices, vec![0, 1, 2]);
        assert_eq!(observer.steps_processed, vec![1, 2, 3]);
        assert_eq!(observer.rewards.len(), summary.steps_processed);
        assert!((observer.rewards[0] - 0.3).abs() < 1e-5);
        assert!((observer.rewards[1] + 0.2).abs() < 1e-5);
        assert_eq!(observer.rewards[2], 0.0);
    }

    #[test]
    fn run_session_with_observer_spike_totals_match_summary() {
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
        let mut observer = RecordingObserver::default();
        let summary = trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect("session");

        let captured: u64 = observer.spike_counts.iter().map(|&n| n as u64).sum();
        assert_eq!(captured, summary.total_spikes);
        assert_eq!(
            observer.running_totals.last().copied().unwrap_or(0),
            summary.total_spikes
        );
        let summed: u64 = summary.per_neuron_spikes.iter().sum();
        assert_eq!(summary.total_spikes, summed);
    }

    #[test]
    fn observer_failure_at_step_n_stops_before_n_plus_one() {
        let batch = subthreshold_batch();
        let fail_at = 1usize;

        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let mut observer = FailAt {
            fail_at,
            seen: Vec::new(),
        };
        let err = trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect_err("observer failure");

        match err {
            TrainerError::Observer {
                step_index,
                steps_processed,
                cause,
            } => {
                assert_eq!(step_index, fail_at);
                assert_eq!(steps_processed, fail_at + 1);
                assert!(cause.contains("injected observer failure"));
            }
            other => panic!("expected Observer error, got {other:?}"),
        }
        assert_eq!(observer.seen, vec![0, 1]);

        // Network state matches a session that processed only examples 0..=N,
        // not N+1.
        let mut prefix_trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut prefix_network = small_network();
        prefix_trainer
            .run_session(&mut prefix_network, &batch[..=fail_at])
            .expect("prefix session");
        assert_networks_match(&network, &prefix_network);

        let mut extra_trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut extra_network = small_network();
        extra_trainer
            .run_session(&mut extra_network, &batch[..=fail_at + 1])
            .expect("prefix plus one");
        assert_ne!(network.global_step, extra_network.global_step);
    }

    #[test]
    fn run_session_matches_noop_observer_path() {
        let batch = subthreshold_batch();

        let mut trainer_a = PlasticityTrainer::new(TrainingConfig::default());
        let mut network_a = small_network();
        let summary_a = trainer_a
            .run_session(&mut network_a, &batch)
            .expect("no observer");

        let mut trainer_b = PlasticityTrainer::new(TrainingConfig::default());
        let mut network_b = small_network();
        let mut observer = RecordingObserver::default();
        let summary_b = trainer_b
            .run_session_with_observer(&mut network_b, &batch, &mut observer)
            .expect("recording observer");

        assert_summaries_match(&summary_a, &summary_b);
        assert_networks_match(&network_a, &network_b);
        assert_eq!(observer.step_indices.len(), summary_a.steps_processed);
    }

    #[test]
    fn empty_batch_does_not_call_observer() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let mut observer = RecordingObserver::default();
        let err = trainer
            .run_session_with_observer(&mut network, &[], &mut observer)
            .expect_err("empty batch");
        assert!(matches!(err, TrainerError::EmptyBatch));
        assert!(observer.step_indices.is_empty());
    }

    #[test]
    fn failed_network_step_does_not_emit_observer_event() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: 0.1,
            },
            TrainingExample {
                stimuli: vec![0.005; 3],
                reward: 0.2,
            },
        ];
        let mut observer = RecordingObserver::default();
        let err = trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect_err("input length mismatch");
        assert!(matches!(err, TrainerError::Step(_)));
        assert_eq!(observer.step_indices, vec![0]);
    }

    #[test]
    fn observer_sees_modulators_applied_by_the_step() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        network.modulators.dopamine = 0.5;
        network.modulators.norepinephrine = 0.5;
        let batch = vec![TrainingExample {
            stimuli: vec![0.005; 8],
            reward: 1.0,
        }];
        let mut observer = RecordingObserver::default();
        trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect("session");

        assert_eq!(observer.dopamine.len(), 1);
        assert!((observer.dopamine[0] - 0.6).abs() < 1e-5);
        assert!((observer.norepinephrine[0] - 0.45).abs() < 1e-5);
        assert!((network.modulators.dopamine - observer.dopamine[0]).abs() < 1e-5);
        assert!((network.modulators.norepinephrine - observer.norepinephrine[0]).abs() < 1e-5);
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
}
