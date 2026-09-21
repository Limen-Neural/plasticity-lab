// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::config::TrainingConfig;
use crate::observer::{TrainingObserver, TrainingStepEvent};
use neuromod::{NeuroModulators, SpikingNetwork, StepError};
use rand::Rng;
use thiserror::Error;

/// Summary metrics collected over a [`PlasticityTrainer::run_session`] call.
///
/// Drifts are relative to network state at the start of the session.
#[derive(Debug, Default, Clone, PartialEq)]
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
/// not run this structural check; scalar step APIs independently reject
/// non-finite rewards before invoking the network.
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
}

/// Errors from batch training sessions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TrainerError {
    /// Underlying network step failed.
    #[error("network step failed: {0:?}")]
    Step(StepError),
    /// A scalar reward was NaN or positive/negative infinity.
    ///
    /// Direct step calls report `index: None`; batch APIs report the rejected
    /// sample's zero-based index. Rejection happens before network or RNG state
    /// changes, even when reward modulation is disabled.
    #[error("non-finite reward{suffix}", suffix = reward_index_suffix(.index))]
    NonFiniteReward { index: Option<usize> },
    /// `run_session` was called with an empty batch.
    ///
    /// Empty is a batch-level condition (no sample index). The network, trainer
    /// config, and any caller-owned metrics are left untouched.
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

#[cfg_attr(test, inline(never))]
fn reward_index_suffix(index: &Option<usize>) -> String {
    index
        .map(|index| format!(" at sample {index}"))
        .unwrap_or_default()
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
    /// When [`TrainingConfig::use_reward_modulation`] is `true` (default) and `reward`
    /// is finite, positive values increase dopamine and decrease norepinephrine;
    /// negative values do the opposite emphasis. Modulator values are clamped to
    /// `[0.0, 1.0]`. The exact deltas come from
    /// [`TrainingConfig::reward_mapping`]. Non-finite rewards (`NaN` and
    /// ±infinity) return [`TrainerError::NonFiniteReward`] before network or RNG
    /// state changes, even when reward modulation is disabled.
    ///
    /// Returns indices of neurons that spiked, or a [`TrainerError`].
    #[cfg_attr(test, inline(never))]
    pub fn train_step(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        reward: f32,
    ) -> Result<Vec<usize>, TrainerError> {
        require_finite_reward(reward, None)?;
        let modulators = self.modulators_for_reward(network, reward);
        network
            .step(stimuli, &modulators)
            .map_err(TrainerError::Step)
    }

    /// Same as [`Self::train_step`], but drives neuromod's stochastic input
    /// encoding from a caller-supplied RNG.
    ///
    /// Use this when a session must be replayable: the same network state,
    /// config, stimuli, reward, and RNG stream produce identical spikes and
    /// plasticity updates. `train_step` keeps the convenience path that uses
    /// neuromod's thread-local RNG.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromod::SpikingNetwork;
    /// use plasticity_lab::{PlasticityTrainer, TrainingConfig};
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    /// let mut network = SpikingNetwork::with_dimensions(4, 2, 8);
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let spikes = trainer
    ///     .train_step_with_rng(&mut network, &[0.25; 8], 0.2, &mut rng)
    ///     .unwrap();
    /// assert!(spikes.iter().all(|&i| i < 4));
    /// ```
    #[cfg_attr(test, inline(never))]
    pub fn train_step_with_rng<R: Rng + ?Sized>(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        reward: f32,
        rng: &mut R,
    ) -> Result<Vec<usize>, TrainerError> {
        require_finite_reward(reward, None)?;
        let modulators = self.modulators_for_reward(network, reward);
        network
            .step_with_rng(stimuli, &modulators, rng)
            .map_err(TrainerError::Step)
    }

    /// Steps the network with explicit neuromodulators (e.g. from the limbic bridge).
    ///
    /// Does not apply scalar reward shaping; callers that already ran a critic
    /// should convert via `crate::to_neuromodulators` (`critic` feature; a
    /// plain code span, not a doc link — that item doesn't exist without the
    /// feature) and pass the result here.
    #[cfg_attr(test, inline(never))]
    pub fn train_step_with_modulators(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<Vec<usize>, TrainerError> {
        network
            .step(stimuli, modulators)
            .map_err(TrainerError::Step)
    }

    /// Same as [`Self::train_step_with_modulators`], with a caller-supplied RNG.
    #[cfg_attr(test, inline(never))]
    pub fn train_step_with_modulators_and_rng<R: Rng + ?Sized>(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        modulators: &NeuroModulators,
        rng: &mut R,
    ) -> Result<Vec<usize>, TrainerError> {
        network
            .step_with_rng(stimuli, modulators, rng)
            .map_err(TrainerError::Step)
    }

    /// Steps the network with a critic [`limbic_critic::ModulatorVector`].
    ///
    /// Converts via [`crate::bridge::to_neuromodulators`] then steps. Available only
    /// with the `critic` feature.
    #[cfg(feature = "critic")]
    #[cfg_attr(test, inline(never))]
    pub fn train_step_from_critic(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        vector: &limbic_critic::ModulatorVector,
    ) -> Result<Vec<usize>, TrainerError> {
        self.train_step_with_modulators(
            network,
            stimuli,
            &crate::bridge::to_neuromodulators(vector),
        )
    }

    /// Replays a batch of generic training examples and returns aggregated metrics.
    ///
    /// Admission is atomic: every example is validated (dimensions, finite
    /// stimuli, finite reward) *before* the first `train_step`. A malformed
    /// sample at index `N` therefore cannot leave samples `0..N` applied.
    /// Examples are then processed in slice order, matching the historical
    /// sequential contract.
    ///
    /// This is the no-observer compatibility path: it does not construct
    /// [`TrainingStepEvent`]s, format or serialize telemetry, or dynamically
    /// dispatch. For per-step callbacks see [`Self::run_session_with_observer`].
    ///
    /// # Errors
    ///
    /// - [`TrainerError::EmptyBatch`] if `data` is empty (no sample index).
    /// - [`TrainerError::NonFiniteReward`] if a reward is NaN or infinite.
    /// - [`TrainerError::InvalidSample`] if any example fails preflight; the
    ///   error names the first failing index and invariant. Network and trainer
    ///   state are unchanged.
    /// - [`TrainerError::Step`] if a network step fails after admission (for
    ///   example a `StepError` that cannot be seen from the example alone).
    #[cfg_attr(test, inline(never))]
    pub fn run_session(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        let mut session = start_session(network, data)?;
        let mut total_reward = 0.0;
        let mut valid_reward_count = 0;

        for example in data {
            let spikes = self.train_step(network, &example.stimuli, example.reward)?;
            Self::accumulate_step(
                &mut session.summary,
                &mut total_reward,
                &mut valid_reward_count,
                example,
                &spikes,
            );
        }

        Ok(Self::finalize_summary(
            session.summary,
            network,
            &session.initial_thresholds,
            &session.initial_weights,
            total_reward,
            valid_reward_count,
        ))
    }

    /// Replays a batch using a caller-supplied RNG for every network step.
    ///
    /// Identical to [`Self::run_session`] except stochastic input spikes are
    /// drawn from `rng` instead of neuromod's thread-local generator. One RNG
    /// stream is used for the whole batch — it is not reseeded per example.
    /// A starting seed replays from the beginning; a mid-session resume needs
    /// that same generator already advanced through the prefix, not a fresh
    /// seed on a deserialized checkpoint.
    ///
    /// # Errors
    ///
    /// - [`TrainerError::EmptyBatch`] if `data` is empty.
    /// - [`TrainerError::NonFiniteReward`] if a reward is NaN or infinite.
    /// - [`TrainerError::InvalidSample`] if any stimulus has the wrong length
    ///   or contains a non-finite value.
    /// - [`TrainerError::Step`] if any network step fails.
    #[cfg_attr(test, inline(never))]
    pub fn run_session_with_rng<R: Rng + ?Sized>(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
        rng: &mut R,
    ) -> Result<TrainingSummary, TrainerError> {
        let mut session = start_session(network, data)?;
        let mut total_reward = 0.0;
        let mut valid_reward_count = 0;

        for example in data {
            let spikes =
                self.train_step_with_rng(network, &example.stimuli, example.reward, rng)?;
            Self::accumulate_step(
                &mut session.summary,
                &mut total_reward,
                &mut valid_reward_count,
                example,
                &spikes,
            );
        }

        Ok(Self::finalize_summary(
            session.summary,
            network,
            &session.initial_thresholds,
            &session.initial_weights,
            total_reward,
            valid_reward_count,
        ))
    }

    /// Computes the modulator vector `train_step` would pass into `network.step`.
    #[cfg_attr(test, inline(never))]
    fn modulators_for_reward(&self, network: &SpikingNetwork, reward: f32) -> NeuroModulators {
        let mut modulators: NeuroModulators = network.modulators;

        if self.config.use_reward_modulation {
            let mapping = self.config.reward_mapping;
            // Positive reward shifts toward dopamine; negative toward norepinephrine
            // (stress/arousal). neuromod replaced the former cortisol field with
            // norepinephrine (see neuromod::NeuroModulators).
            if reward > 0.0 {
                modulators.dopamine =
                    (modulators.dopamine + reward * mapping.dopamine_gain()).clamp(0.0, 1.0);
                modulators.norepinephrine = (modulators.norepinephrine
                    - reward * mapping.positive_norepinephrine_suppression())
                .clamp(0.0, 1.0);
            } else {
                modulators.norepinephrine = (modulators.norepinephrine
                    - reward * mapping.negative_norepinephrine_gain())
                .clamp(0.0, 1.0);
                modulators.dopamine =
                    (modulators.dopamine + reward * mapping.dopamine_gain()).clamp(0.0, 1.0);
            }
        }

        modulators
    }

    pub(crate) fn accumulate_step(
        summary: &mut TrainingSummary,
        total_reward: &mut f32,
        valid_reward_count: &mut u32,
        example: &TrainingExample,
        spikes: &[usize],
    ) {
        if example.reward.is_finite() {
            *total_reward += example.reward;
            *valid_reward_count += 1;
        }
        summary.steps_processed += 1;

        summary.total_spikes += spikes.len() as u64;
        for &idx in spikes {
            if idx < summary.per_neuron_spikes.len() {
                summary.per_neuron_spikes[idx] += 1;
            }
        }
    }

    pub(crate) fn finalize_summary(
        mut summary: TrainingSummary,
        network: &SpikingNetwork,
        initial_thresholds: &[f32],
        initial_weights: &[Vec<f32>],
        total_reward: f32,
        valid_reward_count: u32,
    ) -> TrainingSummary {
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

        summary
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
    /// a simple callback has no dynamic dispatch on the hot path. The
    /// no-observer [`Self::run_session`] path does not construct events.
    ///
    /// # Errors
    ///
    /// - [`TrainerError::EmptyBatch`] if `data` is empty (observer is not called).
    /// - [`TrainerError::NonFiniteReward`] or [`TrainerError::InvalidSample`] if
    ///   preflight rejects any example (observer is not called).
    /// - [`TrainerError::Step`] if a network step fails (observer is not called
    ///   for that failed step; earlier steps have already been observed).
    /// - [`TrainerError::Observer`] if `observer` returns an error.
    #[cfg_attr(test, inline(never))]
    pub fn run_session_with_observer<O: TrainingObserver>(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[TrainingExample],
        observer: &mut O,
    ) -> Result<TrainingSummary, TrainerError> {
        let mut session = start_session(network, data)?;
        let mut total_reward = 0.0;
        let mut valid_reward_count = 0;

        for (step_index, example) in data.iter().enumerate() {
            let spikes = self.train_step(network, &example.stimuli, example.reward)?;
            accumulate_reward(example, &mut total_reward, &mut valid_reward_count);
            session.summary.steps_processed += 1;
            record_step_spikes(&mut session.summary, &spikes);
            observer
                .on_step(step_event(
                    step_index,
                    example.reward,
                    &network.modulators,
                    &spikes,
                    session.summary.steps_processed,
                    session.summary.total_spikes,
                ))
                .map_err(|cause| {
                    observer_abort(
                        step_index,
                        session.summary.steps_processed,
                        cause.to_string(),
                    )
                })?;
        }

        finish_summary(
            &mut session.summary,
            network,
            &session.initial_thresholds,
            &session.initial_weights,
            total_reward,
            valid_reward_count,
        );
        Ok(session.summary)
    }
}

struct SessionPrep {
    summary: TrainingSummary,
    initial_thresholds: Vec<f32>,
    initial_weights: Vec<Vec<f32>>,
}

fn start_session(
    network: &SpikingNetwork,
    data: &[TrainingExample],
) -> Result<SessionPrep, TrainerError> {
    admit_batch(network, data)?;
    let mut summary = TrainingSummary::default();
    let initial_thresholds = network.get_thresholds();
    let initial_weights: Vec<Vec<f32>> =
        network.neurons.iter().map(|n| n.weights.clone()).collect();
    summary.per_neuron_spikes = vec![0; network.neurons.len()];
    Ok(SessionPrep {
        summary,
        initial_thresholds,
        initial_weights,
    })
}

fn accumulate_reward(
    example: &TrainingExample,
    total_reward: &mut f32,
    valid_reward_count: &mut usize,
) {
    if example.reward.is_finite() {
        *total_reward += example.reward;
        *valid_reward_count += 1;
    }
}

fn record_step_spikes(summary: &mut TrainingSummary, spikes: &[usize]) {
    summary.total_spikes += spikes.len() as u64;
    for &idx in spikes {
        if let Some(count) = summary.per_neuron_spikes.get_mut(idx) {
            *count += 1;
        }
    }
}

fn step_event<'a>(
    step_index: usize,
    reward: f32,
    modulators: &'a NeuroModulators,
    spike_indices: &'a [usize],
    steps_processed: usize,
    total_spikes: u64,
) -> TrainingStepEvent<'a> {
    TrainingStepEvent {
        step_index,
        reward,
        modulators,
        spike_indices,
        steps_processed,
        total_spikes,
    }
}

#[inline(never)]
fn observer_abort(step_index: usize, steps_processed: usize, cause: String) -> TrainerError {
    TrainerError::Observer {
        step_index,
        steps_processed,
        cause,
    }
}

fn finish_summary(
    summary: &mut TrainingSummary,
    network: &SpikingNetwork,
    initial_thresholds: &[f32],
    initial_weights: &[Vec<f32>],
    total_reward: f32,
    valid_reward_count: usize,
) {
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
        require_finite_reward(example.reward, Some(index))?;
        if let Some(reason) = sample_invariant(network, example) {
            return Err(TrainerError::InvalidSample { index, reason });
        }
    }
    Ok(())
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
    None
}

#[cfg_attr(test, inline(never))]
fn require_finite_reward(reward: f32, index: Option<usize>) -> Result<(), TrainerError> {
    if reward.is_finite() {
        Ok(())
    } else {
        Err(TrainerError::NonFiniteReward { index })
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
    use crate::config::{RewardMapping, TrainingConfig};
    use crate::observer::{TrainingObserver, TrainingStepEvent};
    use rand::{RngExt as _, SeedableRng, rngs::StdRng};

    fn small_network() -> SpikingNetwork {
        SpikingNetwork::with_dimensions(4, 2, 8)
    }

    fn documented_learning_network() -> SpikingNetwork {
        let mut network = SpikingNetwork::with_dimensions(4, 2, 8);
        for neuron in &mut network.neurons {
            neuron.weights.fill(2.0 / network.num_channels as f32);
        }
        network
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
        example(8, 0.25, 0.2)
    }

    fn example(stimuli_len: usize, fill: f32, reward: f32) -> TrainingExample {
        TrainingExample {
            stimuli: vec![fill; stimuli_len],
            reward,
        }
    }

    fn eligibility_values(network: &SpikingNetwork) -> Vec<Vec<f32>> {
        network
            .neurons
            .iter()
            .map(|n| n.eligibility.iter().map(|t| t.value).collect())
            .collect()
    }

    fn weight_values(network: &SpikingNetwork) -> Vec<Vec<f32>> {
        network.neurons.iter().map(|n| n.weights.clone()).collect()
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
            ..TrainingConfig::default()
        };
        let mut trainer = PlasticityTrainer::new(config);
        let mut network = small_network();
        let stimuli = vec![0.2; 8];
        trainer
            .train_step(&mut network, &stimuli, 0.9)
            .expect("step with modulation disabled");
    }

    #[test]
    fn train_step_rejects_every_non_finite_reward_before_mutation() {
        for reward in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut network = small_network();
            seed_nonzero_network_state(&mut network);
            let before = network_snapshot(&network);
            let mut trainer = PlasticityTrainer::new(TrainingConfig::default());

            let err = trainer
                .train_step(&mut network, &[0.2; 8], reward)
                .expect_err("non-finite reward");

            assert_eq!(err, TrainerError::NonFiniteReward { index: None });
            assert_eq!(network_snapshot(&network), before);
        }
    }

    #[test]
    fn non_finite_reward_display_names_batch_index_when_available() {
        assert_eq!(
            TrainerError::NonFiniteReward { index: None }.to_string(),
            "non-finite reward"
        );
        assert_eq!(
            TrainerError::NonFiniteReward { index: Some(7) }.to_string(),
            "non-finite reward at sample 7"
        );
    }

    #[test]
    fn seeded_train_step_rejects_non_finite_reward_before_advancing_rng() {
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut rng = StdRng::seed_from_u64(42);
        let mut untouched_rng = StdRng::seed_from_u64(42);

        let err = trainer
            .train_step_with_rng(&mut network, &[0.2; 8], f32::NAN, &mut rng)
            .expect_err("non-finite reward");

        assert_eq!(err, TrainerError::NonFiniteReward { index: None });
        assert_eq!(network_snapshot(&network), before);
        assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
    }

    #[test]
    fn reward_modulation_disabled_still_rejects_non_finite_reward() {
        let config = TrainingConfig {
            use_reward_modulation: false,
            ..TrainingConfig::default()
        };
        let mut trainer = PlasticityTrainer::new(config);
        let mut network = small_network();
        let before = network_snapshot(&network);

        let err = trainer
            .train_step(&mut network, &[0.2; 8], f32::INFINITY)
            .expect_err("invalid environment input is independent of modulation policy");

        assert_eq!(err, TrainerError::NonFiniteReward { index: None });
        assert_eq!(network_snapshot(&network), before);
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
    fn custom_reward_mapping_controls_each_scalar_modulator_delta() {
        let mapping = RewardMapping::builder()
            .dopamine_gain(0.3)
            .positive_norepinephrine_suppression(0.4)
            .negative_norepinephrine_gain(0.5)
            .build()
            .expect("valid mapping");
        let config = TrainingConfig::default().with_reward_mapping(mapping);

        let mut positive_network = small_network();
        positive_network.modulators.dopamine = 0.5;
        positive_network.modulators.norepinephrine = 0.5;
        PlasticityTrainer::new(config)
            .train_step(&mut positive_network, &[0.005; 8], 0.5)
            .expect("positive reward");
        assert!((positive_network.modulators.dopamine - 0.65).abs() < 1e-5);
        assert!((positive_network.modulators.norepinephrine - 0.3).abs() < 1e-5);

        let mut negative_network = small_network();
        negative_network.modulators.dopamine = 0.5;
        negative_network.modulators.norepinephrine = 0.5;
        PlasticityTrainer::new(config)
            .train_step(&mut negative_network, &[0.005; 8], -0.5)
            .expect("negative reward");
        assert!((negative_network.modulators.dopamine - 0.35).abs() < 1e-5);
        assert!((negative_network.modulators.norepinephrine - 0.75).abs() < 1e-5);
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

    #[test]
    fn documented_nonzero_initialization_spikes_and_changes_weights() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = documented_learning_network();
        let batch = vec![
            TrainingExample {
                stimuli: vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02],
                reward: 1.0,
            };
            8
        ];
        let mut rng = StdRng::seed_from_u64(0x5EED);

        let summary = trainer
            .run_session_with_rng(&mut network, &batch, &mut rng)
            .expect("documented learning session");

        assert!(
            summary.total_spikes > 0,
            "the documented network must spike"
        );
        assert!(
            summary
                .weight_drifts
                .iter()
                .flatten()
                .any(|delta| delta.abs() > 1e-5),
            "the documented network must show a measurable weight change"
        );
        assert!(
            network
                .neurons
                .iter()
                .flat_map(|neuron| neuron.weights.iter())
                .all(|weight| weight.is_finite())
        );
    }

    // neuromod's `SpikingNetwork::step` only consults RNG to decide, per channel,
    // whether to stamp an input spike time — and only when `|stimulus| > 0.01`
    // (see engine.rs). Below that magnitude, step() is a pure function of network
    // state and inputs. These tests stay on that non-RNG path as a deterministic
    // baseline. Above-threshold replay with a caller-injected RNG is covered in
    // `crate::replay`.

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
    fn run_session_rejects_nan_in_late_sample_atomically() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);
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
        let err = trainer
            .run_session(&mut network, &batch)
            .expect_err("late non-finite reward");

        assert_eq!(err, TrainerError::NonFiniteReward { index: Some(1) });
        assert_eq!(network_snapshot(&network), before);
    }

    #[test]
    fn observer_session_rejects_non_finite_reward_before_any_callback() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: 0.2,
            },
            TrainingExample {
                stimuli: vec![0.005; 8],
                reward: f32::NAN,
            },
        ];
        let mut observer = RecordingObserver::default();
        let err = trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect_err("observer session preflight");

        assert_eq!(err, TrainerError::NonFiniteReward { index: Some(1) });
        assert!(observer.step_indices.is_empty());
        assert_eq!(network_snapshot(&network), before);
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
    fn observer_failure_at_step_n_reports_index_and_does_not_see_n_plus_one() {
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

        match &err {
            TrainerError::Observer {
                step_index,
                steps_processed,
                cause,
            } => {
                assert_eq!(*step_index, fail_at);
                assert_eq!(*steps_processed, fail_at + 1);
                assert!(cause.contains("injected observer failure"));
            }
            other => panic!("expected Observer error, got {other:?}"),
        }
        let displayed = err.to_string();
        assert!(displayed.contains("step 1"));
        assert!(displayed.contains("injected observer failure"));
        assert_eq!(observer.seen, vec![0, 1]);
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
        trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect_err("observer failure");

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
    fn run_session_with_closure_observer_records_step_indices() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let batch = subthreshold_batch();
        let mut seen = Vec::new();
        let mut observer = |event: TrainingStepEvent<'_>| -> Result<(), &'static str> {
            seen.push(event.step_index);
            Ok(())
        };
        let summary = trainer
            .run_session_with_observer(&mut network, &batch, &mut observer)
            .expect("closure observer session");
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(summary.steps_processed, 3);
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
    fn failed_preflight_does_not_emit_observer_event() {
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
        assert!(matches!(
            err,
            TrainerError::InvalidSample {
                index: 1,
                reason: SampleInvariant::StimulusLenMismatch {
                    expected: 8,
                    got: 3,
                }
            }
        ));
        assert!(observer.step_indices.is_empty());
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

    #[test]
    fn record_step_spikes_ignores_out_of_range_indices() {
        let mut summary = TrainingSummary {
            per_neuron_spikes: vec![0, 0],
            ..TrainingSummary::default()
        };
        record_step_spikes(&mut summary, &[0, 99, 1]);
        assert_eq!(summary.total_spikes, 3);
        assert_eq!(summary.per_neuron_spikes, vec![1, 1]);
    }

    #[test]
    fn observer_abort_preserves_step_index_and_cause() {
        let err = observer_abort(2, 3, "boom".to_string());
        match &err {
            TrainerError::Observer {
                step_index,
                steps_processed,
                cause,
            } => {
                assert_eq!(*step_index, 2);
                assert_eq!(*steps_processed, 3);
                assert_eq!(cause, "boom");
            }
            other => panic!("expected Observer error, got {other:?}"),
        }
        assert!(err.to_string().contains("step 2"));
    }

    #[test]
    fn step_event_copies_fields() {
        let mods = NeuroModulators::default();
        let spikes = [1usize];
        let event = step_event(4, 0.5, &mods, &spikes, 5, 7);
        assert_eq!(event.step_index, 4);
        assert!((event.reward - 0.5).abs() < 1e-6);
        assert_eq!(event.spike_indices, &spikes);
        assert_eq!(event.steps_processed, 5);
        assert_eq!(event.total_spikes, 7);
    }

    #[test]
    fn train_step_wraps_step_error_on_length_mismatch() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let err = trainer
            .train_step(&mut network, &[0.2; 3], 0.1)
            .expect_err("single-step error is wrapped consistently");
        assert!(matches!(
            err,
            TrainerError::Step(StepError::InputLenMismatch {
                expected: 8,
                got: 3
            })
        ));
    }

    #[test]
    fn explicit_modulator_step_wraps_step_error_on_length_mismatch() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let err = trainer
            .train_step_with_modulators(&mut network, &[0.2; 3], &NeuroModulators::default())
            .expect_err("explicit-modulator error is wrapped consistently");

        assert!(matches!(
            err,
            TrainerError::Step(StepError::InputLenMismatch {
                expected: 8,
                got: 3
            })
        ));
    }

    #[test]
    fn seeded_explicit_modulator_step_wraps_step_error_on_length_mismatch() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let mut rng = StdRng::seed_from_u64(3);
        let err = trainer
            .train_step_with_modulators_and_rng(
                &mut network,
                &[0.2; 3],
                &NeuroModulators::default(),
                &mut rng,
            )
            .expect_err("seeded explicit-modulator error is wrapped consistently");

        assert!(matches!(
            err,
            TrainerError::Step(StepError::InputLenMismatch {
                expected: 8,
                got: 3
            })
        ));
    }

    #[cfg(feature = "critic")]
    #[test]
    fn critic_step_wraps_step_error_on_length_mismatch() {
        use limbic_critic::ModulatorVector;

        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        let err = trainer
            .train_step_from_critic(&mut network, &[0.2; 3], &ModulatorVector::default())
            .expect_err("critic error is wrapped consistently");

        assert!(matches!(
            err,
            TrainerError::Step(StepError::InputLenMismatch {
                expected: 8,
                got: 3
            })
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
        let eligibility_before = eligibility_values(&network);
        let weights_before = weight_values(&network);

        let batch = vec![
            example(8, 0.4, 0.5),
            example(8, 0.6, -0.2),
            example(3, 0.3, 0.1),
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
        assert_eq!(eligibility_values(&network), eligibility_before);
        assert_eq!(weight_values(&network), weights_before);
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
    fn batch_preflight_rejects_every_non_finite_reward_with_index() {
        for reward in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
            let mut network = small_network();
            seed_nonzero_network_state(&mut network);
            let before = network_snapshot(&network);

            let batch = vec![
                valid_example(),
                TrainingExample {
                    stimuli: vec![0.2; 8],
                    reward,
                },
            ];
            let err = trainer
                .run_session(&mut network, &batch)
                .expect_err("non-finite reward");
            assert_eq!(err, TrainerError::NonFiniteReward { index: Some(1) });
            assert_eq!(network_snapshot(&network), before);
        }
    }

    #[test]
    fn seeded_batch_rejects_non_finite_reward_before_advancing_rng() {
        let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
        let mut network = small_network();
        seed_nonzero_network_state(&mut network);
        let before = network_snapshot(&network);
        let batch = vec![valid_example(), example(8, 0.2, f32::NEG_INFINITY)];
        let mut rng = StdRng::seed_from_u64(91);
        let mut untouched_rng = StdRng::seed_from_u64(91);

        let err = trainer
            .run_session_with_rng(&mut network, &batch, &mut rng)
            .expect_err("preflight rejects before the first random draw");

        assert_eq!(err, TrainerError::NonFiniteReward { index: Some(1) });
        assert_eq!(network_snapshot(&network), before);
        assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
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
