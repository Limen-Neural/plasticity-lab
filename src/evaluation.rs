// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::trainer::{PlasticityTrainer, TrainerError, stimulus_invariant};
use neuromod::{NeuroModulators, SpikingNetwork, StepError};
use rand::Rng;

/// One caller-owned held-out example.
///
/// Split membership and input encoding remain application responsibilities.
/// Evaluation examples intentionally carry no scalar reward.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationExample {
    /// Flat stimulus vector (length must match the network input size).
    pub stimuli: Vec<f32>,
}

/// Spike metrics collected during a plasticity-frozen evaluation session.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EvaluationSummary {
    /// Number of held-out examples processed.
    pub steps_processed: usize,
    /// Total spike events across all evaluation steps.
    pub total_spikes: u64,
    /// Spike count per LIF neuron over the evaluation session.
    pub per_neuron_spikes: Vec<u64>,
}

impl PlasticityTrainer {
    /// Advances one held-out step without retaining plasticity changes.
    ///
    /// The caller supplies the modulators used for runtime dynamics, but the
    /// network's persistent modulator state is restored before this method
    /// returns. Evaluation accepts no scalar reward.
    pub fn eval_step(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<Vec<usize>, TrainerError> {
        network
            .step_frozen(stimuli, modulators)
            .map_err(TrainerError::Step)
    }

    /// Advances one held-out step without retaining plasticity changes, using
    /// caller-supplied modulators and RNG state.
    ///
    /// This delegates to [`SpikingNetwork::step_frozen_with_rng`]. It advances
    /// runtime dynamics and exposes spikes while preserving weights,
    /// eligibility traces, thresholds, adaptive decay state, persistent
    /// modulators, and other plasticity-controlled state. This is stronger than
    /// setting [`crate::TrainingConfig::use_reward_modulation`] to `false`,
    /// which still uses normal plasticity-aware stepping.
    #[cfg_attr(test, inline(never))]
    pub fn eval_step_with_rng<R: Rng + ?Sized>(
        &mut self,
        network: &mut SpikingNetwork,
        stimuli: &[f32],
        modulators: &NeuroModulators,
        rng: &mut R,
    ) -> Result<Vec<usize>, TrainerError> {
        network
            .step_frozen_with_rng(stimuli, modulators, rng)
            .map_err(TrainerError::Step)
    }

    /// Evaluates a caller-owned held-out batch with the thread-local RNG.
    ///
    /// The complete batch is admitted before the first step, so invalid input
    /// cannot leave an earlier held-out example applied. For deterministic
    /// replay, use [`Self::run_eval_with_rng`].
    pub fn run_eval(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[EvaluationExample],
        modulators: &NeuroModulators,
    ) -> Result<EvaluationSummary, TrainerError> {
        admit_evaluation_batch(network, data)?;
        let mut summary = new_evaluation_summary(network);

        for example in data {
            let spikes = self.eval_step(network, &example.stimuli, modulators)?;
            record_evaluation_spikes(&mut summary, &spikes);
        }

        Ok(summary)
    }

    /// Evaluates a caller-owned held-out batch with one caller-owned RNG stream.
    ///
    /// Every step delegates to [`Self::eval_step_with_rng`], so runtime dynamics
    /// and spike observations advance without retaining plasticity changes.
    pub fn run_eval_with_rng<R: Rng + ?Sized>(
        &mut self,
        network: &mut SpikingNetwork,
        data: &[EvaluationExample],
        modulators: &NeuroModulators,
        rng: &mut R,
    ) -> Result<EvaluationSummary, TrainerError> {
        admit_evaluation_batch(network, data)?;
        let mut summary = new_evaluation_summary(network);

        for example in data {
            let spikes = self.eval_step_with_rng(network, &example.stimuli, modulators, rng)?;
            record_evaluation_spikes(&mut summary, &spikes);
        }

        Ok(summary)
    }
}

fn new_evaluation_summary(network: &SpikingNetwork) -> EvaluationSummary {
    EvaluationSummary {
        per_neuron_spikes: vec![0; network.neurons.len()],
        ..EvaluationSummary::default()
    }
}

fn record_evaluation_spikes(summary: &mut EvaluationSummary, spikes: &[usize]) {
    summary.steps_processed += 1;
    summary.total_spikes += spikes.len() as u64;
    for &index in spikes {
        if let Some(count) = summary.per_neuron_spikes.get_mut(index) {
            *count += 1;
        }
    }
}

fn admit_evaluation_batch(
    network: &SpikingNetwork,
    data: &[EvaluationExample],
) -> Result<(), TrainerError> {
    if data.is_empty() {
        return Err(TrainerError::EmptyBatch);
    }

    let step_count = i64::try_from(data.len()).map_err(|_| {
        TrainerError::Step(StepError::StepCounterExhausted {
            global_step: network.global_step,
        })
    })?;
    if network.global_step < 0 || network.global_step.checked_add(step_count).is_none() {
        return Err(TrainerError::Step(StepError::StepCounterExhausted {
            global_step: network.global_step,
        }));
    }

    for (index, example) in data.iter().enumerate() {
        if let Some(reason) = stimulus_invariant(network, &example.stimuli) {
            return Err(TrainerError::InvalidSample { index, reason });
        }
    }
    Ok(())
}
