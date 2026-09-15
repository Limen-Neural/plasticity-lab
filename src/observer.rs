// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-step session observer types for [`crate::PlasticityTrainer::run_session_with_observer`].
//!
//! These types are borrowed snapshots. They never expose `&mut` access to the
//! network, and constructing a [`TrainingStepEvent`] does not heap-allocate.

use neuromod::NeuroModulators;

/// Borrowed telemetry for one successful network step inside a training session.
///
/// All references are valid only for the duration of
/// [`TrainingObserver::on_step`]. The event does not grant mutable access to
/// network state.
#[derive(Debug, Clone, Copy)]
pub struct TrainingStepEvent<'a> {
    /// 0-based index of this example in the session batch.
    pub step_index: usize,
    /// Scalar reward from the [`crate::TrainingExample`].
    pub reward: f32,
    /// Neuromodulator state in effect after the step (borrowed from the network).
    pub modulators: &'a NeuroModulators,
    /// Indices of neurons that spiked on this step.
    pub spike_indices: &'a [usize],
    /// Number of examples processed so far, including this one.
    pub steps_processed: usize,
    /// Cumulative spike count across the session, including this step.
    pub total_spikes: u64,
}

/// Receives one [`TrainingStepEvent`] after each successful network step.
///
/// Returning `Err` aborts the session before the next example is stepped. The
/// failing step's network update has already been applied.
///
/// This trait is generic (statically dispatched). There is no `dyn` call on
/// the session hot path.
pub trait TrainingObserver {
    /// Error type that aborts the session. Displayed in [`crate::TrainerError::Observer`].
    type Error: core::fmt::Display;

    /// Called once after a successful [`crate::PlasticityTrainer::train_step`].
    ///
    /// # Errors
    ///
    /// Any error aborts the session immediately; the next example is not processed.
    fn on_step(&mut self, event: TrainingStepEvent<'_>) -> Result<(), Self::Error>;
}

/// Zero-sized no-op used by [`crate::PlasticityTrainer::run_session`].
///
/// `on_step` is not invoked on that path (`observe = false`); this type
/// exists so the shared generic loop type-checks without passing `()`.
pub(crate) struct NoopObserver;

pub(crate) fn discard_step_event(
    _event: TrainingStepEvent<'_>,
) -> Result<(), core::convert::Infallible> {
    Ok(())
}

impl TrainingObserver for NoopObserver {
    type Error = core::convert::Infallible;

    fn on_step(&mut self, event: TrainingStepEvent<'_>) -> Result<(), Self::Error> {
        discard_step_event(event)
    }
}

impl<F, E> TrainingObserver for F
where
    F: FnMut(TrainingStepEvent<'_>) -> Result<(), E>,
    E: core::fmt::Display,
{
    type Error = E;

    fn on_step(&mut self, event: TrainingStepEvent<'_>) -> Result<(), Self::Error> {
        self(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuromod::NeuroModulators;

    #[test]
    fn event_is_copy_and_has_no_mutable_network_access() {
        let mods = NeuroModulators::default();
        let spikes = [0usize, 2];
        let event = TrainingStepEvent {
            step_index: 0,
            reward: 0.25,
            modulators: &mods,
            spike_indices: &spikes,
            steps_processed: 1,
            total_spikes: 2,
        };
        let copy = event;
        assert_eq!(copy.step_index, 0);
        assert!((copy.reward - 0.25).abs() < 1e-6);
        assert_eq!(copy.spike_indices, &spikes);
        assert_eq!(copy.steps_processed, 1);
        assert_eq!(copy.total_spikes, 2);
        // `&NeuroModulators` / `&[usize]` only — mutating `copy.reward` cannot
        // touch the network, and there is no `&mut` field to reach it.
        let mut local = copy;
        local.reward = 1.0;
        assert!((local.reward - 1.0).abs() < 1e-6);
        assert!((event.reward - 0.25).abs() < 1e-6);
        assert!((mods.dopamine - 0.0).abs() < 1e-6);
    }

    #[test]
    fn closure_observer_receives_event() {
        let mods = NeuroModulators::default();
        let spikes: &[usize] = &[];
        let event = TrainingStepEvent {
            step_index: 3,
            reward: -0.1,
            modulators: &mods,
            spike_indices: spikes,
            steps_processed: 4,
            total_spikes: 0,
        };
        let mut seen = None;
        let mut observer = |e: TrainingStepEvent<'_>| -> Result<(), &'static str> {
            seen = Some(e.step_index);
            Ok(())
        };
        observer.on_step(event).expect("closure observer");
        assert_eq!(seen, Some(3));
    }

    #[test]
    fn noop_observer_on_step_is_ok() {
        let mods = NeuroModulators::default();
        let spikes: &[usize] = &[];
        let event = TrainingStepEvent {
            step_index: 0,
            reward: 0.0,
            modulators: &mods,
            spike_indices: spikes,
            steps_processed: 1,
            total_spikes: 0,
        };
        NoopObserver
            .on_step(event)
            .expect("noop observer cannot fail");
    }
}
