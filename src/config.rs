// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Configuration knobs for [`crate::SpikenautTrainer`].
///
/// Values are serializable (serde) so callers can persist them as part of their
/// own experiment configs or checkpointing setup; this crate does not implement
/// checkpointing itself. Defaults match a light reward-modulated loop. Field
/// docs appear under **Fields** in rustdoc.
///
/// Missing fields deserialize via [`Default`] (`#[serde(default)]` on the struct).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Step size for plasticity-related updates.
    pub learning_rate: f32,
    /// Target average spikes per step (homeostasis setpoint).
    pub target_spikes_per_step: f32,
    /// Strength of homeostatic pull toward [`Self::target_spikes_per_step`].
    pub homeostasis_strength: f32,
    /// Intended batch size for session / orchestration layers.
    pub batch_size: usize,
    /// When `true` (default), `train_step` adjusts neuromodulators from the reward.
    /// When `false`, the network steps with its current modulators unchanged.
    pub use_reward_modulation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            target_spikes_per_step: 0.1,
            homeostasis_strength: 0.001,
            batch_size: 1,
            use_reward_modulation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TrainingConfig;

    #[test]
    fn missing_use_reward_modulation_deserializes_to_default_true() {
        let cfg: TrainingConfig = serde_json::from_str(
            r#"{
                "learning_rate": 0.02,
                "target_spikes_per_step": 0.1,
                "homeostasis_strength": 0.001,
                "batch_size": 1
            }"#,
        )
        .expect("deserialize without use_reward_modulation");
        assert!(cfg.use_reward_modulation);
        assert!((cfg.learning_rate - 0.02).abs() < f32::EPSILON);
    }
}
