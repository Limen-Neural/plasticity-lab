// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Configuration knobs for [`crate::PlasticityTrainer`].
///
/// Every field here drives an explicit runtime code path in
/// [`crate::trainer::PlasticityTrainer::train_step`] — see that method's rustdoc
/// for exactly how each field is consumed. `TrainingConfig` intentionally does
/// not expose knobs without a consumer (learning rate, homeostasis setpoints,
/// and batch size were removed for this reason; see `CHANGELOG.md`). Low-level
/// STDP / homeostasis tuning lives in `neuromod::SpikingNetwork`, which derives
/// its own learning rate and thresholds from the neuromodulator state passed
/// into `step`.
///
/// Values are serializable (serde) so they can be stored with checkpoints or
/// experiment configs. Missing fields deserialize via [`Default`]
/// (`#[serde(default)]` on the struct), and unknown fields (for example from an
/// older config that still carries a since-removed knob) are ignored rather
/// than rejected, since the struct does not use `deny_unknown_fields`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// When `true` (default), `train_step` adjusts neuromodulators from the reward.
    /// When `false`, the network steps with its current modulators unchanged.
    pub use_reward_modulation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            use_reward_modulation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TrainingConfig;

    #[test]
    fn default_enables_reward_modulation() {
        assert!(TrainingConfig::default().use_reward_modulation);
    }

    #[test]
    fn missing_fields_deserialize_to_default() {
        let cfg: TrainingConfig =
            serde_json::from_str("{}").expect("deserialize empty config object");
        assert_eq!(cfg, TrainingConfig::default());
    }

    #[test]
    fn serialization_round_trip_preserves_false() {
        let cfg = TrainingConfig {
            use_reward_modulation: false,
        };
        let json = serde_json::to_string(&cfg).expect("serialize config");
        let round_tripped: TrainingConfig =
            serde_json::from_str(&json).expect("deserialize config");
        assert_eq!(cfg, round_tripped);
    }

    #[test]
    fn serialization_round_trip_preserves_default() {
        let cfg = TrainingConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize default config");
        let round_tripped: TrainingConfig =
            serde_json::from_str(&json).expect("deserialize default config");
        assert_eq!(cfg, round_tripped);
    }

    #[test]
    fn stale_fields_from_pre_0_2_configs_are_ignored_not_rejected() {
        // Configs serialized before the fields were removed (see CHANGELOG.md)
        // may still carry `learning_rate`, `target_spikes_per_step`,
        // `homeostasis_strength`, and `batch_size` in stored checkpoints or
        // experiment configs. None of those fields were ever read by the
        // trainer; deserialization must keep ignoring them rather than error.
        let cfg: TrainingConfig = serde_json::from_str(
            r#"{
                "learning_rate": 0.02,
                "target_spikes_per_step": 0.1,
                "homeostasis_strength": 0.001,
                "batch_size": 4,
                "use_reward_modulation": false
            }"#,
        )
        .expect("deserialize config carrying removed fields");
        assert!(!cfg.use_reward_modulation);
    }
}
