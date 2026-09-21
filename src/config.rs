// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use thiserror::Error;

const DEFAULT_DOPAMINE_GAIN: f32 = 0.1;
const DEFAULT_POSITIVE_NOREPINEPHRINE_SUPPRESSION: f32 = 0.05;
const DEFAULT_NEGATIVE_NOREPINEPHRINE_GAIN: f32 = 0.2;

/// Validated conversion from a caller-defined scalar reward into modulator deltas.
///
/// Rewards are dimensionless and unbounded. Each coefficient is a finite,
/// non-negative modulator delta per unit reward; the resulting dopamine and
/// norepinephrine values are clamped to `[0.0, 1.0]`. Scalar mapping does not
/// modify serotonin or acetylcholine and is not a learned critic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RewardMapping {
    dopamine_gain: f32,
    positive_norepinephrine_suppression: f32,
    negative_norepinephrine_gain: f32,
}

impl RewardMapping {
    /// Starts a fluent builder with the compatibility defaults.
    pub fn builder() -> RewardMappingBuilder {
        RewardMappingBuilder::default()
    }

    /// Dopamine delta per unit positive or negative scalar reward.
    pub fn dopamine_gain(self) -> f32 {
        self.dopamine_gain
    }

    /// Norepinephrine suppression per unit positive reward.
    pub fn positive_norepinephrine_suppression(self) -> f32 {
        self.positive_norepinephrine_suppression
    }

    /// Norepinephrine gain per unit negative reward magnitude.
    pub fn negative_norepinephrine_gain(self) -> f32 {
        self.negative_norepinephrine_gain
    }
}

impl Default for RewardMapping {
    fn default() -> Self {
        Self {
            dopamine_gain: DEFAULT_DOPAMINE_GAIN,
            positive_norepinephrine_suppression: DEFAULT_POSITIVE_NOREPINEPHRINE_SUPPRESSION,
            negative_norepinephrine_gain: DEFAULT_NEGATIVE_NOREPINEPHRINE_GAIN,
        }
    }
}

/// Fluent builder for a validated [`RewardMapping`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardMappingBuilder {
    dopamine_gain: f32,
    positive_norepinephrine_suppression: f32,
    negative_norepinephrine_gain: f32,
}

impl RewardMappingBuilder {
    /// Sets dopamine delta per unit scalar reward.
    pub fn dopamine_gain(mut self, value: f32) -> Self {
        self.dopamine_gain = value;
        self
    }

    /// Sets norepinephrine suppression per unit positive reward.
    pub fn positive_norepinephrine_suppression(mut self, value: f32) -> Self {
        self.positive_norepinephrine_suppression = value;
        self
    }

    /// Sets norepinephrine gain per unit negative reward magnitude.
    pub fn negative_norepinephrine_gain(mut self, value: f32) -> Self {
        self.negative_norepinephrine_gain = value;
        self
    }

    /// Validates and builds the mapping.
    pub fn build(self) -> Result<RewardMapping, RewardMappingError> {
        validate_coefficient("dopamine_gain", self.dopamine_gain)?;
        validate_coefficient(
            "positive_norepinephrine_suppression",
            self.positive_norepinephrine_suppression,
        )?;
        validate_coefficient(
            "negative_norepinephrine_gain",
            self.negative_norepinephrine_gain,
        )?;
        Ok(RewardMapping {
            dopamine_gain: self.dopamine_gain,
            positive_norepinephrine_suppression: self.positive_norepinephrine_suppression,
            negative_norepinephrine_gain: self.negative_norepinephrine_gain,
        })
    }
}

impl Default for RewardMappingBuilder {
    fn default() -> Self {
        let mapping = RewardMapping::default();
        Self {
            dopamine_gain: mapping.dopamine_gain,
            positive_norepinephrine_suppression: mapping.positive_norepinephrine_suppression,
            negative_norepinephrine_gain: mapping.negative_norepinephrine_gain,
        }
    }
}

/// Validation error returned by [`RewardMappingBuilder::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RewardMappingError {
    /// A coefficient was NaN or infinite.
    #[error("reward-mapping coefficient `{name}` must be finite")]
    NonFiniteCoefficient { name: &'static str },
    /// A coefficient was negative.
    #[error("reward-mapping coefficient `{name}` must be non-negative")]
    NegativeCoefficient { name: &'static str },
}

fn validate_coefficient(name: &'static str, value: f32) -> Result<(), RewardMappingError> {
    if !value.is_finite() {
        return Err(RewardMappingError::NonFiniteCoefficient { name });
    }
    if value < 0.0 {
        return Err(RewardMappingError::NegativeCoefficient { name });
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(default, rename = "RewardMapping")]
struct RewardMappingData {
    dopamine_gain: f32,
    positive_norepinephrine_suppression: f32,
    negative_norepinephrine_gain: f32,
}

impl Default for RewardMappingData {
    fn default() -> Self {
        Self {
            dopamine_gain: DEFAULT_DOPAMINE_GAIN,
            positive_norepinephrine_suppression: DEFAULT_POSITIVE_NOREPINEPHRINE_SUPPRESSION,
            negative_norepinephrine_gain: DEFAULT_NEGATIVE_NOREPINEPHRINE_GAIN,
        }
    }
}

impl<'de> Deserialize<'de> for RewardMapping {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = RewardMappingData::deserialize(deserializer)?;
        RewardMappingBuilder {
            dopamine_gain: data.dopamine_gain,
            positive_norepinephrine_suppression: data.positive_norepinephrine_suppression,
            negative_norepinephrine_gain: data.negative_norepinephrine_gain,
        }
        .build()
        .map_err(D::Error::custom)
    }
}

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
/// Values are serializable (serde) so callers can persist them as part of their
/// own experiment configs or checkpointing setup — this crate does not implement
/// checkpointing itself. Missing fields deserialize via [`Default`]
/// (`#[serde(default)]` on the struct), and unknown fields (for example from an
/// older config that still carries a since-removed knob) are ignored rather
/// than rejected, since the struct does not use `deny_unknown_fields`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// When `true` (default), `train_step` adjusts neuromodulators from the reward.
    /// When `false`, the network steps with its current modulators unchanged.
    pub use_reward_modulation: bool,
    /// Validated scalar-reward conversion policy.
    pub reward_mapping: RewardMapping,
}

impl TrainingConfig {
    /// Returns a copy configured with `reward_mapping`.
    pub fn with_reward_mapping(mut self, reward_mapping: RewardMapping) -> Self {
        self.reward_mapping = reward_mapping;
        self
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            use_reward_modulation: true,
            reward_mapping: RewardMapping::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RewardMapping, RewardMappingError, TrainingConfig};
    use serde_test::{Token, assert_tokens};

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
            ..TrainingConfig::default()
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
    fn reward_mapping_serde_representation_uses_f32_coefficients_symmetrically() {
        assert_tokens(
            &RewardMapping::default(),
            &[
                Token::Struct {
                    name: "RewardMapping",
                    len: 3,
                },
                Token::Str("dopamine_gain"),
                Token::F32(0.1),
                Token::Str("positive_norepinephrine_suppression"),
                Token::F32(0.05),
                Token::Str("negative_norepinephrine_gain"),
                Token::F32(0.2),
                Token::StructEnd,
            ],
        );
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

    #[test]
    fn reward_mapping_defaults_preserve_existing_scalar_behavior() {
        let mapping = RewardMapping::default();
        assert_eq!(mapping.dopamine_gain(), 0.1);
        assert_eq!(mapping.positive_norepinephrine_suppression(), 0.05);
        assert_eq!(mapping.negative_norepinephrine_gain(), 0.2);
    }

    #[test]
    fn reward_mapping_builder_sets_each_validated_coefficient() {
        let mapping = RewardMapping::builder()
            .dopamine_gain(0.3)
            .positive_norepinephrine_suppression(0.4)
            .negative_norepinephrine_gain(0.5)
            .build()
            .expect("finite non-negative coefficients");

        assert_eq!(mapping.dopamine_gain(), 0.3);
        assert_eq!(mapping.positive_norepinephrine_suppression(), 0.4);
        assert_eq!(mapping.negative_norepinephrine_gain(), 0.5);
    }

    #[test]
    fn reward_mapping_builder_rejects_non_finite_and_negative_coefficients() {
        assert_eq!(
            RewardMapping::builder().dopamine_gain(f32::NAN).build(),
            Err(RewardMappingError::NonFiniteCoefficient {
                name: "dopamine_gain",
            })
        );
        assert_eq!(
            RewardMapping::builder()
                .negative_norepinephrine_gain(-0.1)
                .build(),
            Err(RewardMappingError::NegativeCoefficient {
                name: "negative_norepinephrine_gain",
            })
        );
    }

    #[test]
    fn training_config_builder_owns_the_validated_mapping() {
        let mapping = RewardMapping::builder()
            .dopamine_gain(0.25)
            .build()
            .expect("valid mapping");
        let config = TrainingConfig::default().with_reward_mapping(mapping);
        assert_eq!(config.reward_mapping, mapping);
    }

    #[test]
    fn old_and_partial_json_default_missing_reward_mapping_fields() {
        let old: TrainingConfig = serde_json::from_str(r#"{"use_reward_modulation":false}"#)
            .expect("old JSON without reward_mapping");
        assert_eq!(old.reward_mapping, RewardMapping::default());

        let partial: TrainingConfig =
            serde_json::from_str(r#"{"reward_mapping":{"dopamine_gain":0.3}}"#)
                .expect("partial reward mapping");
        assert_eq!(partial.reward_mapping.dopamine_gain(), 0.3);
        assert_eq!(
            partial.reward_mapping.positive_norepinephrine_suppression(),
            0.05
        );
        assert_eq!(partial.reward_mapping.negative_norepinephrine_gain(), 0.2);
    }

    #[test]
    fn deserialization_cannot_bypass_reward_mapping_validation() {
        let err =
            serde_json::from_str::<TrainingConfig>(r#"{"reward_mapping":{"dopamine_gain":-0.1}}"#)
                .expect_err("negative mapping coefficient must be rejected");
        assert!(err.to_string().contains("dopamine_gain"));

        let err = serde_json::from_str::<TrainingConfig>(
            r#"{"reward_mapping":{"negative_norepinephrine_gain":3.5e38}}"#,
        )
        .expect_err("non-finite mapping coefficient must be rejected");
        assert!(err.to_string().contains("negative_norepinephrine_gain"));
    }
}
