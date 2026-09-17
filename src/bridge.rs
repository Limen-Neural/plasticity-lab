// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bridge between `limbic-critic` modulator vectors and `neuromod` types.
//!
//! After `limbic-critic` dropped its direct `neuromod` dependency, critics emit
//! [`limbic_critic::ModulatorVector`]. This module owns the conversion into
//! [`neuromod::NeuroModulators`] so training loops in this crate can drive
//! `neuromod`'s reward-modulated STDP without reimplementing it.
//!
//! Enabled only with the `critic` feature (optional `limbic-critic` dep).

use limbic_critic::ModulatorVector;
use neuromod::{NeuroModulators, SpikingNetwork, StepError};

/// Convert a critic [`ModulatorVector`] into neuromod [`NeuroModulators`].
///
/// Field mapping is 1:1 for the supported registry releases:
/// `dopamine`, `serotonin`, `acetylcholine`, `norepinephrine`.
#[inline]
pub fn to_neuromodulators(v: &ModulatorVector) -> NeuroModulators {
    NeuroModulators {
        dopamine: v.dopamine,
        serotonin: v.serotonin,
        acetylcholine: v.acetylcholine,
        norepinephrine: v.norepinephrine,
    }
}

/// Convert neuromod [`NeuroModulators`] back to a critic [`ModulatorVector`].
///
/// Useful for round-trip tests and for feeding neuromod state into critic-side code.
#[inline]
pub fn from_neuromodulators(m: &NeuroModulators) -> ModulatorVector {
    ModulatorVector {
        dopamine: m.dopamine,
        serotonin: m.serotonin,
        acetylcholine: m.acetylcholine,
        norepinephrine: m.norepinephrine,
    }
}

/// Step a network using critic modulators converted via [`to_neuromodulators`].
///
/// Convenience for orchestration layers that hold a `ModulatorVector` from
/// `SimpleCritic` / `TDCritic` and need a single call into `neuromod`.
#[inline]
pub fn apply_modulator_vector(
    network: &mut SpikingNetwork,
    stimuli: &[f32],
    vector: &ModulatorVector,
) -> Result<Vec<usize>, StepError> {
    network.step(stimuli, &to_neuromodulators(vector))
}

#[cfg(test)]
mod tests {
    use super::*;
    use limbic_critic::{Environment, SimpleCritic, TDCritic};

    #[test]
    fn mapping_is_one_to_one() {
        let v = ModulatorVector {
            dopamine: 0.7,
            serotonin: 0.3,
            acetylcholine: 0.55,
            norepinephrine: 0.2,
        };
        let m = to_neuromodulators(&v);
        assert_eq!(m.dopamine, 0.7);
        assert_eq!(m.serotonin, 0.3);
        assert_eq!(m.acetylcholine, 0.55);
        assert_eq!(m.norepinephrine, 0.2);
    }

    #[test]
    fn roundtrip_preserves_fields() {
        let original = ModulatorVector {
            dopamine: 0.11,
            serotonin: 0.22,
            acetylcholine: 0.33,
            norepinephrine: 0.44,
        };
        let roundtrip = from_neuromodulators(&to_neuromodulators(&original));
        assert_eq!(roundtrip, original);
    }

    #[test]
    fn defaults_map_to_defaults() {
        let m = to_neuromodulators(&ModulatorVector::default());
        assert_eq!(m, NeuroModulators::default());
        let v = from_neuromodulators(&NeuroModulators::default());
        assert_eq!(v, ModulatorVector::default());
    }

    struct ConstEnv {
        objective: f32,
        stress: f32,
        volatility: f32,
        surprise: f32,
    }

    impl Environment for ConstEnv {
        fn objective(&self) -> f32 {
            self.objective
        }
        fn stress(&self) -> f32 {
            self.stress
        }
        fn volatility(&self) -> f32 {
            self.volatility
        }
        fn surprise(&self) -> f32 {
            self.surprise
        }
    }

    #[test]
    fn simple_critic_assess_maps_into_neuromodulators() {
        let env = ConstEnv {
            objective: 0.8,
            stress: 0.4,
            volatility: 0.25,
            surprise: 0.6,
        };
        let vector = SimpleCritic::assess(&env);
        let mods = to_neuromodulators(&vector);

        assert_eq!(mods.dopamine, 0.8);
        assert_eq!(mods.norepinephrine, 0.4);
        assert_eq!(mods.serotonin, 0.25);
        assert_eq!(mods.acetylcholine, 0.6);
    }

    #[test]
    /// Verifies that a time-difference critic's output maps to neuromodulators.
    fn td_critic_assess_maps_into_neuromodulators() {
        let env = ConstEnv {
            objective: 0.5,
            stress: 0.1,
            volatility: 0.0,
            surprise: 0.0,
        };
        let mut critic = TDCritic::new(0.1).expect("finite alpha in (0, 1]");
        let vector = critic.assess(&env);
        let mods = to_neuromodulators(&vector);

        // First TD call: td_error = 0.5, ema = 0.05, dopamine = 0.05.tanh()
        assert!((mods.dopamine - 0.05f32.tanh()).abs() < 1e-6);
        assert!((mods.acetylcholine - 0.5f32.tanh()).abs() < 1e-6);
        assert_eq!(mods.norepinephrine, 0.1);
        assert_eq!(mods.serotonin, 0.0);
    }

    #[test]
    fn apply_modulator_vector_steps_network() {
        let mut network = SpikingNetwork::with_dimensions(4, 2, 8);
        let stimuli = vec![0.25; 8];
        let vector = ModulatorVector {
            dopamine: 0.5,
            serotonin: 0.1,
            acetylcholine: 0.4,
            norepinephrine: 0.2,
        };
        let spikes = apply_modulator_vector(&mut network, &stimuli, &vector)
            .expect("step with bridged modulators");
        let _ = spikes;
        assert!((network.modulators.dopamine - 0.5).abs() < 1e-6);
        assert!((network.modulators.norepinephrine - 0.2).abs() < 1e-6);
    }
}
