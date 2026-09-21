// SPDX-License-Identifier: MIT OR Apache-2.0

use neuromod::SpikingNetwork;
use plasticity_lab::{PlasticityTrainer, RewardMapping, RewardMappingError, TrainingConfig};

#[test]
fn reward_mapping_types_are_available_from_the_crate_root() {
    let mapping = RewardMapping::builder()
        .dopamine_gain(0.25)
        .build()
        .expect("valid public reward mapping");

    assert_eq!(mapping.dopamine_gain(), 0.25);
    let _: Option<RewardMappingError> = None;
}

#[test]
fn core_api_steps_without_optional_dependencies() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = SpikingNetwork::with_dimensions(2, 1, 4);

    trainer
        .train_step(&mut network, &[0.0; 4], 0.25)
        .expect("core training step");

    assert_eq!(network.global_step, 1);
}

#[cfg(feature = "critic")]
#[test]
fn critic_api_accepts_limbic_critic_vectors() {
    use limbic_critic::ModulatorVector;
    use plasticity_lab::to_neuromodulators;

    let vector = ModulatorVector {
        dopamine: 0.75,
        serotonin: 0.5,
        acetylcholine: 0.25,
        norepinephrine: 0.125,
    };
    let modulators = to_neuromodulators(&vector);

    assert_eq!(modulators.dopamine, vector.dopamine);
    assert_eq!(modulators.serotonin, vector.serotonin);
    assert_eq!(modulators.acetylcholine, vector.acetylcholine);
    assert_eq!(modulators.norepinephrine, vector.norepinephrine);
}
