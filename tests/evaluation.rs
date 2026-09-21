// SPDX-License-Identifier: MIT OR Apache-2.0

use neuromod::{NeuroModulators, SpikingNetwork, StepError};
use plasticity_lab::{
    EvaluationExample, PlasticityTrainer, SampleInvariant, TrainerError, TrainingConfig,
};
use rand::{RngExt as _, SeedableRng, rngs::StdRng};

#[derive(Debug, PartialEq, Eq)]
struct PlasticityBits {
    modulators: [u32; 4],
    stdp_config: [u32; 4],
    neurons: Vec<NeuronPlasticityBits>,
}

#[derive(Debug, PartialEq, Eq)]
struct NeuronPlasticityBits {
    decay_rate: u32,
    threshold: u32,
    base_threshold: u32,
    weights: Vec<u32>,
    eligibility: Vec<[u32; 2]>,
}

fn plasticity_bits(network: &SpikingNetwork) -> PlasticityBits {
    PlasticityBits {
        modulators: [
            network.modulators.dopamine.to_bits(),
            network.modulators.serotonin.to_bits(),
            network.modulators.acetylcholine.to_bits(),
            network.modulators.norepinephrine.to_bits(),
        ],
        stdp_config: [
            network.stdp_config.tau_eligibility.to_bits(),
            network.stdp_config.reward_lr.to_bits(),
            network.stdp_config.w_min.to_bits(),
            network.stdp_config.w_max.to_bits(),
        ],
        neurons: network
            .neurons
            .iter()
            .map(|neuron| NeuronPlasticityBits {
                decay_rate: neuron.decay_rate.to_bits(),
                threshold: neuron.threshold.to_bits(),
                base_threshold: neuron.base_threshold.to_bits(),
                weights: neuron.weights.iter().map(|value| value.to_bits()).collect(),
                eligibility: neuron
                    .eligibility
                    .iter()
                    .map(|trace| [trace.value.to_bits(), trace.tau.to_bits()])
                    .collect(),
            })
            .collect(),
    }
}

fn active_network() -> SpikingNetwork {
    let mut network = SpikingNetwork::with_dimensions(4, 2, 3);
    network.modulators = NeuroModulators {
        dopamine: 0.25,
        serotonin: 0.5,
        acetylcholine: 0.75,
        norepinephrine: 0.125,
    };
    for neuron in &mut network.neurons {
        neuron.weights.fill(2.0 / 3.0);
        neuron.decay_rate = 0.2;
        neuron.threshold = 0.02;
        neuron.base_threshold = 0.03;
        for trace in &mut neuron.eligibility {
            trace.value = 0.625;
        }
    }
    network
}

#[test]
fn seeded_eval_step_advances_dynamics_and_preserves_plasticity_bitwise() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = active_network();
    let before_plasticity = plasticity_bits(&network);
    let before_step = network.global_step;
    let before_prediction = network.predictive_state.clone();
    let evaluation_modulators = NeuroModulators {
        dopamine: 1.0,
        serotonin: 0.2,
        acetylcholine: 0.8,
        norepinephrine: 0.1,
    };
    let mut rng = StdRng::seed_from_u64(0xE7A1);

    let spikes = trainer
        .eval_step_with_rng(
            &mut network,
            &[1.0, 1.0, 1.0],
            &evaluation_modulators,
            &mut rng,
        )
        .expect("valid frozen evaluation step");

    assert_eq!(plasticity_bits(&network), before_plasticity);
    assert_eq!(network.global_step, before_step + 1);
    assert_ne!(network.predictive_state, before_prediction);
    assert!(!spikes.is_empty(), "frozen evaluation must expose spikes");
}

#[test]
fn unseeded_eval_step_uses_the_same_frozen_contract() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = active_network();
    let before_plasticity = plasticity_bits(&network);
    let before_step = network.global_step;

    let spikes = trainer
        .eval_step(&mut network, &[1.0, 1.0, 1.0], &NeuroModulators::default())
        .expect("valid frozen evaluation step");

    assert_eq!(plasticity_bits(&network), before_plasticity);
    assert_eq!(network.global_step, before_step + 1);
    assert!(!spikes.is_empty());
}

#[test]
fn seeded_eval_batch_reports_spikes_and_preserves_plasticity_across_steps() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = active_network();
    let before_plasticity = plasticity_bits(&network);
    let before_step = network.global_step;
    let held_out = vec![
        EvaluationExample {
            stimuli: vec![1.0, 0.8, 0.6],
        },
        EvaluationExample {
            stimuli: vec![0.6, 1.0, 0.8],
        },
        EvaluationExample {
            stimuli: vec![0.8, 0.6, 1.0],
        },
    ];
    let evaluation_modulators = NeuroModulators {
        dopamine: 1.0,
        serotonin: 0.2,
        acetylcholine: 0.8,
        norepinephrine: 0.1,
    };
    let mut rng = StdRng::seed_from_u64(0xE7A1);

    let summary = trainer
        .run_eval_with_rng(&mut network, &held_out, &evaluation_modulators, &mut rng)
        .expect("valid held-out evaluation");

    assert_eq!(summary.steps_processed, held_out.len());
    assert_eq!(summary.per_neuron_spikes.len(), network.neurons.len());
    assert_eq!(
        summary.total_spikes,
        summary.per_neuron_spikes.iter().sum::<u64>()
    );
    assert!(summary.total_spikes > 0);
    assert_eq!(plasticity_bits(&network), before_plasticity);
    assert_eq!(network.global_step, before_step + held_out.len() as i64);
}

#[test]
fn unseeded_eval_batch_reports_held_out_spike_metrics() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = active_network();
    let before_plasticity = plasticity_bits(&network);
    let held_out = vec![
        EvaluationExample {
            stimuli: vec![1.0, 0.8, 0.6],
        },
        EvaluationExample {
            stimuli: vec![0.6, 1.0, 0.8],
        },
    ];

    let summary = trainer
        .run_eval(&mut network, &held_out, &NeuroModulators::default())
        .expect("valid held-out evaluation");

    assert_eq!(summary.steps_processed, held_out.len());
    assert_eq!(
        summary.total_spikes,
        summary.per_neuron_spikes.iter().sum::<u64>()
    );
    assert_eq!(plasticity_bits(&network), before_plasticity);
}

#[test]
fn eval_batch_rejects_a_late_non_finite_stimulus_before_network_or_rng_mutation() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = active_network();
    let before_network = serde_json::to_string(&network).expect("network snapshot");
    let held_out = vec![
        EvaluationExample {
            stimuli: vec![1.0, 0.8, 0.6],
        },
        EvaluationExample {
            stimuli: vec![0.6, f32::NAN, 0.8],
        },
    ];
    let mut rng = StdRng::seed_from_u64(77);
    let mut untouched_rng = StdRng::seed_from_u64(77);

    let error = trainer
        .run_eval_with_rng(
            &mut network,
            &held_out,
            &NeuroModulators::default(),
            &mut rng,
        )
        .expect_err("late invalid held-out sample");

    assert_eq!(
        error,
        TrainerError::InvalidSample {
            index: 1,
            reason: SampleInvariant::NonFiniteStimulus { channel: 1 },
        }
    );
    assert_eq!(
        serde_json::to_string(&network).expect("network snapshot"),
        before_network
    );
    assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
}

#[test]
fn eval_batch_rejects_a_late_length_mismatch_atomically() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = active_network();
    let before_network = serde_json::to_string(&network).expect("network snapshot");
    let held_out = vec![
        EvaluationExample {
            stimuli: vec![1.0, 0.8, 0.6],
        },
        EvaluationExample {
            stimuli: vec![0.6, 0.8],
        },
    ];

    let error = trainer
        .run_eval(&mut network, &held_out, &NeuroModulators::default())
        .expect_err("late invalid held-out sample");

    assert_eq!(
        error,
        TrainerError::InvalidSample {
            index: 1,
            reason: SampleInvariant::StimulusLenMismatch {
                expected: 3,
                got: 2,
            },
        }
    );
    assert_eq!(
        serde_json::to_string(&network).expect("network snapshot"),
        before_network
    );
}

#[test]
fn eval_batches_reuse_the_existing_empty_batch_error() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut unseeded_network = active_network();
    let mut seeded_network = active_network();
    let before_unseeded = serde_json::to_string(&unseeded_network).expect("network snapshot");
    let before_seeded = serde_json::to_string(&seeded_network).expect("network snapshot");
    let mut rng = StdRng::seed_from_u64(5);
    let mut untouched_rng = StdRng::seed_from_u64(5);

    assert_eq!(
        trainer.run_eval(&mut unseeded_network, &[], &NeuroModulators::default()),
        Err(TrainerError::EmptyBatch)
    );
    assert_eq!(
        trainer.run_eval_with_rng(
            &mut seeded_network,
            &[],
            &NeuroModulators::default(),
            &mut rng,
        ),
        Err(TrainerError::EmptyBatch)
    );
    assert_eq!(
        serde_json::to_string(&unseeded_network).expect("network snapshot"),
        before_unseeded
    );
    assert_eq!(
        serde_json::to_string(&seeded_network).expect("network snapshot"),
        before_seeded
    );
    assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
}

#[test]
fn seeded_eval_replays_runtime_state_and_spike_summary() {
    let mut trainer_a = PlasticityTrainer::new(TrainingConfig::default());
    let mut trainer_b = PlasticityTrainer::new(TrainingConfig::default());
    let mut network_a = active_network();
    let mut network_b = active_network();
    let held_out = vec![
        EvaluationExample {
            stimuli: vec![0.9, 0.4, 0.7],
        },
        EvaluationExample {
            stimuli: vec![0.3, 0.8, 0.5],
        },
    ];
    let modulators = NeuroModulators {
        dopamine: 0.9,
        serotonin: 0.3,
        acetylcholine: 0.7,
        norepinephrine: 0.2,
    };
    let mut rng_a = StdRng::seed_from_u64(0x1297);
    let mut rng_b = StdRng::seed_from_u64(0x1297);

    let summary_a = trainer_a
        .run_eval_with_rng(&mut network_a, &held_out, &modulators, &mut rng_a)
        .expect("first replay");
    let summary_b = trainer_b
        .run_eval_with_rng(&mut network_b, &held_out, &modulators, &mut rng_b)
        .expect("second replay");

    assert_eq!(summary_a, summary_b);
    assert_eq!(
        serde_json::to_string(&network_a).expect("network snapshot"),
        serde_json::to_string(&network_b).expect("network snapshot")
    );
    assert_eq!(rng_a.random::<u64>(), rng_b.random::<u64>());
}

#[test]
fn direct_seeded_eval_wraps_engine_errors_without_consuming_rng() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = active_network();
    let before_network = serde_json::to_string(&network).expect("network snapshot");
    let mut rng = StdRng::seed_from_u64(9);
    let mut untouched_rng = StdRng::seed_from_u64(9);

    let error = trainer
        .eval_step_with_rng(
            &mut network,
            &[1.0, 0.5],
            &NeuroModulators::default(),
            &mut rng,
        )
        .expect_err("length mismatch");

    assert_eq!(
        error,
        TrainerError::Step(StepError::InputLenMismatch {
            expected: 3,
            got: 2,
        })
    );
    assert_eq!(
        serde_json::to_string(&network).expect("network snapshot"),
        before_network
    );
    assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
}
