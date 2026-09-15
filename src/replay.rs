// SPDX-License-Identifier: MIT OR Apache-2.0

//! Above-threshold replay tests for seeded neuromod dynamics.
//!
//! Sub-threshold determinism (no RNG) stays in `trainer::tests`. This module
//! consumes neuromod's caller-injected RNG so real training sessions — stimuli
//! with `|s| > 0.01` — have a reproducible replay contract.

use neuromod::SpikingNetwork;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::config::TrainingConfig;
use crate::trainer::{PlasticityTrainer, TrainingExample, TrainingSummary};

/// Stochastic input encoding is skipped at or below this magnitude (neuromod
/// `SpikingNetwork::step` contract).
const STOCHASTIC_STIMULUS_THRESHOLD: f32 = 0.01;

const SEED_A: u64 = 0xA11CE5EED_u64;
const SEED_B: u64 = 0xB0B5EED11_u64;

/// Serializable experiment record: seed plus the versions a replay depends on.
///
/// This is an example of what an application can persist next to a checkpoint.
/// The library does not ingest replay files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ExperimentManifest {
    seed: u64,
    plasticity_lab_version: String,
    neuromod: DependencyRef,
    training: TrainingConfig,
    network: NetworkSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DependencyRef {
    git: String,
    branch: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct NetworkSpec {
    num_lif: usize,
    num_izh: usize,
    num_channels: usize,
}

#[derive(Debug, Clone)]
struct TickTrace {
    tick: usize,
    spikes: Vec<usize>,
    thresholds: Vec<f32>,
    weights: Vec<Vec<f32>>,
    eligibility: Vec<Vec<f32>>,
    modulators: [f32; 4],
    global_step: i64,
    input_spike_times: Vec<i64>,
}

struct SessionOutcome {
    network: SpikingNetwork,
    summary: TrainingSummary,
    ticks: Vec<TickTrace>,
}

fn example_manifest(seed: u64, spec: NetworkSpec, training: TrainingConfig) -> ExperimentManifest {
    ExperimentManifest {
        seed,
        plasticity_lab_version: env!("CARGO_PKG_VERSION").to_string(),
        neuromod: DependencyRef {
            git: "https://github.com/Limen-Neural/neuromod".to_string(),
            branch: "main".to_string(),
        },
        training,
        network: spec,
    }
}

fn mixed_reward_batch(channels: usize) -> Vec<TrainingExample> {
    let above = |value: f32| -> Vec<f32> {
        assert!(
            value.abs() > STOCHASTIC_STIMULUS_THRESHOLD,
            "replay fixtures must sit above the stochastic threshold"
        );
        vec![value; channels]
    };
    vec![
        TrainingExample {
            stimuli: above(0.35),
            reward: 0.8,
        },
        TrainingExample {
            stimuli: above(0.55),
            reward: -0.45,
        },
        TrainingExample {
            stimuli: above(0.25),
            reward: 0.0,
        },
        TrainingExample {
            stimuli: above(0.80),
            reward: 0.15,
        },
        TrainingExample {
            stimuli: above(0.42),
            reward: -0.9,
        },
        TrainingExample {
            stimuli: above(0.18),
            reward: 0.0,
        },
    ]
}

fn seeded_network(num_lif: usize, num_izh: usize, num_channels: usize) -> SpikingNetwork {
    let mut network = SpikingNetwork::with_dimensions(num_lif, num_izh, num_channels);
    for (n_idx, neuron) in network.neurons.iter_mut().enumerate() {
        for (ch, weight) in neuron.weights.iter_mut().enumerate() {
            *weight = 0.04 + 0.01 * (n_idx as f32) + 0.005 * (ch as f32);
        }
    }
    network
}

fn snapshot_network(network: &SpikingNetwork) -> SpikingNetwork {
    let json = serde_json::to_string(network).expect("SpikingNetwork must serialize");
    serde_json::from_str(&json).expect("checkpoint must deserialize")
}

fn capture_tick(tick: usize, spikes: &[usize], network: &SpikingNetwork) -> TickTrace {
    TickTrace {
        tick,
        spikes: spikes.to_vec(),
        thresholds: network.get_thresholds(),
        weights: network.neurons.iter().map(|n| n.weights.clone()).collect(),
        eligibility: network
            .neurons
            .iter()
            .map(|n| n.eligibility.iter().map(|t| t.value).collect())
            .collect(),
        modulators: [
            network.modulators.dopamine,
            network.modulators.serotonin,
            network.modulators.acetylcholine,
            network.modulators.norepinephrine,
        ],
        global_step: network.global_step,
        input_spike_times: network.input_spike_times.clone(),
    }
}

fn first_diverging_field(left: &TickTrace, right: &TickTrace) -> Option<&'static str> {
    if left.tick != right.tick {
        return Some("tick");
    }
    if left.global_step != right.global_step {
        return Some("global_step");
    }
    if left.spikes != right.spikes {
        return Some("spikes");
    }
    if left.input_spike_times != right.input_spike_times {
        return Some("input_spike_times");
    }
    if left.thresholds != right.thresholds {
        return Some("thresholds");
    }
    if left.weights != right.weights {
        return Some("weights");
    }
    if left.eligibility != right.eligibility {
        return Some("eligibility");
    }
    if left.modulators != right.modulators {
        return Some("modulators");
    }
    None
}

fn assert_ticks_identical(left: &[TickTrace], right: &[TickTrace]) {
    assert_eq!(
        left.len(),
        right.len(),
        "session length diverged: {} vs {} ticks",
        left.len(),
        right.len()
    );
    for (a, b) in left.iter().zip(right) {
        if let Some(field) = first_diverging_field(a, b) {
            panic!(
                "divergence at tick {} (global_step={}), field {field}: {a:?} vs {b:?}",
                a.tick, a.global_step
            );
        }
    }
}

fn assert_summaries_identical(left: &TrainingSummary, right: &TrainingSummary) {
    if left != right {
        let field = if left.steps_processed != right.steps_processed {
            "steps_processed"
        } else if left.total_spikes != right.total_spikes {
            "total_spikes"
        } else if left.avg_reward != right.avg_reward {
            "avg_reward"
        } else if left.threshold_drifts != right.threshold_drifts {
            "threshold_drifts"
        } else if left.weight_drifts != right.weight_drifts {
            "weight_drifts"
        } else if left.per_neuron_spikes != right.per_neuron_spikes {
            "per_neuron_spikes"
        } else {
            "TrainingSummary"
        };
        panic!("divergence after session, field {field}: {left:?} vs {right:?}");
    }
}

fn run_recorded_session(
    spec: NetworkSpec,
    config: TrainingConfig,
    batch: &[TrainingExample],
    seed: u64,
) -> SessionOutcome {
    let mut trainer = PlasticityTrainer::new(config);
    let mut network = seeded_network(spec.num_lif, spec.num_izh, spec.num_channels);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ticks = Vec::with_capacity(batch.len());

    let initial_thresholds = network.get_thresholds();
    let initial_weights: Vec<Vec<f32>> =
        network.neurons.iter().map(|n| n.weights.clone()).collect();
    let mut summary = TrainingSummary {
        per_neuron_spikes: vec![0; network.neurons.len()],
        ..TrainingSummary::default()
    };
    let mut total_reward = 0.0;
    let mut valid_reward_count = 0;

    for (tick, example) in batch.iter().enumerate() {
        let spikes = trainer
            .train_step_with_rng(&mut network, &example.stimuli, example.reward, &mut rng)
            .expect("seeded step");
        ticks.push(capture_tick(tick, &spikes, &network));
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

    SessionOutcome {
        network,
        summary,
        ticks,
    }
}

fn assert_replay_matches(spec: NetworkSpec, config: TrainingConfig, seed: u64) {
    let batch = mixed_reward_batch(spec.num_channels);
    for example in &batch {
        assert!(
            example
                .stimuli
                .iter()
                .all(|s| s.abs() > STOCHASTIC_STIMULUS_THRESHOLD),
            "every fixture stimulus must exceed {STOCHASTIC_STIMULUS_THRESHOLD}"
        );
    }

    let left = run_recorded_session(spec, config, &batch, seed);
    let right = run_recorded_session(spec, config, &batch, seed);
    assert_ticks_identical(&left.ticks, &right.ticks);
    assert_summaries_identical(&left.summary, &right.summary);
    assert_eq!(left.network.global_step, right.network.global_step);

    let mut trainer = PlasticityTrainer::new(config);
    let mut via_session = seeded_network(spec.num_lif, spec.num_izh, spec.num_channels);
    let mut session_rng = StdRng::seed_from_u64(seed);
    let session_summary = trainer
        .run_session_with_rng(&mut via_session, &batch, &mut session_rng)
        .expect("run_session_with_rng");
    assert_summaries_identical(&left.summary, &session_summary);
}

#[test]
fn experiment_manifest_round_trips_seed_and_dependency_versions() {
    let spec = NetworkSpec {
        num_lif: 4,
        num_izh: 2,
        num_channels: 8,
    };
    let manifest = example_manifest(SEED_A, spec, TrainingConfig::default());
    let json = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    let restored: ExperimentManifest = serde_json::from_str(&json).expect("deserialize manifest");
    assert_eq!(manifest, restored);
    assert_eq!(restored.seed, SEED_A);
    assert_eq!(restored.plasticity_lab_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(
        restored.neuromod.git,
        "https://github.com/Limen-Neural/neuromod"
    );
    assert!(json.contains("\"seed\""));
    assert!(json.contains("plasticity_lab_version"));
}

#[test]
fn same_seed_replays_small_network_with_mixed_rewards() {
    assert_replay_matches(
        NetworkSpec {
            num_lif: 4,
            num_izh: 2,
            num_channels: 8,
        },
        TrainingConfig::default(),
        SEED_A,
    );
}

#[test]
fn same_seed_replays_larger_network_with_mixed_rewards() {
    assert_replay_matches(
        NetworkSpec {
            num_lif: 16,
            num_izh: 5,
            num_channels: 16,
        },
        TrainingConfig::default(),
        SEED_A,
    );
}

#[test]
fn same_seed_replays_when_reward_modulation_is_disabled() {
    assert_replay_matches(
        NetworkSpec {
            num_lif: 4,
            num_izh: 2,
            num_channels: 8,
        },
        TrainingConfig {
            use_reward_modulation: false,
        },
        SEED_A,
    );
}

#[test]
fn different_seed_changes_at_least_one_input_spike_trace() {
    let spec = NetworkSpec {
        num_lif: 8,
        num_izh: 3,
        num_channels: 16,
    };
    let batch = mixed_reward_batch(spec.num_channels);
    let left = run_recorded_session(spec, TrainingConfig::default(), &batch, SEED_A);
    let right = run_recorded_session(spec, TrainingConfig::default(), &batch, SEED_B);

    let diverged = left
        .ticks
        .iter()
        .zip(&right.ticks)
        .find_map(|(a, b)| first_diverging_field(a, b).map(|field| (a.tick, field)));
    assert!(
        diverged.is_some(),
        "seeds {SEED_A:#x} and {SEED_B:#x} produced identical stochastic traces"
    );

    let input_trace_changed = left
        .ticks
        .iter()
        .zip(&right.ticks)
        .any(|(a, b)| a.input_spike_times != b.input_spike_times);
    assert!(
        input_trace_changed,
        "different seeds must alter at least one generated input-spike trace; first other divergence was {diverged:?}"
    );
}

#[test]
fn checkpoint_resume_continues_the_seeded_sequence() {
    let spec = NetworkSpec {
        num_lif: 4,
        num_izh: 2,
        num_channels: 8,
    };
    let batch = mixed_reward_batch(spec.num_channels);
    let split = 3;
    assert!(split < batch.len());

    // neuromod does not serialize RNG state (LIM-1221). Resume is the network
    // checkpoint plus a generator advanced through the same prefix.
    let prefix = run_recorded_session(spec, TrainingConfig::default(), &batch[..split], SEED_A);
    let full = run_recorded_session(spec, TrainingConfig::default(), &batch, SEED_A);
    let mut resumed_network = snapshot_network(&prefix.network);
    let mut resumed_rng = StdRng::seed_from_u64(SEED_A);
    let mut catch_up = seeded_network(spec.num_lif, spec.num_izh, spec.num_channels);
    let mut catch_up_trainer = PlasticityTrainer::new(TrainingConfig::default());
    for example in &batch[..split] {
        catch_up_trainer
            .train_step_with_rng(
                &mut catch_up,
                &example.stimuli,
                example.reward,
                &mut resumed_rng,
            )
            .expect("catch up rng");
    }
    assert_eq!(catch_up.global_step, prefix.network.global_step);
    assert_eq!(
        catch_up.input_spike_times, prefix.network.input_spike_times,
        "prefix replay must restore the same input-spike trace before resume"
    );

    let mut resumed_trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut resumed_tail = Vec::new();
    for (offset, example) in batch[split..].iter().enumerate() {
        let tick = split + offset;
        let spikes = resumed_trainer
            .train_step_with_rng(
                &mut resumed_network,
                &example.stimuli,
                example.reward,
                &mut resumed_rng,
            )
            .expect("resumed tail");
        resumed_tail.push(capture_tick(tick, &spikes, &resumed_network));
    }

    assert_ticks_identical(&full.ticks[split..], &resumed_tail);
}

#[test]
fn train_step_with_modulators_and_rng_is_deterministic() {
    let mods = neuromod::NeuroModulators {
        dopamine: 0.7,
        serotonin: 0.2,
        acetylcholine: 0.4,
        norepinephrine: 0.3,
    };
    let stimuli = [0.4f32; 8];

    let mut trainer_a = PlasticityTrainer::new(TrainingConfig::default());
    let mut network_a = seeded_network(4, 2, 8);
    let mut rng_a = StdRng::seed_from_u64(SEED_A);
    let spikes_a = trainer_a
        .train_step_with_modulators_and_rng(&mut network_a, &stimuli, &mods, &mut rng_a)
        .expect("step a");

    let mut trainer_b = PlasticityTrainer::new(TrainingConfig::default());
    let mut network_b = seeded_network(4, 2, 8);
    let mut rng_b = StdRng::seed_from_u64(SEED_A);
    let spikes_b = trainer_b
        .train_step_with_modulators_and_rng(&mut network_b, &stimuli, &mods, &mut rng_b)
        .expect("step b");

    let left = capture_tick(0, &spikes_a, &network_a);
    let right = capture_tick(0, &spikes_b, &network_b);
    if let Some(field) = first_diverging_field(&left, &right) {
        panic!(
            "divergence at tick 0 (global_step={}), field {field}",
            left.global_step
        );
    }
}

#[test]
fn run_session_with_rng_empty_batch_errors() {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = seeded_network(4, 2, 8);
    let mut rng = StdRng::seed_from_u64(SEED_A);
    let err = trainer
        .run_session_with_rng(&mut network, &[], &mut rng)
        .expect_err("empty batch");
    assert!(matches!(err, crate::trainer::TrainerError::EmptyBatch));
}
