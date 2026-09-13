# plasticity-lab

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Generic reward-modulated plasticity loops for spiking neural networks.

## Table of contents

- [Overview](#overview)
- [Ecosystem overview](#ecosystem-overview)
- [Getting started](#getting-started)
- [Choosing features](#choosing-features)
- [Common patterns](#common-patterns)
- [Architecture brief](#architecture-brief)
- [Scope and ownership boundaries](#scope-and-ownership-boundaries)
- [Cross-language notes](#cross-language-notes)
- [Contributing](#contributing)
- [License / REUSE](#license--reuse)

## Overview

`plasticity-lab` provides a small, reusable training loop around [`neuromod::SpikingNetwork`](https://github.com/Limen-Neural/neuromod).
It is intentionally domain-agnostic:

- Input encoding belongs to [`axon-encoder`](https://github.com/Limen-Neural/axon-encoder)
- Reward shaping belongs to [`limbic-critic`](https://github.com/Limen-Neural/limbic-critic)
- This crate runs the loop and tracks training summaries

If you are new to the Limen-Neural stack, start with [Getting started](#getting-started), then skim [Ecosystem overview](#ecosystem-overview) and [Scope and ownership boundaries](#scope-and-ownership-boundaries) so you know which crate owns which piece.

## Ecosystem overview

| Crate | Role | Language | When to use it |
|-------|------|----------|----------------|
| **plasticity-lab** (this crate) | Training loops + plasticity rules (`train_step`, `run_session`) | Rust | You need a reward-modulated SNN training loop and session metrics |
| [neuromod](https://github.com/Limen-Neural/neuromod) | Core SNN + neuromodulator types (`SpikingNetwork`, `NeuroModulators`) | Rust | You need the network, step dynamics, or modulator state |
| [limbic-critic](https://github.com/Limen-Neural/limbic-critic) | Reward shaping | Rust | You need shaped / multi-signal rewards instead of raw scalars |
| [axon-encoder](https://github.com/Limen-Neural/axon-encoder) | Input encoding | Rust | You need to turn raw features into spike stimuli |
| [SynapticDistill.jl](https://github.com/Limen-Neural/SynapticDistill.jl) | Distillation / knowledge transfer | **Julia only** | Teacher–student or differentiable distillation — not STDP |

Typical Rust data path:

```text
raw inputs
  → axon-encoder (optional, feature = "integration")
  → plasticity-lab::train_step / run_session
  → neuromod::SpikingNetwork
  ← limbic-critic reward (optional, feature = "integration")
```

See also the ownership boundary with [SynapticDistill.jl](#boundary-with-synapticdistilljl-linear-lim-25) below.

## Getting started

### Prerequisites

- Rust 1.97.1 toolchain ([rustup](https://rustup.rs/)) — pinned in `rust-toolchain.toml`
- A `Cargo.toml` that can pull git dependencies from GitHub
- Optional: a VS Code Dev Container setup is included under `.devcontainer/`

### 1. Add the dependency

```toml
[dependencies]
plasticity-lab = { git = "https://github.com/Limen-Neural/plasticity-lab" }
neuromod = { git = "https://github.com/Limen-Neural/neuromod" }
```

Pin `rev` values to match this crate’s `Cargo.toml` if you need a locked ecosystem build.

### 2. Minimal reward-modulated session

```rust
use neuromod::SpikingNetwork;
use plasticity_lab::{SpikenautTrainer, TrainingConfig, TrainingExample};

fn main() {
    let mut trainer = SpikenautTrainer::new(TrainingConfig::default());
    let mut network = SpikingNetwork::with_dimensions(32, 8, 64);

    let batch = vec![
        TrainingExample {
            stimuli: vec![0.25; 64],
            reward: 0.2,
        },
        TrainingExample {
            stimuli: vec![0.4; 64],
            reward: -0.1,
        },
    ];

    let summary = trainer.run_session(&mut network, &batch).unwrap();
    println!(
        "processed={}, avg_reward={}, total_spikes={}",
        summary.steps_processed, summary.avg_reward, summary.total_spikes
    );
}
```

### 3. What you get back

`run_session` returns a [`TrainingSummary`](#architecture-brief) with step counts, average reward, spike totals, and threshold/weight drift relative to the session start.

For a single network step with an external reward, call `train_step` directly (see [Architecture brief](#architecture-brief)).

### 4. Optional integrations

To pull in `limbic-critic` and `axon-encoder` as optional deps, enable the `integration` feature — see [Choosing features](#choosing-features).

## Choosing features

| Feature | Default? | What it enables |
|---------|----------|-----------------|
| *(none)* / default | yes | Core loop only: depends on `neuromod` + serde/tracing/rand/thiserror |
| `integration` | no | Optional deps: `limbic-critic` (rewards) and `axon-encoder` (encoding) |

```toml
# Core only (recommended first step)
plasticity-lab = { git = "https://github.com/Limen-Neural/plasticity-lab" }

# With sister-crate integration deps
plasticity-lab = { git = "https://github.com/Limen-Neural/plasticity-lab", features = ["integration"] }
```

**When to use default:** you already shape rewards and encode inputs yourself (or use plain `f32` stimuli and scalar rewards, as in the getting-started example).

**When to enable `integration`:** you want Cargo to resolve `limbic-critic` and `axon-encoder` alongside this crate for a full encoding → train → reward pipeline. The core trainer API does not require the feature; it always takes precomputed `stimuli: &[f32]` and `reward: f32`.

## Common patterns

### Basic reward-modulated session

Use `TrainingExample` batches and `run_session` when you have a fixed list of stimuli/reward pairs (the [Getting started](#getting-started) example).

### Single-step control

Drive the network yourself when rewards are online or adaptive:

```rust
use neuromod::{SpikingNetwork, StepError};
use plasticity_lab::{SpikenautTrainer, TrainingConfig};

fn main() -> Result<(), StepError> {
    let mut trainer = SpikenautTrainer::new(TrainingConfig::default());
    let mut network = SpikingNetwork::with_dimensions(32, 8, 64);

    let stimuli = vec![0.3; 64];
    let reward = 0.15; // from your environment or limbic-critic
    let spikes = trainer.train_step(&mut network, &stimuli, reward)?;
    println!("spikes this step: {:?}", spikes);
    Ok(())
}
```

### Custom rewards (with or without limbic-critic)

`plasticity-lab` never computes rewards. Pass any `f32`:

- Positive → dopamine up / norepinephrine down (clamped)
- Negative → norepinephrine up / dopamine adjusted (clamped)

Shape rewards in application code or via [`limbic-critic`](https://github.com/Limen-Neural/limbic-critic) when using the `integration` feature.

### Custom input encoding (with or without axon-encoder)

`train_step` / `TrainingExample.stimuli` expect a flat `&[f32]` (or `Vec<f32>`) matching the network’s input size. Encode with your own code or [`axon-encoder`](https://github.com/Limen-Neural/axon-encoder).

### Configuring the trainer

```rust
use plasticity_lab::TrainingConfig;

let config = TrainingConfig {
    learning_rate: 0.01,
    target_spikes_per_step: 0.1,
    homeostasis_strength: 0.001,
    batch_size: 1,
    use_reward_modulation: true,
};
```

`TrainingConfig::default()` matches the values above. Set `use_reward_modulation: false` to step the network without adjusting neuromodulators from the reward (stimuli still apply). Other fields are knobs for homeostasis / plasticity as the loop grows.

## Architecture brief

This section describes **this crate only**. Network dynamics live in [neuromod](https://github.com/Limen-Neural/neuromod).

| Item | Role |
|------|------|
| `SpikenautTrainer` | Holds `TrainingConfig`; owns `train_step` and `run_session` |
| `TrainingConfig` | Serializable knobs (learning rate, homeostasis, batch size, reward flag) |
| `TrainingExample` | One sample: `stimuli: Vec<f32>` + `reward: f32` |
| `TrainingSummary` | Session metrics after `run_session` |
| `TrainerError` | `EmptyBatch` or wrapped `StepError` from neuromod |

### `train_step`

1. Reads current neuromodulators from the network.
2. If `use_reward_modulation` is `true` (default), adjusts dopamine / norepinephrine from the scalar `reward` (clamped to `[0, 1]`); otherwise leaves modulators unchanged.
3. Calls `network.step(stimuli, &modulators)`.
4. Returns spike indices (`Vec<usize>`) or `StepError`.

### `run_session`

1. Rejects empty batches (`TrainerError::EmptyBatch`).
2. Snapshots thresholds and weights.
3. Calls `train_step` for each `TrainingExample`.
4. Aggregates spikes and average reward.
5. Records per-neuron threshold and weight drifts vs. session start.
6. Returns `TrainingSummary`.

### `TrainingSummary` fields

| Field | Meaning |
|-------|---------|
| `steps_processed` | Number of examples run |
| `total_spikes` | Sum of spike events across steps |
| `avg_reward` | Mean of example rewards |
| `threshold_drifts` | Per-neuron Δthreshold over the session |
| `weight_drifts` | Per-neuron per-channel Δweight over the session |
| `per_neuron_spikes` | Spike counts per neuron |

API docs: run `cargo doc --open` (or `cargo doc --no-deps` in CI-friendly environments).

## Scope and ownership boundaries

This crate provides generic reward-modulated plasticity loops for spiking neural networks. It is intentionally domain-agnostic.

### Owns
- SNN training loop abstractions
- Plasticity rule implementations (STDP, R-STDP, etc.)
- Integration with `limbic-critic` for reward shaping
- Training progress tracking and metrics
- Checkpointing and model serialization

### Does Not Own
- Domain-specific training logic (mining, trading, etc.)
- Differentiable or online distillation and teacher-student knowledge transfer
- Additional project-specific trainer type names beyond the public `SpikenautTrainer` API

### Boundary with SynapticDistill.jl (Linear LIM-25)
- `plasticity-lab` (Rust): reward-modulated STDP / Hebbian plasticity rules and online low-level weight delta computation.
- `SynapticDistill.jl` (Julia): differentiable or online distillation and teacher-student knowledge transfer.
- `SynapticDistill.jl` must not become the home for STDP logic; `plasticity-lab` must not absorb distillation logic.
- A corresponding note should be aligned in `SynapticDistill.jl`.

### Allowed Dependencies
- `neuromod` (for neuromodulator integration)
- `limbic-critic` (for reward shaping)
- `axon-encoder` (for input encoding)
- Math and statistics libraries
- Serialization libraries

### Forbidden Dependencies
- Domain-specific training logic
- Project-specific naming conventions

(See issues #2, #3, #6 for full planning context and migration notes.)

## Cross-language notes

| Language | Status | Notes |
|----------|--------|--------|
| **Rust** | Supported | This crate; use [Getting started](#getting-started) |
| **Julia** | Sister project | Distillation only in [SynapticDistill.jl](https://github.com/Limen-Neural/SynapticDistill.jl) — not a binding of this crate |
| **Python** | Not planned | No PyO3/maturin bindings exist; tracking issue [#13](https://github.com/Limen-Neural/plasticity-lab/issues/13) was closed as a duplicate without being implemented |

Do not expect a Python package from this repository. There is currently no active plan or open issue tracking Python bindings; if that changes, this README will add a parallel getting-started path.

## Contributing

For coding agents and human contributors:

- [AGENTS.md](AGENTS.md) — project conventions, setup commands, architecture map, allowed deps
- [REVIEW.md](REVIEW.md) — PR review checklist and bot-response expectations

Quick local checks:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

## License / REUSE

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

This repository follows the [REUSE](https://reuse.software/) specification: SPDX identifiers appear in source headers and bulk path annotations in [`REUSE.toml`](REUSE.toml); canonical license texts live under [`LICENSES/`](LICENSES/).
