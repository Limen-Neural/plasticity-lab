# plasticity-lab

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Reusable SNN learning/training orchestration layer: reward-modulated training loops above [`neuromod`](https://github.com/Limen-Neural/neuromod)'s plasticity primitives.

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

`plasticity-lab` is the reusable **SNN learning/training orchestration layer** for the Limen-Neural stack. It provides a small training loop around [`neuromod::SpikingNetwork`](https://github.com/Limen-Neural/neuromod) — the crate that owns neuron/network dynamics, neuromodulator state, and the foundational classical and reward-modulated STDP primitives. `plasticity-lab` calls and configures those primitives through `neuromod`'s public API; it does not reimplement them.

It is intentionally domain-agnostic:

- Neuron/network dynamics and low-level plasticity primitives (STDP, R-STDP) belong to [`neuromod`](https://github.com/Limen-Neural/neuromod)
- Input encoding belongs to [`axon-encoder`](https://github.com/Limen-Neural/axon-encoder)
- Reward shaping belongs to [`limbic-critic`](https://github.com/Limen-Neural/limbic-critic)
- This crate orchestrates the training/session loop, maps rewards or modulator vectors into training steps, and tracks training summaries

If you are new to the Limen-Neural stack, start with [Getting started](#getting-started), then skim [Ecosystem overview](#ecosystem-overview) and [Scope and ownership boundaries](#scope-and-ownership-boundaries) so you know which crate owns which piece.

## Ecosystem overview

| Crate | Role | Language | When to use it |
|-------|------|----------|----------------|
| **plasticity-lab** (this crate) | Training/session orchestration (`train_step`, `run_session`) | Rust | You need a reward-modulated SNN training loop and session metrics |
| [neuromod](https://github.com/Limen-Neural/neuromod) | Core SNN dynamics, neuromodulator types, and foundational (classical + reward-modulated) STDP primitives (`SpikingNetwork`, `NeuroModulators`) | Rust | You need the network, step dynamics, modulator state, or the underlying plasticity rules |
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

`TrainingConfig::default()` matches the values above. Set `use_reward_modulation: false` to step the network without adjusting neuromodulators from the reward (stimuli still apply). Other fields are reserved orchestration-level knobs (homeostasis targets, batch size) for this crate's training loop; they are not yet wired into behavior, and none of them reimplement `neuromod`'s plasticity primitives.

## Architecture brief

This section describes **this crate only**. Network dynamics, neuromodulator state, and the underlying classical / reward-modulated STDP primitives live in [neuromod](https://github.com/Limen-Neural/neuromod) — see its own ownership documentation ([neuromod#readme](https://github.com/Limen-Neural/neuromod#scope-and-ownership-boundaries)) for that crate's boundary commitments.

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

`plasticity-lab` is the reusable **SNN learning/training orchestration layer** above the low-level plasticity primitives in [`neuromod`](https://github.com/Limen-Neural/neuromod). It is intentionally domain-agnostic, and it does not reimplement plasticity algorithms that `neuromod` already owns.

### Layering

```text
        application / supervisor
                  │
                  │  drives experiments, reads TrainingSummary
                  ▼
             plasticity-lab   (this crate)
                  │  training/session orchestration: train_step,
                  │  run_session, reward/modulator-vector mapping,
                  │  batches, metrics, integration adapters
                  ▼
                neuromod
                   SpikingNetwork, neuron/network dynamics,
                   neuromodulator state, foundational classical
                   and reward-modulated STDP primitives
```

### Owns
- Training/session orchestration (`train_step`, `run_session`)
- Mapping externally supplied scalar rewards or modulator vectors (e.g. from `limbic-critic`) into a training step
- Training examples / batches (`TrainingExample`)
- Progress and training summaries (`TrainingSummary`)
- Training/session metrics and invariants (spike counts, threshold/weight drift, empty-batch rejection)
- Integration adapters between independently owned crates (the `bridge` module)
- Checkpoint/session orchestration, if/when it is actually implemented — **not implemented today** (see [Does Not Own](#does-not-own))

### Does Not Own
- Neuron and network dynamics, and `SpikingNetwork` itself — owned by [`neuromod`](https://github.com/Limen-Neural/neuromod)
- Neuromodulator state/types — owned by `neuromod`
- Foundational classical STDP primitives — owned by `neuromod`
- Foundational reward-modulated STDP / eligibility-trace primitives — owned by `neuromod`
- Low-level plasticity configuration applied by the network engine — owned by `neuromod`
- Reward shaping — owned by [`limbic-critic`](https://github.com/Limen-Neural/limbic-critic)
- Input encoding — owned by [`axon-encoder`](https://github.com/Limen-Neural/axon-encoder)
- Differentiable or online distillation and teacher-student knowledge transfer — owned by [`SynapticDistill.jl`](https://github.com/Limen-Neural/SynapticDistill.jl)
- Domain-specific training logic (mining, trading, etc.)
- Checkpointing and model serialization — **not currently implemented** in this crate; do not assume it exists
- Additional project-specific trainer type names beyond the public `SpikenautTrainer` API

### Boundary with neuromod

`plasticity-lab` calls and configures `neuromod`'s plasticity rules through its public API (`SpikingNetwork::step`, `NeuroModulators`) rather than copying the algorithms here. Changes to how STDP or reward-modulated STDP behaves belong in `neuromod`, not in this crate. See [neuromod's own ownership documentation](https://github.com/Limen-Neural/neuromod#scope-and-ownership-boundaries) for its boundary commitments.

### Boundary with SynapticDistill.jl (Linear LIM-25)
- `plasticity-lab` (Rust): training/session orchestration above `neuromod`'s reward-modulated STDP / Hebbian plasticity primitives; it does not implement those primitives itself.
- `SynapticDistill.jl` (Julia): differentiable or online distillation and teacher-student knowledge transfer.
- `SynapticDistill.jl` must not become the home for STDP logic; `plasticity-lab` must not absorb distillation logic.
- A corresponding note should be aligned in `SynapticDistill.jl`.

### Allowed Dependencies
- `neuromod` (network dynamics, neuromodulator state, and low-level plasticity primitives)
- `limbic-critic` (for reward shaping)
- `axon-encoder` (for input encoding)
- Math and statistics libraries
- Serialization libraries

### Forbidden Dependencies
- Domain-specific training logic
- Project-specific naming conventions
- Duplicated STDP/R-STDP rule implementations (call into `neuromod` instead)

(See issues #2, #3, #6, #64 for full planning context and migration notes.)

## Cross-language notes

| Language | Status | Notes |
|----------|--------|--------|
| **Rust** | Supported | This crate; use [Getting started](#getting-started) |
| **Julia** | Sister project | Distillation only in [SynapticDistill.jl](https://github.com/Limen-Neural/SynapticDistill.jl) — not a binding of this crate |
| **Python** | Planned | Bindings tracked in issue [#13](https://github.com/Limen-Neural/plasticity-lab/issues/13); **not yet shipped** |

Do not expect a Python package from this repository today. When #13 lands, this README will add a parallel getting-started path.

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
