# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Removed (breaking)

- `TrainingConfig::learning_rate`, `TrainingConfig::target_spikes_per_step`,
  `TrainingConfig::homeostasis_strength`, and `TrainingConfig::batch_size` (#66).
  None of these fields were ever read by `SpikenautTrainer` or anything it
  calls; only `use_reward_modulation` drove trainer behavior. Before the first
  crates.io release, a public config should not imply behavior it doesn't
  implement, so the placeholders are removed rather than wired up:
  - Low-level learning rate and homeostasis setpoints are owned by
    `neuromod::SpikingNetwork::step`, which already derives its own learning
    rate and per-neuron threshold targets from the `NeuroModulators` passed in
    (see `neuromod::engine`). There is no supported `neuromod` API to
    externally override those internals yet, so proxying the removed fields
    would have meant inventing new behavior rather than exposing an existing
    one — out of scope for this crate (see AGENTS.md ownership boundaries).
  - `batch_size` had no effect: `run_session` takes the batch directly as a
    `&[TrainingExample]` slice, so its size is whatever the caller passes, not
    something to configure separately.

  **Migration:** if you construct `TrainingConfig` with struct-literal syntax,
  drop the four fields — only `use_reward_modulation` remains:

  ```rust
  // Before
  let config = TrainingConfig {
      learning_rate: 0.01,
      target_spikes_per_step: 0.1,
      homeostasis_strength: 0.001,
      batch_size: 1,
      use_reward_modulation: true,
  };

  // After
  let config = TrainingConfig {
      use_reward_modulation: true,
  };
  ```

  Serialized configs (JSON/checkpoints) that still carry the removed fields
  continue to deserialize without error — `TrainingConfig` does not use
  `#[serde(deny_unknown_fields)]`, so the stale keys are silently ignored and
  `use_reward_modulation` (or its default) is read as before.

### Added

- `bridge` module (`integration` feature): 1:1 adapter from `limbic-critic::ModulatorVector` to `neuromod::NeuroModulators` (`to_neuromodulators`, `from_neuromodulators`, `apply_modulator_vector`) (#17)
- `PlasticityTrainer::train_step_with_modulators` for explicit neuromodulator steps
- `PlasticityTrainer::train_step_from_critic` (`integration`) for critic vectors via the bridge
- Expanded README user guides: getting started, ecosystem map, feature choice, common patterns, architecture brief, cross-language notes (#14)
- Crate- and item-level rustdoc for public API (`TrainingConfig`, `PlasticityTrainer`, `TrainingSummary`, etc.)
- CI step: `cargo doc --no-deps --all-features` with broken-doc-link warnings denied

### Deprecated

- `SpikenautTrainer` remains available as a deprecated alias for `PlasticityTrainer` to ease migration for existing git consumers, at both `plasticity_lab::SpikenautTrainer` and `plasticity_lab::trainer::SpikenautTrainer` (covering both the crate-root re-export and the direct module path). It is hidden from generated rustdoc (`#[doc(hidden)]`) and not part of the documented public API — it will be removed in a future release, so do not use it in new code.

### Fixed

- `TrainingConfig::use_reward_modulation` now gates reward-to-neuromodulator updates in `train_step` (was documented but always on)
- `TrainingConfig` deserializes missing fields via `#[serde(default)]` so older configs without `use_reward_modulation` still load (`true` by default)
- Trainer reward path uses `norepinephrine` after neuromod removed `cortisol` (API drift on git `main`)

### Removed

- Qodana Cloud scan workflow and `qodana.yaml` (membership expired; Clippy/Codacy remain)

### Changed

- **Breaking:** the public trainer type is renamed `SpikenautTrainer` → `PlasticityTrainer` (#65). The crate is pre-1.0 and reusable outside the Spikenaut application, so its primary public type should not carry an application-specific name. Migration: replace `plasticity_lab::SpikenautTrainer` with `plasticity_lab::PlasticityTrainer` in imports and usages; the API surface (`new`, `train_step`, `train_step_with_modulators`, `train_step_from_critic`, `run_session`) is unchanged.
- Documentation-only: clarified crate scope as the SNN learning/training orchestration layer above `neuromod`'s network dynamics and plasticity primitives; removed inaccurate claims that this crate owns STDP/R-STDP implementations or checkpointing/model serialization (README, crate rustdoc, `AGENTS.md`) (#64)
- Git deps `neuromod`, `limbic-critic`, and `axon-encoder` re-pinned to current main tips (norepinephrine API / standalone `ModulatorVector`); intentional rev bumps only
- License switched from GPL-3.0 to dual MIT/Apache-2.0 (chore for better adoption and to align with Limen-Neural org standard; see #9 and master neuromod#19)
  - Added `LICENSE-MIT` and `LICENSE-APACHE-2.0`
  - Updated `Cargo.toml` with `license = "MIT OR Apache-2.0"`
  - Updated `README.md` license section and added badge
  - Added SPDX-License-Identifier headers to source files
  - Removed old GPL LICENSE

## [0.1.0] - 2026-04 (initial)

### Added

- Generic reward-modulated plasticity loops for SNNs around `neuromod::SpikingNetwork`
- `SpikenautTrainer`, `TrainingConfig`, `TrainingExample`, `TrainingSummary`
- Integration feature for `limbic-critic` and `axon-encoder`
