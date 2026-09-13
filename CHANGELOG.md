# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- `bridge` module (`integration` feature): 1:1 adapter from `limbic-critic::ModulatorVector` to `neuromod::NeuroModulators` (`to_neuromodulators`, `from_neuromodulators`, `apply_modulator_vector`) (#17)
- `PlasticityTrainer::train_step_with_modulators` for explicit neuromodulator steps
- `PlasticityTrainer::train_step_from_critic` (`integration`) for critic vectors via the bridge
- Expanded README user guides: getting started, ecosystem map, feature choice, common patterns, architecture brief, cross-language notes (#14)
- Crate- and item-level rustdoc for public API (`TrainingConfig`, `PlasticityTrainer`, `TrainingSummary`, etc.)
- CI step: `cargo doc --no-deps --all-features` with broken-doc-link warnings denied

### Deprecated

- `SpikenautTrainer` remains available as a deprecated alias for `PlasticityTrainer` to ease migration for existing git consumers. It is not part of the documented public API and will be removed in a future release — do not use it in new code.

### Fixed

- `TrainingConfig::use_reward_modulation` now gates reward-to-neuromodulator updates in `train_step` (was documented but always on)
- `TrainingConfig` deserializes missing fields via `#[serde(default)]` so older configs without `use_reward_modulation` still load (`true` by default)
- Trainer reward path uses `norepinephrine` after neuromod removed `cortisol` (API drift on git `main`)

### Removed

- Qodana Cloud scan workflow and `qodana.yaml` (membership expired; Clippy/Codacy remain)

### Changed

- **Breaking:** the public trainer type is renamed `SpikenautTrainer` → `PlasticityTrainer` (#65). The crate is pre-1.0 and reusable outside the Spikenaut application, so its primary public type should not carry an application-specific name. Migration: replace `plasticity_lab::SpikenautTrainer` with `plasticity_lab::PlasticityTrainer` in imports and usages; the API surface (`new`, `train_step`, `train_step_with_modulators`, `train_step_from_critic`, `run_session`) is unchanged.
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
