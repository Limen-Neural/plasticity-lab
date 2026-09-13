# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- `bridge` module (`integration` feature): 1:1 adapter from `limbic-critic::ModulatorVector` to `neuromod::NeuroModulators` (`to_neuromodulators`, `from_neuromodulators`, `apply_modulator_vector`) (#17)
- `SpikenautTrainer::train_step_with_modulators` for explicit neuromodulator steps
- `SpikenautTrainer::train_step_from_critic` (`integration`) for critic vectors via the bridge
- Expanded README user guides: getting started, ecosystem map, feature choice, common patterns, architecture brief, cross-language notes (#14)
- Crate- and item-level rustdoc for public API (`TrainingConfig`, `SpikenautTrainer`, `TrainingSummary`, etc.)
- CI step: `cargo doc --no-deps --all-features` with broken-doc-link warnings denied

### Fixed

- `TrainingConfig::use_reward_modulation` now gates reward-to-neuromodulator updates in `train_step` (was documented but always on)
- `TrainingConfig` deserializes missing fields via `#[serde(default)]` so older configs without `use_reward_modulation` still load (`true` by default)
- Trainer reward path uses `norepinephrine` after neuromod removed `cortisol` (API drift on git `main`)

### Removed

- Qodana Cloud scan workflow and `qodana.yaml` (membership expired; Clippy/Codacy remain)

### Changed

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
