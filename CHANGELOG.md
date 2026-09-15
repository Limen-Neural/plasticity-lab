# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Removed (breaking)

- The `integration` feature is renamed `critic`, and no longer pulls
  `axon-encoder` (#67). `axon-encoder` was declared as an optional dependency
  but no code in this crate ever consumed it — it was retained solely to make
  Cargo resolve the sibling crate, which is exactly the kind of feature #67
  flags as not publishable. The bridge to `limbic-critic` is unaffected
  besides the feature name.

  **Migration:**

  ```toml
  # Before
  plasticity-lab = { git = "...", features = ["integration"] }

  # After
  plasticity-lab = { git = "...", features = ["critic"] }
  ```

  If you were relying on this crate to pull in `axon-encoder` transitively,
  add it directly to your own `Cargo.toml` instead.

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

- CI: Build & Test matrix on `ubuntu-latest`, `macos-latest`, and
  `windows-latest` (`fail-fast: false`). `cargo fmt --check` (OS-independent)
  and rustdoc stay Linux-only to save runner minutes; musl `cargo-deny`
  stays Linux-only because the install is the musl Linux binary;
  tarpaulin/Codecov stay Linux-only so coverage is not duplicated.
- `bridge` module (`critic` feature): 1:1 adapter from `limbic-critic::ModulatorVector` to `neuromod::NeuroModulators` (`to_neuromodulators`, `from_neuromodulators`, `apply_modulator_vector`) (#17)
- `PlasticityTrainer::train_step_with_modulators` for explicit neuromodulator steps
- `PlasticityTrainer::train_step_from_critic` (`critic`) for critic vectors via the bridge
- Expanded README user guides: getting started, ecosystem map, feature choice, common patterns, architecture brief, cross-language notes (#14)
- Crate- and item-level rustdoc for public API (`TrainingConfig`, `PlasticityTrainer`, `TrainingSummary`, etc.)
- CI step: `cargo doc --no-deps --all-features` with broken-doc-link warnings denied
- `Cargo.toml` package metadata: `repository`, `readme`, `keywords`, `categories`, and an `exclude` list scoping the published package to release-relevant files (#68)

### Known limitations

- `cargo package` and `cargo publish --dry-run` cannot succeed yet: `limbic-critic`
  is a `branch = "main"` git dependency with no crates.io version, and Cargo
  requires a version requirement for every dependency (including
  feature-gated/optional ones) when packaging a crate for publish. The same
  applies to `neuromod` once dependent resolution reaches it. This is a
  pre-existing, cross-repo blocker (#67) — it is **not** worked around by
  re-pinning to a different mutable git ref, per this repo's own review policy
  (see `REVIEW.md`, `CLAUDE.md`). Publishing `neuromod` and `limbic-critic` to
  crates.io first is a prerequisite for a real `cargo publish` of this crate.

### Deprecated

- `SpikenautTrainer` remains available as a deprecated alias for `PlasticityTrainer` to ease migration for existing git consumers, at both `plasticity_lab::SpikenautTrainer` and `plasticity_lab::trainer::SpikenautTrainer` (covering both the crate-root re-export and the direct module path). It is hidden from generated rustdoc (`#[doc(hidden)]`) and not part of the documented public API — it will be removed in a future release, so do not use it in new code.

### Fixed

- `TrainingConfig::use_reward_modulation` now gates reward-to-neuromodulator updates in `train_step` (was documented but always on)
- `TrainingConfig` deserializes missing fields via `#[serde(default)]` so older configs without `use_reward_modulation` still load (`true` by default)
- Trainer reward path uses `norepinephrine` after neuromod removed `cortisol` (API drift on git `main`)

### Removed

- Qodana Cloud scan workflow and `qodana.yaml` (membership expired; Clippy/Codacy remain)
- `rand` and `tracing` from `[dependencies]` (#67, #68): neither had any code
  in `src/` using them — not a migration concern for consumers, since nothing
  in the public API depended on either. `serde_json` moves from
  `[dependencies]` to `[dev-dependencies]`, since it is only used by
  `config.rs` tests.

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
