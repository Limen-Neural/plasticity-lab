# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`plasticity-lab` is a small, single-package Rust crate (no workspace members) providing reward-modulated plasticity training loops for spiking neural networks (SNNs), as part of the Limen-Neural ecosystem. It is deliberately narrow: network dynamics, input encoding, and reward shaping all live in sibling crates, not here.

## Commands

- Build: `cargo build --all-features`
- Test all: `cargo test --all-features`
- Test one: `cargo test --all-features <test_name>` (e.g. `cargo test --all-features train_step_skips_nan_reward_modulation`)
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Format check: `cargo fmt --check`
- Docs (denies warnings, matches CI): `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`
- Coverage: `cargo tarpaulin --all-features --all-targets --out xml --output-dir coverage`
- License/advisory check: `cargo deny --locked check` (config in `deny.toml`)
- MSRV (Minimum Supported Rust Version) build/test — substitute `Cargo.toml`'s `rust-version` (the same value `.github/workflows/ci.yml`'s `msrv` job hardcodes): `cargo +<version> build --locked --all-features && cargo +<version> test --locked --all-features`

`rust-toolchain.toml` pins the toolchain for local dev, and — less obviously — for the `validate` job too: every step in that job invokes bare `cargo`/`rustc`, and rustup's directory-override resolution means a committed `rust-toolchain.toml` wins over whatever the `dtolnay/rust-toolchain` action set as the rustup default (verified against that action's source: it only runs `rustup default <toolchain>`, never a `+<version>`/PATH override that would beat a directory file). That action's `toolchain: 1.97.1` input mainly guarantees the version is installed and matches `rust-toolchain.toml`'s current value — keep them in sync as a matter of hygiene, but `rust-toolchain.toml` is what's actually driving `validate`'s `fmt`/`clippy`/`build`/`test`/`doc` steps. The `msrv` job is genuinely different: its build/test steps use explicit `cargo +1.97.1 ...` (not bare `cargo`), and an explicit `+toolchain` override wins over any directory file — that job really is decoupled from `rust-toolchain.toml`, by design (see its own comment in the workflow). Two separate rules, don't conflate them: raising the actual MSRV means bumping `Cargo.toml`'s `rust-version` and the `msrv` job's `+<version>` pins together (these must always match each other); advancing the day-to-day toolchain means bumping `rust-toolchain.toml`'s `channel` (and the `validate` job's `dtolnay/rust-toolchain` input, to keep them in sync) independently — that does *not* require touching the `msrv` job.

The `integration` feature (off by default) pulls `limbic-critic` and `axon-encoder` as git dependencies tracking `Limen-Neural/*` `main`. Most feature-gated code (`src/bridge.rs`, `SpikenautTrainer::train_step_from_critic`) only compiles/tests with `--all-features` or `--features integration` — plain `cargo test` will silently skip it.

## Architecture

This crate is the middle orchestration layer in a small vertical stack:

```
neuromod        — SpikingNetwork, NeuroModulators, foundational STDP/R-STDP primitives
    ↓
plasticity-lab  — (this crate) training/session orchestration over neuromod
    ↓
applications / supervisors

axon-encoder    — input encoding (sibling; optional, behind `integration`)
limbic-critic   — reward shaping (sibling; optional, behind `integration`)
```

Source layout (`src/`):

- `lib.rs` — public re-exports only; the `bridge` module and its re-exports are `#[cfg(feature = "integration")]`-gated
- `trainer.rs` — `SpikenautTrainer`, with three step variants and one batch entry point:
  - `train_step` — applies scalar-reward → neuromodulator shift, then steps the network
  - `train_step_with_modulators` — steps with explicit `NeuroModulators`, no reward math
  - `train_step_from_critic` (integration only) — converts a `limbic_critic::ModulatorVector` via `bridge`, then calls `train_step_with_modulators`
  - `run_session` — snapshots per-neuron thresholds/weights before the batch, replays `train_step` over each `TrainingExample`, and diffs against the snapshot to build `TrainingSummary` (`threshold_drifts`, `weight_drifts`, `per_neuron_spikes`, `avg_reward`)
- `config.rs` — `TrainingConfig`; always deserializes via `#[serde(default)]` on the struct so partial/old configs stay forward-compatible
- `bridge.rs` — `to_neuromodulators`/`from_neuromodulators` are pure conversions between `limbic_critic::ModulatorVector` and `neuromod::NeuroModulators`, matched by field *name* (`dopamine: v.dopamine`, etc.) — a named-field struct literal is immune to reordering, so the risk after bumping either sibling dependency isn't a reordered field, it's a field being renamed/removed (a compile error, so it's caught) or a same-named field's meaning quietly changing (not caught by the compiler — re-verify semantics, not just presence). `apply_modulator_vector` is not pure: it takes `&mut SpikingNetwork` and calls `network.step(...)`, so it's the one side-effecting entry point in this module.

Two behavioral details that aren't obvious from the public API alone:

- Of `TrainingConfig`'s five fields, only `use_reward_modulation` is actually read by `trainer.rs`. `learning_rate`, `target_spikes_per_step`, `homeostasis_strength`, and `batch_size` are not wired into any code path yet — don't assume setting them changes training behavior (tracked in issue #66).
- `train_step`'s reward→modulator shift is asymmetric and NaN-guarded: positive reward moves dopamine/norepinephrine by different coefficients (0.1/0.05) than negative reward does (0.1/0.2), all clamped to `[0.0, 1.0]`; `reward.is_nan()` skips modulation entirely because `clamp` doesn't protect against this — a NaN receiver passes straight through unclamped (`NaN.clamp(0.0, 1.0)` is `NaN`, not a panic), which would otherwise poison the modulator state for every subsequent step.

## Ecosystem/ownership boundaries

- STDP/R-STDP primitives and network dynamics belong to `neuromod` — do not reimplement them here even when it would be convenient for a new training feature.
- This crate never encodes inputs or shapes rewards itself. `train_step`/`run_session` take precomputed `stimuli: &[f32]` and a scalar `reward: f32`; `train_step_with_modulators` and, under `integration`, `train_step_from_critic`/`apply_modulator_vector` take `stimuli: &[f32]` plus explicit `NeuroModulators`/`ModulatorVector` instead of a scalar reward. Encoding is `axon-encoder`'s job, reward shaping is `limbic-critic`'s.
- No domain-specific training logic (e.g. mining, trading) and no distillation/teacher-student transfer — that belongs to `SynapticDistill.jl` (Julia sister project, not a binding of this crate).
- No `unsafe` code (enforced by Codacy static analysis).
- Git dependencies are pinned by `branch = "main"`, not `rev` — don't change that without discussion (see `REVIEW.md`), and treat `deny.toml`'s `allow-git` list as the source of truth for which git hosts are permitted.

## Conventions

- Branch naming: `<type>/<short-description>` (e.g. `fix/trainer-panic`); commits are imperative mood, lowercase, concise.
- All GitHub Actions in CI are pinned to commit SHAs, not mutable tags.
- PR merge-blockers (see `REVIEW.md` for the full checklist): `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`, no new `unsafe`, no new dependency without justification in the PR description.

## Active migration context

The crate is mid-refactor toward a v0.2 crates.io release (epic issue #43, sub-issues #64–#68/#46–#49/#54). Two changes are in flight that touch code referenced throughout the repo today:

- The public trainer type `SpikenautTrainer` is being renamed to `PlasticityTrainer` (#65).
- Inert `TrainingConfig` fields (see above) are being audited for removal (#66).

Before assuming current type/field names are final, check whether either issue has already landed.
