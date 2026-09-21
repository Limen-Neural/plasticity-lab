# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`plasticity-lab` is a small, single-package Rust crate (no workspace members) providing reward-modulated plasticity training loops for spiking neural networks (SNNs), as part of the Limen-Neural ecosystem. It is deliberately narrow: network dynamics, input encoding, and reward shaping all live in sibling crates, not here.

## Commands

- Build: `cargo build --all-features`
- Test all: `cargo test --all-features`
- Test one: `cargo test --all-features <test_name>` (e.g. `cargo test --all-features train_step_rejects_every_non_finite_reward_before_mutation`)
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Format check: `cargo fmt --check`
- Docs (denies warnings, matches CI): `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`
- Coverage: `cargo tarpaulin --all-features --all-targets --out xml --output-dir coverage`
- License/advisory check: `cargo deny --locked check` (config in `deny.toml`)
- MSRV (Minimum Supported Rust Version) build/test — substitute `Cargo.toml`'s `rust-version` (the same value `.github/workflows/ci.yml`'s `msrv` job hardcodes): `cargo +<version> build --locked --all-features && cargo +<version> test --locked --all-features`

`rust-toolchain.toml` pins the toolchain for local dev, and — less obviously — for the `validate` job too: every step in that job invokes bare `cargo`/`rustc`, and rustup's directory-override resolution means a committed `rust-toolchain.toml` wins over whatever the `dtolnay/rust-toolchain` action set as the rustup default (verified against that action's source: it only runs `rustup default <toolchain>`, never a `+<version>`/PATH override that would beat a directory file). That action's `toolchain:` input mainly guarantees the version is installed and matches `rust-toolchain.toml`'s current value — keep them in sync as a matter of hygiene, but `rust-toolchain.toml` is what's actually driving `validate`'s `fmt`/`clippy`/`build`/`test`/`doc` steps. The `msrv` job is genuinely different: its build/test steps use explicit `cargo +<version> ...` (not bare `cargo`), and an explicit `+toolchain` override wins over any directory file — that job really is decoupled from `rust-toolchain.toml`, by design (see its own comment in the workflow). Two separate rules, don't conflate them: raising the actual MSRV means bumping `Cargo.toml`'s `rust-version` and *every* version literal in the `msrv` job together — that's three spots, not two: the `Install MSRV toolchain` step's `toolchain:` input (which controls what actually gets installed) as well as both `cargo +<version>` invocations (which select among installed toolchains but can't install one — `+toolchain` requires it to already be present). Bumping only the `cargo +<version>` lines without the install step leaves the new version uninstalled and the job broken. Advancing the day-to-day toolchain means bumping `rust-toolchain.toml`'s `channel` and the `validate` job's `dtolnay/rust-toolchain` input together — but that's not the complete list either: `.devcontainer/Dockerfile`'s `FROM rust:<version>-slim-bookworm`, `.devcontainer/devcontainer.json`'s `name` field, `.cursor/Dockerfile`'s `FROM rust:<version>-slim-bookworm` (the Cursor cloud-agent environment), and `.devin/blueprint.yaml`'s three `rustup ... <version>` lines all hardcode the same version independently and don't derive from `rust-toolchain.toml` or anything else, so they go stale silently if skipped (`.devcontainer/devcontainer.json`'s `name` was missed exactly this way during a past bump — check its current value against `rust-toolchain.toml` rather than trusting it). `README.md` and `AGENTS.md` also state the version in prose. None of this requires touching the `msrv` job. (Deliberately not naming a specific version number anywhere in this paragraph — check `Cargo.toml`'s `rust-version` for the actual current value, since any literal written here will go stale on the next bump, as happened to an earlier draft of this very paragraph.)

The `critic` feature (off by default, renamed from `integration` in #67 — `axon-encoder` was dropped entirely since no code in this crate ever consumed it) pulls the optional registry dependency `limbic-critic = "0.3.0"`. Feature-gated code (`src/bridge.rs`, `PlasticityTrainer::train_step_from_critic`) only compiles/tests with `--all-features` or `--features critic` — plain `cargo test` will silently skip it.

## Architecture

This crate is the middle orchestration layer in a small vertical stack:

```
neuromod        — SpikingNetwork, NeuroModulators, foundational STDP/R-STDP primitives
    ↓
plasticity-lab  — (this crate) training/session orchestration over neuromod
    ↓
applications / supervisors

axon-encoder    — input encoding (sibling; not a dependency of this crate, see #67)
limbic-critic   — reward shaping (sibling; optional, behind `critic`)
```

Source layout (`src/`):

- `lib.rs` — public re-exports only; the `bridge` module and its re-exports are `#[cfg(feature = "critic")]`-gated
- `trainer.rs` — `PlasticityTrainer`, with three step variants and one batch entry point:
  - `train_step` — applies scalar-reward → neuromodulator shift, then steps the network
  - `train_step_with_modulators` — steps with explicit `NeuroModulators`, no reward math
  - `train_step_from_critic` (`critic` feature only) — converts a `limbic_critic::ModulatorVector` via `bridge`, then calls `train_step_with_modulators`
  - `run_session` — admits the whole batch first (dimensions, finite stimuli, finite rewards) so a late invalid sample cannot leave earlier samples applied; then snapshots per-neuron thresholds/weights, replays `train_step` over each `TrainingExample`, and diffs against the snapshot to build `TrainingSummary` (`threshold_drifts`, `weight_drifts`, `per_neuron_spikes`, `avg_reward`)
  - `run_session_with_observer` — same preflight and loop, plus one borrowed `TrainingStepEvent` after each successful step; observer `Err` aborts before the next example (`TrainerError::Observer`)
- `observer.rs` — `TrainingObserver` trait and borrowed `TrainingStepEvent` (no mutable network access)
- `config.rs` — `TrainingConfig`; `#[serde(default)]` on the struct fills omitted fields in map-based / self-describing formats such as JSON so partial/old configs stay compatible. That is not a guarantee for positional or non-self-describing encodings (bincode/postcard).
- `bridge.rs` — `to_neuromodulators`/`from_neuromodulators` are pure conversions between `limbic_critic::ModulatorVector` and `neuromod::NeuroModulators`, matched by field *name* (`dopamine: v.dopamine`, etc.) — a named-field struct literal is immune to reordering, so the risk after bumping either sibling dependency isn't a reordered field, it's a field being renamed/removed (a compile error, so it's caught) or a same-named field's meaning quietly changing (not caught by the compiler — re-verify semantics, not just presence). `apply_modulator_vector` is not pure: it takes `&mut SpikingNetwork` and calls `network.step(...)`, so it's the one side-effecting entry point in this module.

One behavioral detail that isn't obvious from the public API alone:

- `train_step`'s reward→modulator shift is controlled by a validated `RewardMapping`; its compatibility defaults retain the asymmetric 0.1/0.05 positive and 0.1/0.2 negative behavior, all clamped to `[0.0, 1.0]`. Every scalar API rejects NaN and ±infinity as `TrainerError::NonFiniteReward` before network or RNG mutation.

## Ecosystem/ownership boundaries

- STDP/R-STDP primitives and network dynamics belong to `neuromod` — do not reimplement them here even when it would be convenient for a new training feature.
- This crate never encodes inputs or shapes rewards itself. `train_step`/`run_session` take precomputed `stimuli: &[f32]` and a scalar `reward: f32`; `train_step_with_modulators` and, under `critic`, `train_step_from_critic`/`apply_modulator_vector` take `stimuli: &[f32]` plus explicit `NeuroModulators`/`ModulatorVector` instead of a scalar reward. Encoding is `axon-encoder`'s job (not a dependency here — wire it in yourself), reward shaping is `limbic-critic`'s.
- No domain-specific training logic (e.g. mining, trading) and no distillation/teacher-student transfer — that belongs to `SynapticDistill.jl` (Julia sister project, not a binding of this crate).
- No `unsafe` code (enforced by Codacy static analysis).
- Release dependencies are registry-qualified (`neuromod = "0.6.0"`; optional
  `limbic-critic = "0.3.0"`). Do not substitute mutable git, path, or
  placeholder-version sources when qualifying a release.

## Conventions

- Branch naming: `<type>/<short-description>` (e.g. `fix/trainer-panic`); commits are imperative mood, lowercase, concise.
- All GitHub Actions in CI are pinned to commit SHAs, not mutable tags.
- `REVIEW.md`'s own "must pass" checklist: `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`, no new `unsafe`, no new dependency without justification in the PR description. That checklist is narrower than the actual `validate` CI job, though: `cargo deny --locked check` (license/advisory), `cargo doc --no-deps --all-features` (rustdoc warnings denied), and `cargo tarpaulin` are also ordinary failure-producing steps on the **Linux** matrix cell — any of them failing fails that cell (and the merge) even when every `REVIEW.md` item passes. `validate` itself is a three-OS matrix (`ubuntu-latest`, `macos-latest`, `windows-latest`, `fail-fast: false`); clippy/build/test run on every OS, while fmt/deny/doc/tarpaulin/Codecov stay Linux-only.

## v0.2 release context

`0.2.0` was published on crates.io on 2026-09-17 (epic issue #43). The
trainer rename (`SpikenautTrainer` → `PlasticityTrainer`, #65) and the
`TrainingConfig` field cleanup (#66) have already landed — `PlasticityTrainer`
and the single-field `TrainingConfig` above are the current state, not a future
one. `SpikenautTrainer` survives only as a `#[deprecated]`, `#[doc(hidden)`]
migration alias at both `plasticity_lab::SpikenautTrainer` and
`plasticity_lab::trainer::SpikenautTrainer`; do not use it in new code or
assume its presence means the rename is pending. Before publication, run the
locked tests, all-feature package and dry-run, extracted archive tests, and an
independent extracted-package consumer smoke test documented in `RELEASE.md`.
