# AGENTS.md

<!-- version: 2026-07-07 -->

You are a Rust engineering agent working on `plasticity-lab`, a crate for reward-modulated plasticity loops in spiking neural networks (SNNs). Follow the conventions below.

## Project overview

Generic reward-modulated plasticity loops for spiking neural networks.
Single Rust crate; part of the Limen-Neural ecosystem.

## Ecosystem

| Crate               | Role                                                                    | Language |
|----------------------|--------------------------------------------------------------------------|----------|
| `neuromod`           | Core SNN dynamics, neuromodulator types, foundational (classical + reward-modulated) STDP primitives | Rust |
| `plasticity-lab`     | Training/session orchestration above `neuromod` (this crate)           | Rust     |
| `limbic-critic`      | Reward shaping (`critic` feature only)                                  | Rust     |
| `axon-encoder`       | Input encoding — **not a dependency of this crate** (see #67)          | Rust     |
| `SynapticDistill.jl` | Distillation / knowledge transfer                                       | Julia    |

## Setup commands

- Build: `cargo build --all-features`
- Test: `cargo test --all-features`
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Format: `cargo fmt --check`
- Coverage: `cargo tarpaulin --all-features --all-targets --out xml --output-dir coverage`

## Toolchain

- Rust 2024 edition, pinned to `1.98.1` in `rust-toolchain.toml`
- CI uses the same pinned toolchain

## Dev container

- VS Code Dev Container configuration is in `.devcontainer/`
- Base image: `rust:1.98.1-slim-bookworm`
- `cargo fetch` runs on container creation
- Run locally with: `devcontainer up --workspace-folder .`

## Cursor Cloud specific instructions

- The Cursor cloud-agent environment is defined in `.cursor/environment.json` + `.cursor/Dockerfile` (base image `rust:1.98.1-slim-bookworm`, running as the `ubuntu` user). Both files are force-tracked via a `.gitignore` carve-out; the rest of `.cursor/` stays ignored.
- The Build's `install` step runs `cargo fetch --locked && cargo build --locked --all-features && cargo test --no-run --locked --all-features`, so git dependencies plus the all-features build and test-harness caches are warm before an agent starts.
- `.cursor/Dockerfile`'s `FROM rust:<version>` hardcodes the toolchain version independently of `rust-toolchain.toml` — bump it alongside the other version-pinned files on any toolchain change (see CLAUDE.md's toolchain-bump checklist).
- This is a library crate with no binary/server to launch, so there is no `start` command; verify the environment by running the CI checks (`cargo build`/`test`/`clippy`/`fmt`) rather than starting an app.

## Code style

- Rust 2024 edition
- `cargo fmt` and `cargo clippy` must pass before committing
- No `unsafe` code — Codacy flags it via static analysis
- Prefer `thiserror` for error types
- No logging dependency is in use; do not reach for one (`println!`/`eprintln!`) without a concrete need — see #67

## Architecture

- `src/trainer.rs` — core training loop (`PlasticityTrainer`, `run_session`, seeded `*_with_rng` variants, `run_session_with_observer`)
- `src/observer.rs` — per-step session observer (`TrainingObserver`, `TrainingStepEvent`)
- `src/config.rs` — configuration (`TrainingConfig`)
- `src/bridge.rs` — adapter between `limbic-critic` and `neuromod` types (`critic` feature)
- `src/lib.rs` — public API re-exports
- `src/replay.rs` — test-only above-threshold replay coverage (not part of the public crate)
- `plasticity-lab` owns SNN learning/training orchestration only — see the README's [Scope and ownership boundaries](README.md#scope-and-ownership-boundaries)
- Neuron/network dynamics, neuromodulator state, and foundational (classical + reward-modulated) STDP primitives belong in `neuromod` — call/configure them through its public API rather than reimplementing them here
- Reward shaping belongs in `limbic-critic`
- Input encoding belongs in `axon-encoder` — this crate does not depend on it; encode inputs in your own code before calling `train_step`/`run_session`

## What NOT to do

- Do not add domain-specific training logic (mining, trading, etc.)
- Do not add distillation or teacher-student transfer (belongs in `SynapticDistill.jl`)
- Do not implement or duplicate STDP / reward-modulated STDP primitives here — those belong in `neuromod`; call/configure them through its public API
- Do not claim checkpointing or model serialization support unless it is actually implemented in `src/`
- Do not add `unsafe` code
- Do not add heavy or framework-specific dependencies

## Testing

- Unit tests in `src/` alongside source files
- Critic-bridge tests require `--features critic` or `--all-features`
- Run before pushing: `cargo test && cargo test --all-features`. Keep
  `default = []`; the opt-in `wasm-js` feature is covered separately by the
  locked `wasm32-unknown-unknown` CI checks, with and without `critic`.
- CI (`.github/workflows/ci.yml`) runs clippy, build, and both `cargo test`
  variants above on Linux, macOS, and Windows (`fail-fast: false`).
  `cargo fmt --check`, `cargo deny`, rustdoc, and tarpaulin/Codecov stay
  Linux-only (musl cargo-deny binary and tarpaulin ptrace).

## Git conventions

- Branch naming: `<type>/<short-description>` (e.g., `ci/codecov-yaml`, `fix/trainer-panic`)
- Commit messages: imperative mood, lowercase, concise summary
- PRs target `main`
- All actions in CI pinned to commit SHAs

## Dependencies

- Allowed production deps: `neuromod`, `limbic-critic` (`critic` feature only), `serde`, `thiserror`, `rand` (caller-injected RNG for seeded replay)
- `serde_json` is a dev-dependency only (used by `config.rs` and replay-manifest tests) — do not promote it to `[dependencies]` without a real runtime use
- `axon-encoder`, `rand`, and `tracing` were previously unused (#67). `rand`
  is now a production dependency because `train_step_with_rng` /
  `run_session_with_rng` take `&mut impl rand::Rng`. Do not re-add
  `axon-encoder` or `tracing` solely to make Cargo resolve a sibling crate.
- Git deps track `branch = "main"` in `Cargo.toml` (not a `rev` pin); `Cargo.lock` records the currently-resolved commit until `cargo update` bumps it. This intentionally stays a git dependency until `neuromod`/`limbic-critic` are published on crates.io — do not replace it with another mutable git ref as a way to appear crates.io-ready
- Do not add domain-specific or framework-heavy dependencies
