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
| `limbic-critic`      | Reward shaping                                                          | Rust     |
| `axon-encoder`       | Input encoding                                                          | Rust     |
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
- The Build's `install` step runs `cargo fetch --locked && cargo build --locked --all-features`, so git dependencies and the all-features build cache are warm before an agent starts.
- `.cursor/Dockerfile`'s `FROM rust:<version>` hardcodes the toolchain version independently of `rust-toolchain.toml` — bump it alongside the other version-pinned files on any toolchain change (see CLAUDE.md's toolchain-bump checklist).
- This is a library crate with no binary/server to launch, so there is no `start` command; verify the environment by running the CI checks (`cargo build`/`test`/`clippy`/`fmt`) rather than starting an app.

## Code style

- Rust 2024 edition
- `cargo fmt` and `cargo clippy` must pass before committing
- No `unsafe` code — Codacy flags it via static analysis
- Prefer `thiserror` for error types
- Use `tracing` for logging, not `println!`

## Architecture

- `src/trainer.rs` — core training loop (`PlasticityTrainer`, `run_session`)
- `src/config.rs` — configuration (`TrainingConfig`)
- `src/bridge.rs` — integration adapters between `limbic-critic` and `neuromod` types (`integration` feature)
- `src/lib.rs` — public API re-exports
- `plasticity-lab` owns SNN learning/training orchestration only — see the README's [Scope and ownership boundaries](README.md#scope-and-ownership-boundaries)
- Neuron/network dynamics, neuromodulator state, and foundational (classical + reward-modulated) STDP primitives belong in `neuromod` — call/configure them through its public API rather than reimplementing them here
- Reward shaping belongs in `limbic-critic`
- Input encoding belongs in `axon-encoder`

## What NOT to do

- Do not add domain-specific training logic (mining, trading, etc.)
- Do not add distillation or teacher-student transfer (belongs in `SynapticDistill.jl`)
- Do not implement or duplicate STDP / reward-modulated STDP primitives here — those belong in `neuromod`; call/configure them through its public API
- Do not claim checkpointing or model serialization support unless it is actually implemented in `src/`
- Do not add `unsafe` code
- Do not add heavy or framework-specific dependencies

## Testing

- Unit tests in `src/` alongside source files
- Integration tests via `--all-features` flag (requires `integration` feature)
- Run `cargo test --all-features` before pushing
- CI runs clippy, fmt, build, test, and tarpaulin coverage

## Git conventions

- Branch naming: `<type>/<short-description>` (e.g., `ci/codecov-yaml`, `fix/trainer-panic`)
- Commit messages: imperative mood, lowercase, concise summary
- PRs target `main`
- All actions in CI pinned to commit SHAs

## Dependencies

- Allowed: `neuromod`, `limbic-critic`, `axon-encoder`, `serde`, `serde_json`, `tracing`, `thiserror`, `rand`
- Git deps track `branch = "main"` in `Cargo.toml` (not a `rev` pin); `Cargo.lock` records the currently-resolved commit until `cargo update` bumps it
- Do not add domain-specific or framework-heavy dependencies
