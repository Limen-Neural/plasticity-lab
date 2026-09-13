# AGENTS.md

<!-- version: 2026-07-07 -->

You are a Rust engineering agent working on `plasticity-lab`, a crate for reward-modulated plasticity loops in spiking neural networks (SNNs). Follow the conventions below.

## Project overview

Generic reward-modulated plasticity loops for spiking neural networks.
Single Rust crate; part of the Limen-Neural ecosystem.

## Ecosystem

| Crate               | Role                                        | Language |
|----------------------|---------------------------------------------|----------|
| `neuromod`           | Core SNN + neuromodulator types             | Rust     |
| `plasticity-lab`     | Training loops + plasticity rules           | Rust     |
| `limbic-critic`      | Reward shaping                              | Rust     |
| `axon-encoder`       | Input encoding                              | Rust     |
| `SynapticDistill.jl` | Distillation / knowledge transfer           | Julia    |

## Setup commands

- Build: `cargo build --all-features`
- Test: `cargo test --all-features`
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Format: `cargo fmt --check`
- Coverage: `cargo tarpaulin --all-features --all-targets --out xml --output-dir coverage`

## Toolchain

- Rust 2024 edition, pinned to `1.97.1` in `rust-toolchain.toml`
- CI uses the same pinned toolchain

## Dev container

- VS Code Dev Container configuration is in `.devcontainer/`
- Base image: `rust:1.97.1-slim-bookworm`
- `cargo fetch` runs on container creation
- Run locally with: `devcontainer up --workspace-folder .`

## Code style

- Rust 2024 edition
- `cargo fmt` and `cargo clippy` must pass before committing
- No `unsafe` code — Codacy flags it via static analysis
- Prefer `thiserror` for error types
- Use `tracing` for logging, not `println!`

## Architecture

- `src/trainer.rs` — core training loop (`PlasticityTrainer`, `run_session`)
- `src/config.rs` — configuration (`TrainingConfig`)
- `src/lib.rs` — public API re-exports
- `plasticity-lab` owns training loops and plasticity rules only
- Reward shaping belongs in `limbic-critic`
- Input encoding belongs in `axon-encoder`

## What NOT to do

- Do not add domain-specific training logic (mining, trading, etc.)
- Do not add distillation or teacher-student transfer (belongs in `SynapticDistill.jl`)
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
- Git deps are pinned to specific revisions (see Cargo.toml)
- Do not add domain-specific or framework-heavy dependencies
