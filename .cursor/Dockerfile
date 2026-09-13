# Cursor cloud agent environment for plasticity-lab.
#
# Mirrors the pinned toolchain used by rust-toolchain.toml, CI, and the VS Code
# dev container. The Rust version below is hardcoded and does NOT derive from
# rust-toolchain.toml, so it goes stale silently if skipped on a toolchain bump
# (see CLAUDE.md's toolchain-bump checklist). Keep it in sync.
FROM rust:1.98.1-slim-bookworm

# ca-certificates + git resolve the Limen-Neural git dependencies over HTTPS;
# curl/build-essential/pkg-config cover native builds and the cargo-deny /
# cargo-tarpaulin tooling CI installs; sudo lets the agent adjust the box.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Components used by CI, local review, and coverage (llvm-tools-preview).
RUN rustup component add rustfmt clippy llvm-tools-preview

# The base image exposes cargo via ENV PATH, which non-login shells inherit but
# a login shell (bash -l) drops when /etc/profile resets PATH. Add cargo to the
# login PATH too so `cargo`/`rustc` resolve regardless of how a shell starts.
# $PATH is intentionally left literal so it expands when the script is sourced.
# hadolint ignore=SC2016
RUN printf 'export PATH=%s:$PATH\n' "$CARGO_HOME/bin" > /etc/profile.d/cargo.sh \
    && chmod 0644 /etc/profile.d/cargo.sh

# Cloud agents run as the `ubuntu` user. Create it (Debian base has no such
# user), grant passwordless sudo, and hand over the shared Rust toolchain dirs
# so the agent can build and add components without root.
RUN if ! id -u ubuntu >/dev/null 2>&1; then useradd -m -s /bin/bash ubuntu; fi \
    && usermod -aG sudo ubuntu \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && chown -R ubuntu:ubuntu "$CARGO_HOME" "$RUSTUP_HOME"

USER ubuntu
WORKDIR /workspace
