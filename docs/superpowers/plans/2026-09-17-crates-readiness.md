# Crates.io Readiness Implementation Plan

> For agentic workers: use subagent-driven-development for delegated documentation and requesting-code-review for the integrated branch.

**Goal:** Prepare plasticity-lab 0.2.0 for publication using registry neuromod 0.6.0 and limbic-critic 0.3.0, without publishing or merging.
**Architecture:** Preserve trainer behavior and features. Migrate dependencies and test-only provenance; validate the actual archive and enforce package checks in CI.
**Tech Stack:** Rust 2024, Rust 1.98.1, Cargo, GitHub Actions.
**Spec:** User request and preceding crates.io audit in this task; repository AGENTS.md.

## Global Constraints

- Keep default = [], critic optional, and wasm-js opt-in.
- No new production dependencies or training algorithms.
- Preserve the primary checkout and existing branches.
- Use actual published registry versions; no fake overrides.
- Prepare 0.2.0 as an unpublished release candidate; do not claim publication.

## Task 1: Release documentation

Files: README.md, CHANGELOG.md, RELEASE.md, AGENTS.md, CLAUDE.md, REVIEW.md, src/lib.rs.
- Replace current Git install snippets with plasticity-lab = "0.2.0", neuromod = "0.6.0", and optional limbic-critic = "0.3.0" where appropriate.
- Remove conflict marker and contradictory unreleased rand removal claim; keep historical entries accurate.
- Mark 0.2.0 as an unpublished candidate, with final release date assigned at publication.
- Replace obsolete sibling-publication blocker and Git pinning policy with registry dependency qualification.
- Document locked tests, package list, all-feature package/dry-run, extracted archive tests, and external consumer smoke test.
- Preserve scope boundaries and update docs-only feature/deprecation wording in src/lib.rs.
- Verify with targeted stale-language searches and review diff. No behavior changes.

## Task 2: Registry migration and package qualification

Files: Cargo.toml, Cargo.lock, src/bridge.rs, src/replay.rs, deny.toml, .github/workflows/ci.yml.
- Reproduce audit failures with published dependencies before adapting tests.
- Set package version 0.2.0; use neuromod = "0.6.0" and optional limbic-critic version "0.3.0".
- Replace package exclusion list with explicit release file inclusion to prevent dev-file leakage.
- Adapt critic constructor test to Result and replay provenance to exact registry version/checksum and exact rand version from Cargo.lock.
- Remove obsolete Git/GPL exceptions from cargo-deny policy.
- Add locked package/publish dry-run and extracted archive tests in Linux CI, preserving other platform/MSRV/WASM checks.
- Run default/all-feature tests, clippy, fmt, rustdoc, wasm checks, cargo-deny, package and dry-run. Inspect normalized manifest and run an independent consumer using extracted package plus registry dependencies.

## Task 3: Review and delivery

- Independent review of complete diff and validation evidence; fix actionable defects.
- Commit with Codex attribution, push release branch, open one PR against main using GitHub connector.
- Record overlapping Dependabot PR #88 without closing it or claiming to close unrelated tracker work.
- Verify PR metadata and report exact validation and any remaining hosted CI state.
