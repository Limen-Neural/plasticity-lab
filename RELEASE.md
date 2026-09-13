# Release process

Preflight checklist for cutting a `plasticity-lab` release, per #48. Run every
step from a clean checkout of the commit intended for release.

## 1. Confirm publication blockers are closed or explicitly deferred

All of epic #43's blockers must be closed, or deferred with rationale
recorded in the tracking issue. As of this writing:

- #64, #65, #66 — closed (scope, trainer rename, config honesty)
- #67, #68 — the dependency/feature cleanup and metadata hygiene are done;
  the git-dependency → crates.io-version conversion stays **explicitly
  deferred** (see [Known limitation](#known-limitation-cargo-package) below)
- #47 — rustdoc/doctest coverage, closed
- #46 — feature-matrix CI; its original PR (#78) merged into a branch that
  was later squash-merged separately and didn't carry the change through to
  `main`, so the fix is re-landed as part of this PR instead
- #44, #45, #49 — closed

## 2. Run the full validation suite

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny --locked check
```

All must pass. `cargo deny` may print `warn`-level license/yanked-crate
notices (non-blocking per `deny.toml`); it must not print advisory, bans, or
source failures.

## 3. Update CHANGELOG.md

Move the `[Unreleased]` section's content under a new `## [0.2.0] -
<date>` heading, keeping the breaking-change migration notes intact. Leave a
fresh empty `[Unreleased]` section above it.

## 4. Bump the package version

Set `version` in `Cargo.toml` to `0.2.0`, then run `cargo build` once to
update `Cargo.lock`'s own package entry.

## 5. Commit the release edits

Commit the `CHANGELOG.md`, `Cargo.toml`, and `Cargo.lock` changes from steps
3–4. `cargo package` refuses a dirty working directory without
`--allow-dirty` — which this checklist does not use, since packaging
uncommitted state would let the signed tag in step 7 point at a commit that
doesn't match what was actually packaged/published.

## 6. Package validation

```bash
cargo package --list   # manually review — should list only release-relevant files
cargo package
cargo publish --dry-run
```

### <a name="known-limitation-cargo-package"></a>Known limitation: these currently fail

`cargo package` and `cargo publish --dry-run` **cannot succeed today**.
Cargo requires a version requirement for every dependency — including
optional ones — when packaging a crate for publish, and `limbic-critic` and
`neuromod` are still `branch = "main"` git dependencies with no crates.io
version (see `Cargo.toml`). This is a genuine cross-repo blocker: those
sibling crates need to be published to crates.io (or otherwise given a
concrete version) before this crate can be packaged for real.

**Do not work around this by re-pinning to a different mutable git ref, a
path dependency, or a fake version override.** `CLAUDE.md` says plainly not
to change the `branch = "main"` git-dependency pinning without discussion,
and `REVIEW.md`'s reviewer checklist backs that with "No pinned git rev
changes without discussion." Neither document discusses path dependencies or
fake version overrides by name, but the same reasoning applies: it would
just move the dishonesty from "can't package" to "packages, but the
published manifest lies about what it resolves to." Wait for the sibling
crates, or explicitly re-scope this release to depend only on
already-published siblings.

## 7. Publish, then tag

Only after step 6 actually succeeds, publish **before** tagging — a tag
pushed before a successful `cargo publish` advertises a version that isn't
actually installable if publish then fails:

```bash
cargo publish
git tag -s v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

Then create a GitHub Release for the tag. If `.github/workflows/linear-release.yml`
has landed by then (tracked separately, see #79), publishing the Release
fires it automatically and marks the matching release in the
[`plasticity-lab` Linear pipeline](https://linear.app/rpd-34/pipeline/plasticity-lab/releases)
complete — it needs a `LINEAR_ACCESS_KEY` repository secret (a release
pipeline access key, not a personal API key; see the workflow's header
comment). If that workflow hasn't landed yet, mark the Linear release
complete by hand instead.

## 8. Verify the published artifact

- Check the crate page renders correctly on crates.io
- Check docs.rs actually built the `critic`-feature docs. `Cargo.toml`'s
  `[package.metadata.docs.rs] all-features = true` should make this happen
  automatically — docs.rs does **not** enable optional features by default
  on its own — but verify the built site, don't just trust the config
- In a scratch directory, `cargo new` + add `plasticity-lab = "0.2.0"` and
  confirm it builds against the registry, independent of this repository's
  checkout
