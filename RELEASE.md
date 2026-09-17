# Release process

`0.2.0` was published on 2026-09-17. Run this checklist from a clean
checkout of the exact commit intended for release. Do not tag it or describe
it as published until `cargo publish` succeeds. The pre-publish final commit
may carry a planned publication date; it becomes the release date only after a
successful publish.

Capture the release checkout before running any step:

```bash
release_repo=$(git rev-parse --show-toplevel)
cd "$release_repo"
test -z "$(git status --porcelain)"
```

## 1. Confirm the release manifest

The release candidate must use the published registry dependencies:

```toml
plasticity-lab = "0.2.0"
neuromod = "0.6.0"
limbic-critic = "0.3.0" # optional, enabled by the `critic` feature
```

Keep `default = []`; `critic` and `wasm-js` remain opt-in. Do not substitute
git, path, or placeholder-version dependencies to make a package check pass.

## 2. Run the locked validation suite

```bash
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
cargo deny --locked check
```

The locked default and all-feature tests are both required: `critic` is
optional and therefore absent from the default build.

## 3. Inspect and qualify the package

```bash
cargo package --locked --list --all-features
cargo package --locked --all-features
cargo publish --dry-run --locked --all-features
```

Review the list before packaging. It must contain the explicit release files:
`src/**`, `examples/**`, `Cargo.toml`, `Cargo.lock`, `README.md`,
`CHANGELOG.md`, the root license files, `LICENSES/**`, and `REUSE.toml`.
The all-feature commands qualify the optional `limbic-critic` bridge against
its immutable registry release.

## 4. Test the archive that Cargo produced

Use a fresh temporary directory; do not test against this checkout. Derive the
archive path and package version from Cargo metadata so the command remains
correct for the final package version:

```bash
archive_dir=$(cargo metadata --no-deps --format-version 1 | \
  jq -r '.target_directory + "/package"')
package_stem=$(cargo metadata --no-deps --format-version 1 | \
  jq -r '.packages[] | select(.name == "plasticity-lab") | "\(.name)-\(.version)"')
release_tmp=$(mktemp -d)
tar -xzf "$archive_dir/$package_stem.crate" -C "$release_tmp"
(
  cd "$release_tmp/$package_stem"
  cargo test --locked
  cargo test --locked --all-features
)
```

The archive tests catch files that compiled in the repository but were omitted
from the package.

## 5. Run an independent extracted-package consumer smoke test

From another empty temporary directory, create a small consumer whose manifest
uses the extracted package and the candidate's registry sibling versions:

```toml
[dependencies]
plasticity-lab = { path = "../plasticity-lab-<version>" }
neuromod = "0.6.0"
```

Point the path at the directory extracted in step 4. Add a minimal program
that imports `plasticity_lab::PlasticityTrainer` and
`neuromod::SpikingNetwork`. First run `cargo check` to create the consumer's
lockfile, then run `cargo check --locked`. For the optional bridge, add
`limbic-critic = "0.3.0"`, enable `plasticity-lab`'s `critic` feature, import
a bridge item, and run both checks again. This validates the consumer-facing
archive with registry-resolved siblings before `plasticity-lab` itself exists
in the registry.

## 6. Set the final date and publish

After every qualification step passes, update the release-status wording in
`README.md`, `CHANGELOG.md`, `CLAUDE.md`, and this guide from `Unpublished
release candidate` to the planned publication date. Commit those changes and
merge the final dated commit onto `main`. Rerun steps 2–5 from that `main`
checkout, including the package and archive checks, then record the qualified
commit immediately afterward:

```bash
qualified_commit=$(git -C "$release_repo" rev-parse HEAD)
```

If publication fails, restore the candidate wording.

Before publishing, return to the original clean checkout and verify that the
**final dated commit** has not moved locally or on `origin/main`:

```bash
(
  set -euo pipefail
  cd "$release_repo"
  git fetch origin main
  test "$(git branch --show-current)" = "main"
  test -z "$(git status --porcelain)"
  test "$(git rev-parse HEAD)" = "$qualified_commit"
  test "$(git rev-parse origin/main)" = "$qualified_commit"
  cargo publish --locked --all-features
  git tag -s v0.2.0 "$qualified_commit" -m "v0.2.0"
  git push origin v0.2.0
)
```

Finally, verify the crates.io page and the docs.rs build with all features,
then repeat step 5 with `plasticity-lab = "0.2.0"` from crates.io. Create the
GitHub Release for the tag and follow the repository's configured Linear
release workflow, if present.
