# REVIEW.md

PR review standards for `plasticity-lab`. Applied by both automated bots and human reviewers.

## What to check

### Must pass (block merge)

- [ ] `cargo fmt --check` — no formatting issues
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` — no warnings
- [ ] `cargo test --all-features` — all tests green
- [ ] Locked `cargo test` and `cargo test --all-features` pass for a release candidate
- [ ] `cargo package --list --all-features` contains only intended release files
- [ ] All-feature `cargo package` and `cargo publish --dry-run` pass
- [ ] The extracted package tests and an independent extracted-package consumer smoke test pass
- [ ] Locked `wasm32-unknown-unknown` checks pass with `wasm-js`, both with and without `critic`
- [ ] The feature tree contains `getrandom/wasm_js` only when `wasm-js` is enabled
- [ ] No new `unsafe` blocks
- [ ] No new dependencies without justification in PR description

### Should check (warn, don't block)

- [ ] `cargo deny check` — license and advisory clean (pending #15)
- [ ] Coverage doesn't decrease (Codecov status check)
- [ ] New public items have rustdoc
- [ ] Changes to `trainer.rs` include test coverage
- [ ] Registry dependency requirements remain qualified to the versions tested

### Nice to have (note, don't request changes)

- [ ] CHANGELOG entry for user-facing changes
- [ ] README updated if public API changes
- [ ] Browser and non-Web WebAssembly entropy guidance remains accurate when Cargo features change
- [ ] Examples updated if usage patterns change

## Bot reviewer expectations

When responding to bot comments (Devin, Gemini, Amazon Q, CodeRabbit):

- **Pin actions to commit SHAs** — all CI actions must use SHA pins, not mutable tags
- **Cache tool binaries** — avoid reinstalling on every CI run
- **Use `true`/`false`** in YAML, not `yes`/`no` (YAML 1.2 compatibility)
- **Recursive globs** — use `path/**/*` not `path/` for directory ignores

## PR description template

```markdown
## Summary
<1-3 bullet points>

## Changes
<file-by-file breakdown if non-trivial>

## Testing
<how to verify>

Closes #<issue>
```

## Security

- Never commit secrets, tokens, or credentials
- `CODECOV_TOKEN` must stay in GitHub Secrets only
- No `unsafe` code allowed (Codacy enforcement)
