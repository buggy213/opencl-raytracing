# Visual Regression Testing

Snapshot-based visual regression testing for the raytracer. Renders builtin test scenes with deterministic settings and compares results against "blessed" reference images.

## Requirements

- **Rust CLI**: The `cli` binary must be installed and in your PATH:
  ```bash
  cargo install --path crates/cli
  ```
- **Python 3.10+** with [uv](https://github.com/astral-sh/uv)

## Quick Start

```bash
cd crates/visual-testing

# Run all tests with fast settings (1 spp, 1 light sample)
uv run rttest cpu -- -s 1 -l 1

# List available test scenes
uv run rttest --list
```

## Usage

### Running Tests

```bash
# Run all tests with specific renderer settings
uv run rttest cpu -- -s 1 -l 1

# Run specific scenes only
uv run rttest cpu --scenes sphere,cube -- -s 1 -l 1

# Run with higher quality (slower)
uv run rttest cpu -- -s 4 -l 4
```

Arguments passed after `--` are forwarded to underlying `cli` binary. 

### Blessing References

Reference images are not version-controlled. They should be created locally as part of repository setup 
by "blessing" initial outputs after visual verification.

```bash
# Initial setup: interactively review and bless all scenes
uv run rttest cpu --bless -- -s 1 -l 1

# After intentional changes: review and bless interactively
uv run rttest cpu --bless -- -s 1 -l 1

# Auto-bless all without review (use with caution)
uv run rttest cpu --bless-all -- -s 1 -l 1
```

The `--bless` flag opens a matplotlib window for each scene showing:
- The rendered output (with adjustable exposure slider)
- The existing reference (if any) for comparison
- A difference heatmap

Use keyboard shortcuts: **y** to bless, **n** to skip, **q** to quit.

### Machine-Readable Output

For CI or coding agents:

```bash
uv run rttest cpu --json -- -s 1 -l 1
```

Returns JSON with test results, metrics, and artifact paths.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | One or more tests failed (visual differences detected) |
| 2 | Error (render crashed, missing CLI arguments, etc.) |

## Directory Structure

```
crates/visual-testing/
├── references/     # Blessed reference images (not version-controlled)
├── outputs/        # Test outputs and diff images (not version-controlled)
├── src/rttest/     # Python source
└── pyproject.toml
```

## Workflow for Making Changes

1. Make your code changes
2. Run `uv run rttest cpu -- -s 1 -l 1`
3. If tests fail, inspect the diff images in `outputs/`
4. If changes are intentional, bless them:
   - `uv run rttest cpu --bless-failed` for interactive review
   - `uv run rttest cpu --bless -- -s 1 -l 1` to bless all
5. Commit your changes

## Tolerance

By default, any pixel difference causes a failure. This is intentional: the renderer is deterministic, so any change to output indicates a code change that should be reviewed.

To allow small differences (e.g., for floating-point variations across platforms):

```bash
uv run rttest cpu --tolerance 1e-6 -- -s 1 -l 1
```

## Adding New Test Scenes

Add new scenes to `all_test_scenes()` in `crates/raytracing/src/scene/scene.rs`, then:

```bash
cargo install --path crates/cli
uv run rttest cpu --bless -- -s 1 -l 1
```

