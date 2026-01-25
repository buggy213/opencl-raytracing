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

# Run all tests (defined in tests/tests.toml)
uv run rttest cpu -- -s 1 -l 1
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
| 1 | One or more tests failed (visual differences or performance regressions) |
| 2 | Error (render crashed, missing CLI arguments, etc.) |

## Directory Structure

```
crates/visual-testing/
├── references/           # Blessed reference images (not version-controlled)
├── outputs/              # Test outputs and diff images (not version-controlled)
├── tests/                # Example test specification files
├── perf_history.jsonl    # Timing history (not version-controlled)
├── perf_baseline.json    # Blessed timing baseline
├── src/rttest/           # Python source
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

## Performance Testing

Timing is captured by default for all test runs. Use `--no-perf` to disable.

```bash
# Performance timing only (skip visual comparison)
uv run rttest cpu --perf-only -- -s 4 -l 4

# Disable timing capture
uv run rttest cpu --no-perf -- -s 1 -l 1
```

### Baseline and Regression Detection

```bash
# Set current timings as baseline
uv run rttest cpu --perf-baseline -- -s 4 -l 4

# Query timing history for a scene
uv run rttest --perf-history sphere

# Custom regression threshold (default: 10%)
uv run rttest cpu --perf-threshold 5 -- -s 4 -l 4
```

Timing data is stored in:
- `perf_history.jsonl` - Append-only history (not version-controlled)
- `perf_baseline.json` - Blessed baseline (optionally version-controlled)

**Note**: Timing measurements can vary due to system load. For reliable regression detection, run multiple times or use consistent hardware.

## Test Specification (tests/tests.toml)

Tests are defined in `tests/tests.toml`. Use `--tests-file` to specify an alternative:

```bash
uv run rttest cpu --tests-file custom.toml
```

### TOML Format Reference

```toml
# Default settings applied to all tests unless overridden
[defaults]
samples_per_pixel = 1    # -s flag value
light_samples = 1        # -l flag value

# Each [[test]] block defines a single test
[[test]]
name = "sphere"          # Required: unique name (used for output files)
builtin_scene = "sphere" # Builtin scene name (from `cli list-scenes`)
tags = ["basic"]         # Optional: tags for filtering

[[test]]
name = "sponza"
scene_path = "scenes/Sponza.gltf"  # Path to scene file (alternative to builtin_scene)
description = "Complex GLTF scene" # Optional: human-readable description
skip_visual = true                 # Optional: skip visual comparison (perf only)

[test.settings]                    # Optional: override defaults for this test
samples_per_pixel = 4
light_samples = 2
```

### Test Entry Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique test identifier, used for output filenames |
| `builtin_scene` | No* | Name of a builtin scene (from `cli list-scenes`) |
| `scene_path` | No* | Path to a scene file (.gltf, .glb, etc.) |
| `description` | No | Human-readable description |
| `tags` | No | List of strings for filtering |
| `skip_visual` | No | If `true`, skip visual comparison (default: `false`) |
| `settings` | No | Table with `samples_per_pixel` and/or `light_samples` |

*One of `builtin_scene` or `scene_path` should be specified.

## Adding New Test Scenes

1. (For builtin scenes) Add the scene to `all_test_scenes()` in `crates/raytracing/src/scene/scene.rs`
2. Add a `[[test]]` entry to `tests/tests.toml`
3. Rebuild and bless:

```bash
cargo install --path crates/cli
uv run rttest cpu --bless -- -s 1 -l 1
```

