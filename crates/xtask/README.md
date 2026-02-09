# xtask

Build automation tasks; currently only used to bundle CLI.

## Setup

Install `patchelf`:

```bash
sudo apt install patchelf
```

## Usage

```bash
# Bundle CPU-only (debug)
cargo xtask bundle

# Release build
cargo xtask bundle --release

# Include OptiX backend
cargo xtask bundle --optix

# Custom output directory
cargo xtask bundle --output-dir my-bundle
```

## What `bundle` does

1. Builds the CLI binary via `cargo build`
2. Copies the binary and all required shared libraries into a self-contained `dist/` directory
3. Patches the binary's rpath to `$ORIGIN` so it finds its libraries without `LD_LIBRARY_PATH`

### Libraries collected

- **Embree**: `libembree4.so` and `libtbb.so` from `$EMBREE_DIR/lib/` (symlinks preserved)
- **OptiX** (with `--optix`): `libraytracing_optix.so` from cargo's build output directory

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `EMBREE_DIR` | Always | Path to Embree 4 installation |
| `OPTIX91_PATH` | Optional | Path to local OptiX 9.1 SDK (auto-downloaded if unset) |
| `CUDACXX` | With `--optix` | Path to `nvcc` |

## Verification

```bash
cargo xtask bundle
ldd dist/cli          # all libs should resolve via $ORIGIN
./dist/cli --help     # should run without LD_LIBRARY_PATH
```
