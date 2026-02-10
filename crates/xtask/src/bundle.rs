use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, bail};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "reason")]
enum CargoMessage {
    #[serde(rename = "compiler-artifact")]
    CompilerArtifactMessage(CompilerArtifactMessage),
    #[serde(rename = "build-script-executed")]
    BuildScriptMessage(BuildScriptMessage)
}

#[derive(Deserialize)]
struct CompilerArtifactMessage {
    target: CargoTarget,
    executable: String
}

#[derive(Deserialize)]
struct BuildScriptMessage {
    package_id: String,
    linked_paths: Vec<String>
}

#[derive(Deserialize)]
struct CargoTarget {
    name: String,
    kind: Vec<String>,
}

pub fn run(release: bool, optix: bool, output_dir: &Path) -> anyhow::Result<()> {
    check_patchelf()?;

    let workspace_root = workspace_root()?;
    let output_dir = if output_dir.is_relative() {
        workspace_root.join(output_dir)
    } else {
        output_dir.to_path_buf()
    };

    let (binary_path, optix_linked_path) = cargo_build(release, optix)?;

    let binary_path = binary_path.context("cargo build did not produce a CLI binary")?;

    if output_dir.exists() {
        fs::remove_dir_all(&output_dir)
            .with_context(|| format!("failed to clean {}", output_dir.display()))?;
    }
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    // Copy binary
    let dest_binary = output_dir.join("cli");
    fs::copy(&binary_path, &dest_binary).context("failed to copy binary")?;

    // Copy embree libs
    copy_embree_libs(&output_dir)?;

    // Copy optix lib
    if optix {
        copy_optix_lib(&output_dir, optix_linked_path.as_deref(), release)?;
    }

    // Patch rpath
    let status = Command::new("patchelf")
        .args(["--set-rpath", "$ORIGIN"])
        .arg(&dest_binary)
        .status()
        .context("failed to run patchelf")?;
    if !status.success() {
        bail!("patchelf failed with {status}");
    }

    print_summary(&output_dir)?;

    Ok(())
}

fn check_patchelf() -> anyhow::Result<()> {
    let output = Command::new("patchelf").arg("--version").output();
    match output {
        Ok(o) if o.status.success() => Ok(()),
        _ => bail!("patchelf is not installed or not on PATH. Install it with your package manager."),
    }
}

fn workspace_root() -> anyhow::Result<PathBuf> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version=1"])
        .stderr(Stdio::inherit())
        .output()
        .context("failed to run cargo metadata")?;

    #[derive(Deserialize)]
    struct Meta {
        workspace_root: PathBuf,
    }
    let meta: Meta = serde_json::from_slice(&output.stdout).context("failed to parse cargo metadata")?;
    Ok(meta.workspace_root)
}

fn cargo_build(release: bool, optix: bool) -> anyhow::Result<(Option<PathBuf>, Option<PathBuf>)> {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "-p", "cli", "--message-format=json"]);

    if release {
        cmd.arg("--release");
    }
    if optix {
        cmd.args(["--features", "optix"]);
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit());

    let mut child = cmd.spawn().context("failed to spawn cargo build")?;
    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    let mut binary_path: Option<PathBuf> = None;
    let mut optix_linked_path: Option<PathBuf> = None;

    for line in reader.lines() {
        let line = line.context("failed to read cargo output")?;

        let msg: CargoMessage = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(_) => continue,
        };

        match msg {
            CargoMessage::CompilerArtifactMessage(CompilerArtifactMessage { target, executable }) => {
                if target.name == "cli" && target.kind.contains(&"bin".to_string()) {
                    binary_path = Some(PathBuf::from(executable));
                }
            },
            CargoMessage::BuildScriptMessage(BuildScriptMessage { package_id, linked_paths }) => {
                if package_id.contains("raytracing-optix") {
                    for p in &linked_paths {
                        let p = p.strip_prefix("native=").unwrap_or(p);
                        let candidate = PathBuf::from(p);
                        if candidate.exists() {
                            optix_linked_path = Some(candidate);
                            break;
                        }
                    }
                }
            }
        }
    }

    let status = child.wait().context("cargo build failed")?;
    if !status.success() {
        bail!("cargo build exited with {status}");
    }

    Ok((binary_path, optix_linked_path))
}

fn copy_embree_libs(output_dir: &Path) -> anyhow::Result<()> {
    let embree_dir =
        std::env::var("EMBREE_DIR").context("EMBREE_DIR environment variable is not set")?;
    let lib_dir = PathBuf::from(&embree_dir).join("lib");

    if !lib_dir.is_dir() {
        bail!("{} is not a directory", lib_dir.display());
    }

    for entry in fs::read_dir(&lib_dir).context("failed to read EMBREE_DIR/lib")? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if !name_str.starts_with("lib") || !name_str.contains(".so") {
            continue;
        }

        let src = entry.path();
        let dest = output_dir.join(&name);

        if src.is_symlink() {
            let link_target = fs::read_link(&src)
                .with_context(|| format!("failed to read symlink {}", src.display()))?;
            // Recreate relative symlink in dist
            #[cfg(unix)]
            std::os::unix::fs::symlink(&link_target, &dest)
                .with_context(|| format!("failed to create symlink {}", dest.display()))?;
        } else if src.is_file() {
            fs::copy(&src, &dest)
                .with_context(|| format!("failed to copy {}", src.display()))?;
        }
    }

    Ok(())
}

fn copy_optix_lib(
    output_dir: &Path,
    linked_path: Option<&Path>,
    release: bool,
) -> anyhow::Result<()> {
    let so_name = "libraytracing_optix.so";

    // Try the linked_path from cargo build JSON first
    if let Some(dir) = linked_path {
        let candidate = dir.join(so_name);
        if candidate.is_file() {
            let dest = output_dir.join(so_name);
            fs::copy(&candidate, &dest)
                .with_context(|| format!("failed to copy {}", candidate.display()))?;
            return Ok(());
        }
    }

    // Fallback: glob for the library in target build dirs
    let workspace_root = workspace_root()?;
    let profile = if release { "release" } else { "debug" };
    let search_dir = workspace_root.join("target").join(profile).join("build");

    if !search_dir.is_dir() {
        bail!("could not find {so_name}: no linked_path from cargo and {search_dir:?} doesn't exist");
    }

    let mut candidates: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
    for entry in fs::read_dir(&search_dir)? {
        let entry = entry?;
        let dir_name = entry.file_name();
        if !dir_name.to_string_lossy().starts_with("raytracing-optix-") {
            continue;
        }
        let so_path = entry.path().join("out").join(so_name);
        if so_path.is_file() {
            let mtime = fs::metadata(&so_path)?.modified()?;
            candidates.push((so_path, mtime));
        }
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    if let Some((best, _)) = candidates.first() {
        let dest = output_dir.join(so_name);
        fs::copy(best, &dest)
            .with_context(|| format!("failed to copy {}", best.display()))?;
        Ok(())
    } else {
        bail!("could not find {so_name} in {search_dir:?}; try a clean build with --optix");
    }
}

fn print_summary(output_dir: &Path) -> anyhow::Result<()> {
    println!();
    println!("Bundled into {}:", output_dir.display());
    println!("{:<40} {:>10}", "File", "Size");
    println!("{}", "-".repeat(52));

    let mut entries: Vec<_> = fs::read_dir(output_dir)?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let path = entry.path();

        if path.is_symlink() {
            let target = fs::read_link(&path)?;
            println!("  {name_str} -> {}", target.display());
        } else {
            let size = fs::metadata(&path)?.len();
            println!("  {name_str:<38} {}", format_size(size));
        }
    }

    println!();
    println!("Run with: {}/cli --help", output_dir.display());

    Ok(())
}

fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;

    if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}
