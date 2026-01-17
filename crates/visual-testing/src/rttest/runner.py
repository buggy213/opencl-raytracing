"""Test runner: invokes the CLI and orchestrates test execution."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import diff


@dataclass
class TestResult:
    scene: str
    passed: bool
    mse: float | None = None
    max_diff: float | None = None
    error: str | None = None  # fatal error (no output produced)
    missing_reference: bool = False  # output exists but no reference to compare
    output_path: Path | None = None
    reference_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "passed": self.passed,
            "mse": self.mse,
            "max_diff": self.max_diff,
            "error": self.error,
            "missing_reference": self.missing_reference,
            "output_path": str(self.output_path) if self.output_path else None,
            "reference_path": str(self.reference_path) if self.reference_path else None,
        }


def run_tests(
    scenes: list[str],
    cli_binary: str,
    renderer_args: list[str],
    output_dir: Path,
    reference_dir: Path,
    tolerance: float,
    bless_mode: bool,
) -> list[TestResult]:
    """Run visual tests for the given scenes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in scenes:
        result = run_single_test(
            scene=scene,
            cli_binary=cli_binary,
            renderer_args=renderer_args,
            output_dir=output_dir,
            reference_dir=reference_dir,
            tolerance=tolerance,
            bless_mode=bless_mode,
        )
        results.append(result)

    return results


def run_single_test(
    scene: str,
    cli_binary: str,
    renderer_args: list[str],
    output_dir: Path,
    reference_dir: Path,
    tolerance: float,
    bless_mode: bool,
) -> TestResult:
    """Run a single visual test."""
    output_path = output_dir / f"{scene}.exr"
    reference_path = reference_dir / f"{scene}.exr"

    # remove stale output from previous runs
    if output_path.exists():
        output_path.unlink()

    # build CLI command
    cmd = [
        cli_binary,
        "--scene-name", scene,
        "-o", str(output_path),
    ]
    cmd.extend(renderer_args)

    # run render
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            scene=scene,
            passed=False,
            error="render timed out (5 min)",
            output_path=output_path,
            reference_path=reference_path,
        )
    except FileNotFoundError:
        return TestResult(
            scene=scene,
            passed=False,
            error=f"cli binary not found: {cli_binary}",
            output_path=output_path,
            reference_path=reference_path,
        )

    if result.returncode != 0:
        return TestResult(
            scene=scene,
            passed=False,
            error=f"render failed: {result.stderr}",
            output_path=output_path,
            reference_path=reference_path,
        )

    if not output_path.exists():
        return TestResult(
            scene=scene,
            passed=False,
            error="render produced no output",
            output_path=output_path,
            reference_path=reference_path,
        )

    # bless mode: copy output to reference
    if bless_mode:
        import shutil
        shutil.copy(output_path, reference_path)
        return TestResult(
            scene=scene,
            passed=True,
            mse=0.0,
            max_diff=0.0,
            output_path=output_path,
            reference_path=reference_path,
        )

    # compare against reference
    if not reference_path.exists():
        return TestResult(
            scene=scene,
            passed=False,
            missing_reference=True,
            output_path=output_path,
            reference_path=reference_path,
        )

    try:
        comparison = diff.compare_images(output_path, reference_path)
    except Exception as e:
        return TestResult(
            scene=scene,
            passed=False,
            error=f"comparison failed: {e}",
            output_path=output_path,
            reference_path=reference_path,
        )

    passed = comparison.mse <= tolerance

    return TestResult(
        scene=scene,
        passed=passed,
        mse=comparison.mse,
        max_diff=comparison.max_diff,
        output_path=output_path,
        reference_path=reference_path,
    )


def print_results(results: list[TestResult]):
    """Print test results in a human-readable format."""
    print()
    print("=" * 60)
    print("VISUAL TEST RESULTS")
    print("=" * 60)
    print()

    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed and not r.error and not r.missing_reference)
    new_count = sum(1 for r in results if r.missing_reference)
    error_count = sum(1 for r in results if r.error)

    for result in results:
        if result.error:
            status = "ERROR"
            details = result.error
            icon = "!"
        elif result.passed:
            status = "PASS"
            details = ""
            icon = "✓"
        elif result.missing_reference:
            status = "NEW"
            details = "no reference (needs blessing)"
            icon = "?"
        else:
            status = "FAIL"
            details = f"MSE={result.mse:.6e}, max_diff={result.max_diff:.6e}"
            icon = "✗"

        print(f"  {icon} {result.scene:30} {status:6} {details}")

    print()
    print("-" * 60)
    print(f"  {passed_count} passed, {failed_count} failed, {new_count} new, {error_count} errors")
    print("-" * 60)

    needs_blessing = [r for r in results if not r.passed and not r.error]
    if needs_blessing:
        print()
        print("To review and bless, run with --bless")
