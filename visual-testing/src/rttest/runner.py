"""Test runner: invokes the CLI and orchestrates test execution."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import diff
from .perf import PerfRecord
from .test_spec import TestSpec


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
    render_time_seconds: float | None = None
    perf_record: PerfRecord | None = None

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
            "render_time_seconds": self.render_time_seconds,
        }


def run_tests(
    tests: list[TestSpec],
    cli_binary: str,
    renderer_args: list[str],
    output_dir: Path,
    reference_dir: Path,
    tolerance: float,
    bless_mode: bool,
    perf_mode: bool = False,
    perf_only: bool = False,
    backend: str = "cpu",
    scenes_dir: Path | None = None,
) -> list[TestResult]:
    """Run visual tests for the given test specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for test in tests:
        result = run_single_test(
            test=test,
            cli_binary=cli_binary,
            renderer_args=renderer_args,
            output_dir=output_dir,
            reference_dir=reference_dir,
            tolerance=tolerance,
            bless_mode=bless_mode,
            perf_mode=perf_mode,
            perf_only=perf_only,
            backend=backend,
            scenes_dir=scenes_dir,
        )
        results.append(result)

    return results


def run_single_test(
    test: TestSpec,
    cli_binary: str,
    renderer_args: list[str],
    output_dir: Path,
    reference_dir: Path,
    tolerance: float,
    bless_mode: bool,
    perf_mode: bool = False,
    perf_only: bool = False,
    backend: str = "cpu",
    scenes_dir: Path | None = None,
) -> TestResult:
    """Run a single visual test."""
    from .perf import create_perf_record

    scene = test.name
    output_path = output_dir / f"{scene}.exr"
    reference_path = reference_dir / f"{scene}.exr"

    # remove stale output from previous runs
    if output_path.exists():
        output_path.unlink()

    # build CLI command based on test spec
    cmd = [cli_binary]

    if test.builtin_scene:
        cmd.extend(["--scene-name", test.builtin_scene])
    elif test.scene_path:
        scene_path = test.scene_path
        if scenes_dir and not Path(scene_path).is_absolute():
            scene_path = str(scenes_dir / scene_path)
        cmd.extend(["--scene-path", scene_path])
    else:
        return TestResult(
            scene=scene,
            passed=False,
            error="test has neither builtin_scene nor scene_path",
            output_path=output_path,
            reference_path=reference_path,
        )

    cmd.extend(["-o", str(output_path)])

    # apply per-test settings on top of base renderer args
    final_renderer_args = test.get_renderer_args(renderer_args)
    cmd.extend(final_renderer_args)

    # run render with timing
    render_time: float | None = None
    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        render_time = time.perf_counter() - start_time
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
            render_time_seconds=render_time,
        )

    if not output_path.exists():
        return TestResult(
            scene=scene,
            passed=False,
            error="render produced no output",
            output_path=output_path,
            reference_path=reference_path,
            render_time_seconds=render_time,
        )

    # create perf record if in perf mode
    perf_record = None
    if perf_mode and render_time is not None:
        perf_record = create_perf_record(
            scene=scene,
            render_time_seconds=render_time,
            renderer_args=final_renderer_args,
            backend=backend,
        )

    # perf-only mode: skip visual comparison
    if perf_only:
        return TestResult(
            scene=scene,
            passed=True,
            output_path=output_path,
            reference_path=reference_path,
            render_time_seconds=render_time,
            perf_record=perf_record,
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
            render_time_seconds=render_time,
            perf_record=perf_record,
        )

    # compare against reference
    if not reference_path.exists():
        return TestResult(
            scene=scene,
            passed=False,
            missing_reference=True,
            output_path=output_path,
            reference_path=reference_path,
            render_time_seconds=render_time,
            perf_record=perf_record,
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
            render_time_seconds=render_time,
            perf_record=perf_record,
        )

    passed = comparison.mse <= tolerance

    return TestResult(
        scene=scene,
        passed=passed,
        mse=comparison.mse,
        max_diff=comparison.max_diff,
        output_path=output_path,
        reference_path=reference_path,
        render_time_seconds=render_time,
        perf_record=perf_record,
    )


