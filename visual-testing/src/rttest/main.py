"""CLI entrypoint for the visual regression test runner."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import runner, diff, bless
from .perf import PerfHistory, PerfBaseline, format_time
from .test_spec import load_test_suite


def get_cli_binary() -> str:
    """Build the bundled CLI via cargo xtask and return its path."""
    workspace_root = Path(__file__).parent.parent.parent.parent
    dist_binary = workspace_root / "dist" / "cli"

    print("Building CLI bundle via cargo xtask...")
    result = subprocess.run(
        ["cargo", "xtask", "bundle", "--release"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(2)
    print("Build complete.")

    return str(dist_binary)


def get_default_tests_file() -> Path:
    """Get the default tests.toml path."""
    return Path(__file__).parent.parent.parent / "tests" / "tests.toml"


def print_results_with_perf(
    results: list,
    perf_mode: bool,
    regressions: list[tuple[str, float]],
    baseline_mode: bool,
):
    """Print test results with optional performance timing."""
    print()
    print("=" * 70)
    print("VISUAL TEST RESULTS" + (" + PERFORMANCE" if perf_mode else ""))
    print("=" * 70)
    print()

    regression_scenes = {scene for scene, _ in regressions}

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
            icon = "\u2713"
        elif result.missing_reference:
            status = "NEW"
            details = "no reference (needs blessing)"
            icon = "?"
        else:
            status = "FAIL"
            details = f"MSE={result.mse:.6e}, max_diff={result.max_diff:.6e}"
            icon = "\u2717"

        # add timing info
        time_str = ""
        if perf_mode and result.render_time_seconds is not None:
            time_str = f"  [{format_time(result.render_time_seconds):>8}]"
            if result.scene in regression_scenes:
                pct = next(p for s, p in regressions if s == result.scene)
                time_str += f" REGRESSION +{pct:.1f}%"
            elif baseline_mode:
                time_str += " (baselined)"

        print(f"  {icon} {result.scene:30} {status:6}{time_str} {details}")

    print()
    print("-" * 70)
    summary = f"  {passed_count} passed, {failed_count} failed, {new_count} new, {error_count} errors"
    if regressions:
        summary += f", {len(regressions)} regressions"
    print(summary)
    print("-" * 70)

    needs_blessing = [r for r in results if not r.passed and not r.error]
    if needs_blessing:
        print()
        print("To review and bless, run with --bless")


def main():
    # split args on -- to separate rttest args from renderer args
    if "--" in sys.argv:
        split_idx = sys.argv.index("--")
        our_args = sys.argv[1:split_idx]
        renderer_args = sys.argv[split_idx+1:]
    else:
        our_args = sys.argv[1:]
        renderer_args = []

    parser = argparse.ArgumentParser(
        description="Visual regression testing for the raytracer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run rttest cpu -- -s 1 -l 1              Run all tests (from tests/tests.toml)
  uv run rttest cpu --scenes sphere           Run only the sphere test
  uv run rttest cpu --bless                   Interactively bless test outputs as new references
  uv run rttest cpu --perf-only -- -s 4 -l 4  Performance only (skip visual comparison)
  uv run rttest cpu --perf-baseline           Set current timings as baseline
  uv run rttest --perf-history sphere         Query timing history for a scene
""",
    )
    parser.add_argument(
        "backend",
        nargs="?",
        choices=["cpu"],
        help="Rendering backend to use",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        help="Comma-separated list of scenes to test (default: all)",
    )
    parser.add_argument(
        "--bless",
        action="store_true",
        help="Interactively review and bless outputs as references",
    )
    parser.add_argument(
        "--bless-all",
        action="store_true",
        help="Bless all outputs without interactive review (use with caution)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON for programmatic consumption",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="MSE tolerance for pass/fail (default: 0.0, exact match required)",
    )
    # Performance testing options
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Disable timing capture (timing is enabled by default)",
    )
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="Performance testing only (skip visual comparison)",
    )
    parser.add_argument(
        "--perf-baseline",
        action="store_true",
        help="Set current timings as the baseline for regression detection",
    )
    parser.add_argument(
        "--perf-history",
        type=str,
        metavar="SCENE",
        help="Query timing history for a specific scene and exit",
    )
    parser.add_argument(
        "--perf-threshold",
        type=float,
        default=10.0,
        help="Regression threshold in percent (default: 10%%)",
    )
    # Test specification
    parser.add_argument(
        "--tests-file",
        type=Path,
        help="Load test specifications from a TOML file (default: tests/tests.toml)",
    )

    args = parser.parse_args(our_args)
    args.renderer_args = renderer_args

    project_dir = Path(__file__).parent.parent.parent
    output_dir = project_dir / "outputs"
    reference_dir = project_dir / "references"
    perf_history_path = project_dir / "perf_history.jsonl"
    perf_baseline_path = project_dir / "perf_baseline.json"

    # Handle --perf-history query (no backend required)
    if args.perf_history:
        history = PerfHistory(perf_history_path)
        records = history.query_scene(args.perf_history)
        if not records:
            print(f"No history found for scene: {args.perf_history}")
            sys.exit(0)
        print(f"Performance history for '{args.perf_history}':")
        print()
        for r in records:
            print(f"  {r.timestamp[:19]}  {r.commit}  {format_time(r.render_time_seconds):>10}  s={r.samples_per_pixel} l={r.light_samples}")
        sys.exit(0)

    if not args.backend:
        parser.error("backend is required (use 'cpu')")

    # perf is on by default unless --no-perf
    perf_mode = not args.no_perf or args.perf_only or args.perf_baseline

    # load test suite from TOML
    tests_file = args.tests_file or get_default_tests_file()
    try:
        suite = load_test_suite(tests_file)
    except FileNotFoundError:
        print(f"error: tests file not found: {tests_file}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"error: failed to load tests file: {e}", file=sys.stderr)
        sys.exit(2)

    # filter by --scenes if specified
    if args.scenes:
        requested = {s.strip() for s in args.scenes.split(",")}
        available = {t.name for t in suite.tests}
        invalid = requested - available
        if invalid:
            print(f"error: unknown scenes: {', '.join(invalid)}", file=sys.stderr)
            print(f"available: {', '.join(sorted(available))}", file=sys.stderr)
            sys.exit(2)
        suite.tests = [t for t in suite.tests if t.name in requested]

    # run the tests
    workspace_root = Path(__file__).parent.parent.parent.parent
    scenes_dir = workspace_root / "scenes"

    results = runner.run_tests(
        tests=suite.tests,
        cli_binary=get_cli_binary(),
        renderer_args=args.renderer_args,
        output_dir=output_dir,
        reference_dir=reference_dir,
        tolerance=args.tolerance,
        bless_mode=args.bless_all,
        perf_mode=perf_mode,
        perf_only=args.perf_only,
        backend=args.backend,
        scenes_dir=scenes_dir,
    )

    # handle performance recording and regression detection
    regressions = []
    if perf_mode:
        history = PerfHistory(perf_history_path)
        baseline = PerfBaseline(perf_baseline_path)

        for result in results:
            if result.perf_record:
                history.append(result.perf_record)

                if args.perf_baseline:
                    baseline.set(result.perf_record)
                else:
                    is_regression, pct = baseline.check_regression(
                        result.perf_record, args.perf_threshold
                    )
                    if is_regression:
                        regressions.append((result.scene, pct))

    # output results
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print_results_with_perf(results, perf_mode, regressions, args.perf_baseline)

    # handle interactive blessing
    if args.bless:
        to_review = [r for r in results if not r.error and not r.passed]
        if to_review:
            bless.interactive_bless(to_review, output_dir, reference_dir)
        else:
            print("\nNo differences to review.")

    # exit with appropriate code
    if any(r.error for r in results):
        sys.exit(2)
    if regressions:
        sys.exit(1)
    if any(not r.passed for r in results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

