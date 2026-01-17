"""CLI entrypoint for the visual regression test runner."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import runner, diff, bless


def get_cli_binary() -> str:
    """Get the path to the cli binary."""
    return "cli"


def get_available_scenes() -> list[str]:
    """Query the CLI for available builtin test scenes."""
    result = subprocess.run(
        [get_cli_binary(), "list-scenes"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"error: failed to list scenes: {result.stderr}", file=sys.stderr)
        sys.exit(2)
    return json.loads(result.stdout)


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
  uv run rttest cpu -- -s 1 -l 1       Run all tests with 1 spp, 1 light sample
  uv run rttest cpu --scenes sphere    Run only the sphere test
  uv run rttest cpu --bless            Bless all test outputs as new references
  uv run rttest --list                 List available test scenes
""",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test scenes and exit",
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

    args = parser.parse_args(our_args)
    args.renderer_args = renderer_args

    if args.list:
        scenes = get_available_scenes()
        print("Available test scenes:")
        for scene in scenes:
            print(f"  {scene}")
        return

    if not args.backend:
        parser.error("backend is required (use 'cpu')")

    # determine which scenes to test
    all_scenes = get_available_scenes()
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]
        invalid = set(scenes) - set(all_scenes)
        if invalid:
            print(f"error: unknown scenes: {', '.join(invalid)}", file=sys.stderr)
            sys.exit(2)
    else:
        scenes = all_scenes

    # run the tests
    project_dir = Path(__file__).parent.parent.parent
    output_dir = project_dir / "outputs"
    reference_dir = project_dir / "references"
    
    results = runner.run_tests(
        scenes=scenes,
        cli_binary=get_cli_binary(),
        renderer_args=args.renderer_args,
        output_dir=output_dir,
        reference_dir=reference_dir,
        tolerance=args.tolerance,
        bless_mode=args.bless_all,  # only auto-bless with --bless-all
    )

    # output results
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        runner.print_results(results)

    # handle interactive blessing
    if args.bless:
        # bless mode: review outputs that differ or are new (exclude fatal errors)
        to_review = [r for r in results if not r.error and not r.passed]
        if to_review:
            bless.interactive_bless(to_review, output_dir, reference_dir)
        else:
            print("\nNo differences to review.")

    # exit with appropriate code
    if any(r.error for r in results):
        sys.exit(2)
    if any(not r.passed for r in results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

