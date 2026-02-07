"""TOML test specification parsing."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class TestSettings:
    """Per-test render settings.

    Attributes:
        samples_per_pixel: Number of samples per pixel (-s flag)
        light_samples: Number of light samples (-l flag)
    """
    samples_per_pixel: int | None = None
    light_samples: int | None = None


@dataclass
class TestSpec:
    """A single test specification.

    Attributes:
        name: Unique test name, used for output filenames and reporting
        builtin_scene: Name of a builtin scene (from `cli list-scenes`)
        scene_path: Path to a scene file (mutually exclusive with builtin_scene)
        description: Human-readable description of the test
        tags: List of tags for filtering (e.g., ["basic", "materials"])
        skip_visual: If True, skip visual comparison (perf testing only)
        settings: Render settings that override defaults
    """
    name: str
    builtin_scene: str | None = None
    scene_path: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    skip_visual: bool = False
    settings: TestSettings = field(default_factory=TestSettings)

    def get_renderer_args(self, base_args: list[str]) -> list[str]:
        """Build renderer args, overriding with test-specific settings."""
        args = list(base_args)

        if self.settings.samples_per_pixel is not None:
            args = _replace_arg(args, ["-s", "--samples"], str(self.settings.samples_per_pixel))
        if self.settings.light_samples is not None:
            args = _replace_arg(args, ["-l", "--light-samples"], str(self.settings.light_samples))

        return args


def _replace_arg(args: list[str], flags: list[str], value: str) -> list[str]:
    """Replace or append an argument in the args list."""
    result = []
    i = 0
    replaced = False
    while i < len(args):
        if args[i] in flags and i + 1 < len(args):
            result.extend([args[i], value])
            i += 2
            replaced = True
        else:
            result.append(args[i])
            i += 1
    if not replaced:
        result.extend([flags[0], value])
    return result


@dataclass
class TestSuite:
    """A collection of test specifications."""
    tests: list[TestSpec]
    defaults: TestSettings = field(default_factory=TestSettings)

    def filter_by_tags(self, include_tags: list[str] | None, exclude_tags: list[str] | None) -> "TestSuite":
        """Filter tests by tags."""
        tests = self.tests
        if include_tags:
            tests = [t for t in tests if any(tag in t.tags for tag in include_tags)]
        if exclude_tags:
            tests = [t for t in tests if not any(tag in t.tags for tag in exclude_tags)]
        return TestSuite(tests=tests, defaults=self.defaults)


def load_test_suite(path: Path) -> TestSuite:
    """Load a test suite from a TOML file.

    TOML format:

        [defaults]
        samples_per_pixel = 1    # Default samples per pixel for all tests
        light_samples = 1        # Default light samples for all tests

        [[test]]
        name = "sphere"          # Required: unique test name
        builtin_scene = "sphere" # Use a builtin scene (from `cli list-scenes`)
        # OR
        scene_path = "path.gltf" # Use a scene file (mutually exclusive)
        description = "..."      # Optional: human-readable description
        tags = ["basic"]         # Optional: tags for filtering
        skip_visual = false      # Optional: skip visual comparison (perf only)

        [test.settings]          # Optional: override defaults for this test
        samples_per_pixel = 4
        light_samples = 2
    """
    if tomllib is None:
        raise ImportError(
            "tomli is required for Python < 3.11. "
            "Install with: pip install tomli"
        )

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # parse defaults
    defaults_data = data.get("defaults", {})
    defaults = TestSettings(
        samples_per_pixel=defaults_data.get("samples_per_pixel"),
        light_samples=defaults_data.get("light_samples"),
    )

    # parse tests
    tests = []
    for test_data in data.get("test", []):
        settings_data = test_data.get("settings", {})
        settings = TestSettings(
            samples_per_pixel=settings_data.get("samples_per_pixel", defaults.samples_per_pixel),
            light_samples=settings_data.get("light_samples", defaults.light_samples),
        )

        test = TestSpec(
            name=test_data["name"],
            builtin_scene=test_data.get("builtin_scene"),
            scene_path=test_data.get("scene_path"),
            description=test_data.get("description", ""),
            tags=test_data.get("tags", []),
            skip_visual=test_data.get("skip_visual", False),
            settings=settings,
        )
        tests.append(test)

    return TestSuite(tests=tests, defaults=defaults)


