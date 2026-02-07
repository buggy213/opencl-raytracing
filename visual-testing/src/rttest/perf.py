"""Performance tracking: timing capture, history storage, and regression detection."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PerfRecord:
    """A single performance measurement."""
    scene: str
    commit: str
    timestamp: str
    render_time_seconds: float
    settings_hash: str
    samples_per_pixel: int | None
    light_samples: int | None
    backend: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PerfRecord":
        return cls(**data)


def get_git_commit() -> str:
    """Get the current git short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_settings_hash(renderer_args: list[str], backend: str) -> str:
    """Compute a hash of render settings for comparability."""
    key = f"{backend}:{':'.join(sorted(renderer_args))}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def parse_render_args(renderer_args: list[str]) -> tuple[int | None, int | None]:
    """Extract samples_per_pixel and light_samples from renderer args."""
    spp = None
    light = None
    i = 0
    while i < len(renderer_args):
        arg = renderer_args[i]
        if arg in ("-s", "--samples"):
            if i + 1 < len(renderer_args):
                try:
                    spp = int(renderer_args[i + 1])
                except ValueError:
                    pass
            i += 2
        elif arg in ("-l", "--light-samples"):
            if i + 1 < len(renderer_args):
                try:
                    light = int(renderer_args[i + 1])
                except ValueError:
                    pass
            i += 2
        else:
            i += 1
    return spp, light


class PerfHistory:
    """Append-only JSONL history of performance records."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, record: PerfRecord):
        """Append a record to the history file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def load_all(self) -> list[PerfRecord]:
        """Load all records from history."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(PerfRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return records

    def query_scene(self, scene: str, limit: int = 20) -> list[PerfRecord]:
        """Get recent records for a specific scene."""
        all_records = self.load_all()
        scene_records = [r for r in all_records if r.scene == scene]
        return scene_records[-limit:]


@dataclass
class BaselineEntry:
    """A baseline timing for a scene."""
    scene: str
    render_time_seconds: float
    settings_hash: str
    commit: str
    timestamp: str


class PerfBaseline:
    """Blessed baseline timings for regression detection."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, BaselineEntry] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    raw = json.load(f)
                for scene, entry in raw.items():
                    self._data[scene] = BaselineEntry(**entry)
            except (json.JSONDecodeError, TypeError):
                self._data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {scene: asdict(entry) for scene, entry in self._data.items()}
        with open(self.path, "w") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")

    def get(self, scene: str) -> BaselineEntry | None:
        return self._data.get(scene)

    def set(self, record: PerfRecord):
        """Set baseline from a perf record."""
        self._data[record.scene] = BaselineEntry(
            scene=record.scene,
            render_time_seconds=record.render_time_seconds,
            settings_hash=record.settings_hash,
            commit=record.commit,
            timestamp=record.timestamp,
        )
        self._save()

    def check_regression(
        self, record: PerfRecord, threshold_percent: float
    ) -> tuple[bool, float | None]:
        """Check if record represents a regression.

        Returns (is_regression, percent_change).
        Returns (False, None) if no baseline exists or settings don't match.
        """
        baseline = self.get(record.scene)
        if baseline is None:
            return False, None
        if baseline.settings_hash != record.settings_hash:
            return False, None

        if baseline.render_time_seconds <= 0:
            return False, None

        percent_change = (
            (record.render_time_seconds - baseline.render_time_seconds)
            / baseline.render_time_seconds
            * 100
        )
        is_regression = percent_change > threshold_percent
        return is_regression, percent_change


def format_time(seconds: float) -> str:
    """Format time for display."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}us"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def create_perf_record(
    scene: str,
    render_time_seconds: float,
    renderer_args: list[str],
    backend: str,
) -> PerfRecord:
    """Create a performance record with current git state."""
    spp, light = parse_render_args(renderer_args)
    return PerfRecord(
        scene=scene,
        commit=get_git_commit(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        render_time_seconds=render_time_seconds,
        settings_hash=compute_settings_hash(renderer_args, backend),
        samples_per_pixel=spp,
        light_samples=light,
        backend=backend,
    )
