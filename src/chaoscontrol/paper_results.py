"""Paper results registry — append-only JSONL index of runs eligible for
paper tables, tagged as confirmatory or exploratory.

See ``paper_results/README.md`` for the schema and the rationale.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Literal

SCHEMA_VERSION = 1

Status = Literal["exploratory", "confirmatory"]
_VALID_STATUSES: frozenset[str] = frozenset({"exploratory", "confirmatory"})

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "paper_results" / "registry.jsonl"


@dataclass
class RunRecord:
    experiment: str
    condition: str
    seed: int
    status: Status
    metrics: dict
    config_hash: str
    git_sha: str = ""
    git_dirty: bool = False
    timestamp: str = ""
    artifacts: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"invalid status: {self.status!r} "
                f"(expected one of {sorted(_VALID_STATUSES)})"
            )
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def key(self) -> tuple[str, str, int, str]:
        return (self.experiment, self.condition, self.seed, self.status)


def _git_run(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True, text=True, check=True,
        )
        return out.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _current_git_sha() -> str:
    out = _git_run(["rev-parse", "HEAD"])
    return out.strip() if out is not None else "unknown"


def _current_git_dirty() -> bool:
    out = _git_run(["status", "--porcelain"])
    # If git is unavailable, err on the safe side and mark dirty.
    return True if out is None else bool(out.strip())


def _resolve_path(path: Path | str | None) -> Path:
    return Path(path) if path is not None else DEFAULT_REGISTRY_PATH


def register(
    *,
    experiment: str,
    condition: str,
    seed: int,
    status: Status,
    metrics: dict,
    config_hash: str,
    artifacts: Iterable[str] = (),
    extras: dict | None = None,
    registry_path: Path | str | None = None,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> RunRecord:
    """Append a run record to the registry. Auto-fills git_sha/git_dirty/timestamp
    unless explicitly provided (tests pass explicit values to avoid subprocess)."""
    rec = RunRecord(
        experiment=experiment,
        condition=condition,
        seed=seed,
        status=status,
        metrics=dict(metrics),
        config_hash=config_hash,
        git_sha=git_sha if git_sha is not None else _current_git_sha(),
        git_dirty=git_dirty if git_dirty is not None else _current_git_dirty(),
        artifacts=list(artifacts),
        extras=dict(extras) if extras else {},
    )
    path = _resolve_path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
    return rec


def load(registry_path: Path | str | None = None) -> list[RunRecord]:
    path = _resolve_path(registry_path)
    if not path.exists():
        return []
    records: list[RunRecord] = []
    with path.open() as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            d.pop("schema_version", None)
            records.append(RunRecord(**d))
    return records


def query(
    *,
    experiment: str | None = None,
    condition: str | None = None,
    status: Status | None = None,
    registry_path: Path | str | None = None,
) -> list[RunRecord]:
    records = load(registry_path)
    if experiment is not None:
        records = [r for r in records if r.experiment == experiment]
    if condition is not None:
        records = [r for r in records if r.condition == condition]
    if status is not None:
        records = [r for r in records if r.status == status]
    return records


def verify(registry_path: Path | str | None = None) -> dict:
    """Sanity-check the registry. Raises ``ValueError`` on

    (a) duplicate ``(experiment, condition, seed, status)`` keys, or
    (b) any confirmatory record committed from a dirty working tree.

    Exploratory runs MAY be dirty — research scratchpads shouldn't
    require a clean commit. But confirmatory records are paper-binding,
    so a dirty tree there means the ``git_sha`` on file doesn't fully
    pin the code that produced the measurement. Downgrade to
    exploratory, or re-register after committing.

    Returns a summary dict otherwise."""
    records = load(registry_path)
    seen: dict[tuple[str, str, int, str], int] = {}
    duplicates: list[tuple[tuple[str, str, int, str], int, int]] = []
    for i, r in enumerate(records):
        k = r.key()
        if k in seen:
            duplicates.append((k, seen[k], i))
        else:
            seen[k] = i
    if duplicates:
        pretty = ", ".join(
            f"{k} (lines {a + 1} and {b + 1})" for k, a, b in duplicates
        )
        raise ValueError(f"duplicate registry keys: {pretty}")
    dirty_confirmatory = [
        (i, r) for i, r in enumerate(records)
        if r.git_dirty and r.status == "confirmatory"
    ]
    if dirty_confirmatory:
        pretty = ", ".join(
            f"{r.experiment}/{r.condition} seed={r.seed} (line {i + 1})"
            for i, r in dirty_confirmatory
        )
        raise ValueError(
            f"confirmatory records committed from dirty tree: {pretty}. "
            "Commit and re-register, or downgrade status to exploratory."
        )
    return {
        "n_records": len(records),
        "experiments": sorted({r.experiment for r in records}),
        "confirmatory": sum(1 for r in records if r.status == "confirmatory"),
        "exploratory": sum(1 for r in records if r.status == "exploratory"),
        "dirty_records": sum(1 for r in records if r.git_dirty),
    }


__all__ = [
    "RunRecord",
    "Status",
    "SCHEMA_VERSION",
    "DEFAULT_REGISTRY_PATH",
    "register",
    "load",
    "query",
    "verify",
]
