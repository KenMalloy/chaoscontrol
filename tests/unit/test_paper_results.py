from __future__ import annotations

from pathlib import Path

import pytest

from chaoscontrol.paper_results import (
    RunRecord,
    load,
    query,
    register,
    verify,
)


def _reg(path: Path, **overrides):
    base = dict(
        experiment="exp21",
        condition="C_ssm_random",
        seed=1337,
        status="confirmatory",
        metrics={"bpb": 1.492, "wall_clock_s": 600.0},
        config_hash="sha256:abc123",
        registry_path=path,
        git_sha="fakesha",
        git_dirty=False,
    )
    base.update(overrides)
    return register(**base)


def test_register_and_load_roundtrip(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    _reg(reg)
    records = load(reg)
    assert len(records) == 1
    r = records[0]
    assert r.experiment == "exp21"
    assert r.condition == "C_ssm_random"
    assert r.seed == 1337
    assert r.status == "confirmatory"
    assert r.metrics["bpb"] == pytest.approx(1.492)
    assert r.git_sha == "fakesha"
    assert r.git_dirty is False
    # timestamp auto-populated
    assert r.timestamp.endswith("Z")


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load(tmp_path / "does_not_exist.jsonl") == []


def test_query_filters(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    for seed in (1337, 42):
        _reg(reg, seed=seed)
    _reg(
        reg,
        experiment="exp19",
        condition="bf16_all",
        seed=1337,
        status="exploratory",
        metrics={"bpb": 1.7},
    )
    assert len(query(experiment="exp21", registry_path=reg)) == 2
    assert len(query(status="confirmatory", registry_path=reg)) == 2
    assert len(query(experiment="exp19", registry_path=reg)) == 1
    assert (
        len(query(experiment="exp21", condition="C_ssm_random", registry_path=reg))
        == 2
    )


def test_verify_summary(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    _reg(reg)
    _reg(
        reg,
        experiment="exp19",
        condition="bf16_all",
        seed=1,
        status="exploratory",
        metrics={"bpb": 1.7},
        git_dirty=True,
    )
    summary = verify(registry_path=reg)
    assert summary["n_records"] == 2
    assert summary["confirmatory"] == 1
    assert summary["exploratory"] == 1
    assert summary["dirty_records"] == 1
    assert sorted(summary["experiments"]) == ["exp19", "exp21"]


def test_verify_detects_duplicates(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    _reg(reg)
    _reg(reg)  # same (experiment, condition, seed, status)
    with pytest.raises(ValueError, match="duplicate registry keys"):
        verify(registry_path=reg)


def test_verify_allows_same_key_with_different_status(tmp_path: Path) -> None:
    """An exploratory and a confirmatory run at the same seed are distinct —
    e.g., the same config can be re-run for the paper after an initial
    exploratory pass. Uniqueness is on the full 4-tuple."""
    reg = tmp_path / "registry.jsonl"
    _reg(reg, status="exploratory")
    _reg(reg, status="confirmatory")
    summary = verify(registry_path=reg)
    assert summary["n_records"] == 2


def test_verify_raises_on_dirty_confirmatory(tmp_path: Path) -> None:
    """Confirmatory records are paper-binding — a dirty tree means the
    ``git_sha`` on file doesn't fully pin the code that produced the
    measurement. ``verify()`` must fail loud so a downstream paper-stats
    pipeline can't silently inherit un-reproducible rows."""
    reg = tmp_path / "registry.jsonl"
    _reg(reg, status="confirmatory", git_dirty=True)
    with pytest.raises(ValueError, match="dirty tree"):
        verify(registry_path=reg)


def test_verify_allows_dirty_exploratory(tmp_path: Path) -> None:
    """Exploratory runs MAY be dirty — researcher scratchpads shouldn't
    require a clean commit before every data point. ``verify()`` returns
    the summary with ``dirty_records`` > 0 and no raise."""
    reg = tmp_path / "registry.jsonl"
    _reg(reg, status="exploratory", git_dirty=True)
    summary = verify(registry_path=reg)
    assert summary["dirty_records"] == 1
    assert summary["exploratory"] == 1


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValueError, match="invalid status"):
        RunRecord(
            experiment="e",
            condition="c",
            seed=1,
            status="bogus",  # type: ignore[arg-type]
            metrics={},
            config_hash="h",
        )


def test_corrupt_line_raises(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    reg.write_text("not-json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load(reg)


def test_empty_lines_ignored(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    _reg(reg)
    # Simulate a manual edit that left a blank line.
    existing = reg.read_text()
    reg.write_text(existing + "\n\n")
    assert len(load(reg)) == 1
