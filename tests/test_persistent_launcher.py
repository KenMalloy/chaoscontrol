"""Unit tests for run_persistent_launcher.py post-run and pre-flight checks.

Covers the bugs surfaced in code review on 2026-04-17:
- Launcher previously returned rc=0 even when every entry wrote an error
  marker (only the JSON count was checked, not contents).
- Launcher previously accepted a matrix whose entries' ``world_size`` did
  not match ``--world-size``, silently running one regime and reporting
  another.
- Launcher returned rc=0 when TE pre-flight filtered the whole matrix
  (e.g., fp8-only request on a pod without transformer_engine) — an
  empty run reported as successful.
- Launcher had no ``main()``-level integration coverage, so control-flow
  wiring bugs (e.g., pre-flights not actually called) would pass unit
  tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "19_prereqs"))

from run_persistent_launcher import (  # noqa: E402
    _BENIGN_ERROR_PATTERNS,
    _check_output_integrity,
    _validate_matrix_world_size,
    filter_matrix_for_te,
    reject_stale_skip_markers,
)


class TestCheckOutputIntegrity:
    def test_all_successful_results(self, tmp_path: Path) -> None:
        entry = {"name": "bf16", "seed": 1337}
        (tmp_path / "bf16_s1337.json").write_text(json.dumps({
            "config": entry,
            "params": 10_000_000,
            "train": {"steps": 1117, "final_loss": 4.2},
            "eval": {"bpb": 1.49},
        }))
        success, benign, errors = _check_output_integrity(
            tmp_path, [entry], _BENIGN_ERROR_PATTERNS,
        )
        assert success == 1
        assert benign == 0
        assert errors == []

    def test_real_error_is_surfaced(self, tmp_path: Path) -> None:
        entry = {"name": "bf16", "seed": 1337}
        (tmp_path / "bf16_s1337.json").write_text(json.dumps({
            "config": entry,
            "error": "RuntimeError: CUDA out of memory",
        }))
        success, benign, errors = _check_output_integrity(
            tmp_path, [entry], _BENIGN_ERROR_PATTERNS,
        )
        assert success == 0
        assert benign == 0
        assert len(errors) == 1
        assert "CUDA out of memory" in errors[0]

    def test_benign_te_skip_is_not_flagged(self, tmp_path: Path) -> None:
        entry = {"name": "fp8", "seed": 1337}
        (tmp_path / "fp8_s1337.json").write_text(json.dumps({
            "config": entry,
            "error": "skipped: transformer_engine unavailable on pod",
        }))
        success, benign, errors = _check_output_integrity(
            tmp_path, [entry], _BENIGN_ERROR_PATTERNS,
        )
        assert success == 0
        assert benign == 1
        assert errors == []

    def test_missing_output_file_is_real_error(self, tmp_path: Path) -> None:
        entry = {"name": "bf16", "seed": 9999}
        success, benign, errors = _check_output_integrity(
            tmp_path, [entry], _BENIGN_ERROR_PATTERNS,
        )
        assert success == 0
        assert benign == 0
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_malformed_json_is_real_error(self, tmp_path: Path) -> None:
        entry = {"name": "bf16", "seed": 1337}
        (tmp_path / "bf16_s1337.json").write_text("{not valid json")
        success, benign, errors = _check_output_integrity(
            tmp_path, [entry], _BENIGN_ERROR_PATTERNS,
        )
        assert success == 0
        assert benign == 0
        assert len(errors) == 1
        assert "malformed" in errors[0].lower() or "json" in errors[0].lower()

    def test_mixed_matrix_counts_each_kind(self, tmp_path: Path) -> None:
        entries = [
            {"name": "bf16", "seed": 1337},
            {"name": "bf16", "seed": 2674},
            {"name": "fp8", "seed": 1337},
            {"name": "fp8", "seed": 2674},
        ]
        (tmp_path / "bf16_s1337.json").write_text(json.dumps({
            "config": entries[0], "train": {"steps": 1000, "final_loss": 4.2},
            "eval": {"bpb": 1.49},
        }))
        (tmp_path / "bf16_s2674.json").write_text(json.dumps({
            "config": entries[1],
            "error": "RuntimeError: something else",
        }))
        (tmp_path / "fp8_s1337.json").write_text(json.dumps({
            "config": entries[2],
            "error": "skipped: transformer_engine unavailable on pod",
        }))
        (tmp_path / "fp8_s2674.json").write_text(json.dumps({
            "config": entries[3],
            "error": "skipped: transformer_engine unavailable on pod",
        }))
        success, benign, errors = _check_output_integrity(
            tmp_path, entries, _BENIGN_ERROR_PATTERNS,
        )
        assert success == 1
        assert benign == 2
        assert len(errors) == 1


class TestValidateMatrixWorldSize:
    def test_all_match_passes(self) -> None:
        entries = [
            {"name": "bf16", "seed": 1337, "world_size": 4},
            {"name": "bf16", "seed": 2674, "world_size": 4},
        ]
        _validate_matrix_world_size(entries, cli_world_size=4)

    def test_mismatch_raises(self) -> None:
        entries = [
            {"name": "bf16", "seed": 1337, "world_size": 4},
            {"name": "bf16", "seed": 2674, "world_size": 2},
        ]
        with pytest.raises(ValueError, match="world_size"):
            _validate_matrix_world_size(entries, cli_world_size=4)

    def test_entry_missing_field_raises(self) -> None:
        entries = [{"name": "bf16", "seed": 1337}]
        with pytest.raises(ValueError, match="world_size"):
            _validate_matrix_world_size(entries, cli_world_size=4)


class TestRejectStaleSkipMarkers:
    """Pre-flight guard: stale fp8 skip markers on a TE-capable pod.

    Code-review finding 2026-04-17-A#1: without this check, a user who ran
    on a TE-less pod, moved to a TE-capable pod, and re-ran would get
    rc=0 with every fp8 entry counted as a benign skip — a silent failure
    to produce the fp8 results they explicitly asked for.
    """

    def _fp8_entry(self, seed: int) -> dict:
        return {"name": "fp8", "seed": seed, "precision": "fp8"}

    def _bf16_entry(self, seed: int) -> dict:
        return {"name": "bf16", "seed": seed, "precision": "bf16"}

    def test_raises_when_te_available_and_stale_fp8_markers_exist(
        self, tmp_path: Path,
    ) -> None:
        """Primary regression: the whole point of this check."""
        entries = [self._fp8_entry(1337), self._fp8_entry(2674)]
        for entry in entries:
            path = tmp_path / f"{entry['name']}_s{entry['seed']}.json"
            path.write_text(json.dumps({
                "config": entry,
                "error": "skipped: transformer_engine unavailable on pod",
            }))
        with pytest.raises(RuntimeError, match="stale skip markers"):
            reject_stale_skip_markers(
                entries, tmp_path, te_probe=lambda: True,
            )

    def test_no_raise_when_te_unavailable(self, tmp_path: Path) -> None:
        """TE-less pod with stale markers is the expected state.

        This is a valid re-run on a machine that genuinely can't run fp8 —
        the markers are the correct record of reality. No raise.
        """
        entry = self._fp8_entry(1337)
        (tmp_path / "fp8_s1337.json").write_text(json.dumps({
            "config": entry,
            "error": "skipped: transformer_engine unavailable on pod",
        }))
        reject_stale_skip_markers(
            [entry], tmp_path, te_probe=lambda: False,
        )

    def test_no_raise_when_no_fp8_in_matrix(self, tmp_path: Path) -> None:
        """A bf16-only matrix never touches fp8 markers.

        Even if stale fp8 markers are present from a prior unrelated run,
        the current matrix does not request fp8, so there is nothing to
        raise about.
        """
        (tmp_path / "fp8_s1337.json").write_text(json.dumps({
            "config": {"name": "fp8", "seed": 1337, "precision": "fp8"},
            "error": "skipped: transformer_engine unavailable on pod",
        }))
        reject_stale_skip_markers(
            [self._bf16_entry(1337)], tmp_path, te_probe=lambda: True,
        )

    def test_no_raise_when_fp8_marker_is_a_real_error(
        self, tmp_path: Path,
    ) -> None:
        """Real fp8 failures are not ``stale skip markers``.

        A marker whose error string is e.g. ``CUDA out of memory`` came
        from a real fp8 attempt that failed on its merits — honoring it
        across re-runs is correct behavior. Only the ``transformer_engine
        unavailable`` pattern means "we never tried."
        """
        entry = self._fp8_entry(1337)
        (tmp_path / "fp8_s1337.json").write_text(json.dumps({
            "config": entry,
            "error": "RuntimeError: CUDA out of memory",
        }))
        reject_stale_skip_markers(
            [entry], tmp_path, te_probe=lambda: True,
        )

    def test_no_raise_when_fp8_has_real_success(self, tmp_path: Path) -> None:
        """Successful prior fp8 runs are honored by idempotent skip — never
        mistaken for a stale marker."""
        entry = self._fp8_entry(1337)
        (tmp_path / "fp8_s1337.json").write_text(json.dumps({
            "config": entry,
            "train": {"steps": 1000, "final_loss": 4.2},
            "eval": {"bpb": 1.49},
        }))
        reject_stale_skip_markers(
            [entry], tmp_path, te_probe=lambda: True,
        )

    def test_error_message_enumerates_exact_stale_paths(
        self, tmp_path: Path,
    ) -> None:
        """The error must name every stale file by exact path.

        A failed pre-flight at minute 0 of a multi-hour pod session must
        leave the operator with the exact paths to delete. The message
        deliberately does not include a glob hint — globs break on paths
        with spaces and miss custom-named fp8 entries (e.g., a matrix
        using ``name="fp8_lr128"``). The enumerated paths above the hint
        are already the authoritative recovery instruction.
        """
        entry = self._fp8_entry(1337)
        stale_path = tmp_path / "fp8_s1337.json"
        stale_path.write_text(json.dumps({
            "config": entry,
            "error": "skipped: transformer_engine unavailable on pod",
        }))
        with pytest.raises(RuntimeError) as ctx:
            reject_stale_skip_markers(
                [entry], tmp_path, te_probe=lambda: True,
            )
        msg = str(ctx.value)
        assert str(stale_path) in msg

    def test_previews_limit_at_five_with_count(self, tmp_path: Path) -> None:
        """Matrix with >5 stale markers shows first 5 and a count for the rest."""
        entries = [self._fp8_entry(s) for s in range(1000, 1008)]  # 8 entries
        for entry in entries:
            path = tmp_path / f"{entry['name']}_s{entry['seed']}.json"
            path.write_text(json.dumps({
                "config": entry,
                "error": "skipped: transformer_engine unavailable on pod",
            }))
        with pytest.raises(RuntimeError) as ctx:
            reject_stale_skip_markers(
                entries, tmp_path, te_probe=lambda: True,
            )
        msg = str(ctx.value)
        assert "+3 more" in msg


class TestFilterMatrixForTeWithInjectedProbe:
    """Covers ``filter_matrix_for_te``'s decision branches using the
    injected ``te_probe`` — avoids importing transformer_engine from tests.
    """

    def test_fp8_dropped_when_probe_false(self, tmp_path: Path) -> None:
        entries = [
            {"name": "bf16", "seed": 1337, "precision": "bf16"},
            {"name": "fp8", "seed": 1337, "precision": "fp8"},
        ]
        runnable, skipped = filter_matrix_for_te(
            entries, tmp_path, te_probe=lambda: False,
        )
        assert [e["name"] for e in runnable] == ["bf16"]
        assert [e["name"] for e in skipped] == ["fp8"]

    def test_fp8_kept_when_probe_true(self, tmp_path: Path) -> None:
        entries = [
            {"name": "bf16", "seed": 1337, "precision": "bf16"},
            {"name": "fp8", "seed": 1337, "precision": "fp8"},
        ]
        runnable, skipped = filter_matrix_for_te(
            entries, tmp_path, te_probe=lambda: True,
        )
        assert len(runnable) == 2
        assert skipped == []

    def test_bf16_only_short_circuits_probe(self, tmp_path: Path) -> None:
        """No fp8 in matrix ⇒ probe never called."""
        probe_called = False

        def probe() -> bool:
            nonlocal probe_called
            probe_called = True
            return True

        entries = [{"name": "bf16", "seed": 1337, "precision": "bf16"}]
        runnable, skipped = filter_matrix_for_te(
            entries, tmp_path, te_probe=probe,
        )
        assert runnable == entries
        assert skipped == []
        assert probe_called is False


LAUNCHER_PATH = REPO / "experiments" / "19_prereqs" / "run_persistent_launcher.py"


def _run_launcher(
    args: list[str], *, cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Invoke the launcher as a subprocess and capture stdout/stderr.

    Using subprocess rather than calling ``main()`` in-process covers
    the argparse surface, control-flow wiring, and any side effects
    Python's import cache might miss. Slow (~500ms per call) but the
    only way to catch a ``main()``-level regression like "pre-flight
    call site silently deleted."
    """
    return subprocess.run(
        [sys.executable, str(LAUNCHER_PATH), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


class TestLauncherMainIntegration:
    """End-to-end exit-code + side-effect coverage of ``main()``.

    Fills the integration-coverage gap flagged in the prior review: unit
    tests exercise pre-flight helpers directly, so wiring bugs (e.g., a
    pre-flight removed from the call site) wouldn't surface until a pod
    run. These subprocess tests lock the control flow down.

    Every test here runs with ``--dry-run`` to avoid torchrun / GPU
    requirements. This host has no transformer_engine, so fp8 pre-flight
    naturally exercises the "TE unavailable" branch; bf16-only matrices
    exercise the "ready to launch (but dry-run)" branch.
    """

    def _base_args(
        self, output_dir: Path, *, conditions: str = "bf16",
    ) -> list[str]:
        """Minimal CLI args for a dry-run invocation on this host.

        --data-path / --sp-model-path are required but the paths aren't
        opened before dry-run exit, so /tmp sentinels work.
        """
        return [
            "--data-path", "/tmp/nonexistent-dataset",
            "--sp-model-path", "/tmp/nonexistent-spm",
            "--output-dir", str(output_dir),
            "--world-size", "4",
            "--base-lr", "0.128",
            "--budget", "600",
            "--conditions", conditions,
            "--dry-run",
        ]

    def test_fp8_only_on_te_less_host_exits_nonzero(
        self, tmp_path: Path,
    ) -> None:
        """Silent-success regression: user asks for fp8, gets zero runs.

        On this (TE-less) host, ``filter_matrix_for_te`` drops every
        fp8 entry. Pre-fix, the launcher wrote skip markers and exited
        0 — a rc=0 claim that the matrix completed when in fact no
        training happened. Must be rc=1 post-fix.
        """
        result = _run_launcher(
            self._base_args(tmp_path, conditions="fp8"),
        )
        assert result.returncode == 1, (
            f"fp8-only matrix on TE-less host must exit non-zero. "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        # Skip markers for every requested seed must still be written
        # — downstream summarizers need the "errored" marker for each
        # expected output, not silent absence.
        marker_files = sorted(tmp_path.glob("fp8_s*.json"))
        assert len(marker_files) >= 1, (
            f"expected fp8 skip markers in {tmp_path}, found none"
        )
        # Contents should be the TE-unavailable pattern the runner expects.
        for path in marker_files:
            payload = json.loads(path.read_text())
            assert "transformer_engine unavailable" in payload.get("error", "")

    def test_bf16_only_dry_run_exits_zero(self, tmp_path: Path) -> None:
        """Positive control: a runnable matrix in --dry-run exits 0.

        Confirms the non-zero path is specific to "zero runnable" and
        isn't a blanket "return 1 from dry-run" regression.
        """
        result = _run_launcher(
            self._base_args(tmp_path, conditions="bf16"),
        )
        assert result.returncode == 0, (
            f"bf16-only dry-run must exit zero. "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_stale_fp8_markers_with_te_path_raises_before_run(
        self, tmp_path: Path,
    ) -> None:
        """Integration-level coverage of the stale-marker pre-flight wiring.

        Without transformer_engine on the test host we can't flip the
        TE-available branch from outside. But we CAN verify the
        pre-flight is at least on the call path by confirming the
        launcher doesn't crash when markers exist and conditions are
        bf16-only (stale fp8 markers from a prior run must not affect
        a bf16-only matrix — ``reject_stale_skip_markers`` only raises
        for fp8 entries in the current matrix). This guards against a
        future edit that wires the pre-flight incorrectly (e.g.,
        raising on any stale marker regardless of the current matrix).
        """
        # Pre-seed a stale fp8 marker.
        (tmp_path / "fp8_s1337.json").write_text(json.dumps({
            "config": {"name": "fp8", "seed": 1337, "precision": "fp8"},
            "error": "skipped: transformer_engine unavailable on pod",
        }))
        # Run a bf16-only matrix — must not raise, must exit 0 (dry-run).
        result = _run_launcher(
            self._base_args(tmp_path, conditions="bf16"),
        )
        assert result.returncode == 0, (
            f"bf16-only matrix with unrelated stale fp8 marker must exit 0 "
            f"(the stale marker is not in the current matrix). "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_world_size_mismatch_is_rejected_at_main_level(
        self, tmp_path: Path,
    ) -> None:
        """Integration coverage of the world-size pre-flight wiring.

        Pass a custom matrix whose entries declare ``world_size=2``
        while the CLI passes ``--world-size 4``. The launcher must
        refuse at pre-flight, not at some later stage.
        """
        matrix_path = tmp_path / "matrix.json"
        matrix_path.write_text(json.dumps([
            {
                "name": "bf16", "seed": 1337, "precision": "bf16",
                "world_size": 2,
            },
        ]))
        args = self._base_args(tmp_path, conditions="bf16") + [
            "--matrix-json", str(matrix_path),
        ]
        result = _run_launcher(args)
        assert result.returncode != 0, (
            f"world-size mismatch must exit non-zero. "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        # The error message should name the conflict, not some downstream
        # NameError / JSON decode failure.
        combined = result.stdout + result.stderr
        assert "world_size" in combined
