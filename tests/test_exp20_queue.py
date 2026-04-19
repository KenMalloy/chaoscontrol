import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("experiments/20_ssm_native_ttt/run_queue.py")


def _write_config(config_dir: Path, name: str, summary_path: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}.json"
    path.write_text(json.dumps({
        "name": name,
        "summary_path": str(summary_path),
        "output_path": str(summary_path.with_suffix(".jsonl")),
    }))
    return path


def test_dry_run_respects_resume_and_assigns_gpus(tmp_path):
    config_dir = tmp_path / "configs"
    done_summary = tmp_path / "done_summary.json"
    done_summary.write_text("{}\n")
    _write_config(config_dir, "done", done_summary)
    _write_config(config_dir, "todo", tmp_path / "todo_summary.json")
    (config_dir / "manifest.json").write_text("{}\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config-dir",
            str(config_dir),
            "--gpus",
            "0",
            "1",
            "--resume",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "SKIP done" in result.stdout
    assert "RUN todo gpu=0" in result.stdout
    assert "manifest" not in result.stdout


def test_queue_runs_configs_and_records_logs(tmp_path):
    config_dir = tmp_path / "configs"
    logs = tmp_path / "logs"
    first_summary = tmp_path / "first_summary.json"
    second_summary = tmp_path / "second_summary.json"
    _write_config(config_dir, "first", first_summary)
    _write_config(config_dir, "second", second_summary)

    runner = tmp_path / "runner.py"
    runner.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())\n"
        "Path(cfg['summary_path']).write_text(json.dumps({\n"
        "    'name': cfg['name'],\n"
        "    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "    'local_rank': os.environ.get('LOCAL_RANK'),\n"
        "}) + '\\n')\n"
        "print('ran ' + cfg['name'])\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config-dir",
            str(config_dir),
            "--gpus",
            "2",
            "--runner",
            str(runner),
            "--log-dir",
            str(logs),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    first = json.loads(first_summary.read_text())
    second = json.loads(second_summary.read_text())
    assert first["cuda_visible_devices"] == "2"
    assert second["cuda_visible_devices"] == "2"
    assert first["local_rank"] == "0"
    assert second["local_rank"] == "0"
    assert "ran first" in (logs / "first.log").read_text()
    assert "ran second" in (logs / "second.log").read_text()
