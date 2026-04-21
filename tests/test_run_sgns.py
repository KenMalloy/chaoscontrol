from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN_SGNS = REPO / "experiments" / "21_sgns_tokenizer" / "run_sgns.sh"


def test_run_sgns_honors_python_override_and_sets_src_path(tmp_path):
    """The pod runner must use the configured interpreter, not bare python."""
    marker = tmp_path / "python_invocation.txt"
    good_python = tmp_path / "good_python"
    bad_bin = tmp_path / "bad_bin"
    bad_bin.mkdir()
    bad_python = bad_bin / "python"

    good_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'exe=%s\\n' \"$0\" > \"$RUN_SGNS_MARKER\"\n"
        "printf 'args=%s\\n' \"$*\" >> \"$RUN_SGNS_MARKER\"\n"
        "printf 'pythonpath=%s\\n' \"${PYTHONPATH:-}\" >> \"$RUN_SGNS_MARKER\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    bad_python.write_text("#!/usr/bin/env bash\nexit 42\n", encoding="utf-8")
    good_python.chmod(0o755)
    bad_python.chmod(0o755)

    env = os.environ.copy()
    env["PY"] = str(good_python)
    env["RUN_SGNS_MARKER"] = str(marker)
    env["PATH"] = str(bad_bin) + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        ["bash", str(RUN_SGNS), "data-dir", str(tmp_path / "out.pt")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    invocation = marker.read_text(encoding="utf-8")
    assert f"exe={good_python}" in invocation
    assert str(REPO / "src") in invocation
    assert "scripts/train_sgns.py" in invocation
