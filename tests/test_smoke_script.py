import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_smoke_module():
    specification = importlib.util.spec_from_file_location(
        "smoke_test", Path("scripts/smoke_test.py")
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules["smoke_test"] = module
    specification.loader.exec_module(module)
    return module


def test_smoke_test_passes_end_to_end_on_cpu(tmp_path):
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_test.py",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "smoke"),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    assert "SMOKE TEST PASS" in completed.stdout
    assert '"verdict": "PASS"' in report.read_text()


def test_unavailable_device_fails_with_a_summary_and_nonzero_exit(tmp_path):
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "--device", "tpu", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 1
    assert "SMOKE TEST FAIL" in completed.stdout


def test_a_failing_step_fails_the_run_and_skips_what_depends_on_it(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "smoke_test", raising=False)
    import quantrl.training

    def broken_train(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(quantrl.training, "train", broken_train)
    monkeypatch.setattr(
        sys, "argv", ["smoke_test.py", "--device", "cpu", "--output-dir", str(tmp_path / "smoke")]
    )
    exit_code = load_smoke_module().main()
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "[FAIL] train fp32 baseline" in output
    assert "injected failure" in output
    assert "SKIP  plots from logs" in output
    assert "SMOKE TEST FAIL" in output


@pytest.mark.parametrize("flag", ["--help"])
def test_smoke_script_documents_its_options(flag):
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", flag], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0
    assert "--device" in completed.stdout and "--render" in completed.stdout
