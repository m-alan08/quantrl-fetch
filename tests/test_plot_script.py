import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

METRICS = {"eval_success_rate": 0.5, "eval_mean_return": -20.0}


def fake_run(directory, seed):
    directory.mkdir(parents=True)
    steps = np.arange(1, 6) * 100
    pd.DataFrame(
        {
            "step": steps,
            "eval_success_rate": np.linspace(0, 1, 5) * (0.9 + 0.05 * seed),
            "eval_mean_return": np.linspace(-50, -2, 5),
        }
    ).to_csv(directory / "progress.csv", index=False)


def fake_bit_rows(methods, extra=None):
    rows = []
    for seed in range(3):
        rows.append(
            {
                "method": "fp32",
                "bits": 32,
                "seed": seed,
                "success_rate": 1.0,
                "mean_final_distance": 0.02,
                **(extra or {}),
            }
        )
        for method in methods:
            for bits in (8, 4, 2):
                rows.append(
                    {
                        "method": method,
                        "bits": bits,
                        "seed": seed,
                        "success_rate": bits / 8 - 0.05 * seed,
                        "mean_final_distance": 0.02 * (9 - bits),
                        **(extra or {}),
                    }
                )
    return rows


def run_plot_script(*arguments):
    completed = subprocess.run(
        [sys.executable, "scripts/make_plots.py", *arguments],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return completed.stdout


def assert_png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 10_000


def test_training_curves_from_fake_progress_logs(tmp_path):
    runs = [tmp_path / f"run_seed{seed}" for seed in range(3)]
    for seed, run in enumerate(runs):
        fake_run(run, seed)
    output = tmp_path / "curves.png"
    combined = tmp_path / "combined.csv"
    run_plot_script(
        "--runs", *map(str, runs), "--out", str(output), "--csv", str(combined), "--title", "fake"
    )
    assert_png(output)
    assert len(pd.read_csv(combined)) == 15


def test_method_comparison_from_a_fake_sweep(tmp_path):
    sweep = tmp_path / "sweep.csv"
    pd.DataFrame(fake_bit_rows(["ptq", "qat_finetune", "qat_scratch"])).to_csv(sweep, index=False)
    output = tmp_path / "methods.png"
    run_plot_script("--sweep-csv", str(sweep), "--out", str(output), "--title", "fake")
    assert_png(output)


@pytest.mark.parametrize("grids", [["unsigned"], ["signed", "unsigned"]])
def test_ptq_plot_from_a_fake_ptq_grid(tmp_path, grids):
    rows = []
    for grid in grids:
        for target in ("both", "weights"):
            rows += fake_bit_rows(
                ["ptq"],
                {
                    "target": target,
                    "weight_granularity": "per_channel",
                    "relu_activation_grid": grid,
                },
            )
    ptq = tmp_path / "ptq.csv"
    pd.DataFrame(rows).to_csv(ptq, index=False)
    output = tmp_path / "ptq.png"
    run_plot_script("--ptq-csv", str(ptq), "--out", str(output), "--title", "fake")
    assert_png(output)


def test_plot_script_without_inputs_fails_with_a_message(tmp_path):
    completed = subprocess.run(
        [sys.executable, "scripts/make_plots.py", "--out", str(tmp_path / "x.png")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
    assert "Pass --runs, --ptq-csv, or --sweep-csv" in completed.stderr
