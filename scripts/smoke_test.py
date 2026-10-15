from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantrl.rendering import RENDER_BACKENDS, configure_rendering

TINY_TRAINING = [
    "sac.hidden_sizes=[64, 64]",
    "sac.batch_size=64",
    "train.random_steps=150",
    "train.update_after=150",
    "train.updates_per_step=0.5",
    "train.eval_interval=300",
    "train.eval_episodes=3",
    "eval.episodes=5",
    "eval.calibration_episodes=3",
    "eval.bootstrap_resamples=200",
    "her.buffer_episodes=50",
]


@dataclass
class StepResult:
    name: str
    status: str
    seconds: float
    detail: str


@dataclass
class SmokeTest:
    device: str
    output_dir: Path
    results: list[StepResult] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def run(self, name: str, action: Callable[[], str], needs: tuple[str, ...] = ()) -> None:
        failed = [
            item.name for item in self.results if item.name in needs and item.status != "PASS"
        ]
        if failed:
            self.results.append(StepResult(name, "SKIP", 0.0, f"needs {', '.join(failed)}"))
            print(f"[SKIP] {name}: needs {', '.join(failed)}", flush=True)
            return
        print(f"[....] {name}", flush=True)
        started = time.perf_counter()
        try:
            detail = action()
            status = "PASS"
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            status = "FAIL"
            traceback.print_exc()
        seconds = time.perf_counter() - started
        self.results.append(StepResult(name, status, seconds, detail))
        print(f"[{status}] {name} ({seconds:.1f}s): {detail}", flush=True)

    def overrides(self, *extra: str) -> list[str]:
        return [*TINY_TRAINING, f"device={self.device}", f"output_dir={self.output_dir}", *extra]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny end-to-end check: baseline, two bit widths, PTQ, QAT, resume, plots."
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:<index>, or mps")
    parser.add_argument("--output-dir", default=None, help="Defaults to runs/smoke_<device>")
    parser.add_argument("--render", action="store_true", help="Also render a short video")
    parser.add_argument("--render-backend", default="auto", choices=list(RENDER_BACKENDS))
    parser.add_argument("--report", default=None, help="Optional JSON file for the results")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.render:
        configure_rendering(arguments.render_backend)

    import pandas as pd
    import torch

    from quantrl.checkpoint import deterministic_policy_function, load_policy
    from quantrl.config import load_config
    from quantrl.envs.factory import check_goal_env, make_env
    from quantrl.eval.evaluate import evaluate_policy, evaluation_seeds
    from quantrl.eval.quantized import evaluate_ptq_settings, ptq_settings
    from quantrl.sweep import baseline_rows_for_methods
    from quantrl.training import train
    from quantrl.utils.device import describe_device, resolve_device
    from quantrl.viz.plots import load_progress, plot_success_vs_bits, plot_training_curves

    try:
        device = resolve_device(arguments.device)
    except (ValueError, RuntimeError) as error:
        print(f"[FAIL] environment and device: {type(error).__name__}: {error}")
        print("SMOKE TEST FAIL (device could not be selected)")
        return 1
    output_dir = Path(arguments.output_dir or f"runs/smoke_{device.type}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    smoke = SmokeTest(arguments.device, output_dir)
    config_file = "configs/fetch_reach.json"
    started = time.perf_counter()

    def environment() -> str:
        description = describe_device(device)
        for env_id in ("FetchReach-v4", "FetchPush-v4"):
            env = make_env(load_config(None, [f"env.env_id={env_id}"]).env)
            check_goal_env(env)
            env.close()
        smoke.context["device"] = description
        return (
            f"device {description['device']} (torch {description['torch']}, CUDA build "
            f"{description['torch_cuda_build']}, ROCm build {description['torch_rocm_build']}, "
            f"visible {description['available_devices']}); FetchReach and FetchPush contracts ok"
        )

    def train_baseline() -> str:
        config = load_config(
            config_file,
            smoke.overrides("name=fp32", "train.total_steps=600", "train.checkpoint_interval=300"),
        )
        summary = train(config, verbose=False)
        smoke.context["fp32"] = config
        smoke.context["fp32_summary"] = summary
        progress = pd.read_csv(config.run_dir / "progress.csv")
        if list(progress["step"]) != [300, 600]:
            raise AssertionError(
                f"expected evaluations at 300 and 600, got {list(progress['step'])}"
            )
        for name in ("checkpoint.pt", "checkpoint_300.pt", "checkpoint_600.pt", "resume_state.pt"):
            if not (config.run_dir / name).is_file():
                raise AssertionError(f"{name} was not written")
        final = summary["final_eval"]
        return (
            f"600 steps, final success {final['success_rate']:.2f} "
            f"on {final['episodes']} episodes, "
            f"{summary['train_wall_seconds']:.1f}s of training"
        )

    def reload_and_evaluate() -> str:
        config = smoke.context["fp32"]
        loaded = load_policy(config.run_dir / "checkpoint.pt")
        env = make_env(loaded.config.env)
        result, _ = evaluate_policy(
            env,
            deterministic_policy_function(loaded.actor),
            loaded.normalizer,
            evaluation_seeds(config.eval.seed_offset, config.eval.episodes),
            config.eval.bootstrap_resamples,
            config.eval.confidence,
        )
        env.close()
        recorded = smoke.context["fp32_summary"]["final_eval"]
        if device.type == "cpu" and result.to_dict() != recorded:
            raise AssertionError(f"reloaded evaluation {result.to_dict()} differs from {recorded}")
        if abs(result.success_rate - recorded["success_rate"]) > 0.2:
            raise AssertionError(
                f"reloaded success {result.success_rate} "
                f"far from recorded {recorded['success_rate']}"
            )
        match = "identical to" if result.to_dict() == recorded else "close to"
        return f"checkpoint reloaded on CPU, evaluation {match} the one recorded during training"

    def resume() -> str:
        first = load_config(config_file, smoke.overrides("name=resume", "train.total_steps=300"))
        train(first, verbose=False)
        config = load_config(
            config_file,
            smoke.overrides("name=resume", "train.total_steps=450", "train.resume=true"),
        )
        summary = train(config, verbose=False)
        metadata = json.loads((config.run_dir / "metadata.json").read_text())
        if metadata.get("resumed_from_step") != 300 or summary["total_steps"] != 450:
            raise AssertionError(
                f"resume went from {metadata.get('resumed_from_step')} to {summary['total_steps']}"
            )
        steps = list(pd.read_csv(config.run_dir / "progress.csv")["step"])
        if steps != [300, 450]:
            raise AssertionError(f"progress after resume has steps {steps}, expected [300, 450]")
        return "saved state at step 300 resumed and continued to 450 with a continuous log"

    def ptq() -> str:
        loaded = load_policy(smoke.context["fp32"].run_dir / "checkpoint.pt")
        rows = evaluate_ptq_settings(loaded, ptq_settings([32, 8, 4], ["both"], [True]))
        smoke.context["ptq_rows"] = rows
        recorded = smoke.context["fp32_summary"]["final_eval"]["success_rate"]
        if device.type == "cpu" and rows[0]["success_rate"] != recorded:
            raise AssertionError(
                f"PTQ at 32 bits gave success {rows[0]['success_rate']}, the fp32 actor {recorded}"
            )
        return ", ".join(f"{row['bits']} bits: success {row['success_rate']:.2f}" for row in rows)

    def qat() -> str:
        rows = []
        fp32 = smoke.context["fp32"]
        runs = [
            ("qat_scratch", 8, []),
            ("qat_scratch", 4, []),
            (
                "qat_finetune",
                4,
                [
                    f"train.init_checkpoint={fp32.run_dir / 'checkpoint.pt'}",
                    "train.eval_at_start=true",
                    "train.random_steps=0",
                    "sac.actor_lr=0.0001",
                    "sac.alpha_lr=0",
                ],
            ),
        ]
        for method, bits, extra in runs:
            config = load_config(
                config_file,
                smoke.overrides(
                    f"name={method}_w{bits}",
                    "train.total_steps=300",
                    f"quant.method={method}",
                    f"quant.weight_bits={bits}",
                    f"quant.activation_bits={bits}",
                    "quant.per_channel_weights=true",
                    *extra,
                ),
            )
            summary = train(config, verbose=False)
            loaded = load_policy(config.run_dir / "checkpoint.pt")
            if loaded.config.quant.weight_bits != bits:
                raise AssertionError(f"{method} checkpoint reloaded with the wrong bit width")
            rows.append({"method": method, "bits": bits, "seed": 0, **summary["final_eval"]})
            smoke.context.setdefault("qat_runs", []).append(config.run_dir)
        smoke.context["qat_rows"] = rows
        return ", ".join(
            f"{row['method']} {row['bits']} bits: success {row['success_rate']:.2f}" for row in rows
        )

    def plots() -> str:
        plot_dir = output_dir / "plots"
        progress = load_progress([smoke.context["fp32"].run_dir, *smoke.context["qat_runs"]])
        plot_training_curves(progress, plot_dir / "training_curves.png", "smoke test")
        rows = [
            {
                key: row[key]
                for key in ("method", "bits", "seed", "success_rate", "mean_final_distance")
            }
            for row in [*smoke.context["ptq_rows"], *smoke.context["qat_rows"]]
        ]
        frame = pd.DataFrame(
            baseline_rows_for_methods(rows, ["ptq", "qat_finetune", "qat_scratch"])
        )
        frame = frame.assign(series=frame["method"])
        plot_success_vs_bits(frame, plot_dir / "success_vs_bits.png", "smoke test", "series")
        sweep_csv = plot_dir / "sweep.csv"
        pd.DataFrame(rows).to_csv(sweep_csv, index=False)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/make_plots.py",
                "--sweep-csv",
                str(sweep_csv),
                "--out",
                str(plot_dir / "from_cli.png"),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(f"make_plots.py failed: {completed.stderr[-500:]}")
        for name in ("training_curves.png", "success_vs_bits.png", "from_cli.png"):
            if (plot_dir / name).stat().st_size < 10_000:
                raise AssertionError(f"{name} looks empty")
        return f"3 figures written to {plot_dir}"

    def video() -> str:
        from quantrl.viz.video import render_policy_episodes, write_video

        loaded = load_policy(smoke.context["fp32"].run_dir / "checkpoint.pt")
        frames, _ = render_policy_episodes(
            loaded.config.env,
            deterministic_policy_function(loaded.actor),
            loaded.normalizer,
            [loaded.config.eval.seed_offset],
            "smoke",
            size=96,
        )
        path = output_dir / "smoke.mp4"
        write_video(frames, path)
        return f"{len(frames)} frames rendered to {path}"

    smoke.run("environment and device", environment)
    smoke.run("train fp32 baseline", train_baseline, ("environment and device",))
    smoke.run("reload checkpoint and evaluate", reload_and_evaluate, ("train fp32 baseline",))
    smoke.run("resume from saved state", resume, ("train fp32 baseline",))
    smoke.run("PTQ at 8 and 4 bits", ptq, ("train fp32 baseline",))
    smoke.run("QAT at 8 and 4 bits", qat, ("train fp32 baseline",))
    smoke.run("plots from logs", plots, ("PTQ at 8 and 4 bits", "QAT at 8 and 4 bits"))
    if arguments.render:
        smoke.run("render a video", video, ("train fp32 baseline",))

    total = time.perf_counter() - started
    failures = [item for item in smoke.results if item.status != "PASS"]
    print()
    print(
        f"Smoke test summary (device {device}, "
        f"{platform.system()} {platform.machine()}, {total:.0f}s)"
    )
    for item in smoke.results:
        print(f"  {item.status:<4}  {item.name:<32} {item.seconds:6.1f}s  {item.detail}")
    verdict = "PASS" if not failures else f"FAIL ({len(failures)} of {len(smoke.results)} steps)"
    print(f"SMOKE TEST {verdict}")
    if arguments.report:
        Path(arguments.report).write_text(
            json.dumps(
                {
                    "verdict": "PASS" if not failures else "FAIL",
                    "device": str(device),
                    "torch": torch.__version__,
                    "seconds": round(total, 1),
                    "steps": [item.__dict__ for item in smoke.results],
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
