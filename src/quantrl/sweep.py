from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantrl.checkpoint import deterministic_policy_function, load_policy
from quantrl.config import RunConfig, load_config
from quantrl.envs.factory import make_env
from quantrl.eval.evaluate import evaluate_policy, evaluation_seeds
from quantrl.eval.quantized import PTQSetting, evaluate_ptq_settings
from quantrl.quant.quantizer import FULL_PRECISION_BITS
from quantrl.training import train

METHODS = ("ptq", "qat_finetune", "qat_scratch")


@dataclass
class SweepPlan:
    base_config: str
    fp32_run_dirs: list[str]
    bit_widths: list[int]
    methods: list[str]
    per_channel_weights: bool = True
    output_dir: str = "runs"
    finetune_overrides: list[str] = field(default_factory=list)
    scratch_overrides: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown methods {sorted(unknown)}, expected some of {METHODS}")
        if FULL_PRECISION_BITS in self.bit_widths:
            raise ValueError("Bit widths for the sweep are low-bit only; fp32 is always included")


def fp32_checkpoint(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoint.pt"


def quantized_run_name(base_name: str, method: str, bits: int) -> str:
    return f"{base_name}_{method}_w{bits}a{bits}"


def quantization_overrides(plan: SweepPlan, method: str, bits: int) -> list[str]:
    return [
        f"quant.method={method}",
        f"quant.weight_bits={bits}",
        f"quant.activation_bits={bits}",
        f"quant.per_channel_weights={json.dumps(plan.per_channel_weights)}",
        f"output_dir={plan.output_dir}",
    ]


def training_configs(plan: SweepPlan) -> list[RunConfig]:
    configs = []
    for run_dir in plan.fp32_run_dirs:
        fp32_config = load_config(Path(run_dir) / "config.json")
        base_name = fp32_config.name
        for bits in plan.bit_widths:
            if "qat_finetune" in plan.methods:
                configs.append(
                    load_config(
                        Path(run_dir) / "config.json",
                        [
                            *quantization_overrides(plan, "qat_finetune", bits),
                            f"name={quantized_run_name(base_name, 'qat_finetune', bits)}",
                            f"train.init_checkpoint={fp32_checkpoint(run_dir)}",
                            "train.eval_at_start=true",
                            *plan.finetune_overrides,
                        ],
                    )
                )
            if "qat_scratch" in plan.methods:
                configs.append(
                    load_config(
                        plan.base_config,
                        [
                            *quantization_overrides(plan, "qat_scratch", bits),
                            f"name={quantized_run_name(base_name, 'qat_scratch', bits)}",
                            f"seed={fp32_config.seed}",
                            *plan.scratch_overrides,
                        ],
                    )
                )
    return configs


def train_if_missing(config: RunConfig) -> dict[str, Any]:
    summary_path = config.run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["reused"] = True
        return summary
    summary = train(config, verbose=False)
    summary["reused"] = False
    return summary


def evaluate_quantized_run(run_dir: str | Path, episodes: int | None = None) -> dict[str, Any]:
    loaded = load_policy(Path(run_dir) / "checkpoint.pt")
    config = loaded.config
    env = make_env(config.env)
    try:
        result, _ = evaluate_policy(
            env,
            deterministic_policy_function(loaded.actor),
            loaded.normalizer,
            evaluation_seeds(config.eval.seed_offset, episodes or config.eval.episodes),
            config.eval.bootstrap_resamples,
            config.eval.confidence,
        )
    finally:
        env.close()
    return {
        "env_id": config.env.env_id,
        "name": config.name,
        "seed": config.seed,
        "method": config.quant.method,
        "bits": config.quant.weight_bits,
        "target": "both",
        "weight_granularity": "per_channel" if config.quant.per_channel_weights else "per_tensor",
        "weight_bits": config.quant.weight_bits,
        "activation_bits": config.quant.activation_bits,
        **result.to_dict(),
    }


def evaluate_fp32_and_ptq(
    run_dir: str | Path, plan: SweepPlan, episodes: int | None = None
) -> list[dict[str, Any]]:
    loaded = load_policy(fp32_checkpoint(run_dir))
    bit_widths = [FULL_PRECISION_BITS]
    if "ptq" in plan.methods:
        bit_widths += list(plan.bit_widths)
    settings = [PTQSetting(bits, "both", plan.per_channel_weights) for bits in bit_widths]
    rows = evaluate_ptq_settings(loaded, settings, episodes)
    for row in rows:
        row.pop("calibration_inputs", None)
    return rows


def run_sweep(
    plan: SweepPlan, workers: int = 4, episodes: int | None = None, verbose: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs = training_configs(plan)
    training_summaries = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(train_if_missing, config): config for config in configs}
        for future in as_completed(futures):
            summary = future.result()
            training_summaries.append(summary)
            if verbose:
                print(
                    f"trained {summary['name']} seed {summary['seed']}: "
                    f"success {summary['final_eval']['success_rate']:.2f}, "
                    f"{summary['train_wall_seconds']:.0f}s"
                    f"{' (reused)' if summary['reused'] else ''}, "
                    f"{time.perf_counter() - started:.0f}s elapsed",
                    flush=True,
                )
        evaluation_futures = [
            executor.submit(evaluate_fp32_and_ptq, run_dir, plan, episodes)
            for run_dir in plan.fp32_run_dirs
        ] + [
            executor.submit(evaluate_quantized_run, config.run_dir, episodes) for config in configs
        ]
        rows: list[dict[str, Any]] = []
        for future in evaluation_futures:
            result = future.result()
            rows.extend(result if isinstance(result, list) else [result])
    return rows, training_summaries


def baseline_rows_for_methods(
    rows: Sequence[dict[str, Any]], methods: Sequence[str]
) -> list[dict[str, Any]]:
    expanded = []
    for row in rows:
        if row["method"] != "fp32":
            expanded.append(dict(row))
            continue
        for method in methods:
            expanded.append({**row, "method": method, "is_fp32_reference": True})
    for row in expanded:
        row.setdefault("is_fp32_reference", False)
    return expanded
