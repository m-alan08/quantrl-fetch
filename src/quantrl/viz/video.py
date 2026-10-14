from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from quantrl.config import EnvConfig
from quantrl.envs.factory import make_env
from quantrl.envs.normalizer import GoalObservationNormalizer
from quantrl.eval.evaluate import PolicyFunction

SUCCESS_COLOR = (46, 160, 67)
FAILURE_COLOR = (200, 50, 50)
NEUTRAL_COLOR = (30, 30, 30)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_label(frame: np.ndarray, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = _font(max(14, frame.shape[0] // 22))
    padding = font.size // 2
    y = padding
    for text, color in lines:
        left, top, right, bottom = draw.textbbox((padding, y), text, font=font)
        draw.rectangle(
            (left - padding // 2, top - padding // 3, right + padding // 2, bottom + padding // 3),
            fill=(255, 255, 255),
        )
        draw.text((padding, y), text, font=font, fill=color)
        y = bottom + padding
    return np.asarray(image)


def render_episode(
    env,
    policy: PolicyFunction,
    normalizer: GoalObservationNormalizer,
    seed: int,
) -> tuple[list[np.ndarray], bool]:
    observation, _ = env.reset(seed=seed)
    frames = [env.render()]
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        inputs = normalizer(observation["observation"], observation["desired_goal"])
        action = policy(inputs[None])[0]
        observation, _, terminated, truncated, info = env.step(action)
        frames.append(env.render())
    return frames, bool(info["is_success"])


def render_policy_episodes(
    env_config: EnvConfig,
    policy: PolicyFunction,
    normalizer: GoalObservationNormalizer,
    seeds: Sequence[int],
    title: str,
    size: int = 480,
    hold_frames: int = 10,
) -> tuple[list[np.ndarray], list[bool]]:
    env = make_env(env_config, render_mode="rgb_array", render_width=size, render_height=size)
    all_frames: list[np.ndarray] = []
    outcomes: list[bool] = []
    try:
        for episode_number, seed in enumerate(seeds, start=1):
            frames, success = render_episode(env, policy, normalizer, seed)
            outcomes.append(success)
            for step, frame in enumerate(frames):
                is_last = step == len(frames) - 1
                status_line = (
                    (
                        "success" if success else "failure",
                        SUCCESS_COLOR if success else FAILURE_COLOR,
                    )
                    if is_last
                    else (f"step {step}", NEUTRAL_COLOR)
                )
                labeled = draw_label(
                    frame,
                    [
                        (title, NEUTRAL_COLOR),
                        (f"episode {episode_number}", NEUTRAL_COLOR),
                        status_line,
                    ],
                )
                repeats = hold_frames if is_last else 1
                all_frames.extend([labeled] * repeats)
    finally:
        env.close()
    return all_frames, outcomes


def write_video(frames: Sequence[np.ndarray], path: str | Path, fps: int = 25) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, list(frames), fps=fps, macro_block_size=8, quality=7)


def write_gif(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: int = 25,
    frame_stride: int = 2,
    width: int = 320,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(frames)[::frame_stride]
    resized = []
    for frame in selected:
        image = Image.fromarray(frame)
        height = round(image.height * width / image.width)
        resized.append(image.resize((width, height), Image.LANCZOS))
    duration_ms = round(1000 * frame_stride / fps)
    resized[0].save(
        path, save_all=True, append_images=resized[1:], duration=duration_ms, loop=0, optimize=True
    )


def check_same_environment(left: EnvConfig, right: EnvConfig) -> None:
    if left != right:
        raise ValueError(
            f"Side-by-side policies must share an environment config, got {left} and {right}"
        )


def compose_side_by_side(
    left_frames: Sequence[np.ndarray],
    right_frames: Sequence[np.ndarray],
    divider_width: int = 8,
    divider_color: tuple[int, int, int] = (255, 255, 255),
) -> list[np.ndarray]:
    if not left_frames or not right_frames:
        raise ValueError("Both frame sequences must be non-empty")
    if left_frames[0].shape != right_frames[0].shape:
        raise ValueError(f"Frame shapes differ: {left_frames[0].shape} and {right_frames[0].shape}")
    length = max(len(left_frames), len(right_frames))
    height = left_frames[0].shape[0]
    divider = np.empty((height, divider_width, 3), dtype=np.uint8)
    divider[:] = divider_color
    composed = []
    for index in range(length):
        left = left_frames[min(index, len(left_frames) - 1)]
        right = right_frames[min(index, len(right_frames) - 1)]
        composed.append(np.concatenate([left, divider, right], axis=1))
    return composed
