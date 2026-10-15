from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping, MutableMapping

RENDER_BACKENDS = ("auto", "glfw", "egl", "osmesa")
LINUX_ONLY_BACKENDS = ("egl", "osmesa")


class RenderingSetupError(RuntimeError):
    pass


def choose_render_backend(
    requested: str,
    system: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    system = system or platform.system()
    environ = os.environ if environ is None else environ
    if requested not in RENDER_BACKENDS:
        raise ValueError(f"render backend must be one of {RENDER_BACKENDS}, got {requested!r}")
    if requested in LINUX_ONLY_BACKENDS and system != "Linux":
        raise ValueError(f"render backend {requested!r} only works on Linux, this is {system}")
    if requested != "auto":
        return requested
    if environ.get("MUJOCO_GL"):
        return environ["MUJOCO_GL"]
    if system == "Linux" and not (environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY")):
        return "egl"
    return None


def configure_rendering(
    requested: str = "auto",
    environ: MutableMapping[str, str] | None = None,
    system: str | None = None,
) -> str | None:
    environ = os.environ if environ is None else environ
    backend = choose_render_backend(requested, system, environ)
    if backend is None:
        return None
    current = environ.get("MUJOCO_GL")
    if "mujoco" in sys.modules and current != backend:
        raise RenderingSetupError(
            f"MuJoCo was already imported with MUJOCO_GL={current!r}; the render backend "
            f"{backend!r} has to be chosen before anything imports mujoco or gymnasium_robotics"
        )
    environ["MUJOCO_GL"] = backend
    if backend in LINUX_ONLY_BACKENDS:
        environ["PYOPENGL_PLATFORM"] = backend
    return backend
