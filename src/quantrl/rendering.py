from __future__ import annotations

import ctypes.util
import os
import platform
import sys
from collections.abc import Callable, Mapping, MutableMapping

RENDER_BACKENDS = ("auto", "glfw", "egl", "osmesa")
LINUX_ONLY_BACKENDS = ("egl", "osmesa")


SYSTEM_LIBRARIES = {"egl": "EGL", "osmesa": "OSMesa"}
INSTALL_HINTS = {
    "egl": "install the EGL library (for example libegl1 on Debian or Ubuntu, mesa-libEGL on RHEL)",
    "osmesa": "install the OSMesa library (for example libosmesa6 on Debian or Ubuntu)",
}


class RenderingSetupError(RuntimeError):
    pass


def library_available(backend: str) -> bool:
    library = SYSTEM_LIBRARIES.get(backend)
    return library is None or ctypes.util.find_library(library) is not None


def check_backend_library(backend: str, finder: Callable[[str], bool] = library_available) -> None:
    if backend in SYSTEM_LIBRARIES and not finder(backend):
        raise RenderingSetupError(
            f"MUJOCO_GL={backend} needs the {SYSTEM_LIBRARIES[backend]} system library, which was "
            f"not found; {INSTALL_HINTS[backend]}, choose another render backend, or unset "
            "MUJOCO_GL when not rendering (training and evaluation never render)"
        )


def choose_render_backend(
    requested: str,
    system: str | None = None,
    environ: Mapping[str, str] | None = None,
    finder: Callable[[str], bool] = library_available,
) -> str | None:
    system = system or platform.system()
    environ = os.environ if environ is None else environ
    if requested not in RENDER_BACKENDS:
        raise ValueError(f"render backend must be one of {RENDER_BACKENDS}, got {requested!r}")
    if requested in LINUX_ONLY_BACKENDS and system != "Linux":
        raise ValueError(f"render backend {requested!r} only works on Linux, this is {system}")
    if requested != "auto":
        check_backend_library(requested, finder)
        return requested
    if environ.get("MUJOCO_GL"):
        check_backend_library(environ["MUJOCO_GL"], finder)
        return environ["MUJOCO_GL"]
    if system == "Linux" and not (environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY")):
        for backend in LINUX_ONLY_BACKENDS:
            if finder(backend):
                return backend
        raise RenderingSetupError(
            "This Linux machine has no display and neither the EGL nor the OSMesa library was "
            f"found, so MuJoCo cannot render; {INSTALL_HINTS['egl']} or {INSTALL_HINTS['osmesa']}"
        )
    return None


def configure_rendering(
    requested: str = "auto",
    environ: MutableMapping[str, str] | None = None,
    system: str | None = None,
    finder: Callable[[str], bool] = library_available,
) -> str | None:
    environ = os.environ if environ is None else environ
    backend = choose_render_backend(requested, system, environ, finder)
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


def check_environment_before_mujoco_import(
    environ: Mapping[str, str] | None = None,
    finder: Callable[[str], bool] = library_available,
    loaded_modules: Mapping[str, object] | None = None,
) -> None:
    environ = os.environ if environ is None else environ
    loaded_modules = sys.modules if loaded_modules is None else loaded_modules
    backend = environ.get("MUJOCO_GL", "").lower().strip()
    if backend in SYSTEM_LIBRARIES and "mujoco" not in loaded_modules:
        check_backend_library(backend, finder)
