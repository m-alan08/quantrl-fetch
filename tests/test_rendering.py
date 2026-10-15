import importlib
import json
import os
import platform
import subprocess
import sys

import pytest

from quantrl.rendering import RenderingSetupError, choose_render_backend, configure_rendering

RENDER_ONE_FRAME = """
import json, sys
from quantrl.rendering import configure_rendering
backend = configure_rendering(sys.argv[1])
from quantrl.config import load_config
from quantrl.envs.factory import make_env
env_config = load_config(None, []).env
env = make_env(env_config, render_mode="rgb_array", render_width=64, render_height=64)
env.reset(seed=0)
frame = env.render()
env.close()
print(json.dumps({"backend": backend, "shape": list(frame.shape), "std": float(frame.std())}))
"""


@pytest.mark.parametrize(
    "requested, system, environ, expected",
    [
        ("auto", "Linux", {}, "egl"),
        ("auto", "Linux", {"DISPLAY": ":0"}, None),
        ("auto", "Linux", {"WAYLAND_DISPLAY": "wayland-0"}, None),
        ("auto", "Linux", {"MUJOCO_GL": "osmesa"}, "osmesa"),
        ("auto", "Darwin", {}, None),
        ("osmesa", "Linux", {"DISPLAY": ":0"}, "osmesa"),
        ("glfw", "Darwin", {}, "glfw"),
    ],
)
def test_backend_choice(requested, system, environ, expected):
    assert choose_render_backend(requested, system, environ) == expected


@pytest.mark.parametrize("requested", ["egl", "osmesa"])
def test_linux_only_backends_are_refused_elsewhere(requested):
    with pytest.raises(ValueError, match="only works on Linux"):
        choose_render_backend(requested, "Darwin", {})


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="render backend must be one of"):
        choose_render_backend("vulkan", "Linux", {})


def test_configure_sets_the_environment_for_headless_linux():
    environ = {}
    if "mujoco" in sys.modules:
        environ["MUJOCO_GL"] = "egl"
    assert configure_rendering("auto", environ, "Linux") == "egl"
    assert environ["MUJOCO_GL"] == "egl"
    assert environ["PYOPENGL_PLATFORM"] == "egl"


def test_choosing_a_backend_after_mujoco_is_imported_is_refused():
    importlib.import_module("mujoco")
    with pytest.raises(RenderingSetupError, match="already imported"):
        configure_rendering("osmesa", {"MUJOCO_GL": "egl"}, "Linux")


def render_in_subprocess(backend):
    environment = {key: value for key, value in os.environ.items() if key != "MUJOCO_GL"}
    completed = subprocess.run(
        [sys.executable, "-c", RENDER_ONE_FRAME, backend],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_rendering_works_with_the_automatic_backend():
    result = render_in_subprocess("auto")
    assert result["shape"] == [64, 64, 3]
    assert result["std"] > 1.0


@pytest.mark.skipif(platform.system() != "Linux", reason="EGL and OSMesa only exist on Linux")
@pytest.mark.parametrize("backend", ["egl", "osmesa"])
def test_headless_linux_rendering(backend):
    if os.environ.get("QUANTRL_TEST_HEADLESS_RENDER") != "1":
        pytest.skip(
            "set QUANTRL_TEST_HEADLESS_RENDER=1 where EGL and OSMesa libraries are installed"
        )
    result = render_in_subprocess(backend)
    assert result["backend"] == backend
    assert result["shape"] == [64, 64, 3]
    assert result["std"] > 1.0
