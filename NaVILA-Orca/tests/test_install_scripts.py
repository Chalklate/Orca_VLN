from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
PYSIDE_SHA256 = "30c50babaa8825be4519c9613166e595d97d2a1ce799f186667bb4c767ecffef"
PAK_SHA256 = "11f292569ed54f2be5991b3a3f6e60fac2d34a52a384c3cbf97ef9b2f9a6af88"


def _make_executable(path: Path, body: str = "#!/usr/bin/env bash\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _load_runtime_preparer():
    module_path = SCRIPTS / "prepare_orcalab_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_orcalab_runtime", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_resolver_prefers_project_prefix_over_active_conda(
    tmp_path: Path,
) -> None:
    project_prefix = tmp_path / "project-orcalab"
    active_prefix = tmp_path / "active-navila"
    _make_executable(project_prefix / "bin/python")
    _make_executable(project_prefix / "bin/orcalab")
    _make_executable(active_prefix / "bin/python", "#!/usr/bin/env bash\nexit 1\n")

    command = (
        'source "$1"; navila_orca_resolve_runtime; '
        'printf "%s\\n%s\\n" "$NAVILA_ORCA_PYTHON" "$NAVILA_ORCA_ORCALAB_BIN"'
    )
    env = {
        **os.environ,
        "NAVILA_ORCALAB_ENV_PREFIX": str(project_prefix),
        "CONDA_PREFIX": str(active_prefix),
    }
    env.pop("NAVILA_ORCA_PYTHON", None)
    env.pop("NAVILA_ORCA_ORCALAB_BIN", None)
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(SCRIPTS / "orcalab_env.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == [
        str(project_prefix / "bin/python"),
        str(project_prefix / "bin/orcalab"),
    ]


def test_runtime_resolver_explains_missing_project_environment(
    tmp_path: Path,
) -> None:
    command = 'source "$1"; navila_orca_resolve_runtime'
    env = {
        **os.environ,
        "NAVILA_ORCALAB_ENV_PREFIX": str(tmp_path / "missing"),
        "CONDA_PREFIX": str(tmp_path / "also-missing"),
    }
    env.pop("NAVILA_ORCA_PYTHON", None)
    env.pop("NAVILA_ORCA_ORCALAB_BIN", None)
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(SCRIPTS / "orcalab_env.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "setup_orcalab_env.sh" in result.stderr


def test_navila_server_rejects_partial_model_directory(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    env = {
        **os.environ,
        "NAVVLM_MODEL_PATH": str(model),
        "NAVVLM_PYTHON": "/bin/true",
    }
    result = subprocess.run(
        [str(SCRIPTS / "start_navvlm_server.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "missing or incomplete" in result.stderr


def test_transformers_reinstall_cannot_reresolve_locked_dependencies() -> None:
    script = (SCRIPTS / "setup_navila_env.sh").read_text()

    assert 'pip install --force-reinstall --no-deps \\' in script


def test_nvidia_preflight_explains_driver_library_mismatch(tmp_path: Path) -> None:
    fake_smi = tmp_path / "nvidia-smi"
    _make_executable(
        fake_smi,
        "#!/usr/bin/env bash\n"
        "echo 'Failed to initialize NVML: Driver/library version mismatch' >&2\n"
        "echo 'NVML library version: 580.173' >&2\n"
        "exit 1\n",
    )
    result = subprocess.run(
        [str(SCRIPTS / "check_nvidia_driver.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "NVIDIA_SMI_BIN": str(fake_smi), "LD_LIBRARY_PATH": ""},
    )

    assert result.returncode == 2
    assert "Driver/library version mismatch" in result.stderr
    assert "Reboot the computer once" in result.stderr
    assert "Do not delete or reinstall" in result.stderr


def test_nvidia_preflight_detects_library_path_pollution(tmp_path: Path) -> None:
    fake_smi = tmp_path / "nvidia-smi"
    _make_executable(
        fake_smi,
        "#!/usr/bin/env bash\n"
        'if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then exit 1; fi\n'
        "echo 'GPU 0: test'\n",
    )
    result = subprocess.run(
        [str(SCRIPTS / "check_nvidia_driver.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NVIDIA_SMI_BIN": str(fake_smi),
            "LD_LIBRARY_PATH": "/fake/cuda/stubs",
        },
    )

    assert result.returncode == 2
    assert "unset LD_LIBRARY_PATH" in result.stderr


def test_system_dependency_check_reports_missing_qt_xcb_package(
    tmp_path: Path,
) -> None:
    fake_dpkg = tmp_path / "dpkg-query"
    _make_executable(fake_dpkg, "#!/usr/bin/env bash\nexit 1\n")
    result = subprocess.run(
        [str(SCRIPTS / "setup_system_deps.sh"), "--verify"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DPKG_QUERY_BIN": str(fake_dpkg)},
    )

    assert result.returncode == 2
    assert "libxcb-cursor0" in result.stderr
    assert "libopengl0" in result.stderr
    assert "setup_system_deps.sh" in result.stderr


def test_navila_install_and_server_verify_the_real_builder_import() -> None:
    constraints = (PROJECT_ROOT / "constraints/navila-rss2025.txt").read_text()
    setup = (SCRIPTS / "setup_navila_env.sh").read_text()
    server = (SCRIPTS / "start_navvlm_server.sh").read_text()

    assert "deepspeed==0.9.5" in constraints
    assert "torch==" not in constraints
    assert 'TORCH_VERSION="2.7.0"' in setup
    assert 'TORCHVISION_VERSION="0.22.0"' in setup
    assert "/whl/cu128" in setup
    assert 'FLASH_ATTN_VERSION="2.8.3"' in setup
    assert "torch2.7cxx11abiTRUE-cp310" in setup
    assert 'pip install --force-reinstall --no-deps "${FLASH_ATTN_WHEEL}"' in setup
    assert "from llava.model.builder import load_pretrained_model" in setup
    assert "from llava.model.builder import load_pretrained_model" in server
    assert "check_navila_cuda.py" in server


def test_orcalab_runtime_pins_the_cuda_12_8_wheel_pair() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    constraints = (
        PROJECT_ROOT / "constraints/orcalab-26.7.1.txt"
    ).read_text()
    setup = (SCRIPTS / "setup_orcalab_env.sh").read_text()

    assert '"torch==2.11.0"' in pyproject
    assert '"torchvision==0.26.0"' in pyproject
    assert "torch==2.11.0+cu128" in constraints
    assert "torchvision==0.26.0+cu128" in constraints
    assert 'TORCH_VERSION="2.11.0"' in setup
    assert 'TORCHVISION_VERSION="0.26.0"' in setup
    assert "/whl/cu128" in setup
    assert '"torch==${TORCH_VERSION}+cu128"' in setup
    assert 'require_equal("torch CUDA build", torch.version.cuda, "12.8")' in setup


def test_orcalab_runtime_recognizes_both_glvnd_frontend_abis() -> None:
    preparer = _load_runtime_preparer()

    assert preparer.glvnd_dependencies(
        ["libQt6Gui.so.6", "libGL.so.1", "libc.so.6"]
    ) == ("libGL.so.1",)
    assert preparer.glvnd_dependencies(
        [
            "libQt6Gui.so.6",
            "/lib/x86_64-linux-gnu/libOpenGL.so.0",
            "libc.so.6",
        ]
    ) == ("/lib/x86_64-linux-gnu/libOpenGL.so.0",)


def test_orcalab_runtime_reuses_the_verified_legacy_archive(
    tmp_path: Path, monkeypatch,
) -> None:
    preparer = _load_runtime_preparer()
    legacy = tmp_path / "python-project-unknown.tar.xz"
    legacy.touch()
    monkeypatch.setattr(
        preparer,
        "sha256",
        lambda path: preparer.PYSIDE_SHA256 if path == legacy else "invalid",
    )

    assert preparer.runtime_archive(tmp_path) == legacy


def test_orcalab_runtime_resolves_the_requested_host_glvnd_library(
    tmp_path: Path, monkeypatch,
) -> None:
    preparer = _load_runtime_preparer()
    host_libgl = tmp_path / "libGL.so.1"
    host_libopengl = tmp_path / "libOpenGL.so.0"
    host_libgl.touch()
    host_libopengl.touch()
    cache = (
        f"\tlibOpenGL.so.0 (libc6,x86-64) => {host_libopengl}\n"
        f"\tlibGL.so.1 (libc6,x86-64) => {host_libgl}\n"
    )

    monkeypatch.setattr(preparer.shutil, "which", lambda _name: "/sbin/ldconfig")
    monkeypatch.setattr(
        preparer.subprocess,
        "check_output",
        lambda _command, text: cache,
    )

    assert preparer.system_glvnd_library("libGL.so.1") == host_libgl
    assert preparer.system_glvnd_library("libOpenGL.so.0") == host_libopengl


def test_orcalab_runtime_rpath_includes_qt_and_shiboken_libraries() -> None:
    preparer = _load_runtime_preparer()
    pyside6 = Path("/env/site-packages/PySide6")
    shiboken6 = Path("/env/site-packages/shiboken6")

    entries = preparer.native_rpath(
        pyside6,
        shiboken6,
        Path("/env/lib"),
        Path("/runtime/dist"),
    ).split(":")

    assert entries == [
        "$ORIGIN",
        str(pyside6),
        str(pyside6 / "Qt" / "lib"),
        str(shiboken6),
        "/env/lib",
        "/runtime/dist",
    ]


def test_blackwell_cuda_preflight_rejects_a_pre_cuda_12_8_torch_build() -> None:
    import importlib.util
    import sys
    from types import SimpleNamespace

    module_path = SCRIPTS / "check_navila_cuda.py"
    spec = importlib.util.spec_from_file_location("check_navila_cuda", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_capability=lambda _index: (12, 0),
        get_device_name=lambda _index: "NVIDIA GeForce RTX 5090 Laptop GPU",
        get_arch_list=lambda: ["sm_50", "sm_60", "sm_70", "sm_80", "sm_90"],
    )
    fake_torch = SimpleNamespace(
        __version__="2.3.0+cu121",
        version=SimpleNamespace(cuda="12.1"),
        cuda=fake_cuda,
    )

    try:
        module.inspect_and_test_cuda(fake_torch)
    except RuntimeError as exc:
        assert "Blackwell requires" in str(exc)
        assert "CUDA 12.1" in str(exc)
    else:
        raise AssertionError("Blackwell must reject a pre-CUDA 12.8 PyTorch build")


def test_orcalab_launcher_opens_editor_without_forcing_runtime_mode() -> None:
    launcher = (SCRIPTS / "start_orcalab_gui.sh").read_text()

    assert 'exec "${ORCALAB_BIN}" "${WORKSPACE}" --verbose "$@"' in launcher
    assert "--full-screen" not in launcher
    assert "--sim-config" not in launcher
    assert "--scene orcalab_day" not in launcher
    assert "PROFILE_WATCHER" not in launcher


def test_scene_launcher_preserves_original_navila_camera_defaults() -> None:
    launcher = (SCRIPTS / "run_orcalab_scene_locomotion.sh").read_text()

    assert "--camera-mount-position" not in launcher
    assert "--stabilize-camera-horizon" not in launcher


def test_scene_launcher_uses_editable_prompt_file_unless_overridden(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    _make_executable(
        fake_python,
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == '-c' ]]; then\n"
        "  if [[ \"${2:-}\" == *'print(version'* ]]; then echo '26.7.1'; fi\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$@\"\n",
    )
    env = {
        **os.environ,
        "NAVILA_ORCA_PYTHON": str(fake_python),
    }
    launcher = str(SCRIPTS / "run_orcalab_scene_locomotion.sh")

    default_run = subprocess.run(
        [launcher],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    default_args = default_run.stdout.splitlines()
    instruction_index = default_args.index("--instruction-file")
    assert default_args[instruction_index + 1] == str(
        PROJECT_ROOT / "prompts/orcalab_scene_locomotion.txt"
    )
    output_index = default_args.index("--output")
    assert default_args[output_index + 1].startswith(
        str(PROJECT_ROOT / "outputs/scene_locomotion_runs/")
    )

    override_run = subprocess.run(
        [
            launcher,
            "--instruction",
            "Turn left.",
            "--output",
            str(tmp_path / "custom-output"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    override_args = override_run.stdout.splitlines()
    assert "--instruction-file" not in override_args
    assert override_args.count("--output") == 1
    assert override_args[-4:] == [
        "--instruction",
        "Turn left.",
        "--output",
        str(tmp_path / "custom-output"),
    ]


def test_memory_guide_supervisor_has_expected_commands() -> None:
    supervisor = SCRIPTS / "memory_guide_dev.sh"
    result = subprocess.run(
        ["bash", str(supervisor), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for command in ("start", "status", "run", "inspect", "stop"):
        assert f" {command}" in result.stdout
    assert "NAVILA_SERVER_MODE" in result.stdout
    assert "NAVILA_AWS_INSTANCE_ID" in result.stdout
    assert "SimpleMovement_DiningTable" not in result.stdout


def test_orcalab_setup_prepares_native_viewport_before_first_gui() -> None:
    constraints = (PROJECT_ROOT / "constraints/orcalab-26.7.1.txt").read_text()
    setup = (SCRIPTS / "setup_orcalab_env.sh").read_text()
    resolver = (SCRIPTS / "orcalab_env.sh").read_text()
    preparer = (SCRIPTS / "prepare_orcalab_runtime.py").read_text()

    assert "orca-gym==26.7.1" in constraints
    assert "orca-lab==26.7.1" in constraints
    assert "orcalab-pyside==26.7.1" in constraints
    assert "patchelf==0.17.2.4" in constraints
    assert "prepare_orcalab_runtime.py" in setup
    assert "env -u LD_LIBRARY_PATH" in setup
    assert 'glvnd_sonames = ("libGL.so.1", "libOpenGL.so.0")' in setup
    assert "is not bound to host GLVND" in setup
    assert 'expected_pyside / "Qt" / "lib"' in setup
    assert "import shiboken6" in setup
    assert 'export PATH="$(dirname "${resolved_python}"):${PATH}"' in resolver
    assert "unset LD_LIBRARY_PATH" in resolver
    assert 'version("orcalab-pyside") == "26.7.1"' in resolver
    assert PYSIDE_SHA256 in preparer
    assert PAK_SHA256 in preparer
    assert '"--replace-needed"' in preparer
    assert "libPySideGameLauncher.so" in preparer
    assert 'GLVND_SONAMES = ("libGL.so.1", "libOpenGL.so.0")' in preparer
    assert 'f"orcalab-pyside-{url_version}"' in preparer
    assert "url_version = ORCALAB_VERSION" in preparer
    assert 'pyside6 / "Qt" / "lib"' in preparer
    assert "has unresolved libraries" in preparer
    assert "_glapi_tls_Current" in preparer
