"""Command-line entry point for the NaVILA-to-Orca pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
from importlib import import_module
import json
from pathlib import Path
import socket
import sys
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from .backends.mjlab_go2 import (
    DEFAULT_CHECKPOINT,
    MjlabGo2Backend,
)
from .contracts import RenderFrame, RobotState
from .episodes import DEFAULT_SCENARIO, load_episode
from .live_monitor import LiveNavigationMonitor
from .paths import BUNDLED_GO2_XML, DEFAULT_GLOBAL_SETTINGS
from .render.grpc_bridge import GrpcRenderBridge, ProgrammableRenderServer
from .render.orca import DEFAULT_GO2_ASSET, OrcaLabRenderBridge
from .render.orca_camera import (
    DEFAULT_CAMERA_ACTOR_NAME,
    DEFAULT_CAMERA_ASSET,
    DEFAULT_CAMERA_MOUNT_POSITION,
    DEFAULT_CAMERA_MOUNT_QUAT_WXYZ,
    OrcaGrpcPngCamera,
    OrcaMujocoCameraFollower,
    OrcaMujocoPngCamera,
)
from .runner import NavigationRunner
from .teleop import KeyboardTeleopRunner, TerminalKeyboard
from .training import (
    COMPATIBLE_VERSION_SPECS,
    compatibility_errors,
    current_installed_versions,
)
from .vlm_client import LengthPrefixedJsonVLMClient


class ScriptedVLMClient:
    """Deterministic VLM substitute used only for orchestration smoke tests."""

    def __init__(self, outputs: Sequence[str]) -> None:
        if not outputs:
            raise ValueError("at least one scripted action is required")
        self._outputs = iter(outputs)

    def infer(self, _images: Sequence[Image.Image], _instruction: str) -> str:
        try:
            return next(self._outputs)
        except StopIteration:
            return "stop"


class RecordingRenderBridge:
    """Keep returned RGB frames for reproducible smoke evidence."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.frames: list[RenderFrame] = []
        self.pose_pushes = 0

    def render(self, state: RobotState, qpos_batch: Any = None) -> RenderFrame:
        frame = self.delegate.render(state, qpos_batch=qpos_batch)
        self.frames.append(frame)
        return frame

    def push_state(self, state: RobotState, qpos_batch: Any = None) -> None:
        push_state = getattr(self.delegate, "push_state", None)
        if callable(push_state):
            push_state(state, qpos_batch=qpos_batch)
            self.pose_pushes += 1

    def capture(self, state: RobotState, qpos_batch: Any = None) -> RenderFrame:
        capture = getattr(self.delegate, "capture", None)
        if callable(capture):
            frame = capture(state, qpos_batch=qpos_batch)
        else:
            frame = self.delegate.render(state, qpos_batch=qpos_batch)
        self.frames.append(frame)
        return frame

    def capture_transient(
        self, state: RobotState, qpos_batch: Any = None
    ) -> RenderFrame:
        """Capture a live-display frame without retaining it in the run artifact."""

        capture = getattr(self.delegate, "capture", None)
        if callable(capture):
            return capture(state, qpos_batch=qpos_batch)
        return self.delegate.render(state, qpos_batch=qpos_batch)

    def close(self) -> None:
        self.delegate.close()


def _procedural_rgb(state: RobotState, _qpos_batch: np.ndarray | None) -> np.ndarray:
    """Draw a state-dependent diagnostic image; this is not scene rendering."""

    height = width = 512
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((x + int(state.step_id)) % 256).astype(np.uint8)
    position_code = int(round(state.root_pos_world[0] * 32)) % 256
    image[..., 1] = ((y + position_code) % 256).astype(np.uint8)
    image[..., 2] = np.uint8(int(round((state.base_rpy[2] + np.pi) * 32)) % 256)
    # A high-contrast horizon makes transport/color-order mistakes visible.
    image[height // 2 - 2 : height // 2 + 2, :, :] = 255
    return image


def _doctor(args: argparse.Namespace) -> int:
    versions = current_installed_versions()
    report: dict[str, Any] = {
        "python": sys.executable,
        "versions": versions,
        "expected_versions": COMPATIBLE_VERSION_SPECS,
        "version_errors": list(compatibility_errors(versions)),
        "paths": {
            "scenario": _path_status(DEFAULT_SCENARIO),
            "global_settings": _path_status(DEFAULT_GLOBAL_SETTINGS),
            "go2_checkpoint": _path_status(DEFAULT_CHECKPOINT),
            "go2_xml": _path_status(BUNDLED_GO2_XML),
        },
        "imports": {},
        "cuda": {},
        "orcagym_grpc": _port_status("127.0.0.1", 50051),
    }
    for module_name in (
        "grpc",
        "av",
        "cv2",
        "orca_gym.sensor.rgbd_camera",
        "navila_orca.go2_task.tasks",
        "mjlab",
        "mujoco_warp",
    ):
        try:
            module = import_module(module_name)
            report["imports"][module_name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:  # doctor must report the complete stack
            report["imports"][module_name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    try:
        import torch

        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "torch": torch.__version__,
        }
        if torch.cuda.is_available():
            report["cuda"]["devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        report["cuda"] = {"available": False, "error": str(exc)}

    report["requirements"] = {
        "gpu_required": bool(args.require_gpu),
        "orca_required": bool(args.require_orca),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    required_paths_ok = all(item["exists"] for item in report["paths"].values())
    imports_ok = all(item["ok"] for item in report["imports"].values())
    ready = not report["version_errors"] and required_paths_ok and imports_ok
    if args.require_gpu:
        ready = ready and bool(report["cuda"].get("available"))
    if args.require_orca:
        ready = ready and bool(report["orcagym_grpc"].get("listening"))
    return 0 if ready else 2


def _path_status(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "exists": resolved.exists()}


def _port_status(host: str, port: int) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return {"address": f"{host}:{port}", "listening": True}
    except OSError as exc:
        return {
            "address": f"{host}:{port}",
            "listening": False,
            "detail": str(exc),
        }


def _make_vlm(args: argparse.Namespace) -> Any:
    if args.vlm_backend == "tcp":
        return LengthPrefixedJsonVLMClient(
            args.vlm_host, args.vlm_port, timeout_s=args.vlm_timeout
        )
    actions = args.scripted_action or [
        "move forward 25 cm",
        "turn left 15 degrees",
        "stop",
    ]
    return ScriptedVLMClient(actions)


def _make_renderer(
    args: argparse.Namespace,
    backend: MjlabGo2Backend,
    output_dir: Path | None = None,
) -> tuple[Any, ProgrammableRenderServer | None]:
    if args.render_backend == "grpc-loopback":
        server = ProgrammableRenderServer(_procedural_rgb).start()
        return GrpcRenderBridge(
            address=server.address, timeout_s=args.render_timeout
        ), server
    if args.render_backend == "grpc":
        if not args.grpc_render_address:
            raise ValueError(
                "--grpc-render-address is required for --render-backend grpc"
            )
        return (
            GrpcRenderBridge(
                address=args.grpc_render_address, timeout_s=args.render_timeout
            ),
            None,
        )
    camera_factory = None
    follower_factory = None
    if args.camera_transport == "grpc-png":
        camera_class = OrcaGrpcPngCamera
        if args.orcalab_camera_mode == "mujoco-png":
            camera_class = OrcaMujocoPngCamera
            follower_factory = OrcaMujocoCameraFollower
        camera_factory = partial(
            camera_class,
            edit_address=args.orcalab_edit_address,
            timeout_s=args.render_timeout,
        )
    publish_scene = bool(args.publish_scene) and not bool(args.no_publish)
    discover_agents = args.robot_actor_name == "auto" and not publish_scene
    robot_actor_name = None if discover_agents else args.robot_actor_name
    if robot_actor_name == "auto":
        robot_actor_name = "go2_000"
    aligned_xml_output = None
    if args.manual_xml_override:
        if output_dir is None:
            raise ValueError("output_dir is required for --manual-xml-override")
        aligned_xml_output = output_dir / "scene_alignment" / "aligned_scene.xml"
    return (
        OrcaLabRenderBridge(
            orcagym_address=args.orcagym_address,
            camera_port=args.camera_port,
            camera_name=args.camera_name,
            timeout_s=args.render_timeout,
            num_envs=backend.num_envs,
            joint_qpos_addr=backend.joint_qpos_addr,
            agent_name=robot_actor_name,
            discover_agents=discover_agents,
            asset_path=args.robot_asset_path,
            terrain_asset_path=args.terrain_asset_path,
            publish=publish_scene,
            anchor_to_scene=args.anchor_existing_scene,
            scene_timestep=args.scene_timestep,
            scene_profile=args.scene_profile,
            strict_scene_options=args.strict_scene_alignment,
            manual_xml_override=args.manual_xml_override,
            aligned_xml_output=aligned_xml_output,
            bind_camera=not args.no_camera_bind,
            edit_address=args.orcalab_edit_address,
            camera_actor_name=args.camera_actor_name,
            camera_asset_path=args.camera_asset_path,
            camera_mount_position=args.camera_mount_position,
            camera_mount_quat_wxyz=args.camera_mount_quat_wxyz,
            stabilize_camera_horizon=args.stabilize_camera_horizon,
            camera_factory=camera_factory,
            camera_follower_factory=follower_factory,
        ),
        None,
    )


def _run(args: argparse.Namespace) -> int:
    if args.scene_ready:
        raise ValueError(
            "--scene-ready cannot be asserted yet: MjlabGo2Backend still uses "
            "flat physics and does not apply the episode scene/start transform"
        )
    episode = load_episode(args.scenario)
    waypoint_instructions = _resolve_waypoint_instructions(args)
    instruction_provider = _make_instruction_provider(args)
    episode = replace(
        episode,
        instruction=_resolve_instruction(
            args,
            episode.instruction,
            waypoint_instructions=waypoint_instructions,
        ),
    )
    output_dir = _resolve_output_dir(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = MjlabGo2Backend(
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=1,
        deterministic_play=not args.randomized_play,
        warmup_steps=args.warmup_steps,
    )
    raw_renderer: Any | None = None
    renderer: RecordingRenderBridge | None = None
    live_monitor: LiveNavigationMonitor | None = None
    loopback_server: ProgrammableRenderServer | None = None
    try:
        # OrcaLab layout construction needs joint addresses, so start physics
        # before choosing the renderer. NavigationRunner performs the episode reset.
        backend.start()
        raw_renderer, loopback_server = _make_renderer(
            args,
            backend,
            output_dir=output_dir,
        )
        renderer = RecordingRenderBridge(raw_renderer)
        vlm = _make_vlm(args)
        if args.live_monitor:
            live_monitor = LiveNavigationMonitor()
        # This stays fail-closed until the backend applies a versioned visual,
        # collision and coordinate manifest plus the episode reset pose.
        scene_fidelity = False
        runner = NavigationRunner(
            backend,
            renderer,
            vlm,
            scene_fidelity=scene_fidelity,
            image_interval_s=args.image_interval,
            state_stream_interval_s=args.state_stream_interval,
            max_control_steps=args.max_control_steps,
            max_decisions=args.max_decisions,
            monitor=live_monitor,
            monitor_interval_s=args.monitor_interval,
            decouple_vlm=args.decouple_vlm,
            waypoint_instructions=waypoint_instructions,
            instruction_provider=instruction_provider,
        )
        result = runner.run(episode)
        artifact_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        frames_dir = output_dir / "frames" / artifact_id
        saved_frames = _save_frames(renderer.frames, frames_dir)
        render_details: dict[str, Any] = {"backend": args.render_backend}
        if args.render_backend == "grpc-loopback":
            render_details.update(
                {
                    "address": getattr(raw_renderer, "address", None),
                    "source": "state-dependent procedural diagnostic RGB",
                }
            )
        elif args.render_backend == "grpc":
            render_details["address"] = args.grpc_render_address
        else:
            render_details.update(
                {
                    "orcagym_address": args.orcagym_address,
                    "camera_port": args.camera_port,
                    "camera_label": args.camera_name,
                    "camera_transport": args.camera_transport,
                    "camera_binding": "disabled"
                    if args.no_camera_bind
                    else "rigid Go2 base-pose follower",
                    "camera_actor": None
                    if args.no_camera_bind
                    else f"/{args.camera_actor_name}",
                    "camera_asset_path": args.camera_asset_path,
                    "camera_mount_position": args.camera_mount_position,
                    "camera_mount_quat_wxyz": args.camera_mount_quat_wxyz,
                    "stabilize_camera_horizon": args.stabilize_camera_horizon,
                    "orcalab_edit_address": args.orcalab_edit_address,
                    "robot_actor_name": args.robot_actor_name,
                    "robot_asset_path": args.robot_asset_path,
                    "terrain_asset_path": args.terrain_asset_path,
                    "publish": bool(args.publish_scene) and not bool(args.no_publish),
                    "anchor_existing_scene": args.anchor_existing_scene,
                    "scene_profile": args.scene_profile,
                    "scene_timestep": args.scene_timestep,
                    "strict_scene_alignment": args.strict_scene_alignment,
                    "manual_xml_override": args.manual_xml_override,
                    "alignment": getattr(raw_renderer, "alignment_report", None),
                }
            )
        vlm_details: dict[str, Any] = {"backend": args.vlm_backend}
        if args.vlm_backend == "tcp":
            vlm_details.update(
                {
                    "host": args.vlm_host,
                    "port": args.vlm_port,
                    "timeout_s": args.vlm_timeout,
                }
            )
        payload = {
            "pipeline_status": "completed",
            "termination_reason": result.termination_reason,
            "control_steps": result.control_steps,
            "decisions": result.decisions,
            "waypoints_completed": result.waypoints_completed,
            "waypoint_count": result.waypoint_count,
            "waypoint_stop_rejections": result.waypoint_stop_rejections,
            "vlm_outputs": list(result.vlm_outputs),
            "motion_chunks": [asdict(chunk) for chunk in result.motion_chunks],
            "metrics": result.metrics,
            "final_state": _state_dict(result.final_state),
            "episode": {
                "episode_id": episode.episode_id,
                "scene_id": episode.scene_id,
                "instruction": episode.instruction,
                "waypoint_instructions": list(waypoint_instructions),
            },
            "runtime": {
                "physics": "mjlab Unitree-Go2-Flat / MuJoCo Warp",
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "scenario": str(Path(args.scenario).expanduser().resolve()),
                "device": args.device,
                "control_dt": backend.control_dt,
                "warmup_steps": args.warmup_steps,
                "state_stream_interval": args.state_stream_interval,
                "image_interval": args.image_interval,
                "max_control_steps": args.max_control_steps,
                "max_decisions": args.max_decisions,
                "live_monitor": args.live_monitor,
                "monitor_interval": args.monitor_interval,
                "render_backend": args.render_backend,
                "vlm_backend": args.vlm_backend,
                "render": render_details,
                "vlm": vlm_details,
                "captured_frames": len(renderer.frames),
                "frames_directory": str(frames_dir),
                "frame_files": [path.name for path in saved_frames],
                "pose_pushes": renderer.pose_pushes,
                "scene_fidelity": scene_fidelity,
                "physics_alignment": backend.alignment_report,
            },
            "limitations": []
            if scene_fidelity
            else [
                "Matterport 3DGS and matching collision geometry are not installed.",
                "Navigation Success/SPL are emitted for contract compatibility but are not scene-quality evidence.",
            ],
        }
        result_path = output_dir / "measurements.json"
        alignment_path = output_dir / "scene_alignment.json"
        alignment_path.write_text(
            json.dumps(
                {
                    "physics": backend.alignment_report,
                    "render": render_details.get("alignment"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"result: {result_path}")
        return 0
    finally:
        if live_monitor is not None:
            live_monitor.close()
        if renderer is not None:
            try:
                renderer.close()
            except Exception as exc:
                print(f"warning: renderer close failed: {exc}", file=sys.stderr)
        elif raw_renderer is not None:
            try:
                raw_renderer.close()
            except Exception:
                pass
        if loopback_server is not None:
            loopback_server.stop()
        backend.close()


def _teleop(args: argparse.Namespace) -> int:
    """Run simulator physics from a terminal keyboard, without NaVILA."""

    episode = load_episode(args.scenario)
    output_dir = _resolve_output_dir(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = MjlabGo2Backend(
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=1,
        deterministic_play=not args.randomized_play,
        warmup_steps=args.warmup_steps,
    )
    raw_renderer: Any | None = None
    renderer: RecordingRenderBridge | None = None
    live_monitor: LiveNavigationMonitor | None = None
    loopback_server: ProgrammableRenderServer | None = None
    try:
        backend.start()
        if not args.no_preview:
            raw_renderer, loopback_server = _make_renderer(
                args,
                backend,
                output_dir=output_dir,
            )
            renderer = RecordingRenderBridge(raw_renderer)
        if args.live_monitor:
            if renderer is None:
                raise ValueError("--live-monitor requires preview rendering")
            live_monitor = LiveNavigationMonitor(window_name="keyboard teleop")

        teleop = KeyboardTeleopRunner(
            backend,
            renderer,
            output_dir=output_dir,
            forward_speed_mps=args.forward_speed,
            strafe_speed_mps=args.strafe_speed,
            turn_rate_rad_s=args.turn_rate,
            command_hold_s=args.command_hold,
            capture_interval_s=args.capture_interval,
            live_monitor=live_monitor,
            realtime=not args.no_realtime,
        )
        with TerminalKeyboard() as keyboard:
            result = teleop.run(
                episode,
                keyboard,
                max_control_steps=args.max_control_steps,
                max_sim_time_s=args.max_sim_time,
            )
        payload = {
            "pipeline_status": "completed",
            "termination_reason": result.termination_reason,
            "control_steps": result.control_steps,
            "sim_time_s": result.sim_time_s,
            "anchors": [asdict(anchor) for anchor in result.anchors],
            "output_directory": str(output_dir),
            "runtime": {
                "physics": "mjlab Unitree-Go2-Flat / MuJoCo Warp",
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "scenario": str(Path(args.scenario).expanduser().resolve()),
                "device": args.device,
                "control_dt": backend.control_dt,
                "warmup_steps": args.warmup_steps,
                "render_backend": None if args.no_preview else args.render_backend,
                "live_monitor": bool(args.live_monitor),
                "preview": not args.no_preview,
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"teleop artifacts: {output_dir}")
        return 0
    finally:
        if live_monitor is not None:
            live_monitor.close()
        if renderer is not None:
            try:
                renderer.close()
            except Exception as exc:
                print(f"warning: renderer close failed: {exc}", file=sys.stderr)
        elif raw_renderer is not None:
            try:
                raw_renderer.close()
            except Exception:
                pass
        if loopback_server is not None:
            loopback_server.stop()
        backend.close()


def _resolve_output_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (Path.cwd() / "outputs" / run_id).resolve()


def _resolve_waypoint_instructions(args: argparse.Namespace) -> tuple[str, ...]:
    value = getattr(args, "waypoint_instruction_file", None)
    if value is None:
        return ()
    prompt_path = Path(value).expanduser().resolve()
    try:
        lines = prompt_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read waypoint file {prompt_path}: {exc}") from exc
    instructions = tuple(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(instructions) < 2:
        raise ValueError(
            f"waypoint file {prompt_path} must contain at least two non-empty lines"
        )
    return instructions


def _read_instruction_file(prompt_path: Path) -> str:
    try:
        instruction = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(
            f"cannot read instruction file {prompt_path}: {exc}"
        ) from exc
    if not instruction:
        raise ValueError(f"navigation instruction from {prompt_path} is empty")
    return instruction


def _make_instruction_provider(
    args: argparse.Namespace,
) -> Callable[[], str] | None:
    """Reload a file-backed instruction before every VLM decision."""

    value = getattr(args, "instruction_file", None)
    if value is None:
        return None
    prompt_path = Path(value).expanduser().resolve()
    return partial(_read_instruction_file, prompt_path)


def _resolve_instruction(
    args: argparse.Namespace,
    scenario_instruction: str,
    *,
    waypoint_instructions: Sequence[str] = (),
) -> str:
    """Resolve an optional CLI/file prompt without modifying the scenario."""

    if waypoint_instructions:
        instruction = " ".join(
            f"Waypoint {index}: {text}"
            for index, text in enumerate(waypoint_instructions, start=1)
        )
        source = "--waypoint-instruction-file"
    elif args.instruction_file is not None:
        prompt_path = Path(args.instruction_file).expanduser().resolve()
        instruction = _read_instruction_file(prompt_path)
        source = str(prompt_path)
    elif args.instruction is not None:
        instruction = str(args.instruction).strip()
        source = "--instruction"
    else:
        instruction = str(scenario_instruction).strip()
        source = "scenario"
    if not instruction:
        raise ValueError(f"navigation instruction from {source} is empty")
    return instruction


def _save_frames(frames: Sequence[RenderFrame], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, frame in enumerate(frames):
        name = f"{index:03d}_step_{frame.step_id:06d}.jpg"
        path = directory / name
        Image.fromarray(frame.rgb, mode="RGB").save(path, quality=90)
        saved.append(path)
    return saved


def _state_dict(state: RobotState) -> dict[str, Any]:
    payload = asdict(state)
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in payload.items()
    }


def _add_renderer_options(parser: argparse.ArgumentParser) -> None:
    """Add the renderer flags shared by ``run`` and ``teleop``."""

    parser.add_argument(
        "--render-backend",
        choices=("grpc-loopback", "grpc", "orcalab"),
        default="grpc-loopback",
    )
    parser.add_argument("--grpc-render-address")
    parser.add_argument("--orcagym-address", default="127.0.0.1:50051")
    parser.add_argument("--orcalab-edit-address", default="127.0.0.1:50151")
    parser.add_argument(
        "--orcalab-camera-mode",
        choices=("agent-data-png", "mujoco-png"),
        default="mujoco-png",
    )
    parser.add_argument("--camera-port", type=int, default=7070)
    parser.add_argument("--camera-name", default="navila_ego")
    parser.add_argument(
        "--camera-transport",
        choices=("grpc-png", "websocket"),
        default="grpc-png",
        help="RGB capture transport; grpc-png works with the current local OrcaStudio build",
    )
    parser.add_argument("--camera-actor-name", default=DEFAULT_CAMERA_ACTOR_NAME)
    parser.add_argument("--camera-asset-path", default=DEFAULT_CAMERA_ASSET)
    parser.add_argument(
        "--camera-mount-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=list(DEFAULT_CAMERA_MOUNT_POSITION),
        help="Go2 base-frame camera translation (original NaVILA default: 0.1 0 0.5)",
    )
    parser.add_argument(
        "--camera-mount-quat-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        default=list(DEFAULT_CAMERA_MOUNT_QUAT_WXYZ),
        help="Go2 base-frame camera rotation in wxyz order",
    )
    parser.add_argument(
        "--stabilize-camera-horizon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="follow Go2 yaw while rejecting base roll/pitch in ego-camera orientation",
    )
    parser.add_argument(
        "--no-camera-bind",
        action="store_true",
        help="use a manually configured camera stream instead of provisioning/following one",
    )
    parser.add_argument("--robot-asset-path", default=DEFAULT_GO2_ASSET)
    parser.add_argument(
        "--robot-actor-name",
        default="auto",
        help="existing OrcaLab Go2 actor name; 'auto' requires exactly one complete Go2",
    )
    parser.add_argument("--terrain-asset-path")
    publish_group = parser.add_mutually_exclusive_group()
    publish_group.add_argument(
        "--publish-scene",
        action="store_true",
        help="destructively republish the Orca scene; disabled by default to preserve authored objects",
    )
    publish_group.add_argument(
        "--no-publish",
        action="store_true",
        help="legacy explicit spelling for the safe default: reuse the current scene",
    )
    parser.add_argument(
        "--anchor-existing-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply local locomotion relative to the authored Go2 scene XY/yaw",
    )
    parser.add_argument(
        "--scene-profile",
        choices=("orca-train", "orca-runtime"),
        default="orca-train",
        help="MuJoCo option profile used for both downloaded XML and remote scene",
    )
    parser.add_argument(
        "--scene-timestep",
        type=float,
        help="override the selected profile timestep; normally leave unset",
    )
    parser.add_argument(
        "--strict-scene-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail before motion when OrcaLab rejects a MuJoCo scene option",
    )
    parser.add_argument(
        "--manual-xml-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write and load a corrected copy of OrcaLab's downloaded combined XML",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navila-orca",
        description="Run NaVILA with a Go2 MJWarp backend and an Orca-compatible render bridge.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser(
        "doctor", help="inspect the pinned local stack and assets"
    )
    doctor.add_argument(
        "--require-gpu",
        action="store_true",
        help="return exit 2 unless CUDA is usable",
    )
    doctor.add_argument(
        "--require-orca",
        action="store_true",
        help="return exit 2 unless the OrcaGym gRPC endpoint is reachable",
    )
    doctor.set_defaults(func=_doctor)

    run = subparsers.add_parser("run", help="run a finite navigation pipeline")
    run.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO),
        help="project-owned navigation scenario JSON",
    )
    instruction_group = run.add_mutually_exclusive_group()
    instruction_group.add_argument(
        "--instruction",
        help="override the scenario navigation instruction with this text",
    )
    instruction_group.add_argument(
        "--instruction-file",
        help="UTF-8 file containing the navigation instruction sent to NaVILA",
    )
    instruction_group.add_argument(
        "--waypoint-instruction-file",
        help="UTF-8 file with one staged waypoint instruction per non-empty line",
    )
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    _add_renderer_options(run)
    run.add_argument(
        "--randomized-play",
        action="store_true",
        help="retain MJLab reset/domain randomization instead of the deterministic scene test profile",
    )
    run.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="zero-command Go2 policy steps after reset (original NaVILA default: 100)",
    )
    run.add_argument("--render-timeout", type=float, default=10.0)
    run.add_argument(
        "--scene-ready",
        action="store_true",
        help="reserved; rejected until visual/collision/episode transforms are implemented",
    )
    run.add_argument("--vlm-backend", choices=("scripted", "tcp"), default="scripted")
    run.add_argument("--vlm-host", default="127.0.0.1")
    run.add_argument("--vlm-port", type=int, default=54321)
    run.add_argument("--vlm-timeout", type=float, default=180.0)
    run.add_argument(
        "--scripted-action",
        action="append",
        help="canonical action; repeat for a deterministic smoke sequence",
    )
    run.add_argument("--image-interval", type=float, default=0.5)
    run.add_argument(
        "--state-stream-interval",
        type=float,
        default=0.04,
        help="simulated seconds between pose pushes (0.04 = 25 Hz at 50 Hz control)",
    )
    run.add_argument(
        "--live-monitor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show live Go2 ego RGB with instruction and VLM command text",
    )
    run.add_argument(
        "--monitor-interval",
        type=float,
        default=0.1,
        help="simulated seconds between live ego monitor refreshes",
    )
    run.add_argument(
        "--decouple-vlm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "run physics under a zero-velocity watchdog while the next NaVILA "
            "inference is pending"
        ),
    )
    run.add_argument(
        "--max-control-steps",
        type=int,
        default=500,
        help="total control-step limit; use 0 for unlimited",
    )
    run.add_argument(
        "--max-decisions",
        type=int,
        default=8,
        help="VLM decision limit; use 0 for unlimited",
    )
    run.add_argument("--output")
    run.set_defaults(func=_run)

    teleop = subparsers.add_parser(
        "teleop",
        help="drive simulator locomotion from keyboard and record pose-tagged anchors",
    )
    teleop.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO),
        help="project-owned navigation scenario JSON used for the initial reset contract",
    )
    teleop.add_argument("--device", default="cuda:0")
    teleop.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    _add_renderer_options(teleop)
    teleop.add_argument(
        "--randomized-play",
        action="store_true",
        help="retain MJLab reset/domain randomization instead of the deterministic scene test profile",
    )
    teleop.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="zero-command Go2 policy steps after reset",
    )
    teleop.add_argument(
        "--forward-speed",
        type=float,
        default=0.5,
        help="forward/backward keyboard speed in m/s",
    )
    teleop.add_argument(
        "--strafe-speed",
        type=float,
        default=0.35,
        help="Q/E keyboard strafe speed in m/s",
    )
    teleop.add_argument(
        "--turn-rate",
        type=float,
        default=0.8,
        help="A/D keyboard yaw rate in rad/s",
    )
    teleop.add_argument(
        "--command-hold",
        type=float,
        default=0.35,
        help="seconds a movement key remains active without terminal key-repeat",
    )
    teleop.add_argument(
        "--capture-interval",
        type=float,
        default=0.2,
        help="simulated seconds between camera captures",
    )
    teleop.add_argument(
        "--live-monitor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the captured ego RGB beside teleoperation diagnostics",
    )
    teleop.add_argument(
        "--no-preview",
        action="store_true",
        help="run physics and pose logging without connecting to an Orca/camera renderer",
    )
    teleop.add_argument(
        "--no-realtime",
        action="store_true",
        help="run as fast as possible instead of pacing physics to wall-clock time",
    )
    teleop.add_argument(
        "--max-control-steps",
        type=int,
        default=0,
        help="control-step limit; use 0 for unlimited",
    )
    teleop.add_argument(
        "--max-sim-time",
        type=float,
        default=0.0,
        help="simulated-time limit in seconds; use 0 for unlimited",
    )
    teleop.add_argument("--output")
    teleop.set_defaults(func=_teleop)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
