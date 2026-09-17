#!/usr/bin/env python3
"""Run the portable NaVILA camera/inference/motion loop on a Unitree robot.

Motor execution is disabled unless ``--execute-actions`` is supplied.  The
expected deployment is: this lightweight client on the robot, and the existing
NaVILA TCP inference server on the remote NVIDIA workstation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navila_orca.actions import ActionParseError, parse_velocity_command  # noqa: E402
from navila_orca.frames import brighten_images  # noqa: E402
from navila_orca.hardware.unitree_gateway import (  # noqa: E402
    A2_GSTREAMER_PIPELINE,
    CameraCaptureWorker,
    GStreamerCameraSource,
    Go2VideoClientCameraSource,
    HardwareDecisionRecorder,
    JpegFrameHistory,
    SafeMotionExecutor,
    SafetyLimits,
    UnitreeSportController,
)
from navila_orca.vlm_client import LengthPrefixedJsonVLMClient  # noqa: E402


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portable A2/Go2 camera to remote-NaVILA gateway."
    )
    instruction_group = parser.add_mutually_exclusive_group(required=True)
    instruction_group.add_argument(
        "--instruction",
        help="single navigation instruction",
    )
    instruction_group.add_argument(
        "--waypoint-instruction-file",
        type=Path,
        help="UTF-8 file containing one navigation instruction per line",
    )
    parser.add_argument("--robot-model", choices=("a2", "go2"), default="go2")
    parser.add_argument(
        "--network-interface",
        required=True,
        help=(
            "local interface carrying Unitree DDS, such as enx.../enp2s0 for "
            "Ethernet or wlp... for Wi-Fi"
        ),
    )
    parser.add_argument(
        "--vlm-host",
        help="remote NaVILA host; optional only with --camera-check-output",
    )
    parser.add_argument("--vlm-port", type=int, default=54321)
    parser.add_argument("--vlm-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--capture-hz", type=_positive_float, default=5.0)
    parser.add_argument(
        "--camera-rpc-timeout", type=_positive_float, default=2.0,
        help="SDK2 VideoClient request timeout in seconds",
    )
    parser.add_argument(
        "--camera-frame-retries", type=_nonnegative_int, default=3,
        help="retries for one malformed or failed Go2 JPEG sample",
    )
    parser.add_argument(
        "--camera-error-retries", type=_nonnegative_int, default=3,
        help="bad capture cycles tolerated before the camera fails closed",
    )
    parser.add_argument(
        "--image-brightness",
        type=_positive_float,
        default=1.0,
        help="brightness multiplier applied to frames before VLM inference (1.0 unchanged)",
    )
    parser.add_argument("--camera-warmup-frames", type=int, default=8)
    parser.add_argument("--camera-timeout", type=_positive_float, default=15.0)
    parser.add_argument(
        "--camera-pipeline",
        help="override the A2 multicast GStreamer pipeline",
    )
    parser.add_argument(
        "--camera-check-output",
        type=Path,
        help="capture one warmed-up frame here and exit without inference/motion",
    )
    parser.add_argument(
        "--max-history-frames",
        type=int,
        default=0,
        help="0 keeps the complete compressed history, matching NaVILA",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=1,
        help="maximum VLM decisions per instruction/waypoint",
    )
    parser.add_argument("--decision-pause", type=_nonnegative_float, default=0.2)
    parser.add_argument("--max-forward-mps", type=_nonnegative_float, default=0.20)
    parser.add_argument("--max-lateral-mps", type=_nonnegative_float, default=0.0)
    parser.add_argument("--max-yaw-rps", type=_nonnegative_float, default=0.35)
    parser.add_argument("--max-action-seconds", type=_positive_float, default=0.75)
    parser.add_argument("--command-hz", type=_positive_float, default=50.0)
    parser.add_argument(
        "--print-timings",
        action="store_true",
        help="print per-decision capture, VLM, action, pause, and total timings",
    )
    parser.add_argument(
        "--execute-actions",
        action="store_true",
        help="ARM MOTORS; omitted by default, so inference is recorded only",
    )
    parser.add_argument(
        "--balance-stand",
        action="store_true",
        help="call BalanceStand once before execution; requires --execute-actions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="decision-sample directory (a timestamped default is used otherwise)",
    )
    parser.add_argument("--scene-id", default="physical_site")
    parser.add_argument("--episode-id")
    return parser


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "outputs" / "unitree" / stamp / "decision_samples"


def _load_instructions(args: argparse.Namespace) -> list[str]:
    if args.instruction is not None:
        return [str(args.instruction).strip()]
    path = args.waypoint_instruction_file.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"waypoint instruction file does not exist: {path}")
    instructions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    instructions = [line for line in instructions if line]
    if not instructions:
        raise ValueError(f"waypoint instruction file is empty: {path}")
    return instructions


def _validate_args(args: argparse.Namespace) -> None:
    if args.camera_warmup_frames <= 0:
        raise ValueError("--camera-warmup-frames must be positive")
    if args.max_history_frames < 0:
        raise ValueError("--max-history-frames must be non-negative")
    if args.max_decisions <= 0:
        raise ValueError("--max-decisions must be positive")
    if not 1 <= args.vlm_port <= 65535:
        raise ValueError("--vlm-port must be between 1 and 65535")
    if args.balance_stand and not args.execute_actions:
        raise ValueError("--balance-stand requires --execute-actions")
    if args.execute_actions and args.camera_check_output is not None:
        raise ValueError("--execute-actions cannot be used with --camera-check-output")
    if args.camera_check_output is None and not str(args.vlm_host or "").strip():
        raise ValueError("--vlm-host is required unless --camera-check-output is used")
    if args.robot_model == "go2" and args.camera_pipeline:
        raise ValueError("--camera-pipeline applies only to the A2 camera")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        instructions = _load_instructions(args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    waypoint_count = len(instructions)

    output = (args.output or _default_output()).expanduser().resolve()
    episode_id = args.episode_id or output.parent.name
    limits = SafetyLimits(
        max_forward_mps=args.max_forward_mps,
        max_lateral_mps=args.max_lateral_mps,
        max_yaw_rps=args.max_yaw_rps,
        max_duration_s=args.max_action_seconds,
    )

    controller = None
    executor = None
    camera_worker = None
    interrupted = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if executor is not None:
            executor.request_stop()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sighup = signal.signal(signal.SIGHUP, request_stop)
    try:
        if args.robot_model == "go2":
            camera = Go2VideoClientCameraSource(
                network_interface=args.network_interface,
                timeout_s=args.camera_rpc_timeout,
                initialize_channel=True,
                frame_retries=args.camera_frame_retries,
            )
        else:
            pipeline = args.camera_pipeline or A2_GSTREAMER_PIPELINE.format(
                interface=args.network_interface
            )
            camera = GStreamerCameraSource(pipeline)

        if args.execute_actions:
            controller = UnitreeSportController(
                robot_model=args.robot_model,
                network_interface=args.network_interface,
                initialize_channel=args.robot_model != "go2",
            )
            executor = SafeMotionExecutor(
                controller, limits=limits, command_hz=args.command_hz
            )
            if args.balance_stand:
                controller.balance_stand()

        history = JpegFrameHistory(max_frames=args.max_history_frames)
        camera_worker = CameraCaptureWorker(
            camera,
            history,
            capture_hz=args.capture_hz,
            max_consecutive_errors=args.camera_error_retries,
        )
        camera_worker.start()
        if not history.wait_for(args.camera_warmup_frames, args.camera_timeout):
            camera_worker.ensure_healthy()
            raise RuntimeError(
                f"camera produced only {len(history)} frame(s) within "
                f"{args.camera_timeout:g}s"
            )

        if args.camera_check_output is not None:
            images, _ = history.sample()
            camera_check_output = args.camera_check_output.expanduser().resolve()
            camera_check_output.parent.mkdir(parents=True, exist_ok=True)
            images[-1].save(camera_check_output, format="JPEG", quality=95)
            print(
                f"camera check passed ({len(history)} frames); "
                f"latest frame={camera_check_output}",
                flush=True,
            )
            return 0

        vlm = LengthPrefixedJsonVLMClient(
            args.vlm_host, args.vlm_port, timeout_s=args.vlm_timeout
        )
        recorder = HardwareDecisionRecorder(
            output,
            robot_model=args.robot_model,
            episode_id=episode_id,
            scene_id=args.scene_id,
        )
        print(
            f"camera ready ({len(history)} frames); motors="
            f"{'ARMED' if args.execute_actions else 'DRY-RUN'}; output={output}",
            flush=True,
        )

        total_decisions = 0
        for waypoint_index, raw_instruction in enumerate(instructions, start=1):
            if interrupted:
                break
            instruction = raw_instruction
            if waypoint_count > 1:
                instruction = (
                    f"Waypoint {waypoint_index} of {waypoint_count}. "
                    f"{raw_instruction}"
                )
            print(
                f"waypoint={waypoint_index}/{waypoint_count} "
                f"instruction={raw_instruction!r}",
                flush=True,
            )
            waypoint_stopped = False
            for waypoint_decision in range(1, args.max_decisions + 1):
                if interrupted:
                    break
                decision_started = time.perf_counter()
                timings: dict[str, float] = {}

                phase_started = time.perf_counter()
                camera_worker.ensure_healthy()
                timings["health"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                images, history_frames = history.sample()
                timings["sample"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                images = brighten_images(images, args.image_brightness)
                timings["preprocess"] = time.perf_counter() - phase_started

                if args.print_timings:
                    print(
                        f"timing waypoint={waypoint_index}/{waypoint_count} "
                        f"decision={waypoint_decision}/{args.max_decisions} "
                        "phase=vlm_start",
                        flush=True,
                    )
                phase_started = time.perf_counter()
                baseline_output = vlm.infer(images, instruction)
                timings["vlm"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                try:
                    parsed = parse_velocity_command(baseline_output)
                except ActionParseError:
                    if controller is not None:
                        controller.stop_move()
                    raise
                timings["parse"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                bounded = limits.apply(parsed)
                total_decisions += 1
                recorder.record(
                    decision=total_decisions,
                    instruction=instruction,
                    images=images,
                    history_frames=history_frames,
                    baseline_output=baseline_output,
                    parsed_command=parsed,
                    bounded_command=bounded,
                    executed=args.execute_actions,
                    waypoint_index=waypoint_index,
                    waypoint_count=waypoint_count,
                )
                timings["record"] = time.perf_counter() - phase_started
                print(
                    f"waypoint={waypoint_index}/{waypoint_count} "
                    f"decision={waypoint_decision}/{args.max_decisions} "
                    f"model={baseline_output!r} "
                    f"bounded=(vx={bounded.vx:.3f}, vy={bounded.vy:.3f}, "
                    f"wz={bounded.wz:.3f}, duration={bounded.duration_s:.2f}, "
                    f"stop={bounded.stop})",
                    flush=True,
                )
                if bounded.stop:
                    phase_started = time.perf_counter()
                    if controller is not None:
                        controller.stop_move()
                    timings["action"] = time.perf_counter() - phase_started
                    timings["pause"] = 0.0
                    waypoint_stopped = True
                    print(
                        f"waypoint={waypoint_index}/{waypoint_count} complete",
                        flush=True,
                    )
                    if args.print_timings:
                        timings["processing"] = sum(
                            timings[name]
                            for name in ("health", "sample", "preprocess", "vlm", "parse", "record")
                        )
                        timings["total"] = time.perf_counter() - decision_started
                        print(
                            f"timing waypoint={waypoint_index}/{waypoint_count} "
                            f"decision={waypoint_decision}/{args.max_decisions} "
                            + " ".join(
                                f"{name}={timings[name]:.3f}s"
                                for name in (
                                    "health", "sample", "preprocess", "vlm",
                                    "parse", "record", "processing", "action",
                                    "pause", "total",
                                )
                            ),
                            flush=True,
                        )
                    break
                phase_started = time.perf_counter()
                if executor is not None:
                    executor.execute(parsed)
                timings["action"] = time.perf_counter() - phase_started
                if interrupted:
                    break
                phase_started = time.perf_counter()
                time.sleep(args.decision_pause)
                timings["pause"] = time.perf_counter() - phase_started
                if args.print_timings:
                    timings["processing"] = sum(
                        timings[name]
                        for name in ("health", "sample", "preprocess", "vlm", "parse", "record")
                    )
                    timings["total"] = time.perf_counter() - decision_started
                    print(
                        f"timing waypoint={waypoint_index}/{waypoint_count} "
                        f"decision={waypoint_decision}/{args.max_decisions} "
                        + " ".join(
                            f"{name}={timings[name]:.3f}s"
                            for name in (
                                "health", "sample", "preprocess", "vlm",
                                "parse", "record", "processing", "action",
                                "pause", "total",
                            )
                        ),
                        flush=True,
                    )

            if interrupted:
                break
            if not waypoint_stopped:
                raise RuntimeError(
                    f"waypoint {waypoint_index}/{waypoint_count} reached "
                    f"--max-decisions ({args.max_decisions}) without a stop action"
                )
            if waypoint_index < waypoint_count:
                history.clear()
                if not history.wait_for(args.camera_warmup_frames, args.camera_timeout):
                    camera_worker.ensure_healthy()
                    raise RuntimeError(
                        f"camera produced only {len(history)} frame(s) while "
                        f"starting waypoint {waypoint_index + 1}/{waypoint_count}"
                    )
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        print(f"unitree gateway failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGHUP, previous_sighup)
        if camera_worker is not None:
            camera_worker.close()
            if camera_worker.dropped_frames:
                print(
                    f"camera warning: recovered after {camera_worker.dropped_frames} "
                    "dropped capture cycle(s)",
                    file=sys.stderr,
                )
        if controller is not None:
            try:
                controller.stop_move()
            except Exception as exc:
                print(f"warning: final StopMove failed: {exc}", file=sys.stderr)

    print("stopped safely" if interrupted else "run complete", flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
