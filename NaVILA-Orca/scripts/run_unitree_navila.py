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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portable A2/Go2 camera to remote-NaVILA gateway."
    )
    parser.add_argument("--instruction", required=True)
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
    parser.add_argument("--max-decisions", type=int, default=1)
    parser.add_argument("--decision-pause", type=_nonnegative_float, default=0.2)
    parser.add_argument("--max-forward-mps", type=_nonnegative_float, default=0.20)
    parser.add_argument("--max-lateral-mps", type=_nonnegative_float, default=0.0)
    parser.add_argument("--max-yaw-rps", type=_nonnegative_float, default=0.35)
    parser.add_argument("--max-action-seconds", type=_positive_float, default=0.75)
    parser.add_argument("--command-hz", type=_positive_float, default=50.0)
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
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

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
    try:
        if args.robot_model == "go2":
            camera = Go2VideoClientCameraSource(
                network_interface=args.network_interface,
                initialize_channel=True,
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
            camera, history, capture_hz=args.capture_hz
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

        for decision in range(1, args.max_decisions + 1):
            if interrupted:
                break
            camera_worker.ensure_healthy()
            images, history_frames = history.sample()
            baseline_output = vlm.infer(images, args.instruction)
            try:
                parsed = parse_velocity_command(baseline_output)
            except ActionParseError:
                if controller is not None:
                    controller.stop_move()
                raise
            bounded = limits.apply(parsed)
            recorder.record(
                decision=decision,
                instruction=args.instruction,
                images=images,
                history_frames=history_frames,
                baseline_output=baseline_output,
                parsed_command=parsed,
                bounded_command=bounded,
                executed=args.execute_actions,
            )
            print(
                f"decision={decision} model={baseline_output!r} "
                f"bounded=(vx={bounded.vx:.3f}, vy={bounded.vy:.3f}, "
                f"wz={bounded.wz:.3f}, duration={bounded.duration_s:.2f}, "
                f"stop={bounded.stop})",
                flush=True,
            )
            if bounded.stop:
                if controller is not None:
                    controller.stop_move()
                break
            if executor is not None:
                executor.execute(parsed)
            if interrupted:
                break
            time.sleep(args.decision_pause)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        print(f"unitree gateway failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if camera_worker is not None:
            camera_worker.close()
        if controller is not None:
            try:
                controller.stop_move()
            except Exception as exc:
                print(f"warning: final StopMove failed: {exc}", file=sys.stderr)

    print("stopped safely" if interrupted else "run complete", flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
