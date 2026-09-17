#!/usr/bin/env python3
"""Run the portable NaVILA camera/inference/motion loop on a Unitree robot.

Motor execution is disabled unless ``--execute-actions`` is supplied.  The
expected deployment is: this lightweight client on the robot, and the existing
NaVILA TCP inference server on the remote NVIDIA workstation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navila_orca.actions import ActionParseError, parse_velocity_command  # noqa: E402
from navila_orca.contracts import VelocityCommand  # noqa: E402
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
from navila_orca.openai_router import (  # noqa: E402
    GoalSeerResult,
    LandmarkSeerResult,
    OpenAIRouterError,
    OpenAIQueryRouter,
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
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
    parser.add_argument(
        "--landmark-plan",
        type=Path,
        help=(
            "Memory Guide plan JSON containing landmark_steps and scan reference "
            "images; required with --landmark-seer"
        ),
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
    parser.add_argument(
        "--landmark-seer",
        "--goal-seer",
        dest="landmark_seer",
        action="store_true",
        help=(
            "use OpenAI vision to verify landmarks and the requested item, and "
            "supervise approach/inspection; requires --landmark-plan"
        ),
    )
    parser.add_argument(
        "--landmark-seer-model-id",
        default=os.environ.get("NAVILA_OPENAI_MODEL", "gpt-5.6-luna"),
        help="OpenAI model used by the landmark seer",
    )
    parser.add_argument(
        "--landmark-seer-base-url",
        default=os.environ.get("NAVILA_OPENAI_BASE_URL"),
        help="optional OpenAI-compatible base URL for the landmark seer",
    )
    parser.add_argument(
        "--landmark-seer-max-turns",
        type=_positive_int,
        default=24,
        help="maximum in-place sweep turns per landmark",
    )
    parser.add_argument(
        "--landmark-seer-retries",
        type=_nonnegative_int,
        default=2,
        help="retries for a transient seer API or structured-output failure",
    )
    parser.add_argument(
        "--landmark-seer-turn-degrees",
        type=_positive_float,
        default=30.0,
        help="nominal in-place sweep increment before rechecking the landmark",
    )
    parser.add_argument(
        "--landmark-seer-check-interval",
        type=_positive_float,
        default=2.0,
        help="seconds between visual visibility checks during approach",
    )
    parser.add_argument(
        "--landmark-seer-min-confidence",
        type=_nonnegative_float,
        default=0.60,
        help="minimum seer confidence required to permit approach",
    )
    parser.add_argument(
        "--landmark-seer-center-bearing-degrees",
        type=_positive_float,
        default=25.0,
        help="maximum accepted absolute image bearing before centering in place",
    )
    parser.add_argument(
        "--landmark-seer-max-forward-action-seconds",
        type=_positive_float,
        default=0.50,
        help="maximum forward action duration while approaching a scanned landmark",
    )
    parser.add_argument(
        "--landmark-seer-sweep-direction",
        choices=("left", "right"),
        default="right",
        help="direction used for blind in-place landmark reacquisition",
    )
    parser.add_argument(
        "--goal-seer-max-inspection-turns",
        type=_nonnegative_int,
        default=24,
        help="maximum in-place inspection turns per landmark after arrival",
    )
    parser.add_argument(
        "--goal-seer-min-item-confidence",
        type=_nonnegative_float,
        default=0.70,
        help="minimum item confidence required to declare the search successful",
    )
    parser.add_argument(
        "--goal-seer-inspection-turn-degrees",
        type=_positive_float,
        default=30.0,
        help="in-place turn increment while searching the current landmark area",
    )
    parser.add_argument("--decision-pause", type=_nonnegative_float, default=0.2)
    parser.add_argument("--max-forward-mps", type=_nonnegative_float, default=0.8)
    parser.add_argument("--max-lateral-mps", type=_nonnegative_float, default=0.0)
    parser.add_argument("--max-yaw-rps", type=_nonnegative_float, default=0.8)
    parser.add_argument("--max-action-seconds", type=_positive_float, default=2.00)
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


def _load_landmark_plan(
    path: Path | None,
    *,
    waypoint_count: int,
) -> tuple[str, list[dict[str, Any]]]:
    if path is None:
        raise ValueError("--landmark-seer requires --landmark-plan")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"landmark plan does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read landmark plan: {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"landmark plan is not a JSON object: {resolved}")
    raw_target = payload.get("target_display_name") or payload.get("target")
    target_name = str(raw_target or "requested item").strip().replace("_", " ")
    if not target_name:
        target_name = "requested item"
    raw_steps = payload.get("landmark_steps") if isinstance(payload, Mapping) else None
    if not isinstance(raw_steps, list) or len(raw_steps) != waypoint_count:
        raise ValueError(
            "landmark plan landmark_steps must contain exactly one entry per "
            f"waypoint ({waypoint_count} expected)"
        )
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"landmark plan step {index} is not an object")
        name = str(raw_step.get("search_location", "")).strip()
        description = str(raw_step.get("description", "")).strip()
        reference_image = str(raw_step.get("reference_image", "")).strip()
        if not name:
            raise ValueError(f"landmark plan step {index} has no search_location")
        if not reference_image:
            raise ValueError(
                f"landmark plan step {index} ({name!r}) has no reference_image; "
                "regenerate the plan after the landmark scan"
            )
        steps.append(
            {
                "name": name,
                "description": description,
                "reference_image": reference_image,
            }
        )
    return target_name, steps


def _load_reference_image(step: Mapping[str, Any]) -> Image.Image:
    path = Path(str(step["reference_image"])).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"landmark reference image does not exist: {path}; copy the scan "
            "images to this machine or regenerate the scan here"
        )
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise ValueError(f"could not decode landmark reference image {path}: {exc}") from exc


def _seer_target_visible(
    result: LandmarkSeerResult,
    *,
    minimum_confidence: float,
) -> bool:
    return result.target_visible and result.confidence >= minimum_confidence


def _seer_target_centered(
    result: LandmarkSeerResult,
    *,
    minimum_confidence: float,
    maximum_bearing: float,
) -> bool:
    return _seer_target_visible(
        result, minimum_confidence=minimum_confidence
    ) and abs(result.bearing_degrees) <= maximum_bearing


def _seer_requires_inspection_stop(
    result: LandmarkSeerResult,
    *,
    minimum_confidence: float,
    maximum_bearing: float,
) -> bool:
    """Return true when the landmark is centered but advancing is unsafe."""

    return _seer_target_centered(
        result,
        minimum_confidence=minimum_confidence,
        maximum_bearing=maximum_bearing,
    ) and not result.safe_to_advance


def _limit_landmark_approach_command(
    command: VelocityCommand,
    *,
    max_forward_action_seconds: float,
) -> VelocityCommand:
    """Prevent one landmark approach decision from becoming a long drive."""

    if command.stop or command.vx <= 0.0:
        return command
    return VelocityCommand(
        command.vx,
        command.vy,
        command.wz,
        min(command.duration_s, max_forward_action_seconds),
    )


def _goal_item_found(
    result: GoalSeerResult,
    *,
    minimum_confidence: float,
) -> bool:
    return result.item_visible and result.item_confidence >= minimum_confidence


def _goal_landmark_centered(
    result: GoalSeerResult,
    *,
    minimum_confidence: float,
    maximum_bearing: float,
) -> bool:
    return (
        result.landmark_visible
        and result.landmark_confidence >= minimum_confidence
        and abs(result.landmark_bearing_degrees) <= maximum_bearing
    )


def _goal_requires_inspection(
    result: GoalSeerResult,
    *,
    minimum_landmark_confidence: float,
    maximum_bearing: float,
) -> bool:
    return _goal_landmark_centered(
        result,
        minimum_confidence=minimum_landmark_confidence,
        maximum_bearing=maximum_bearing,
    ) and not result.safe_to_advance


def _landmark_result_from_goal(result: GoalSeerResult) -> LandmarkSeerResult:
    """Adapt the combined verifier result for the existing acquisition sweep."""

    return LandmarkSeerResult(
        target_visible=result.landmark_visible,
        confidence=result.landmark_confidence,
        relative_position=result.landmark_relative_position,
        bearing_degrees=result.landmark_bearing_degrees,
        distance_state=result.landmark_distance_state,
        safe_to_advance=result.safe_to_advance,
        rationale=result.rationale,
    )


def _run_landmark_seer_check(
    *,
    seer: OpenAIQueryRouter,
    step: Mapping[str, Any],
    reference_image: Image.Image,
    history: JpegFrameHistory,
    camera_worker: CameraCaptureWorker,
    brightness: float,
    minimum_confidence: float,
    maximum_bearing: float,
    retries: int,
    waypoint_index: int,
    waypoint_count: int,
) -> LandmarkSeerResult:
    camera_worker.ensure_healthy()
    live_images, _ = history.sample()
    live_image = brighten_images(live_images, brightness)[-1]
    result = None
    for attempt in range(retries + 1):
        try:
            result = seer.see_landmark(
                target_name=str(step["name"]),
                target_description=str(step.get("description", "")),
                reference_image=reference_image,
                live_image=live_image,
            )
            break
        except OpenAIRouterError as exc:
            if attempt >= retries:
                raise
            print(
                f"landmark-seer retry={attempt + 1}/{retries} "
                f"reason={exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.20)
    assert result is not None
    visible = _seer_target_visible(result, minimum_confidence=minimum_confidence)
    centered = _seer_target_centered(
        result,
        minimum_confidence=minimum_confidence,
        maximum_bearing=maximum_bearing,
    )
    print(
        f"landmark-seer waypoint={waypoint_index}/{waypoint_count} "
        f"visible={visible} centered={centered} raw_visible={result.target_visible} "
        f"confidence={result.confidence:.2f} position={result.relative_position} "
        f"bearing={result.bearing_degrees:+.1f} distance={result.distance_state} "
        f"safe_to_advance={result.safe_to_advance} rationale={result.rationale!r}",
        flush=True,
    )
    return result


def _run_goal_seer_check(
    *,
    seer: OpenAIQueryRouter,
    item_name: str,
    step: Mapping[str, Any],
    reference_image: Image.Image,
    history: JpegFrameHistory,
    camera_worker: CameraCaptureWorker,
    brightness: float,
    minimum_item_confidence: float,
    minimum_landmark_confidence: float,
    maximum_bearing: float,
    retries: int,
    waypoint_index: int,
    waypoint_count: int,
) -> GoalSeerResult:
    camera_worker.ensure_healthy()
    live_images, _ = history.sample()
    live_image = brighten_images(live_images, brightness)[-1]
    result = None
    for attempt in range(retries + 1):
        try:
            result = seer.see_goal(
                item_name=item_name,
                item_description="",
                landmark_name=str(step["name"]),
                landmark_description=str(step.get("description", "")),
                reference_image=reference_image,
                live_image=live_image,
            )
            break
        except OpenAIRouterError as exc:
            if attempt >= retries:
                raise
            print(
                f"goal-seer retry={attempt + 1}/{retries} reason={exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.20)
    assert result is not None
    landmark_centered = _goal_landmark_centered(
        result,
        minimum_confidence=minimum_landmark_confidence,
        maximum_bearing=maximum_bearing,
    )
    item_found = _goal_item_found(
        result, minimum_confidence=minimum_item_confidence
    )
    print(
        f"goal-seer waypoint={waypoint_index}/{waypoint_count} "
        f"landmark_visible={result.landmark_visible} "
        f"landmark_centered={landmark_centered} "
        f"landmark_confidence={result.landmark_confidence:.2f} "
        f"landmark_bearing={result.landmark_bearing_degrees:+.1f} "
        f"distance={result.landmark_distance_state} "
        f"safe_to_advance={result.safe_to_advance} "
        f"item_visible={result.item_visible} item_found={item_found} "
        f"item_confidence={result.item_confidence:.2f} "
        f"item_position={result.item_relative_position} "
        f"item_bearing={result.item_bearing_degrees:+.1f} "
        f"rationale={result.rationale!r}",
        flush=True,
    )
    return result


def _inspect_landmark_for_item(
    *,
    args: argparse.Namespace,
    seer: OpenAIQueryRouter,
    item_name: str,
    step: Mapping[str, Any],
    reference_image: Image.Image,
    history: JpegFrameHistory,
    camera_worker: CameraCaptureWorker,
    executor: SafeMotionExecutor | None,
    initial_result: GoalSeerResult,
    waypoint_index: int,
    waypoint_count: int,
) -> tuple[GoalSeerResult, bool]:
    """Search a reached landmark in place, stopping immediately if the item appears."""

    if executor is not None:
        executor.controller.stop_move()
    print(
        f"goal-inspection-start waypoint={waypoint_index}/{waypoint_count} "
        f"landmark={step['name']!r} item={item_name!r}",
        flush=True,
    )
    effective_turn_degrees = min(
        args.goal_seer_inspection_turn_degrees,
        math.degrees(args.max_yaw_rps * args.max_action_seconds),
    )
    if effective_turn_degrees <= 0.0:
        raise ValueError("goal seer could not construct a positive inspection turn")
    inspection_turns = min(
        args.goal_seer_max_inspection_turns,
        max(1, math.ceil(360.0 / effective_turn_degrees)),
    )
    turn_sign = 1.0 if args.landmark_seer_sweep_direction == "left" else -1.0
    result = initial_result
    for turn_index in range(inspection_turns + 1):
        if _goal_item_found(
            result, minimum_confidence=args.goal_seer_min_item_confidence
        ):
            if executor is not None:
                executor.controller.stop_move()
            print(
                f"goal-found waypoint={waypoint_index}/{waypoint_count} "
                f"item={item_name!r} confidence={result.item_confidence:.2f}",
                flush=True,
            )
            return result, True
        if turn_index >= inspection_turns:
            break
        if executor is None:
            print(
                f"goal-inspection skipped waypoint={waypoint_index}/{waypoint_count} "
                "because --execute-actions was not supplied",
                flush=True,
            )
            break
        duration_s = math.radians(effective_turn_degrees) / args.max_yaw_rps
        turn_direction = "left" if turn_sign > 0.0 else "right"
        print(
            f"goal-inspection-turn waypoint={waypoint_index}/{waypoint_count} "
            f"turn={turn_index + 1}/{inspection_turns} "
            f"direction={turn_direction} degrees={effective_turn_degrees:.1f} "
            f"duration={duration_s:.2f}s",
            flush=True,
        )
        executor.execute(
            VelocityCommand(0.0, 0.0, turn_sign * args.max_yaw_rps, duration_s)
        )
        result = _run_goal_seer_check(
            seer=seer,
            item_name=item_name,
            step=step,
            reference_image=reference_image,
            history=history,
            camera_worker=camera_worker,
            brightness=args.image_brightness,
            minimum_item_confidence=args.goal_seer_min_item_confidence,
            minimum_landmark_confidence=args.landmark_seer_min_confidence,
            maximum_bearing=args.landmark_seer_center_bearing_degrees,
            retries=args.landmark_seer_retries,
            waypoint_index=waypoint_index,
            waypoint_count=waypoint_count,
        )
    print(
        f"goal-inspection-complete waypoint={waypoint_index}/{waypoint_count} "
        f"item={item_name!r} found=False",
        flush=True,
    )
    return result, False


def _acquire_landmark(
    *,
    args: argparse.Namespace,
    seer: OpenAIQueryRouter,
    step: Mapping[str, Any],
    reference_image: Image.Image,
    history: JpegFrameHistory,
    camera_worker: CameraCaptureWorker,
    executor: SafeMotionExecutor | None,
    waypoint_index: int,
    waypoint_count: int,
    initial_result: LandmarkSeerResult | None = None,
) -> LandmarkSeerResult:
    """Sweep in place and verify the target before permitting forward motion."""

    if args.max_yaw_rps <= 0.0:
        raise ValueError("landmark seer requires --max-yaw-rps greater than zero")
    effective_turn_degrees = min(
        args.landmark_seer_turn_degrees,
        math.degrees(args.max_yaw_rps * args.max_action_seconds),
    )
    if effective_turn_degrees <= 0.0:
        raise ValueError("landmark seer could not construct a positive turn increment")
    turn_sign = 1.0 if args.landmark_seer_sweep_direction == "left" else -1.0
    result = initial_result
    for turn_index in range(args.landmark_seer_max_turns + 1):
        if result is None:
            result = _run_landmark_seer_check(
                seer=seer,
                step=step,
                reference_image=reference_image,
                history=history,
                camera_worker=camera_worker,
                brightness=args.image_brightness,
                minimum_confidence=args.landmark_seer_min_confidence,
                maximum_bearing=args.landmark_seer_center_bearing_degrees,
                retries=args.landmark_seer_retries,
                waypoint_index=waypoint_index,
                waypoint_count=waypoint_count,
            )
        if _seer_target_centered(
            result,
            minimum_confidence=args.landmark_seer_min_confidence,
            maximum_bearing=args.landmark_seer_center_bearing_degrees,
        ):
            return result
        if turn_index >= args.landmark_seer_max_turns:
            break
        if executor is None:
            raise RuntimeError(
                f"landmark {step['name']!r} is not visible; dry-run cannot perform "
                "the in-place reacquisition sweep"
            )
        if _seer_target_visible(
            result, minimum_confidence=args.landmark_seer_min_confidence
        ):
            # Seer convention: negative bearing is left, positive is right.
            # SportClient uses positive wz for left, so turn against the
            # observed bearing to center the landmark before approaching.
            correction_degrees = min(
                abs(result.bearing_degrees), effective_turn_degrees
            )
            turn_sign = -1.0 if result.bearing_degrees > 0.0 else 1.0
            turn_reason = "center"
        else:
            correction_degrees = effective_turn_degrees
            turn_reason = "sweep"
        duration_s = math.radians(correction_degrees) / args.max_yaw_rps
        turn_direction = "left" if turn_sign > 0.0 else "right"
        print(
            f"landmark-acquire waypoint={waypoint_index}/{waypoint_count} "
            f"turn={turn_index + 1}/{args.landmark_seer_max_turns} "
            f"reason={turn_reason} "
            f"direction={turn_direction} "
            f"degrees={correction_degrees:.1f} duration={duration_s:.2f}s",
            flush=True,
        )
        executor.execute(
            # The action executor applies the same configured yaw and duration
            # safety caps used by NaVILA actions.
            VelocityCommand(0.0, 0.0, turn_sign * args.max_yaw_rps, duration_s)
        )
        result = None
    raise RuntimeError(
        f"landmark seer could not reacquire {step['name']!r} after "
        f"{args.landmark_seer_max_turns} in-place turns; stopped safely"
    )


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
    if args.landmark_seer and args.landmark_plan is None:
        raise ValueError("--landmark-seer requires --landmark-plan")
    if not 0.0 <= args.landmark_seer_min_confidence <= 1.0:
        raise ValueError("--landmark-seer-min-confidence must be between 0 and 1")
    if not 0.0 <= args.goal_seer_min_item_confidence <= 1.0:
        raise ValueError("--goal-seer-min-item-confidence must be between 0 and 1")
    if args.landmark_seer_turn_degrees > 90.0:
        raise ValueError("--landmark-seer-turn-degrees must be <= 90")
    if args.goal_seer_inspection_turn_degrees > 90.0:
        raise ValueError("--goal-seer-inspection-turn-degrees must be <= 90")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        instructions = _load_instructions(args)
        landmark_item_name, landmark_steps = (
            _load_landmark_plan(args.landmark_plan, waypoint_count=len(instructions))
            if args.landmark_seer
            else (None, None)
        )
        landmark_reference_images = (
            [_load_reference_image(step) for step in landmark_steps]
            if landmark_steps is not None
            else None
        )
        landmark_seer = (
            OpenAIQueryRouter.from_environment(
                model_id=args.landmark_seer_model_id,
                base_url=args.landmark_seer_base_url,
            )
            if args.landmark_seer
            else None
        )
    except (OSError, ValueError) as exc:
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
        mission_found = False
        for waypoint_index, raw_instruction in enumerate(instructions, start=1):
            if interrupted:
                break
            instruction = raw_instruction
            print(
                f"waypoint={waypoint_index}/{waypoint_count} "
                f"instruction={raw_instruction!r}",
                flush=True,
            )
            visibility_result: LandmarkSeerResult | None = None
            goal_result: GoalSeerResult | None = None
            last_goal_check = 0.0
            landmark_step = None
            landmark_reference = None
            waypoint_stopped = False
            if landmark_seer is not None:
                assert landmark_steps is not None
                assert landmark_reference_images is not None
                landmark_step = landmark_steps[waypoint_index - 1]
                landmark_reference = landmark_reference_images[waypoint_index - 1]
                visibility_result = _acquire_landmark(
                    args=args,
                    seer=landmark_seer,
                    step=landmark_step,
                    reference_image=landmark_reference,
                    history=history,
                    camera_worker=camera_worker,
                    executor=executor,
                    waypoint_index=waypoint_index,
                    waypoint_count=waypoint_count,
                )
                goal_result = _run_goal_seer_check(
                    seer=landmark_seer,
                    item_name=str(landmark_item_name),
                    step=landmark_step,
                    reference_image=landmark_reference,
                    history=history,
                    camera_worker=camera_worker,
                    brightness=args.image_brightness,
                    minimum_item_confidence=args.goal_seer_min_item_confidence,
                    minimum_landmark_confidence=args.landmark_seer_min_confidence,
                    maximum_bearing=args.landmark_seer_center_bearing_degrees,
                    retries=args.landmark_seer_retries,
                    waypoint_index=waypoint_index,
                    waypoint_count=waypoint_count,
                )
                last_goal_check = time.monotonic()
                if _goal_item_found(
                    goal_result, minimum_confidence=args.goal_seer_min_item_confidence
                ):
                    if controller is not None:
                        controller.stop_move()
                    print(
                        f"goal-found waypoint={waypoint_index}/{waypoint_count} "
                        f"item={landmark_item_name!r} "
                        f"confidence={goal_result.item_confidence:.2f}; search complete",
                        flush=True,
                    )
                    mission_found = True
                    waypoint_stopped = True
                else:
                    # The combined seer is the final centering authority. If it
                    # disagrees with the landmark-only pass, reacquire in place
                    # before allowing NaVILA to issue a forward action.
                    for _ in range(2):
                        if _goal_landmark_centered(
                            goal_result,
                            minimum_confidence=args.landmark_seer_min_confidence,
                            maximum_bearing=args.landmark_seer_center_bearing_degrees,
                        ):
                            break
                        visibility_result = _acquire_landmark(
                            args=args,
                            seer=landmark_seer,
                            step=landmark_step,
                            reference_image=landmark_reference,
                            history=history,
                            camera_worker=camera_worker,
                            executor=executor,
                            waypoint_index=waypoint_index,
                            waypoint_count=waypoint_count,
                            initial_result=_landmark_result_from_goal(goal_result),
                        )
                        goal_result = _run_goal_seer_check(
                            seer=landmark_seer,
                            item_name=str(landmark_item_name),
                            step=landmark_step,
                            reference_image=landmark_reference,
                            history=history,
                            camera_worker=camera_worker,
                            brightness=args.image_brightness,
                            minimum_item_confidence=args.goal_seer_min_item_confidence,
                            minimum_landmark_confidence=args.landmark_seer_min_confidence,
                            maximum_bearing=args.landmark_seer_center_bearing_degrees,
                            retries=args.landmark_seer_retries,
                            waypoint_index=waypoint_index,
                            waypoint_count=waypoint_count,
                        )
                        last_goal_check = time.monotonic()
                        if _goal_item_found(
                            goal_result,
                            minimum_confidence=args.goal_seer_min_item_confidence,
                        ):
                            if controller is not None:
                                controller.stop_move()
                            print(
                                f"goal-found waypoint={waypoint_index}/{waypoint_count} "
                                f"item={landmark_item_name!r} "
                                f"confidence={goal_result.item_confidence:.2f}; search complete",
                                flush=True,
                            )
                            mission_found = True
                            waypoint_stopped = True
                            break
                    if not mission_found and not _goal_landmark_centered(
                        goal_result,
                        minimum_confidence=args.landmark_seer_min_confidence,
                        maximum_bearing=args.landmark_seer_center_bearing_degrees,
                    ):
                        raise RuntimeError(
                            f"goal seer could not center landmark {landmark_step['name']!r}; "
                            "refusing NaVILA forward motion"
                        )
                    if (
                        not mission_found
                        and _goal_requires_inspection(
                            goal_result,
                            minimum_landmark_confidence=args.landmark_seer_min_confidence,
                            maximum_bearing=args.landmark_seer_center_bearing_degrees,
                        )
                    ):
                        goal_result, found = _inspect_landmark_for_item(
                            args=args,
                            seer=landmark_seer,
                            item_name=str(landmark_item_name),
                            step=landmark_step,
                            reference_image=landmark_reference,
                            history=history,
                            camera_worker=camera_worker,
                            executor=executor,
                            initial_result=goal_result,
                            waypoint_index=waypoint_index,
                            waypoint_count=waypoint_count,
                        )
                        waypoint_stopped = True
                        mission_found = found

            decision_limit = 0 if waypoint_stopped else args.max_decisions
            for waypoint_decision in range(1, decision_limit + 1):
                if interrupted:
                    break
                decision_started = time.perf_counter()
                timings: dict[str, float] = {}

                if (
                    landmark_seer is not None
                    and landmark_step is not None
                    and landmark_reference is not None
                    and (
                        goal_result is None
                        or time.monotonic() - last_goal_check
                        >= args.landmark_seer_check_interval
                    )
                ):
                    goal_result = _run_goal_seer_check(
                        seer=landmark_seer,
                        item_name=str(landmark_item_name),
                        step=landmark_step,
                        reference_image=landmark_reference,
                        history=history,
                        camera_worker=camera_worker,
                        brightness=args.image_brightness,
                        minimum_item_confidence=args.goal_seer_min_item_confidence,
                        minimum_landmark_confidence=args.landmark_seer_min_confidence,
                        maximum_bearing=args.landmark_seer_center_bearing_degrees,
                        retries=args.landmark_seer_retries,
                        waypoint_index=waypoint_index,
                        waypoint_count=waypoint_count,
                    )
                    last_goal_check = time.monotonic()
                    visibility_result = _landmark_result_from_goal(goal_result)
                    if _goal_item_found(
                        goal_result, minimum_confidence=args.goal_seer_min_item_confidence
                    ):
                        if controller is not None:
                            controller.stop_move()
                        print(
                            f"goal-found waypoint={waypoint_index}/{waypoint_count} "
                            f"item={landmark_item_name!r} "
                            f"confidence={goal_result.item_confidence:.2f}; search complete",
                            flush=True,
                        )
                        mission_found = True
                        waypoint_stopped = True
                        break
                    if not _goal_landmark_centered(
                        goal_result,
                        minimum_confidence=args.landmark_seer_min_confidence,
                        maximum_bearing=args.landmark_seer_center_bearing_degrees,
                    ):
                        visibility_result = _acquire_landmark(
                            args=args,
                            seer=landmark_seer,
                            step=landmark_step,
                            reference_image=landmark_reference,
                            history=history,
                            camera_worker=camera_worker,
                            executor=executor,
                            waypoint_index=waypoint_index,
                            waypoint_count=waypoint_count,
                            initial_result=visibility_result,
                        )
                        goal_result = _run_goal_seer_check(
                            seer=landmark_seer,
                            item_name=str(landmark_item_name),
                            step=landmark_step,
                            reference_image=landmark_reference,
                            history=history,
                            camera_worker=camera_worker,
                            brightness=args.image_brightness,
                            minimum_item_confidence=args.goal_seer_min_item_confidence,
                            minimum_landmark_confidence=args.landmark_seer_min_confidence,
                            maximum_bearing=args.landmark_seer_center_bearing_degrees,
                            retries=args.landmark_seer_retries,
                            waypoint_index=waypoint_index,
                            waypoint_count=waypoint_count,
                        )
                        last_goal_check = time.monotonic()
                        if _goal_item_found(
                            goal_result,
                            minimum_confidence=args.goal_seer_min_item_confidence,
                        ):
                            if controller is not None:
                                controller.stop_move()
                            print(
                                f"goal-found waypoint={waypoint_index}/{waypoint_count} "
                                f"item={landmark_item_name!r} "
                                f"confidence={goal_result.item_confidence:.2f}; search complete",
                                flush=True,
                            )
                            mission_found = True
                            waypoint_stopped = True
                            break
                        if not _goal_landmark_centered(
                            goal_result,
                            minimum_confidence=args.landmark_seer_min_confidence,
                            maximum_bearing=args.landmark_seer_center_bearing_degrees,
                        ):
                            raise RuntimeError(
                                f"goal seer could not center landmark {landmark_step['name']!r}; "
                                "refusing NaVILA forward motion"
                            )
                    if (
                        not mission_found
                        and _goal_requires_inspection(
                            goal_result,
                            minimum_landmark_confidence=args.landmark_seer_min_confidence,
                            maximum_bearing=args.landmark_seer_center_bearing_degrees,
                        )
                    ):
                        goal_result, found = _inspect_landmark_for_item(
                            args=args,
                            seer=landmark_seer,
                            item_name=str(landmark_item_name),
                            step=landmark_step,
                            reference_image=landmark_reference,
                            history=history,
                            camera_worker=camera_worker,
                            executor=executor,
                            initial_result=goal_result,
                            waypoint_index=waypoint_index,
                            waypoint_count=waypoint_count,
                        )
                        waypoint_stopped = True
                        mission_found = found
                        break

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
                action_command = parsed
                if landmark_seer is not None:
                    action_command = _limit_landmark_approach_command(
                        parsed,
                        max_forward_action_seconds=args.landmark_seer_max_forward_action_seconds,
                    )
                bounded = limits.apply(action_command)
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
                    executor.execute(action_command)
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
            if mission_found:
                print(
                    f"search complete: item={landmark_item_name!r} "
                    f"waypoint={waypoint_index}/{waypoint_count}",
                    flush=True,
                )
                break
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
