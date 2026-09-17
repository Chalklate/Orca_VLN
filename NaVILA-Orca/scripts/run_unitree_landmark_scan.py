#!/usr/bin/env python3
"""Capture a safe Go2 visual sweep and build a topological landmark map.

The robot rotates in place in small bounded increments.  The resulting map
contains named visual references and likely search surfaces, not metric poses;
NaVILA is still responsible for visually approaching each reference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navila_orca.frames import brighten_image  # noqa: E402
from navila_orca.hardware.unitree_gateway import (  # noqa: E402
    Go2VideoClientCameraSource,
    SafeMotionExecutor,
    SafetyLimits,
    UnitreeSportController,
)
from navila_orca.contracts import VelocityCommand  # noqa: E402
from navila_orca.openai_router import (  # noqa: E402
    DEFAULT_MODEL_ID,
    OpenAIQueryRouter,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a bounded in-place Go2 landmark scan"
    )
    parser.add_argument("--robot-model", choices=("go2",), default="go2")
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--scan-views", type=_positive_int, default=8)
    parser.add_argument(
        "--scan-step-degrees",
        type=_positive_float,
        help="open-loop turn per step; default is 360/scan-views",
    )
    parser.add_argument("--scan-yaw-rps", type=_positive_float, default=0.20)
    parser.add_argument("--scan-direction", choices=("left", "right"), default="left")
    parser.add_argument("--settle-seconds", type=_nonnegative_float, default=0.40)
    parser.add_argument("--camera-warmup-frames", type=_positive_int, default=3)
    parser.add_argument("--camera-timeout", type=_positive_float, default=15.0)
    parser.add_argument("--camera-rpc-timeout", type=_positive_float, default=2.0)
    parser.add_argument("--camera-frame-retries", type=_nonnegative_int, default=3)
    parser.add_argument("--image-brightness", type=_positive_float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for scan frames and metadata (timestamped by default)",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "memory_guide" / "latest_landmark_map.json",
        help="JSON map consumed by run_unitree_memory_guide.sh",
    )
    parser.add_argument(
        "--openai-model-id",
        default=DEFAULT_MODEL_ID,
        help="OpenAI vision/extraction model (default: gpt-5.6-luna)",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="optional OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--skip-landmark-extraction",
        action="store_true",
        help="capture frames only; do not call OpenAI or write landmark entries",
    )
    parser.add_argument(
        "--execute-actions",
        action="store_true",
        help="ARM MOTORS for the in-place scan; omit for a camera-only dry run",
    )
    return parser


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "outputs" / "landmark_scan" / stamp


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _normalise_landmarks(raw_landmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate repeated views and assign local IDs after validation."""

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_landmarks:
        name = str(raw.get("name", "")).strip()
        kind = str(raw.get("kind", "other")).strip() or "other"
        if not name:
            continue
        key = (" ".join(name.lower().split()), kind.lower())
        confidence = float(raw.get("confidence", 0.0))
        candidate = {
            "name": name,
            "kind": kind,
            "view_index": int(raw.get("view_index", 0)),
            "description": str(raw.get("description", "")).strip(),
            "search_surface": bool(raw.get("search_surface", False)),
            "confidence": confidence,
        }
        previous = selected.get(key)
        if previous is None or candidate["confidence"] > previous["confidence"]:
            selected[key] = candidate
    ordered = sorted(
        selected.values(),
        key=lambda item: (item["view_index"], -item["confidence"], item["name"].lower()),
    )
    for index, landmark in enumerate(ordered, start=1):
        landmark["id"] = f"landmark_{index:03d}"
    return ordered


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.scan_views < 2:
        print("configuration error: --scan-views must be at least 2", file=sys.stderr)
        return 2
    step_degrees = args.scan_step_degrees or (360.0 / args.scan_views)
    if step_degrees > 180.0:
        print("configuration error: --scan-step-degrees must be <= 180", file=sys.stderr)
        return 2
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaw = abs(args.scan_yaw_rps)
    if args.scan_direction == "right":
        yaw = -yaw
    step_duration = math.radians(step_degrees) / abs(args.scan_yaw_rps)

    camera = None
    controller = None
    executor = None
    images = []
    views: list[dict[str, Any]] = []
    try:
        camera = Go2VideoClientCameraSource(
            network_interface=args.network_interface,
            timeout_s=args.camera_rpc_timeout,
            frame_retries=args.camera_frame_retries,
            initialize_channel=True,
        )
        if args.execute_actions:
            controller = UnitreeSportController(
                robot_model="go2",
                network_interface=args.network_interface,
                initialize_channel=False,
            )
            executor = SafeMotionExecutor(
                controller,
                limits=SafetyLimits(
                    max_forward_mps=0.0,
                    max_lateral_mps=0.0,
                    max_yaw_rps=abs(args.scan_yaw_rps),
                    max_duration_s=step_duration,
                ),
            )

        warmup_deadline = time.monotonic() + args.camera_timeout
        for _ in range(args.camera_warmup_frames):
            if time.monotonic() >= warmup_deadline:
                raise RuntimeError(
                    f"camera warmup exceeded {args.camera_timeout:g} seconds"
                )
            camera.read()

        print(
            f"landmark scan: views={args.scan_views}, step={step_degrees:g} degrees, "
            f"yaw={yaw:.3f} rps, motors={'ARMED' if args.execute_actions else 'DRY-RUN'}",
            flush=True,
        )
        if not args.execute_actions:
            print(
                "warning: without --execute-actions every view is from the same heading",
                file=sys.stderr,
                flush=True,
            )

        for view_index in range(args.scan_views):
            if view_index > 0:
                if executor is not None:
                    executor.execute(VelocityCommand(0.0, 0.0, yaw, step_duration))
                if args.settle_seconds > 0.0:
                    time.sleep(args.settle_seconds)
            image = brighten_image(camera.read(), args.image_brightness)
            image_path = output_dir / f"view_{view_index:03d}.jpg"
            image.save(image_path, format="JPEG", quality=95)
            images.append(image)
            views.append(
                {
                    "index": view_index,
                    "heading_estimate_deg": round(
                        (view_index * step_degrees * (1.0 if yaw > 0 else -1.0)) % 360.0,
                        3,
                    ),
                    "image_path": str(image_path),
                }
            )
            print(f"captured view {view_index + 1}/{args.scan_views}: {image_path}", flush=True)

        landmarks: list[dict[str, Any]] = []
        if not args.skip_landmark_extraction:
            router = OpenAIQueryRouter.from_environment(
                model_id=args.openai_model_id,
                base_url=args.openai_base_url,
            )
            landmarks = _normalise_landmarks(
                router.describe_landmarks(
                    images,
                    view_indices=[view["index"] for view in views],
                )
            )

        map_payload: dict[str, Any] = {
            "version": 1,
            "map_type": "visual_landmark_scan",
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "robot_model": "go2",
            "coordinate_frame": "relative_open_loop_heading",
            "warning": (
                "Visual references only; heading is open-loop and there are no metric "
                "poses or obstacle-clearance guarantees."
            ),
            "scan": {
                "views": args.scan_views,
                "step_degrees": step_degrees,
                "yaw_rps": yaw,
                "direction": args.scan_direction,
                "execute_actions": bool(args.execute_actions),
            },
            "views": views,
            "landmarks": landmarks,
        }
        _write_json(map_payload, output_dir / "scan.json")
        _write_json(map_payload, args.map_output)
        print(
            f"landmark map: {args.map_output.expanduser().resolve()} "
            f"({len(landmarks)} landmark(s))",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"landmark scan failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if controller is not None:
            try:
                controller.stop_move()
            except Exception as exc:
                print(f"warning: final StopMove failed: {exc}", file=sys.stderr)
        if camera is not None:
            camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
