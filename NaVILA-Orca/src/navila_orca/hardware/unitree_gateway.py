"""Portable, fail-closed camera and motion primitives for Unitree robots.

The inference model intentionally stays off the robot.  This module owns only
camera capture, the eight-frame NaVILA history, bounded SportClient execution,
and reviewable decision recording.  Unitree and OpenCV imports are lazy so the
module and its tests also run on an operator laptop without robot packages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module
import io
import json
from pathlib import Path
import threading
import time
from typing import Any, Protocol, Sequence

from PIL import Image

from ..contracts import VelocityCommand
from ..frames import NUM_VIDEO_FRAMES


A2_GSTREAMER_PIPELINE = (
    "udpsrc address=230.1.1.1 port=1720 multicast-iface={interface} "
    "! application/x-rtp, media=video, encoding-name=H264 "
    "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert "
    "! video/x-raw,width=1280,height=720,format=BGR "
    "! appsink drop=1 sync=false"
)


class FrameSource(Protocol):
    """Blocking RGB frame source."""

    def read(self) -> Image.Image: ...

    def close(self) -> None: ...


class MotionController(Protocol):
    """Small subset of Unitree's high-level motion client used by NaVILA."""

    def move(self, vx: float, vy: float, wz: float) -> None: ...

    def stop_move(self) -> None: ...

    def balance_stand(self) -> None: ...

    def damp(self) -> None: ...


class GStreamerCameraSource:
    """Read the A2 onboard multicast H.264 stream through OpenCV/GStreamer."""

    def __init__(self, pipeline: str) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is unavailable. Run this in the robot-provided virtual "
                "environment whose cv2 build includes GStreamer support."
            ) from exc
        self._cv2 = cv2
        if not bool(getattr(cv2, "CAP_GSTREAMER", None)):
            raise RuntimeError("this OpenCV build does not expose CAP_GSTREAMER")
        self.pipeline = str(pipeline)
        self._capture = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self._capture.isOpened():
            raise RuntimeError(
                "could not open the Unitree camera stream; verify the robot "
                "network interface, multicast route, and GStreamer-enabled cv2"
            )

    def read(self) -> Image.Image:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Unitree camera stream returned no frame")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")

    def close(self) -> None:
        self._capture.release()


class Go2VideoClientCameraSource:
    """Read JPEG frames from the Go2 front camera through SDK2 VideoClient.

    The VideoClient already returns a compressed JPEG byte sequence, so Pillow
    is sufficient for decoding.  OpenCV is not required by this adapter.
    """

    def __init__(
        self,
        *,
        network_interface: str,
        timeout_s: float = 2.0,
        initialize_channel: bool = True,
    ) -> None:
        if not network_interface.strip():
            raise ValueError("network_interface must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        try:
            channel_module = import_module("unitree_sdk2py.core.channel")
            video_module = import_module("unitree_sdk2py.go2.video.video_client")
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py with Go2 VideoClient is unavailable in this "
                "Python environment"
            ) from exc
        if initialize_channel:
            channel_module.ChannelFactoryInitialize(0, network_interface)
        self.network_interface = network_interface
        self._client = video_module.VideoClient()
        self._client.SetTimeout(float(timeout_s))
        self._client.Init()

    def read(self) -> Image.Image:
        code, data = self._client.GetImageSample()
        if code != 0:
            raise RuntimeError(f"Go2 VideoClient GetImageSample failed with code {code}")
        encoded = bytes(data)
        if not encoded:
            raise RuntimeError("Go2 VideoClient returned an empty image")
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                return image.convert("RGB").copy()
        except Exception as exc:
            raise RuntimeError("Go2 VideoClient returned an invalid JPEG image") from exc

    def close(self) -> None:
        # The SDK2 Python VideoClient exposes no close method.
        return None


@dataclass(frozen=True, slots=True)
class HistoryFrame:
    sequence: int
    captured_monotonic_s: float
    captured_utc: str
    jpeg: bytes
    size: tuple[int, int]


class JpegFrameHistory:
    """Thread-safe, compressed full-run history with exact NaVILA sampling."""

    def __init__(self, *, jpeg_quality: int = 85, max_frames: int = 0) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        self.jpeg_quality = int(jpeg_quality)
        self.max_frames = int(max_frames)
        self._frames: list[HistoryFrame] = []
        self._condition = threading.Condition()
        self._next_sequence = 0

    def append(self, image: Image.Image) -> HistoryFrame:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=self.jpeg_quality)
        frame = HistoryFrame(
            sequence=self._next_sequence,
            captured_monotonic_s=time.monotonic(),
            captured_utc=datetime.now(timezone.utc).isoformat(),
            jpeg=buffer.getvalue(),
            size=rgb.size,
        )
        with self._condition:
            self._next_sequence += 1
            self._frames.append(frame)
            if self.max_frames and len(self._frames) > self.max_frames:
                del self._frames[: len(self._frames) - self.max_frames]
            self._condition.notify_all()
        return frame

    def __len__(self) -> int:
        with self._condition:
            return len(self._frames)

    def wait_for(self, count: int, timeout_s: float) -> bool:
        if count <= 0:
            return True
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while len(self._frames) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def sample(self) -> tuple[list[Image.Image], list[HistoryFrame | None]]:
        """Return seven uniform historical frames plus the latest frame.

        Histories shorter than eight frames are left-padded with black images,
        matching :func:`navila_orca.frames.sample_history`.
        """

        with self._condition:
            frames = list(self._frames)
        if not frames:
            raise ValueError("cannot sample an empty frame history")

        if len(frames) < NUM_VIDEO_FRAMES:
            missing = NUM_VIDEO_FRAMES - len(frames)
            metadata: list[HistoryFrame | None] = [None] * missing + frames
            sampled_metadata = metadata[: NUM_VIDEO_FRAMES - 1] + [metadata[-1]]
        else:
            indices = [
                int(index * (len(frames) - 1) / 7)
                for index in range(NUM_VIDEO_FRAMES - 1)
            ]
            indices.append(len(frames) - 1)
            sampled_metadata = [frames[index] for index in indices]

        images: list[Image.Image] = []
        latest_size = frames[-1].size
        for frame in sampled_metadata:
            if frame is None:
                images.append(Image.new("RGB", latest_size, (0, 0, 0)))
            else:
                with Image.open(io.BytesIO(frame.jpeg)) as image:
                    images.append(image.convert("RGB").copy())
        return images, sampled_metadata


class CameraCaptureWorker:
    """Continuously collect frames while inference and motion are in progress."""

    def __init__(
        self,
        source: FrameSource,
        history: JpegFrameHistory,
        *,
        capture_hz: float = 5.0,
    ) -> None:
        if capture_hz <= 0:
            raise ValueError("capture_hz must be positive")
        self.source = source
        self.history = history
        self.capture_hz = float(capture_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("camera capture worker has already been started")
        self._thread = threading.Thread(
            target=self._run, name="unitree-camera-capture", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / self.capture_hz
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                self.history.append(self.source.read())
                self._stop.wait(max(0.0, period - (time.monotonic() - started)))
        except BaseException as exc:
            self.error = exc
            self._stop.set()

    def ensure_healthy(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"camera capture failed: {self.error}") from self.error

    def close(self) -> None:
        self._stop.set()
        self.source.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Deployment limits deliberately below Unitree's hardware capabilities."""

    max_forward_mps: float = 0.20
    max_lateral_mps: float = 0.0
    max_yaw_rps: float = 0.35
    max_duration_s: float = 0.75

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(float(value) < 0.0 for value in values.values()):
            raise ValueError("safety limits must be non-negative")
        if self.max_duration_s <= 0.0:
            raise ValueError("max_duration_s must be positive")

    def apply(self, command: VelocityCommand) -> VelocityCommand:
        if command.stop:
            return command
        return VelocityCommand(
            vx=_clip(command.vx, self.max_forward_mps),
            vy=_clip(command.vy, self.max_lateral_mps),
            wz=_clip(command.wz, self.max_yaw_rps),
            duration_s=min(command.duration_s, self.max_duration_s),
        )


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


class UnitreeSportController:
    """Lazy wrapper around A2 or Go2 ``unitree_sdk2py`` SportClient."""

    _CLIENT_MODULES = {
        "a2": "unitree_sdk2py.a2.sport.sport_client",
        "go2": "unitree_sdk2py.go2.sport.sport_client",
    }

    def __init__(
        self,
        *,
        robot_model: str = "a2",
        network_interface: str = "br0",
        timeout_s: float = 2.0,
        initialize_channel: bool = True,
    ) -> None:
        robot_model = robot_model.lower()
        if robot_model not in self._CLIENT_MODULES:
            raise ValueError("robot_model must be 'a2' or 'go2'")
        if not network_interface.strip():
            raise ValueError("network_interface must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        try:
            channel_module = import_module("unitree_sdk2py.core.channel")
            client_module = import_module(self._CLIENT_MODULES[robot_model])
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is unavailable. Run this gateway inside the "
                "robot-provided robot-env virtual environment."
            ) from exc

        if initialize_channel:
            channel_module.ChannelFactoryInitialize(0, network_interface)
        self.robot_model = robot_model
        self.network_interface = network_interface
        self._client = client_module.SportClient()
        self._client.SetTimeout(float(timeout_s))
        self._client.Init()

    @staticmethod
    def _check_result(operation: str, result: Any) -> None:
        if isinstance(result, int) and result != 0:
            raise RuntimeError(f"Unitree {operation} failed with code {result}")

    def move(self, vx: float, vy: float, wz: float) -> None:
        self._check_result("Move", self._client.Move(vx, vy, wz))

    def stop_move(self) -> None:
        self._check_result("StopMove", self._client.StopMove())

    def balance_stand(self) -> None:
        self._check_result("BalanceStand", self._client.BalanceStand())

    def damp(self) -> None:
        self._check_result("Damp", self._client.Damp())


class SafeMotionExecutor:
    """Execute one bounded body-frame command and always finish with StopMove."""

    def __init__(
        self,
        controller: MotionController,
        *,
        limits: SafetyLimits | None = None,
        command_hz: float = 50.0,
    ) -> None:
        if command_hz <= 0:
            raise ValueError("command_hz must be positive")
        self.controller = controller
        self.limits = limits or SafetyLimits()
        self.command_hz = float(command_hz)
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def execute(self, command: VelocityCommand) -> VelocityCommand:
        bounded = self.limits.apply(command)
        if bounded.stop:
            self.controller.stop_move()
            return bounded

        self._stop_requested.clear()
        period = 1.0 / self.command_hz
        deadline = time.monotonic() + bounded.duration_s
        original_error: BaseException | None = None
        try:
            while time.monotonic() < deadline and not self._stop_requested.is_set():
                started = time.monotonic()
                self.controller.move(bounded.vx, bounded.vy, bounded.wz)
                self._stop_requested.wait(
                    max(0.0, min(period, deadline - time.monotonic()) - (time.monotonic() - started))
                )
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            try:
                self.controller.stop_move()
            except BaseException:
                if original_error is None:
                    raise
        return bounded

    def emergency_stop(self, *, damp: bool = False) -> None:
        self.request_stop()
        self.controller.stop_move()
        if damp:
            self.controller.damp()


class HardwareDecisionRecorder:
    """Write physical-robot decisions in the existing review/export schema."""

    RECORD_VERSION = 1

    def __init__(
        self,
        output_root: Path,
        *,
        robot_model: str,
        episode_id: str,
        scene_id: str,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_root / "manifest.jsonl"
        self.robot_model = str(robot_model)
        self.episode_id = str(episode_id)
        self.scene_id = str(scene_id)

    def record(
        self,
        *,
        decision: int,
        instruction: str,
        images: Sequence[Image.Image],
        history_frames: Sequence[HistoryFrame | None],
        baseline_output: str,
        parsed_command: VelocityCommand,
        bounded_command: VelocityCommand,
        executed: bool,
    ) -> dict[str, Any]:
        if len(images) != NUM_VIDEO_FRAMES:
            raise ValueError("a hardware decision requires exactly eight images")
        if len(history_frames) != len(images):
            raise ValueError("history_frames must match images")
        if decision <= 0:
            raise ValueError("decision numbers are one-based")

        sample_dir = self.output_root / f"decision_{decision:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_files: list[str] = []
        for index, image in enumerate(images):
            path = sample_dir / f"frame_{index:03d}.jpg"
            image.convert("RGB").save(path, format="JPEG", quality=95)
            image_files.append(str(path.relative_to(self.output_root)))

        record = {
            "record_version": self.RECORD_VERSION,
            "source": "unitree_hardware_decision_sample",
            "review_status": "unreviewed",
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "robot_model": self.robot_model,
            "decision": int(decision),
            "instruction": instruction.strip(),
            "image_files": image_files,
            "frame_sequences": [
                None if frame is None else frame.sequence for frame in history_frames
            ],
            "frame_captured_utc": [
                None if frame is None else frame.captured_utc for frame in history_frames
            ],
            "baseline_output": str(baseline_output),
            "baseline_command": asdict(parsed_command),
            "bounded_command": asdict(bounded_command),
            "executed": bool(executed),
            "target_action": None,
            "reviewer": None,
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with self.manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
