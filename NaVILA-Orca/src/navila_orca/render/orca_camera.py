"""Go2 ego-camera provisioning and rigid-pose following for OrcaLab.

OrcaLab's public scene API cannot parent an independent camera asset below an
internal MuJoCo body in a robot ``AssetActor``.  The supported prototype is
therefore a root camera actor whose world pose is updated from the Go2 free
joint before every ``UpdateLocalEnv`` call.  The resulting transform is the
same rigid-body relation as an engine-side parent constraint::

    T_world_camera = T_world_base @ T_base_camera

The high-level VLN path consumes RGB only.  Depth and the locomotion policy are
intentionally outside this adapter.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image


DEFAULT_CAMERA_ACTOR_NAME = "navila_ego"
DEFAULT_CAMERA_ASSET = "prefabs/agentcamera"
DEFAULT_CAMERA_MOUNT_POSITION = (0.1, 0.0, 0.5)
# The NavVLM camera uses (-0.5, 0.5, -0.5, 0.5) under its source camera-frame
# convention. Orca's CameraSensor post-multiplies AtomToRos, so
# the equivalent forward +X / image-up +Z entity rotation is yaw -90 degrees.
_SQRT_HALF = float(2.0**-0.5)
DEFAULT_CAMERA_MOUNT_QUAT_WXYZ = (_SQRT_HALF, 0.0, 0.0, -_SQRT_HALF)
_PNG_IEND_TRAILER = b"\x00\x00\x00\x00IEND\xaeB`\x82"


class OrcaCameraError(RuntimeError):
    """Raised when the OrcaLab ego-camera cannot be created or configured."""


def _read_complete_png(
    path: str, *, timeout_s: float, poll_interval_s: float = 0.01
) -> np.ndarray:
    """Wait for OrcaLab's asynchronous file write, then decode the PNG once."""

    deadline = time.monotonic() + timeout_s
    last_error: OSError | EOFError | None = None
    while True:
        try:
            with open(path, "rb") as stream:
                stream.seek(-len(_PNG_IEND_TRAILER), os.SEEK_END)
                if stream.read(len(_PNG_IEND_TRAILER)) != _PNG_IEND_TRAILER:
                    raise EOFError("PNG IEND chunk is not available yet")
            break
        except (OSError, EOFError) as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise OrcaCameraError(
                    f"OrcaLab PNG was not completed within {timeout_s:.1f}s: "
                    f"{path}: {last_error}"
                ) from exc
            time.sleep(poll_interval_s)

    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, SyntaxError, ValueError) as exc:
        raise OrcaCameraError(
            f"OrcaLab completed an unreadable PNG: {path}: {exc}"
        ) from exc


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def normalize_quat_wxyz(value: Sequence[float], name: str = "quaternion") -> np.ndarray:
    """Return a finite unit quaternion in ``(w, x, y, z)`` order."""

    quat = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero norm")
    return quat / norm


def multiply_quat_wxyz(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    """Hamilton product for quaternions in ``(w, x, y, z)`` order."""

    lw, lx, ly, lz = normalize_quat_wxyz(left, "left quaternion")
    rw, rx, ry, rz = normalize_quat_wxyz(right, "right quaternion")
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return normalize_quat_wxyz(result, "quaternion product")


def rotate_vector_wxyz(
    quat_wxyz: Sequence[float], vector: Sequence[float]
) -> np.ndarray:
    """Rotate a 3-vector without importing SciPy or the optional Orca stack."""

    quat = normalize_quat_wxyz(quat_wxyz)
    vec = _finite_vector(vector, 3, "vector")
    # q * v * conjugate(q), simplified to avoid normalizing the pure-vector
    # quaternion (which would incorrectly discard vector magnitude).
    scalar = quat[0]
    axis = quat[1:]
    return (
        2.0 * np.dot(axis, vec) * axis
        + (scalar * scalar - np.dot(axis, axis)) * vec
        + 2.0 * scalar * np.cross(axis, vec)
    )


def compose_camera_pose(
    base_position: Sequence[float],
    base_quat_wxyz: Sequence[float],
    mount_position: Sequence[float] = DEFAULT_CAMERA_MOUNT_POSITION,
    mount_quat_wxyz: Sequence[float] = DEFAULT_CAMERA_MOUNT_QUAT_WXYZ,
    *,
    stabilize_horizon: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the world camera pose from the world base pose and mount pose.

    With ``stabilize_horizon`` enabled the camera position remains rigidly
    attached to the base, but its orientation follows base yaw only.  This is
    equivalent to a two-axis roll/pitch gimbal and prevents locomotion body
    sway from rotating the image horizon.
    """

    base_pos = _finite_vector(base_position, 3, "base_position")
    base_quat = normalize_quat_wxyz(base_quat_wxyz, "base_quat_wxyz")
    mount_pos = _finite_vector(mount_position, 3, "mount_position")
    mount_quat = normalize_quat_wxyz(mount_quat_wxyz, "mount_quat_wxyz")
    camera_pos = base_pos + rotate_vector_wxyz(base_quat, mount_pos)
    orientation_quat = base_quat
    if stabilize_horizon:
        w, x, y, z = base_quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        orientation_quat = np.array(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
            dtype=np.float64,
        )
    camera_quat = multiply_quat_wxyz(orientation_quat, mount_quat)
    return camera_pos, camera_quat


@dataclass(frozen=True, slots=True)
class _OrcaRuntime:
    service_factory: Callable[[], Any]
    asset_actor_type: type
    transform_type: type
    path_type: type
    add_actor_request_type: type
    property_key_type: type


def _load_orca_runtime() -> _OrcaRuntime:
    """Import OrcaLab lazily so normal/unit-test imports stay lightweight."""

    try:
        actor_module = importlib.import_module("orcalab.actor")
        try:
            math_module = importlib.import_module("orcalab.transform")
        except ModuleNotFoundError:
            math_module = importlib.import_module("orcalab.math")
        path_module = importlib.import_module("orcalab.path")
        property_module = importlib.import_module("orcalab.actor_property")
        edit_types_module = importlib.import_module("orcalab.scene_edit_types")
        wrapper_module = importlib.import_module("orcalab.protos.edit_service_wrapper")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OrcaCameraError(
            "OrcaLab edit-service Python modules are unavailable. Activate the "
            "local 'orcalab' environment before using --render-backend orcalab."
        ) from exc
    return _OrcaRuntime(
        service_factory=wrapper_module.EditServiceWrapper,
        asset_actor_type=actor_module.AssetActor,
        transform_type=math_module.Transform,
        path_type=path_module.Path,
        add_actor_request_type=edit_types_module.AddActorRequest,
        property_key_type=property_module.ActorPropertyKey,
    )


class OrcaEgoCameraFollower:
    """Provision one RGB camera and rigidly follow the first rendered Go2."""

    _REQUIRED_PROPERTIES = {
        "IsRecording",
        "Width",
        "Height",
        "ColorCamera",
        "ColorPort",
    }

    def __init__(
        self,
        *,
        edit_address: str,
        color_port: int,
        actor_name: str = DEFAULT_CAMERA_ACTOR_NAME,
        asset_path: str = DEFAULT_CAMERA_ASSET,
        width: int = 512,
        height: int = 512,
        mount_position: Sequence[float] = DEFAULT_CAMERA_MOUNT_POSITION,
        mount_quat_wxyz: Sequence[float] = DEFAULT_CAMERA_MOUNT_QUAT_WXYZ,
        stabilize_horizon: bool = False,
        event_loop: asyncio.AbstractEventLoop | None = None,
        runtime_factory: Callable[[], _OrcaRuntime] = _load_orca_runtime,
    ) -> None:
        if not edit_address or ":" not in edit_address:
            raise ValueError("edit_address must be an explicit 'host:port'")
        if not actor_name.isascii() or not actor_name.isidentifier():
            raise ValueError("actor_name must be an ASCII actor identifier")
        if not asset_path:
            raise ValueError("asset_path must be non-empty")
        if not (1 <= int(color_port) <= 65535):
            raise ValueError("color_port must be in [1, 65535]")
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("camera width and height must be positive")

        self.edit_address = edit_address
        self.color_port = int(color_port)
        self.actor_name = actor_name
        self.asset_path = asset_path
        self.width = int(width)
        self.height = int(height)
        self.mount_position = _finite_vector(mount_position, 3, "mount_position").copy()
        self.mount_quat_wxyz = normalize_quat_wxyz(mount_quat_wxyz, "mount_quat_wxyz")
        self.stabilize_horizon = bool(stabilize_horizon)
        self._loop = event_loop
        self._owns_loop = event_loop is None
        self._runtime_factory = runtime_factory
        self._runtime: _OrcaRuntime | None = None
        self._service: Any | None = None
        self._actor_path: Any | None = None

    @property
    def started(self) -> bool:
        return self._service is not None

    @property
    def actor_path(self) -> str:
        return f"/{self.actor_name}"

    def _run(self, awaitable: Any) -> Any:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True
        if self._loop.is_running():
            raise OrcaCameraError(
                "Orca camera follower requires a synchronous owner event loop"
            )
        asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(awaitable)

    def start(self) -> None:
        """Create/reuse the camera actor and enforce the RGB stream settings."""

        if self.started:
            return
        runtime = self._runtime_factory()
        service = runtime.service_factory()
        try:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._owns_loop = True
            asyncio.set_event_loop(self._loop)
            service.init_grpc(self.edit_address)
            self._runtime = runtime
            self._service = service
            self._actor_path = runtime.path_type(self.actor_path)
            self._run(self._start_async())
        except Exception:
            self._close_service_best_effort()
            raise

    async def _start_async(self) -> None:
        assert self._runtime is not None
        assert self._service is not None
        assert self._actor_path is not None
        if not await self._service.aloha():
            raise OrcaCameraError(
                f"OrcaLab edit service is not reachable at {self.edit_address}"
            )

        groups = await self._try_property_groups()
        if groups is None:
            transform = self._runtime.transform_type(
                position=self.mount_position.copy(),
                rotation=self.mount_quat_wxyz.copy(),
                scale=1.0,
            )
            actor = self._runtime.asset_actor_type(self.actor_name, self.asset_path)
            actor.transform = transform
            request = self._runtime.add_actor_request_type(
                actor, self._runtime.path_type.root_path()
            )
            if hasattr(self._service, "add_actor_batch"):
                added, errors = await self._service.add_actor_batch([request], True)
                if not added:
                    detail = "; ".join(error for error in errors if error)
                    # A previous interrupted run may have created the actor
                    # between our query and add. Re-query before failing.
                    groups = await self._try_property_groups()
                    if groups is None:
                        raise OrcaCameraError(
                            f"failed to add camera actor {self.actor_path}: {detail}"
                        )
            elif hasattr(self._service, "add_asset_actor"):
                await self._service.add_asset_actor(
                    actor, self._runtime.path_type.root_path()
                )
            else:
                raise OrcaCameraError(
                    "installed OrcaLab edit wrapper has no supported add-actor API"
                )
            if groups is None:
                groups = await self._wait_for_property_groups()

        await self._configure_camera(groups)

    async def _try_property_groups(self) -> list[Any] | None:
        assert self._service is not None
        assert self._actor_path is not None
        try:
            groups = await self._service.get_property_groups(self._actor_path)
        except Exception:
            return None
        return list(groups)

    async def _wait_for_property_groups(self) -> list[Any]:
        for _ in range(100):
            groups = await self._try_property_groups()
            if groups:
                return groups
            await asyncio.sleep(0.05)
        raise OrcaCameraError(
            f"camera actor {self.actor_path} spawned without editable properties"
        )

    def _property_keys(self, groups: Sequence[Any]) -> dict[str, Any]:
        assert self._runtime is not None
        assert self._actor_path is not None
        keys: dict[str, Any] = {}
        for group in groups:
            for prop in group.properties:
                name = prop.name()
                keys[name] = self._runtime.property_key_type(
                    self._actor_path,
                    group.prefix,
                    name,
                    prop.value_type(),
                )
        return keys

    async def _configure_camera(self, groups: Sequence[Any]) -> None:
        assert self._service is not None
        keys = self._property_keys(groups)
        missing = sorted(self._REQUIRED_PROPERTIES.difference(keys))
        if missing:
            raise OrcaCameraError(
                f"camera asset {self.asset_path!r} is missing properties: "
                + ", ".join(missing)
            )

        # CameraCapture rejects most mutations while recording. Stop first,
        # apply one batch, then restart the encoder/WebSocket stream.
        await self._service.set_properties([keys["IsRecording"]], [False])
        await asyncio.sleep(0.05)

        requested: dict[str, Any] = {
            "Width": self.width,
            "Height": self.height,
            "RandomObjectColor": False,
            "ColorCamera": True,
            "DepthCamera": False,
            "NormalCamera": False,
            "ObjectColorCamera": False,
            "UseNvEnc": True,
            "NvencGpuIndex": 0,
            "ColorPort": self.color_port,
        }
        if self.color_port < 65535:
            requested["DepthPort"] = self.color_port + 1
        selected = [
            (keys[name], value) for name, value in requested.items() if name in keys
        ]
        await self._service.set_properties(
            [key for key, _value in selected],
            [value for _key, value in selected],
        )
        await self._service.set_properties([keys["IsRecording"]], [True])

        verify_names = [
            "IsRecording",
            "Width",
            "Height",
            "ColorCamera",
            "ColorPort",
        ]
        values = await self._service.get_properties(
            [keys[name] for name in verify_names]
        )
        expected = [True, self.width, self.height, True, self.color_port]
        mismatches = [
            f"{name}={actual!r} (expected {wanted!r})"
            for name, actual, wanted in zip(verify_names, values, expected)
            if actual != wanted
        ]
        if mismatches:
            raise OrcaCameraError(
                "OrcaLab rejected ego-camera configuration: " + "; ".join(mismatches)
            )

    def update(
        self,
        base_position: Sequence[float],
        base_quat_wxyz: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move the camera to the rigidly composed world pose."""

        if not self.started or self._runtime is None or self._actor_path is None:
            raise OrcaCameraError("camera follower has not been started")
        camera_pos, camera_quat = compose_camera_pose(
            base_position,
            base_quat_wxyz,
            self.mount_position,
            self.mount_quat_wxyz,
            stabilize_horizon=self.stabilize_horizon,
        )
        transform = self._runtime.transform_type(
            position=camera_pos.copy(),
            rotation=camera_quat.copy(),
            scale=1.0,
        )
        if hasattr(self._service, "set_actor_transform_batch"):
            awaitable = self._service.set_actor_transform_batch(
                [self._actor_path], [transform]
            )
        elif hasattr(self._service, "set_actor_transform"):
            awaitable = self._service.set_actor_transform(
                self._actor_path, transform, local=False
            )
        else:
            raise OrcaCameraError(
                "installed OrcaLab edit wrapper has no supported transform API"
            )
        self._run(awaitable)
        return camera_pos, camera_quat

    def wait_for_stream(self, timeout_s: float = 5.0) -> None:
        """Wait until the configured color TCP/WebSocket listener is reachable."""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        host = self.edit_address.rsplit(":", 1)[0]
        if host in {"0.0.0.0", "::", "localhost"}:
            host = "127.0.0.1"
        deadline = time.monotonic() + float(timeout_s)
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, self.color_port), timeout=0.25):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        raise OrcaCameraError(
            f"camera actor {self.actor_path} did not open color WebSocket "
            f"{host}:{self.color_port} within {timeout_s:.1f}s: {last_error}"
        )

    def _close_service_best_effort(self) -> None:
        service = self._service
        self._service = None
        self._actor_path = None
        self._runtime = None
        if service is not None:
            try:
                self._run(service.destroy_grpc())
            except Exception:
                pass

    def close(self) -> None:
        """Close only the edit channel; leave the requested scene actor intact."""

        self._close_service_best_effort()
        if self._owns_loop and self._loop is not None:
            try:
                self._loop.close()
            finally:
                self._loop = None

    def __enter__(self) -> "OrcaEgoCameraFollower":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class OrcaGrpcPngCamera:
    """Pull RGB frames through OrcaLab's edit gRPC camera capture API.

    The OrcaStudio build currently installed on this machine starts its H.264
    WebSocket but does not enqueue encoded packets (the StreamingHandler save
    call is disabled in that engine build). ``GetCameraDataPNG`` remains a
    functional renderer-owned capture path and returns the same 512px camera.
    This adapter mirrors ``CameraWrapper``'s small interface so the navigation
    bridge can select it without touching VLN or locomotion code.
    """

    pull_capture = True

    def __init__(
        self,
        name: str,
        _port: int,
        *,
        edit_address: str,
        remote_camera_name: str = "AgentCamera",
        timeout_s: float = 10.0,
        output_dir: str | None = None,
        runtime_factory: Callable[[], _OrcaRuntime] = _load_orca_runtime,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.name = name
        self.edit_address = edit_address
        self.remote_camera_name = remote_camera_name
        self.timeout_s = float(timeout_s)
        self.output_dir = output_dir
        self._runtime_factory = runtime_factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._service: Any | None = None
        self._request_index = 0
        self._frame_index = -1
        self._received_first_frame = False
        self.last_transform: Any | None = None

    def _run(self, awaitable: Any) -> Any:
        if self._loop is None:
            raise OrcaCameraError("gRPC PNG camera has not been started")
        asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(awaitable)

    def start(self) -> None:
        if self._service is not None:
            return
        runtime = self._runtime_factory()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        service = runtime.service_factory()
        try:
            service.init_grpc(self.edit_address)
            self._loop = loop
            self._service = service
            if not self._run(service.aloha()):
                raise OrcaCameraError(
                    f"OrcaLab edit service is not reachable at {self.edit_address}"
                )
            if self.output_dir is None:
                self.output_dir = tempfile.mkdtemp(prefix="navila_orca_camera_")
        except Exception:
            self._service = None
            self._loop = None
            loop.close()
            raise

    def is_first_frame_received(self) -> bool:
        return self._received_first_frame

    def get_frame(self, format: str = "rgb24") -> tuple[np.ndarray, int]:
        if format not in {"rgb24", "bgr24"}:
            raise ValueError("OrcaGrpcPngCamera supports rgb24 or bgr24")
        if self._service is None or self.output_dir is None:
            raise OrcaCameraError("gRPC PNG camera has not been started")

        request_index = self._request_index
        self._request_index += 1
        result = self._run(
            self._service.get_camera_data_png(
                self.remote_camera_name, self.output_dir, request_index
            )
        )
        if not result.has_color:
            raise OrcaCameraError(
                f"Orca camera {self.remote_camera_name!r} returned no color frame"
            )
        image_path = (
            f"{self.output_dir}/color/"
            f"{self.remote_camera_name}_color_{request_index}.png"
        )
        array = _read_complete_png(image_path, timeout_s=self.timeout_s)

        if format == "bgr24":
            array = np.ascontiguousarray(array[..., ::-1])
        self._frame_index += 1
        self._received_first_frame = True
        self.last_transform = result.transform
        return array, self._frame_index

    def stop(self) -> None:
        service, loop = self._service, self._loop
        self._service = None
        self._loop = None
        if service is not None and loop is not None:
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(service.destroy_grpc())
            finally:
                loop.close()


class OrcaMujocoPngCamera(OrcaGrpcPngCamera):
    """Persistent ``mujococamera*`` actor captured through ``GetCameraPNG``."""

    def __init__(self, *args: Any, remote_camera_name: str = "mujococamera1080", **kwargs: Any) -> None:
        super().__init__(*args, remote_camera_name=remote_camera_name, **kwargs)

    def get_frame(self, format: str = "rgb24") -> tuple[np.ndarray, int]:
        if format not in {"rgb24", "bgr24"}:
            raise ValueError("OrcaMujocoPngCamera supports rgb24 or bgr24")
        if self._service is None or self.output_dir is None:
            raise OrcaCameraError("gRPC PNG camera has not been started")
        index = self._request_index
        self._request_index += 1
        filename = f"{self.remote_camera_name}_{index}.png"
        path = os.path.join(self.output_dir, filename)
        if not self._run(self._service.get_camera_png(self.remote_camera_name, self.output_dir, filename)):
            raise OrcaCameraError(f"Orca camera {self.remote_camera_name!r} refused PNG capture")
        array = _read_complete_png(path, timeout_s=self.timeout_s)
        if format == "bgr24":
            array = np.ascontiguousarray(array[..., ::-1])
        self._frame_index += 1
        self._received_first_frame = True
        return array, self._frame_index


class OrcaMujocoCameraFollower(OrcaEgoCameraFollower):
    """Persistent OrcaLab MuJoCo camera without agentcamera stream properties."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._created_actor = False

    async def _start_async(self) -> None:
        assert self._runtime is not None
        assert self._service is not None
        assert self._actor_path is not None
        # OrcaLab 26.7.1's wrapper can report ``aloha() == False`` even while
        # its public AddActor/GetCameraPNG RPCs are available.  The add call
        # below is the authoritative reachability check.
        try:
            await self._service.get_actor_property_groups_batch([self._actor_path])
            return
        except Exception:
            pass
        actor = self._runtime.asset_actor_type(self.actor_name, self.asset_path)
        actor.transform = self._runtime.transform_type(
            position=self.mount_position.copy(), rotation=self.mount_quat_wxyz.copy(), scale=1.0
        )
        request = self._runtime.add_actor_request_type(actor, self._runtime.path_type.root_path())
        try:
            await self._service.add_actor_batch([request])
        except TypeError:
            added, errors = await self._service.add_actor_batch([request], True)
            if not added:
                raise OrcaCameraError("failed to add persistent MuJoCo camera: " + "; ".join(errors))
        self._created_actor = True

    def close(self) -> None:
        if self._created_actor and self._service is not None and self._actor_path is not None:
            try:
                self._run(self._service.delete_actor_batch([self._actor_path]))
            except Exception:
                pass
        self._created_actor = False
        super().close()
