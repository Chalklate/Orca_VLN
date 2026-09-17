"""Simulator-independent data contracts for the NaVILA navigation loop.

All poses exposed by this module use a right-handed world frame and quaternions
in ``(w, x, y, z)`` order.  Simulator adapters are responsible for converting
their native conventions at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image


FloatArray = NDArray[np.floating]
UInt8Array = NDArray[np.uint8]


def _float_array(value: Any, *, shape: tuple[int, ...] | None, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class EpisodeSpec:
    """The engine-neutral subset of a VLN-CE episode used at evaluation time."""

    episode_id: str
    scene_id: str
    instruction: str
    start_position: FloatArray
    start_quat_wxyz: FloatArray
    goal_position: FloatArray
    goal_radius: float
    reference_path: FloatArray
    gt_locations: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", str(self.episode_id))
        object.__setattr__(
            self,
            "start_position",
            _float_array(self.start_position, shape=(3,), name="start_position"),
        )
        quat = _float_array(self.start_quat_wxyz, shape=(4,), name="start_quat_wxyz")
        if np.linalg.norm(quat) <= 1.0e-12:
            raise ValueError("start_quat_wxyz must have non-zero norm")
        object.__setattr__(self, "start_quat_wxyz", quat)
        object.__setattr__(
            self,
            "goal_position",
            _float_array(self.goal_position, shape=(3,), name="goal_position"),
        )
        radius = float(self.goal_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("goal_radius must be a positive finite value")
        object.__setattr__(self, "goal_radius", radius)
        for field_name in ("reference_path", "gt_locations"):
            array = _float_array(getattr(self, field_name), shape=None, name=field_name)
            if array.ndim != 2 or array.shape[1:] != (3,) or len(array) == 0:
                raise ValueError(
                    f"{field_name} must have shape (N, 3) with N > 0, got {array.shape}"
                )
            object.__setattr__(self, field_name, array)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class RobotState:
    """A synchronized robot state at one physics control tick."""

    step_id: int
    sim_time_s: float
    root_pos_world: FloatArray
    root_quat_wxyz: FloatArray
    body_ang_vel: FloatArray
    base_rpy: FloatArray
    joint_pos: FloatArray
    joint_vel: FloatArray
    last_raw_action: FloatArray

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if not np.isfinite(self.sim_time_s) or self.sim_time_s < 0.0:
            raise ValueError("sim_time_s must be non-negative and finite")
        for name, shape in (
            ("root_pos_world", (3,)),
            ("root_quat_wxyz", (4,)),
            ("body_ang_vel", (3,)),
            ("base_rpy", (3,)),
        ):
            object.__setattr__(
                self, name, _float_array(getattr(self, name), shape=shape, name=name)
            )
        for name in ("joint_pos", "joint_vel", "last_raw_action"):
            array = _float_array(getattr(self, name), shape=None, name=name)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
            object.__setattr__(self, name, array)
        if not (
            self.joint_pos.shape == self.joint_vel.shape == self.last_raw_action.shape
        ):
            raise ValueError(
                "joint_pos, joint_vel, and last_raw_action must have the same shape"
            )


@dataclass(frozen=True)
class RenderFrame:
    """An RGB frame synchronized to a particular physics state."""

    step_id: int
    sim_time_s: float
    camera_id: str
    rgb: UInt8Array
    frame_id: str = ""

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if not np.isfinite(self.sim_time_s) or self.sim_time_s < 0.0:
            raise ValueError("sim_time_s must be non-negative and finite")
        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must be uint8 with shape (H, W, 3), got {rgb.dtype} {rgb.shape}"
            )
        rgb = rgb.copy()
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self.rgb, mode="RGB")


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame velocity command and its exact simulated duration."""

    vx: float
    vy: float
    wz: float
    duration_s: float
    stop: bool = False

    def __post_init__(self) -> None:
        values = (self.vx, self.vy, self.wz, self.duration_s)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("velocity command values must be finite")
        if self.duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")
        if self.stop and (
            self.duration_s != 0.0
            or any(value != 0.0 for value in (self.vx, self.vy, self.wz))
        ):
            raise ValueError("a stop command must have zero velocity and zero duration")


@dataclass(frozen=True)
class PhysicsStep:
    state: RobotState
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class VelocityPhysicsBackend(Protocol):
    """Preferred Orca facade: physics and the low-level policy stay together."""

    control_dt: float
    qpos_batch: Any

    def reset(self, episode: EpisodeSpec | None = None) -> RobotState: ...

    def set_velocity_command(self, command: VelocityCommand) -> None: ...

    def step(self) -> RobotState | PhysicsStep: ...

    def close(self) -> None: ...


@runtime_checkable
class JointActionPhysicsBackend(Protocol):
    """Optional lower-level facade for separately hosted locomotion policies."""

    control_dt: float

    def reset(self, episode: EpisodeSpec | None = None) -> RobotState: ...

    def terrain_features(self, state: RobotState) -> FloatArray: ...

    def step(self, joint_action: FloatArray) -> RobotState | PhysicsStep: ...

    def close(self) -> None: ...


# Backwards-compatible short name for code that explicitly uses the lower-level
# adapter.  NavigationRunner prefers VelocityPhysicsBackend when available.
PhysicsBackend = JointActionPhysicsBackend


@runtime_checkable
class RenderBridge(Protocol):
    """A renderer that returns a frame carrying the requested physics token."""

    def render(self, state: RobotState, qpos_batch: Any = None) -> RenderFrame: ...

    def close(self) -> None: ...


@runtime_checkable
class LocomotionPolicy(Protocol):
    """Maps state, terrain features, and a velocity command to backend actions."""

    def reset(self) -> None: ...

    def act(
        self,
        state: RobotState,
        terrain_features: FloatArray,
        command: VelocityCommand,
    ) -> FloatArray: ...


@runtime_checkable
class VLMClient(Protocol):
    """Infers the next textual action from exactly eight RGB images."""

    def infer(self, images: Sequence[Image.Image], instruction: str) -> str: ...
