"""Keyboard teleoperation and pose-tagged scene-anchor collection.

This module deliberately sits beside the VLM navigation runner.  It uses the
same velocity-command facade and physics ticks, but does not ask NaVILA for an
action.  That makes it useful for bootstrapping a semantic/topological map in
the simulator without requiring VR hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any, Protocol, Sequence

import numpy as np

from .contracts import (
    EpisodeSpec,
    PhysicsStep,
    RenderBridge,
    RobotState,
    VelocityCommand,
    VelocityPhysicsBackend,
)
from .runner import duration_to_ticks


class TeleopError(RuntimeError):
    """Raised when keyboard teleoperation cannot be started safely."""


class KeyboardSource(Protocol):
    """Small input interface so the control loop can be tested without a TTY."""

    def poll(self) -> Sequence[str]: ...

    def read_label(self) -> str: ...


class TerminalKeyboard:
    """Read single-key commands from a POSIX terminal in cbreak mode."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved_attrs: list[Any] | None = None

    def __enter__(self) -> "TerminalKeyboard":
        if not sys.stdin.isatty():
            raise TeleopError(
                "keyboard teleop needs an interactive terminal on stdin; "
                "run it in the terminal where the stack was started"
            )
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (OSError, termios.error) as exc:
            raise TeleopError(f"could not configure terminal keyboard input: {exc}") from exc
        self._fd = fd
        self._saved_attrs = saved
        return self

    def __exit__(self, *_exc: object) -> None:
        self._restore()

    def _restore(self) -> None:
        if self._fd is not None and self._saved_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
            finally:
                self._fd = None
                self._saved_attrs = None

    def poll(self) -> list[str]:
        if self._fd is None:
            raise TeleopError("keyboard input is not active")
        events: list[str] = []
        while True:
            readable, _writable, _exceptional = select.select([self._fd], [], [], 0.0)
            if not readable:
                return events
            data = os.read(self._fd, 1)
            if not data:
                return events
            events.append(data.decode("utf-8", errors="ignore"))

    def read_label(self) -> str:
        """Temporarily return to canonical terminal input for an anchor label."""

        self._restore()
        try:
            return input("\nAnchor label (blank cancels): ").strip()
        except EOFError:
            return ""
        finally:
            fd = sys.stdin.fileno()
            self._fd = fd
            self._saved_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)


@dataclass(frozen=True, slots=True)
class TeleopAnchor:
    """A semantic label tied to a simulator pose and optional RGB snapshot."""

    label: str
    step_id: int
    sim_time_s: float
    root_pos_world: tuple[float, float, float]
    root_quat_wxyz: tuple[float, float, float, float]
    base_rpy: tuple[float, float, float]
    image_path: str | None = None


@dataclass(frozen=True, slots=True)
class TeleopResult:
    """Summary written by a completed teleoperation session."""

    termination_reason: str
    control_steps: int
    sim_time_s: float
    anchors: tuple[TeleopAnchor, ...]
    final_state: RobotState


class KeyboardTeleopRunner:
    """Run Go2 physics from short, safe keyboard velocity pulses.

    A key press renews a command for ``command_hold_s``.  If key-repeat stops
    or the terminal disappears, the command automatically becomes zero.  The
    simulator is still advanced at its normal control rate, independently of
    the preview capture rate.
    """

    def __init__(
        self,
        physics: VelocityPhysicsBackend,
        renderer: RenderBridge | None = None,
        *,
        output_dir: str | Path | None = None,
        forward_speed_mps: float = 0.5,
        strafe_speed_mps: float = 0.35,
        turn_rate_rad_s: float = 0.8,
        command_hold_s: float = 0.35,
        capture_interval_s: float = 0.2,
        live_monitor: Any | None = None,
        realtime: bool = True,
    ) -> None:
        self.physics = physics
        self.renderer = renderer
        self.output_dir = None if output_dir is None else Path(output_dir).expanduser().resolve()
        self.forward_speed_mps = self._positive(forward_speed_mps, "forward_speed_mps")
        self.strafe_speed_mps = self._positive(strafe_speed_mps, "strafe_speed_mps")
        self.turn_rate_rad_s = self._positive(turn_rate_rad_s, "turn_rate_rad_s")
        self.command_hold_s = self._positive(command_hold_s, "command_hold_s")
        self.capture_interval_s = self._positive(
            capture_interval_s, "capture_interval_s"
        )
        self.live_monitor = live_monitor
        self.realtime = bool(realtime)

    @staticmethod
    def _positive(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    @staticmethod
    def controls_text() -> str:
        return (
            "W/S forward/back | A/D turn left/right | Q/E strafe left/right | "
            "Space brake | M mark anchor | +/- speed | X or Esc quit"
        )

    def run(
        self,
        episode: EpisodeSpec | None,
        keyboard: KeyboardSource,
        *,
        max_control_steps: int = 0,
        max_sim_time_s: float = 0.0,
    ) -> TeleopResult:
        requested_steps = int(max_control_steps)
        if requested_steps < 0:
            raise ValueError("max_control_steps must be non-negative")
        max_sim_time_s = float(max_sim_time_s)
        if not np.isfinite(max_sim_time_s) or max_sim_time_s < 0.0:
            raise ValueError("max_sim_time_s must be finite and non-negative")

        control_dt = float(self.physics.control_dt)
        capture_ticks = duration_to_ticks(self.capture_interval_s, control_dt)
        if capture_ticks <= 0:
            raise ValueError("capture_interval_s must span at least one control tick")

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "anchors").mkdir(parents=True, exist_ok=True)

        state = self.physics.reset(episode)
        anchors: list[TeleopAnchor] = []
        command_events: list[dict[str, Any]] = []
        control_steps = 0
        termination_reason = "keyboard_quit"
        active_velocity = np.zeros(3, dtype=np.float64)
        command_deadline = 0.0
        speed_scale = 1.0
        next_tick = time.monotonic()

        def capture(*, record: bool = True):
            if self.renderer is None:
                return None
            qpos_batch = getattr(self.physics, "qpos_batch", None)
            push_state = getattr(self.renderer, "push_state", None)
            if callable(push_state):
                push_state(state, qpos_batch=qpos_batch)
            capture_method = getattr(self.renderer, "capture", None)
            if callable(capture_method):
                frame = capture_method(state, qpos_batch=qpos_batch)
            else:
                frame = self.renderer.render(state, qpos_batch=qpos_batch)
            if int(frame.step_id) != int(state.step_id):
                raise RuntimeError(
                    f"renderer returned stale frame for step {frame.step_id}; "
                    f"requested {state.step_id}"
                )
            if self.live_monitor is not None:
                self.live_monitor.update(
                    frame,
                    instruction="keyboard teleop",
                    vlm_output="none",
                    command=self._command_text(active_velocity, speed_scale),
                    status="teleoperating",
                    decision=0,
                    chunk_result=(
                        f"pose x={state.root_pos_world[0]:+.2f}m "
                        f"y={state.root_pos_world[1]:+.2f}m "
                        f"yaw={np.degrees(state.base_rpy[2]):+.1f}deg"
                    ),
                )
            return frame

        capture()
        print(self.controls_text(), flush=True)
        print("Press M to label the current view and save its pose.", flush=True)

        running = True
        try:
            while running:
                for key in keyboard.poll():
                    key = key.lower()
                    if key in ("x", "\x1b", "\x03"):
                        active_velocity[:] = 0.0
                        command_deadline = 0.0
                        running = False
                        termination_reason = "keyboard_quit"
                        break
                    if key == " ":
                        active_velocity[:] = 0.0
                        command_deadline = 0.0
                        self._record_event(command_events, key, state, active_velocity)
                        continue
                    if key == "+":
                        speed_scale = min(3.0, speed_scale + 0.1)
                        print(f"speed scale: {speed_scale:.1f}", flush=True)
                        continue
                    if key == "-":
                        speed_scale = max(0.1, speed_scale - 0.1)
                        print(f"speed scale: {speed_scale:.1f}", flush=True)
                        continue
                    if key == "m":
                        label = keyboard.read_label()
                        if label:
                            anchor = self._save_anchor(
                                label=label,
                                state=state,
                                capture=capture,
                                anchor_index=len(anchors),
                            )
                            anchors.append(anchor)
                            print(
                                "ANCHOR_SAVED "
                                f"index={len(anchors) - 1} label={anchor.label!r} "
                                f"step={anchor.step_id} "
                                f"position={list(anchor.root_pos_world)}",
                                flush=True,
                            )
                        continue
                    if key in ("h", "?"):
                        print(self.controls_text(), flush=True)
                        continue
                    motion = self._motion_for_key(key, speed_scale)
                    if motion is None:
                        continue
                    active_velocity[:] = motion
                    command_deadline = time.monotonic() + self.command_hold_s
                    self._record_event(command_events, key, state, active_velocity)

                if not running:
                    break
                # Reading a label temporarily switches to canonical terminal
                # input, and a slow terminal poll can itself take time. Check
                # the watchdog against a fresh wall-clock value after input.
                now = time.monotonic()
                if now >= command_deadline:
                    active_velocity[:] = 0.0

                command = self._velocity_command(active_velocity, control_dt)
                set_velocity_command = getattr(self.physics, "set_velocity_command", None)
                if not callable(set_velocity_command):
                    raise TypeError("keyboard teleop requires set_velocity_command()")
                set_velocity_command(command)
                raw_step = self.physics.step()
                step = raw_step if isinstance(raw_step, PhysicsStep) else PhysicsStep(raw_step)
                state = step.state
                control_steps += 1

                if control_steps % capture_ticks == 0:
                    capture()
                if step.terminated or step.truncated:
                    termination_reason = "terminated" if step.terminated else "truncated"
                    break
                if requested_steps and control_steps >= requested_steps:
                    termination_reason = "max_control_steps"
                    break
                if max_sim_time_s and state.sim_time_s >= max_sim_time_s:
                    termination_reason = "max_sim_time"
                    break

                if self.realtime:
                    next_tick += control_dt
                    time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            try:
                set_velocity_command = getattr(self.physics, "set_velocity_command", None)
                if callable(set_velocity_command):
                    set_velocity_command(VelocityCommand(0.0, 0.0, 0.0, 0.0, stop=True))
            finally:
                self._write_artifacts(
                    state=state,
                    anchors=anchors,
                    command_events=command_events,
                    control_steps=control_steps,
                    termination_reason=termination_reason,
                )

        return TeleopResult(
            termination_reason=termination_reason,
            control_steps=control_steps,
            sim_time_s=float(state.sim_time_s),
            anchors=tuple(anchors),
            final_state=state,
        )

    def _motion_for_key(self, key: str, speed_scale: float) -> tuple[float, float, float] | None:
        forward = self.forward_speed_mps * speed_scale
        strafe = self.strafe_speed_mps * speed_scale
        turn = self.turn_rate_rad_s * speed_scale
        return {
            "w": (forward, 0.0, 0.0),
            "s": (-forward, 0.0, 0.0),
            "a": (0.0, 0.0, turn),
            "d": (0.0, 0.0, -turn),
            "q": (0.0, strafe, 0.0),
            "e": (0.0, -strafe, 0.0),
        }.get(key)

    @staticmethod
    def _velocity_command(velocity: np.ndarray, duration_s: float) -> VelocityCommand:
        if np.allclose(velocity, 0.0):
            return VelocityCommand(0.0, 0.0, 0.0, 0.0, stop=True)
        return VelocityCommand(
            float(velocity[0]),
            float(velocity[1]),
            float(velocity[2]),
            float(duration_s),
        )

    @staticmethod
    def _command_text(velocity: np.ndarray, speed_scale: float) -> str:
        return (
            f"vx={velocity[0]:+.2f} m/s, vy={velocity[1]:+.2f} m/s, "
            f"wz={velocity[2]:+.2f} rad/s, scale={speed_scale:.1f}"
        )

    @staticmethod
    def _record_event(
        events: list[dict[str, Any]], key: str, state: RobotState, velocity: np.ndarray
    ) -> None:
        events.append(
            {
                "key": key,
                "step_id": int(state.step_id),
                "sim_time_s": float(state.sim_time_s),
                "velocity": [float(value) for value in velocity],
            }
        )

    def _save_anchor(
        self,
        *,
        label: str,
        state: RobotState,
        capture: Any,
        anchor_index: int,
    ) -> TeleopAnchor:
        image_path: str | None = None
        frame = capture(record=True)
        if frame is not None and self.output_dir is not None:
            safe_label = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in label.strip()
            ).strip("_") or f"anchor_{anchor_index:03d}"
            relative = Path("anchors") / f"{anchor_index:03d}_{safe_label}.png"
            frame.to_pil().save(self.output_dir / relative, format="PNG")
            image_path = str(relative)
        return TeleopAnchor(
            label=label,
            step_id=int(state.step_id),
            sim_time_s=float(state.sim_time_s),
            root_pos_world=tuple(float(value) for value in state.root_pos_world),
            root_quat_wxyz=tuple(float(value) for value in state.root_quat_wxyz),
            base_rpy=tuple(float(value) for value in state.base_rpy),
            image_path=image_path,
        )

    def _write_artifacts(
        self,
        *,
        state: RobotState,
        anchors: Sequence[TeleopAnchor],
        command_events: Sequence[dict[str, Any]],
        control_steps: int,
        termination_reason: str,
    ) -> None:
        if self.output_dir is None:
            return
        payload = {
            "version": 1,
            "termination_reason": termination_reason,
            "control_steps": int(control_steps),
            "sim_time_s": float(state.sim_time_s),
            "anchors": [asdict(anchor) for anchor in anchors],
            "final_state": {
                "step_id": int(state.step_id),
                "sim_time_s": float(state.sim_time_s),
                "root_pos_world": state.root_pos_world.tolist(),
                "root_quat_wxyz": state.root_quat_wxyz.tolist(),
                "base_rpy": state.base_rpy.tolist(),
            },
            "controls": {
                "forward_speed_mps": self.forward_speed_mps,
                "strafe_speed_mps": self.strafe_speed_mps,
                "turn_rate_rad_s": self.turn_rate_rad_s,
                "command_hold_s": self.command_hold_s,
                "capture_interval_s": self.capture_interval_s,
            },
            "command_events": list(command_events),
        }
        (self.output_dir / "teleop.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
