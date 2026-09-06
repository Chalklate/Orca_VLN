"""Engine-independent NaVILA navigation orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

import numpy as np

from .actions import parse_velocity_command
from .contracts import (
    EpisodeSpec,
    JointActionPhysicsBackend,
    LocomotionPolicy,
    PhysicsStep,
    RenderBridge,
    RobotState,
    VLMClient,
    VelocityPhysicsBackend,
    VelocityCommand,
)
from .frames import sample_history
from .metrics import MetricValue, NavigationMetrics


class TimingError(ValueError):
    """Raised when a simulated duration is not an exact control-tick count."""


@dataclass(frozen=True, slots=True)
class MotionChunkReport:
    """Commanded versus measured base motion for one NaVILA action chunk."""

    decision: int
    start_step: int
    end_step: int
    executed_duration_s: float
    target_distance_m: float
    measured_distance_m: float
    distance_error_m: float
    forward_progress_m: float
    lateral_drift_m: float
    target_forward_velocity_mps: float
    measured_forward_velocity_mps: float
    forward_velocity_error_mps: float
    target_yaw_deg: float
    measured_yaw_deg: float
    yaw_error_deg: float
    target_yaw_rate_rad_s: float
    measured_yaw_rate_rad_s: float
    yaw_rate_error_rad_s: float
    cumulative_forward_error_m: float
    cumulative_signed_yaw_error_deg: float
    cumulative_yaw_magnitude_error_deg: float


@dataclass(frozen=True, slots=True)
class RunResult:
    metrics: dict[str, MetricValue]
    termination_reason: str
    control_steps: int
    decisions: int
    vlm_outputs: tuple[str, ...]
    motion_chunks: tuple[MotionChunkReport, ...]
    final_state: RobotState
    waypoints_completed: int = 0
    waypoint_count: int = 0
    waypoint_stop_rejections: int = 0


def duration_to_ticks(duration_s: float, control_dt: float) -> int:
    """Convert simulated seconds to an exact integer number of control ticks."""

    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise TimingError("control_dt must be positive and finite")
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise TimingError("duration_s must be non-negative and finite")
    ratio = duration_s / control_dt
    ticks = int(round(ratio))
    if not np.isclose(ratio, ticks, rtol=0.0, atol=1.0e-9):
        raise TimingError(
            f"duration {duration_s} is not divisible by control_dt {control_dt}"
        )
    return ticks


class NavigationRunner:
    """Compose physics, rendering, locomotion, VLM inference, and metrics."""

    def __init__(
        self,
        physics: VelocityPhysicsBackend | JointActionPhysicsBackend,
        renderer: RenderBridge,
        vlm_client: VLMClient,
        *,
        locomotion_policy: LocomotionPolicy | None = None,
        scene_fidelity: bool,
        image_interval_s: float = 0.5,
        state_stream_interval_s: float = 0.04,
        max_control_steps: int = 5_000,
        max_decisions: int = 200,
        monitor: Any | None = None,
        monitor_interval_s: float = 0.1,
        decouple_vlm: bool = False,
        action_parser: Callable[[str], VelocityCommand] = parse_velocity_command,
        waypoint_instructions: Sequence[str] | None = None,
        instruction_provider: Callable[[], str] | None = None,
    ) -> None:
        self.physics = physics
        self.renderer = renderer
        self.locomotion_policy = locomotion_policy
        self.vlm_client = vlm_client
        self.scene_fidelity = bool(scene_fidelity)
        self.image_interval_s = float(image_interval_s)
        self.state_stream_interval_s = float(state_stream_interval_s)
        requested_control_steps = int(max_control_steps)
        requested_decisions = int(max_decisions)
        self.max_control_steps = (
            None if requested_control_steps == 0 else requested_control_steps
        )
        self.max_decisions = None if requested_decisions == 0 else requested_decisions
        self.monitor = monitor
        self.monitor_interval_s = float(monitor_interval_s)
        self.decouple_vlm = bool(decouple_vlm)
        self.action_parser = action_parser
        if instruction_provider is not None and not callable(instruction_provider):
            raise TypeError("instruction_provider must be callable")
        self.instruction_provider = instruction_provider
        if not waypoint_instructions:
            self.waypoint_instructions: tuple[str, ...] = ()
        else:
            self.waypoint_instructions = tuple(
                str(instruction).strip() for instruction in waypoint_instructions
            )
            if not self.waypoint_instructions or any(
                not instruction for instruction in self.waypoint_instructions
            ):
                raise ValueError("waypoint instructions must be non-empty strings")
        if self.waypoint_instructions and self.instruction_provider is not None:
            raise ValueError(
                "instruction_provider and waypoint_instructions are mutually exclusive"
            )
        if requested_control_steps < 0 or requested_decisions < 0:
            raise ValueError("runner limits must be non-negative; zero means unlimited")
        self._velocity_facade = callable(
            getattr(self.physics, "set_velocity_command", None)
        )
        if not self._velocity_facade and self.locomotion_policy is None:
            raise TypeError("a joint-action physics backend requires locomotion_policy")

    def _capture(self, state: RobotState, *, record: bool = True):
        qpos_batch = getattr(self.physics, "qpos_batch", None)
        capture = (
            getattr(self.renderer, "capture", None)
            if record
            else getattr(self.renderer, "capture_transient", None)
        )
        if not callable(capture):
            capture = getattr(self.renderer, "capture", None)
        if callable(capture):
            frame = capture(state, qpos_batch=qpos_batch)
        else:
            frame = self.renderer.render(state, qpos_batch=qpos_batch)
        if frame.step_id != state.step_id:
            raise RuntimeError(
                f"renderer returned stale frame for step {frame.step_id}; requested {state.step_id}"
            )
        return frame

    def _push_state(self, state: RobotState) -> bool:
        push_state = getattr(self.renderer, "push_state", None)
        if not callable(push_state):
            return False
        qpos_batch = getattr(self.physics, "qpos_batch", None)
        push_state(state, qpos_batch=qpos_batch)
        return True

    def run(self, episode: EpisodeSpec) -> RunResult:
        control_dt = float(self.physics.control_dt)
        capture_ticks = duration_to_ticks(self.image_interval_s, control_dt)
        stream_ticks = duration_to_ticks(self.state_stream_interval_s, control_dt)
        monitor_ticks = (
            duration_to_ticks(self.monitor_interval_s, control_dt)
            if self.monitor is not None
            else None
        )
        if capture_ticks <= 0:
            raise TimingError("image_interval_s must span at least one control tick")
        if stream_ticks <= 0:
            raise TimingError(
                "state_stream_interval_s must span at least one control tick"
            )
        if monitor_ticks is not None and monitor_ticks <= 0:
            raise TimingError(
                "monitor_interval_s must span at least one control tick"
            )

        state = self.physics.reset(episode)
        if self.locomotion_policy is not None:
            self.locomotion_policy.reset()
        metrics = NavigationMetrics(
            episode, state.root_pos_world, scene_fidelity=self.scene_fidelity
        )
        self._push_state(state)
        initial_frame = self._capture(state)
        frame_history = [initial_frame]
        last_frame = initial_frame
        raw_outputs: list[str] = []
        motion_chunks: list[MotionChunkReport] = []
        control_steps = 0
        decisions = 0
        cumulative_forward_error_m = 0.0
        cumulative_signed_yaw_error_deg = 0.0
        cumulative_yaw_magnitude_error_deg = 0.0
        termination_reason = "max_decisions"
        monitor_output = "Waiting for first VLM decision..."
        monitor_command = "none"
        monitor_chunk_result = "none completed yet"
        waypoint_index = 0
        waypoint_count = len(self.waypoint_instructions)
        waypoints_completed = 0
        waypoint_stop_rejections = 0
        waypoint_motion_chunks = 0
        waypoint_stop_rejected = False

        def active_instruction() -> str:
            if not self.waypoint_instructions:
                instruction = (
                    self.instruction_provider()
                    if self.instruction_provider is not None
                    else episode.instruction
                )
                instruction = str(instruction).strip()
                if not instruction:
                    raise ValueError("active navigation instruction is empty")
                return instruction
            instruction = (
                f"Waypoint {waypoint_index + 1} of {waypoint_count}. "
                f"{self.waypoint_instructions[waypoint_index]}"
            )
            if waypoint_stop_rejected:
                instruction += (
                    " The previous stop was rejected because no new motion has "
                    "been executed for this waypoint. Do not output stop yet; "
                    "choose a turn or forward action toward this waypoint."
                )
            return instruction

        current_instruction = active_instruction()
        ready_output: str | None = None
        ready_instruction: str | None = None
        if self.monitor is not None:
            self.monitor.update(
                initial_frame,
                instruction=current_instruction,
                vlm_output=monitor_output,
                command=monitor_command,
                status="ready",
                decision=decisions,
                chunk_result=monitor_chunk_result,
            )

        def advance_one_tick(
            command: VelocityCommand,
            *,
            status: str,
        ) -> tuple[bool, bool, bool]:
            """Advance one physics tick and service the render/monitor cadence."""

            nonlocal state, last_frame, control_steps, termination_reason
            if (
                self.max_control_steps is not None
                and control_steps >= self.max_control_steps
            ):
                termination_reason = "max_control_steps"
                return False, True, False

            if self._velocity_facade:
                raw_step = self.physics.step()
            else:
                terrain = np.asarray(
                    self.physics.terrain_features(state), dtype=np.float64
                )
                action = np.asarray(
                    self.locomotion_policy.act(state, terrain, command),
                    dtype=np.float64,
                )
                if action.ndim != 1 or not np.all(np.isfinite(action)):
                    raise ValueError(
                        "locomotion policy action must be a finite one-dimensional array"
                    )
                raw_step = self.physics.step(action)
            step = (
                raw_step
                if isinstance(raw_step, PhysicsStep)
                else PhysicsStep(raw_step)
            )
            if step.state.step_id <= state.step_id:
                raise RuntimeError(
                    "physics state step_id must increase monotonically"
                )
            state = step.state
            control_steps += 1
            auto_reset_state = bool(step.info.get("auto_reset_state", False))
            measured = not ((step.terminated or step.truncated) and auto_reset_state)
            if measured:
                metrics.update(state.root_pos_world)

                capture_due = control_steps % capture_ticks == 0
                monitor_due = (
                    monitor_ticks is not None
                    and control_steps % monitor_ticks == 0
                )
                stream_due = control_steps % stream_ticks == 0
                if stream_due or capture_due or monitor_due:
                    self._push_state(state)
                if capture_due or monitor_due:
                    # A split bridge captures the frame produced after the most
                    # recent pose push. Legacy bridges render synchronously here.
                    last_frame = self._capture(state, record=capture_due)
                    if capture_due:
                        frame_history.append(last_frame)
                    if self.monitor is not None:
                        self.monitor.update(
                            last_frame,
                            instruction=current_instruction,
                            vlm_output=monitor_output,
                            command=monitor_command,
                            status=status,
                            decision=decisions,
                            chunk_result=monitor_chunk_result,
                        )

            physics_done = bool(step.terminated or step.truncated)
            if physics_done:
                termination_reason = (
                    "terminated" if step.terminated else "truncated"
                )
            return True, physics_done, measured

        while self.max_decisions is None or decisions < self.max_decisions:
            if ready_output is not None:
                raw_output = ready_output
                current_instruction = ready_instruction or current_instruction
                ready_output = None
                ready_instruction = None
            else:
                if decisions:
                    current_instruction = active_instruction()
                sampled_images = sample_history(frame_history)
                if self.monitor is not None:
                    self.monitor.update(
                        last_frame,
                        instruction=current_instruction,
                        vlm_output=monitor_output,
                        command=monitor_command,
                        status="waiting for VLM response",
                        decision=decisions + 1,
                        chunk_result=monitor_chunk_result,
                    )

                def infer():
                    return self.vlm_client.infer(sampled_images, current_instruction)

                responsive_infer = (
                    getattr(self.monitor, "run_while_responsive", None)
                    if self.monitor is not None
                    else None
                )
                raw_output = (
                    responsive_infer(infer) if callable(responsive_infer) else infer()
                )

            raw_outputs.append(raw_output)
            decisions += 1
            command = self.action_parser(raw_output)
            monitor_output = raw_output
            monitor_command = self._command_text(command)

            if command.stop:
                if waypoint_count and waypoint_motion_chunks == 0:
                    waypoint_stop_rejections += 1
                    waypoint_stop_rejected = True
                    monitor_output = (
                        f"Ignored premature stop for waypoint "
                        f"{waypoint_index + 1}/{waypoint_count}: no new motion "
                        "has been executed for this waypoint"
                    )
                    monitor_command = "stop rejected; request a motion action"
                    current_instruction = active_instruction()
                    print(
                        "WAYPOINT_STOP_REJECTED "
                        f"waypoint={waypoint_index + 1}/{waypoint_count} "
                        "reason=no_motion_since_stage_start",
                        flush=True,
                    )
                    if self.monitor is not None:
                        self.monitor.update(
                            last_frame,
                            instruction=current_instruction,
                            vlm_output=monitor_output,
                            command=monitor_command,
                            status="premature waypoint stop rejected",
                            decision=decisions,
                            chunk_result=monitor_chunk_result,
                        )
                    continue
                if waypoint_count and waypoint_index + 1 < waypoint_count:
                    waypoints_completed = waypoint_index + 1
                    waypoint_index += 1
                    waypoint_motion_chunks = 0
                    waypoint_stop_rejected = False
                    if self._velocity_facade:
                        self.physics.set_velocity_command(
                            VelocityCommand(0.0, 0.0, 0.0, 0.0)
                        )
                    # Start the next visual subgoal from the confirmed waypoint
                    # instead of carrying an image history biased toward the old one.
                    frame_history = [last_frame]
                    monitor_output = (
                        f"Waypoint {waypoints_completed}/{waypoint_count} completed"
                    )
                    monitor_command = "advance to next waypoint"
                    current_instruction = active_instruction()
                    if self.monitor is not None:
                        self.monitor.update(
                            last_frame,
                            instruction=current_instruction,
                            vlm_output=monitor_output,
                            command=monitor_command,
                            status="intermediate waypoint confirmed",
                            decision=decisions,
                            chunk_result=monitor_chunk_result,
                        )
                    continue
                if waypoint_count:
                    waypoints_completed = waypoint_count
                metrics.update(state.root_pos_world, stop_called=True)
                termination_reason = "stop"
                if self.monitor is not None:
                    self.monitor.update(
                        last_frame,
                        instruction=current_instruction,
                        vlm_output=monitor_output,
                        command=monitor_command,
                        status="VLM requested stop",
                        decision=decisions,
                        chunk_result=monitor_chunk_result,
                    )
                break

            if self.monitor is not None:
                self.monitor.update(
                    last_frame,
                    instruction=current_instruction,
                    vlm_output=monitor_output,
                    command=monitor_command,
                    status="executing motion chunk",
                    decision=decisions,
                    chunk_result=monitor_chunk_result,
                )

            command_ticks = duration_to_ticks(command.duration_s, control_dt)
            if command_ticks <= 0:
                raise TimingError(
                    "non-stop actions must span at least one control tick"
                )

            if self._velocity_facade:
                # One high-level action owns a fixed velocity command. The
                # backend keeps it latched across policy ticks.
                self.physics.set_velocity_command(command)
            chunk_start_state = state
            chunk_measured_state = state
            executed_ticks = 0
            physics_done = False
            for _ in range(command_ticks):
                did_step, physics_done, measured = advance_one_tick(
                    command,
                    status="executing motion chunk",
                )
                if not did_step:
                    break
                executed_ticks += 1
                if measured:
                    chunk_measured_state = state
                if physics_done:
                    break

            if executed_ticks:
                if waypoint_count:
                    waypoint_motion_chunks += 1
                    waypoint_stop_rejected = False
                report = self._motion_chunk_report(
                    decision=decisions,
                    command=command,
                    control_dt=control_dt,
                    executed_ticks=executed_ticks,
                    start=chunk_start_state,
                    end=chunk_measured_state,
                    previous_cumulative_forward_error_m=cumulative_forward_error_m,
                    previous_cumulative_signed_yaw_error_deg=(
                        cumulative_signed_yaw_error_deg
                    ),
                    previous_cumulative_yaw_magnitude_error_deg=(
                        cumulative_yaw_magnitude_error_deg
                    ),
                )
                cumulative_forward_error_m = report.cumulative_forward_error_m
                cumulative_signed_yaw_error_deg = (
                    report.cumulative_signed_yaw_error_deg
                )
                cumulative_yaw_magnitude_error_deg = (
                    report.cumulative_yaw_magnitude_error_deg
                )
                motion_chunks.append(report)
                monitor_chunk_result = self._chunk_report_text(report)
                print(
                    "MOTION_CHUNK " + monitor_chunk_result,
                    flush=True,
                )
                if self.monitor is not None:
                    self.monitor.update(
                        last_frame,
                        instruction=current_instruction,
                        vlm_output=monitor_output,
                        command=monitor_command,
                        status="motion chunk completed",
                        decision=decisions,
                        chunk_result=monitor_chunk_result,
                    )

            if (
                not physics_done
                and self.decouple_vlm
                and (self.max_decisions is None or decisions < self.max_decisions)
            ):
                # The next request is based on the completed action and its
                # newest captured frame history. While NaVILA is thinking, the
                # command deadline has expired, so physics continues under a
                # zero-velocity watchdog instead of extending the old action.
                next_instruction = active_instruction()
                current_instruction = next_instruction
                sampled_images = sample_history(frame_history)
                if self.monitor is not None:
                    monitor_command = "zero-velocity watchdog"
                    self.monitor.update(
                        last_frame,
                        instruction=next_instruction,
                        vlm_output=monitor_output,
                        command="zero-velocity watchdog",
                        status="waiting for VLM response; physics continues",
                        decision=decisions + 1,
                        chunk_result=monitor_chunk_result,
                    )

                inference_pool = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="navila-vlm",
                )
                inference_future = inference_pool.submit(
                    self.vlm_client.infer,
                    sampled_images,
                    next_instruction,
                )
                zero_command = VelocityCommand(0.0, 0.0, 0.0, 0.0)
                if self._velocity_facade:
                    self.physics.set_velocity_command(zero_command)
                try:
                    while not inference_future.done():
                        did_step, physics_done, _ = advance_one_tick(
                            zero_command,
                            status="waiting for VLM response; physics continues",
                        )
                        if not did_step or physics_done:
                            break
                        time.sleep(0.001)
                    if not physics_done:
                        ready_output = inference_future.result()
                        ready_instruction = next_instruction
                finally:
                    inference_pool.shutdown(wait=True, cancel_futures=True)

            if (
                not physics_done
                and self.max_control_steps is not None
                and control_steps >= self.max_control_steps
            ):
                termination_reason = "max_control_steps"
                physics_done = True
            if physics_done:
                break

        return RunResult(
            metrics=metrics.as_dict(),
            termination_reason=termination_reason,
            control_steps=control_steps,
            decisions=decisions,
            vlm_outputs=tuple(raw_outputs),
            motion_chunks=tuple(motion_chunks),
            final_state=state,
            waypoints_completed=waypoints_completed,
            waypoint_count=waypoint_count,
            waypoint_stop_rejections=waypoint_stop_rejections,
        )

    @staticmethod
    def _motion_chunk_report(
        *,
        decision: int,
        command: VelocityCommand,
        control_dt: float,
        executed_ticks: int,
        start: RobotState,
        end: RobotState,
        previous_cumulative_forward_error_m: float,
        previous_cumulative_signed_yaw_error_deg: float,
        previous_cumulative_yaw_magnitude_error_deg: float,
    ) -> MotionChunkReport:
        duration_s = executed_ticks * control_dt
        delta_xy = np.asarray(end.root_pos_world[:2] - start.root_pos_world[:2])
        measured_distance_m = float(np.linalg.norm(delta_xy))
        target_distance_m = float(np.hypot(command.vx, command.vy) * duration_s)
        start_yaw = float(start.base_rpy[2])
        heading = np.array([np.cos(start_yaw), np.sin(start_yaw)])
        left = np.array([-np.sin(start_yaw), np.cos(start_yaw)])
        forward_progress_m = float(np.dot(delta_xy, heading))
        lateral_drift_m = float(np.dot(delta_xy, left))
        yaw_delta = float(
            (float(end.base_rpy[2]) - start_yaw + np.pi) % (2.0 * np.pi) - np.pi
        )
        target_yaw = float(command.wz * duration_s)
        measured_forward_velocity_mps = forward_progress_m / duration_s
        measured_yaw_rate_rad_s = yaw_delta / duration_s
        target_yaw_deg = float(np.degrees(target_yaw))
        measured_yaw_deg = float(np.degrees(yaw_delta))
        cumulative_forward_error_m = previous_cumulative_forward_error_m
        if target_distance_m > 0.0:
            cumulative_forward_error_m += forward_progress_m - target_distance_m
        cumulative_signed_yaw_error_deg = previous_cumulative_signed_yaw_error_deg
        cumulative_yaw_magnitude_error_deg = (
            previous_cumulative_yaw_magnitude_error_deg
        )
        if abs(target_yaw_deg) > 0.0:
            cumulative_signed_yaw_error_deg += measured_yaw_deg - target_yaw_deg
            cumulative_yaw_magnitude_error_deg += (
                abs(measured_yaw_deg) - abs(target_yaw_deg)
            )
        return MotionChunkReport(
            decision=decision,
            start_step=start.step_id,
            end_step=end.step_id,
            executed_duration_s=duration_s,
            target_distance_m=target_distance_m,
            measured_distance_m=measured_distance_m,
            distance_error_m=measured_distance_m - target_distance_m,
            forward_progress_m=forward_progress_m,
            lateral_drift_m=lateral_drift_m,
            target_forward_velocity_mps=float(command.vx),
            measured_forward_velocity_mps=measured_forward_velocity_mps,
            forward_velocity_error_mps=(
                measured_forward_velocity_mps - float(command.vx)
            ),
            target_yaw_deg=target_yaw_deg,
            measured_yaw_deg=measured_yaw_deg,
            yaw_error_deg=measured_yaw_deg - target_yaw_deg,
            target_yaw_rate_rad_s=float(command.wz),
            measured_yaw_rate_rad_s=measured_yaw_rate_rad_s,
            yaw_rate_error_rad_s=measured_yaw_rate_rad_s - float(command.wz),
            cumulative_forward_error_m=cumulative_forward_error_m,
            cumulative_signed_yaw_error_deg=cumulative_signed_yaw_error_deg,
            cumulative_yaw_magnitude_error_deg=cumulative_yaw_magnitude_error_deg,
        )

    @staticmethod
    def _chunk_report_text(report: MotionChunkReport) -> str:
        return (
            f"decision={report.decision} "
            f"distance ideal={report.target_distance_m:.3f}m "
            f"actual={report.measured_distance_m:.3f}m "
            f"error={report.distance_error_m:+.3f}m "
            f"forward={report.forward_progress_m:.3f}m "
            f"lateral={report.lateral_drift_m:+.3f}m; "
            f"vx ideal={report.target_forward_velocity_mps:+.3f}m/s "
            f"actual={report.measured_forward_velocity_mps:+.3f}m/s "
            f"error={report.forward_velocity_error_mps:+.3f}m/s; "
            f"yaw ideal={report.target_yaw_deg:+.1f}deg "
            f"actual={report.measured_yaw_deg:+.1f}deg "
            f"error={report.yaw_error_deg:+.1f}deg; "
            f"wz ideal={report.target_yaw_rate_rad_s:+.3f}rad/s "
            f"actual={report.measured_yaw_rate_rad_s:+.3f}rad/s "
            f"error={report.yaw_rate_error_rad_s:+.3f}rad/s; "
            f"cumulative forward={report.cumulative_forward_error_m:+.3f}m "
            f"signed-yaw={report.cumulative_signed_yaw_error_deg:+.1f}deg "
            f"turn-magnitude={report.cumulative_yaw_magnitude_error_deg:+.1f}deg"
        )

    @staticmethod
    def _command_text(command: VelocityCommand) -> str:
        if command.stop:
            return "stop"
        return (
            f"vx={command.vx:.2f} m/s, vy={command.vy:.2f} m/s, "
            f"wz={command.wz:.3f} rad/s, duration={command.duration_s:.2f}s"
        )
