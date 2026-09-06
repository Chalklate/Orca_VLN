import threading

import numpy as np
import pytest

from navila_orca.contracts import EpisodeSpec, PhysicsStep, RenderFrame, RobotState
from navila_orca.runner import NavigationRunner, TimingError, duration_to_ticks


def _state(step_id: int, time_s: float, position) -> RobotState:
    zeros3 = np.zeros(3)
    zeros12 = np.zeros(12)
    return RobotState(
        step_id=step_id,
        sim_time_s=time_s,
        root_pos_world=np.asarray(position),
        root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        body_ang_vel=zeros3,
        base_rpy=zeros3,
        joint_pos=zeros12,
        joint_vel=zeros12,
        last_raw_action=zeros12,
    )


class FakePhysics:
    control_dt = 0.02

    def __init__(self):
        self.state = _state(0, 0.0, [0.0, 0.0, 0.0])
        self.qpos_batch = np.zeros((1, 19))
        self.command = None
        self.command_updates = 0

    def reset(self, episode):
        self.state = _state(0, 0.0, episode.start_position)
        return self.state

    def set_velocity_command(self, command):
        self.command = command
        self.command_updates += 1

    def step(self):
        position = self.state.root_pos_world.copy()
        position[0] += self.command.vx * self.control_dt
        self.state = _state(
            self.state.step_id + 1, self.state.sim_time_s + self.control_dt, position
        )
        return self.state

    def close(self):
        pass


class FakeRenderer:
    def __init__(self):
        self.steps = []

    def render(self, state, qpos_batch=None):
        assert qpos_batch.shape == (1, 19)
        self.steps.append(state.step_id)
        rgb = np.full((8, 8, 3), state.step_id % 255, dtype=np.uint8)
        return RenderFrame(
            state.step_id, state.sim_time_s, "ego", rgb, str(state.step_id)
        )

    def close(self):
        pass


class ScriptedVLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def infer(self, images, instruction):
        self.requests.append((images, instruction))
        assert len(images) == 8
        return next(self.outputs)


def _episode():
    return EpisodeSpec(
        episode_id="1",
        scene_id="synthetic",
        instruction="move to the target",
        start_position=np.array([0.0, 0.0, 0.0]),
        start_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        goal_position=np.array([0.75, 0.0, 0.0]),
        goal_radius=0.1,
        reference_path=np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        gt_locations=np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]),
    )


def test_runner_executes_exact_ticks_samples_history_and_stops_normally():
    physics = FakePhysics()
    renderer = FakeRenderer()
    vlm = ScriptedVLM(
        [
            "The next action is move forward 75 cm.",
            "The next action is turn left 30 degrees.",
            "The next action is stop.",
        ]
    )
    result = NavigationRunner(
        physics,
        renderer,
        vlm,
        scene_fidelity=False,
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert result.control_steps == 75 + 50
    assert result.decisions == 3
    assert renderer.steps == [0, 25, 50, 75, 100, 125]
    assert len(vlm.requests) == 3
    assert result.metrics["scene_fidelity"] is False
    assert result.metrics["success"] == 1.0
    assert result.metrics["spl"] == pytest.approx(1.0)
    assert result.metrics["path_length"] == pytest.approx(0.75)
    assert physics.command_updates == 2
    assert len(result.motion_chunks) == 2
    forward, turn = result.motion_chunks
    assert forward.target_distance_m == pytest.approx(0.75)
    assert forward.measured_distance_m == pytest.approx(0.75)
    assert forward.distance_error_m == pytest.approx(0.0)
    assert forward.forward_progress_m == pytest.approx(0.75)
    assert forward.target_forward_velocity_mps == pytest.approx(0.5)
    assert forward.measured_forward_velocity_mps == pytest.approx(0.5)
    assert forward.forward_velocity_error_mps == pytest.approx(0.0)
    assert forward.cumulative_forward_error_m == pytest.approx(0.0)
    assert turn.target_yaw_deg == pytest.approx(30.0)
    assert turn.measured_yaw_deg == pytest.approx(0.0)
    assert turn.yaw_error_deg == pytest.approx(-30.0)
    assert turn.target_yaw_rate_rad_s == pytest.approx(np.pi / 6.0)
    assert turn.measured_yaw_rate_rad_s == pytest.approx(0.0)
    assert turn.yaw_rate_error_rad_s == pytest.approx(-np.pi / 6.0)
    assert turn.cumulative_signed_yaw_error_deg == pytest.approx(-30.0)
    assert turn.cumulative_yaw_magnitude_error_deg == pytest.approx(-30.0)


def test_empty_waypoint_sequence_disables_staged_mode():
    vlm = ScriptedVLM(["stop"])
    result = NavigationRunner(
        FakePhysics(),
        FakeRenderer(),
        vlm,
        scene_fidelity=False,
        waypoint_instructions=(),
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert result.waypoint_count == 0
    assert result.waypoints_completed == 0
    assert vlm.requests[0][1] == "move to the target"


def test_runner_refreshes_file_backed_instruction_between_decisions():
    current = {"instruction": "Move toward the orange bin."}

    class UpdatingVLM(ScriptedVLM):
        def infer(self, images, instruction):
            output = super().infer(images, instruction)
            if len(self.requests) == 1:
                current["instruction"] = "Turn right toward the blue barrel."
            return output

    vlm = UpdatingVLM(["move forward 25 cm", "stop"])
    result = NavigationRunner(
        FakePhysics(),
        FakeRenderer(),
        vlm,
        scene_fidelity=False,
        instruction_provider=lambda: current["instruction"],
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert [request[1] for request in vlm.requests] == [
        "Move toward the orange bin.",
        "Turn right toward the blue barrel.",
    ]


def test_runner_treats_intermediate_waypoint_stops_as_stage_completion(capsys):
    physics = FakePhysics()
    vlm = ScriptedVLM(
        [
            "move forward 25 cm",
            "stop",
            "move forward 25 cm",
            "stop",
            "move forward 25 cm",
            "stop",
        ]
    )
    result = NavigationRunner(
        physics,
        FakeRenderer(),
        vlm,
        scene_fidelity=False,
        waypoint_instructions=(
            "Reach the orange bin, then stop.",
            "Reach the blue barrel, then stop.",
            "Reach the yellow truck, then stop.",
        ),
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert result.control_steps == 75
    assert result.decisions == 6
    assert result.waypoints_completed == 3
    assert result.waypoint_count == 3
    assert result.waypoint_stop_rejections == 0
    assert [request[1] for request in vlm.requests] == [
        "Waypoint 1 of 3. Reach the orange bin, then stop.",
        "Waypoint 1 of 3. Reach the orange bin, then stop.",
        "Waypoint 2 of 3. Reach the blue barrel, then stop.",
        "Waypoint 2 of 3. Reach the blue barrel, then stop.",
        "Waypoint 3 of 3. Reach the yellow truck, then stop.",
        "Waypoint 3 of 3. Reach the yellow truck, then stop.",
    ]
    output = capsys.readouterr().out
    assert "WAYPOINT_STOP_ACCEPTED waypoint=1/3 decision=2" in output
    assert "WAYPOINT_STOP_ACCEPTED waypoint=2/3 decision=4" in output
    assert "MISSION_STOP_ACCEPTED decision=6 waypoint=3/3" in output
    # Two intermediate zero commands plus three forward commands.
    assert physics.command_updates == 5


def test_runner_rejects_consecutive_stop_after_waypoint_switch():
    physics = FakePhysics()
    vlm = ScriptedVLM(
        [
            "move forward 25 cm",
            "stop",
            "stop",
            "move forward 25 cm",
            "stop",
            "move forward 25 cm",
            "stop",
        ]
    )
    result = NavigationRunner(
        physics,
        FakeRenderer(),
        vlm,
        scene_fidelity=False,
        waypoint_instructions=(
            "Reach the orange bin, then stop.",
            "Reach the blue barrel, then stop.",
            "Reach the yellow truck, then stop.",
        ),
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert result.control_steps == 75
    assert result.decisions == 7
    assert result.waypoints_completed == 3
    assert result.waypoint_stop_rejections == 1
    rejected_retry_instruction = vlm.requests[3][1]
    assert rejected_retry_instruction.startswith(
        "Waypoint 2 of 3. Reach the blue barrel, then stop."
    )
    assert "previous stop was rejected" in rejected_retry_instruction


def test_duration_conversion_rejects_fractional_control_ticks():
    assert duration_to_ticks(1.5, 0.02) == 75
    with pytest.raises(TimingError):
        duration_to_ticks(0.03, 0.02)


def test_split_renderer_streams_pose_faster_than_vlm_capture():
    class StreamingRenderer:
        def __init__(self):
            self.pushes = []
            self.captures = []

        def push_state(self, state, qpos_batch=None):
            assert qpos_batch.shape == (1, 19)
            self.pushes.append(state.step_id)

        def capture(self, state, qpos_batch=None):
            self.captures.append(state.step_id)
            rgb = np.full((8, 8, 3), state.step_id % 255, dtype=np.uint8)
            return RenderFrame(
                state.step_id, state.sim_time_s, "ego", rgb, str(state.step_id)
            )

        def close(self):
            pass

    physics = FakePhysics()
    renderer = StreamingRenderer()
    vlm = ScriptedVLM(["move forward 25 cm", "stop"])
    result = NavigationRunner(
        physics,
        renderer,
        vlm,
        scene_fidelity=False,
        state_stream_interval_s=0.04,
    ).run(_episode())

    assert result.control_steps == 25
    assert renderer.captures == [0, 25]
    assert renderer.pushes == [0, *range(2, 25, 2), 25]


def test_auto_reset_terminal_state_does_not_add_reset_teleport_to_metrics():
    class AutoResetPhysics(FakePhysics):
        def step(self):
            reset_state = _state(1, self.control_dt, [100.0, 0.0, 0.0])
            return PhysicsStep(
                reset_state,
                terminated=True,
                info={"auto_reset_state": True},
            )

    physics = AutoResetPhysics()
    result = NavigationRunner(
        physics,
        FakeRenderer(),
        ScriptedVLM(["move forward 25 cm"]),
        scene_fidelity=False,
    ).run(_episode())

    assert result.termination_reason == "terminated"
    assert result.control_steps == 1
    assert result.metrics["path_length"] == 0.0


def test_exact_control_step_limit_does_not_request_an_extra_vlm_action():
    vlm = ScriptedVLM(["move forward 25 cm"])
    result = NavigationRunner(
        FakePhysics(),
        FakeRenderer(),
        vlm,
        scene_fidelity=False,
        max_control_steps=25,
    ).run(_episode())

    assert result.termination_reason == "max_control_steps"
    assert result.control_steps == 25
    assert result.decisions == 1
    assert len(vlm.requests) == 1


def test_zero_limits_run_until_vlm_stop():
    result = NavigationRunner(
        FakePhysics(),
        FakeRenderer(),
        ScriptedVLM(["move forward 25 cm", "stop"]),
        scene_fidelity=False,
        max_control_steps=0,
        max_decisions=0,
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert result.control_steps == 25
    assert result.decisions == 2


def test_decoupled_vlm_runs_zero_velocity_watchdog_while_waiting():
    class CountingPhysics(FakePhysics):
        def __init__(self):
            super().__init__()
            self.zero_steps = 0
            self.release = threading.Event()

        def step(self):
            if self.command is not None and (
                self.command.vx == 0.0
                and self.command.vy == 0.0
                and self.command.wz == 0.0
            ):
                self.zero_steps += 1
            state = super().step()
            if self.zero_steps >= 5:
                self.release.set()
            return state

    physics = CountingPhysics()

    class GatedVLM(ScriptedVLM):
        def infer(self, images, instruction):
            output = super().infer(images, instruction)
            if len(self.requests) == 2:
                assert physics.release.wait(timeout=1.0)
            return output

    vlm = GatedVLM(["move forward 25 cm", "stop"])
    result = NavigationRunner(
        physics,
        FakeRenderer(),
        vlm,
        scene_fidelity=False,
        decouple_vlm=True,
        max_control_steps=200,
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert result.control_steps >= 30
    assert physics.zero_steps >= 5
    assert result.metrics["path_length"] == pytest.approx(0.25)
    assert len(vlm.requests) == 2


def test_live_monitor_refreshes_faster_without_changing_vlm_history_interval():
    class FakeMonitor:
        def __init__(self):
            self.updates = []

        def update(self, frame, **details):
            self.updates.append((frame.step_id, details))

        def run_while_responsive(self, operation):
            return operation()

    monitor = FakeMonitor()
    renderer = FakeRenderer()
    vlm = ScriptedVLM(["move forward 25 cm", "stop"])
    result = NavigationRunner(
        FakePhysics(),
        renderer,
        vlm,
        scene_fidelity=False,
        image_interval_s=0.5,
        monitor=monitor,
        monitor_interval_s=0.1,
    ).run(_episode())

    assert result.termination_reason == "stop"
    assert renderer.steps == [0, 5, 10, 15, 20, 25]
    assert len(vlm.requests) == 2
    assert monitor.updates[-1][1]["status"] == "VLM requested stop"
