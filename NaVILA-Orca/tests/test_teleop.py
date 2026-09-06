import json
import time

import numpy as np

from navila_orca.contracts import EpisodeSpec, RenderFrame, RobotState
from navila_orca.teleop import KeyboardTeleopRunner


def _state(step_id: int, time_s: float, position) -> RobotState:
    return RobotState(
        step_id=step_id,
        sim_time_s=time_s,
        root_pos_world=np.asarray(position, dtype=np.float64),
        root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        body_ang_vel=np.zeros(3),
        base_rpy=np.zeros(3),
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        last_raw_action=np.zeros(12),
    )


class FakePhysics:
    control_dt = 0.02

    def __init__(self):
        self.state = _state(0, 0.0, [0.0, 0.0, 0.0])
        self.qpos_batch = np.zeros((1, 19))
        self.commands = []

    def reset(self, _episode):
        self.state = _state(0, 0.0, [0.0, 0.0, 0.0])
        return self.state

    def set_velocity_command(self, command):
        self.commands.append(command)

    def step(self):
        position = self.state.root_pos_world.copy()
        command = self.commands[-1]
        position[0] += command.vx * self.control_dt
        self.state = _state(
            self.state.step_id + 1,
            self.state.sim_time_s + self.control_dt,
            position,
        )
        return self.state


class FakeRenderer:
    def __init__(self):
        self.pushed = []
        self.captures = 0

    def push_state(self, state, qpos_batch=None):
        self.pushed.append((state.step_id, qpos_batch.shape))

    def capture(self, state, qpos_batch=None):
        self.captures += 1
        return RenderFrame(
            state.step_id,
            state.sim_time_s,
            "ego",
            np.full((4, 5, 3), state.step_id, dtype=np.uint8),
            str(state.step_id),
        )

    def close(self):
        pass


class FakeKeyboard:
    def __init__(self, events, labels=(), delays=()):
        self.events = iter(events)
        self.labels = iter(labels)
        self.delays = iter(delays)

    def poll(self):
        time.sleep(next(self.delays, 0.0))
        return next(self.events, ())

    def read_label(self):
        return next(self.labels, "")


def _episode():
    return EpisodeSpec(
        episode_id="teleop",
        scene_id="synthetic",
        instruction="",
        start_position=np.zeros(3),
        start_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        goal_position=np.ones(3),
        goal_radius=0.1,
        reference_path=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        gt_locations=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )


def test_keyboard_teleop_records_pose_and_png_anchor(tmp_path):
    physics = FakePhysics()
    renderer = FakeRenderer()
    result = KeyboardTeleopRunner(
        physics,
        renderer,
        output_dir=tmp_path,
        capture_interval_s=0.02,
        realtime=False,
    ).run(
        _episode(),
        FakeKeyboard([("w",), ("m",), ("x",)], labels=("dining_table",)),
    )

    assert result.termination_reason == "keyboard_quit"
    assert result.control_steps == 2
    assert len(result.anchors) == 1
    assert result.anchors[0].label == "dining_table"
    assert result.anchors[0].step_id == 1
    assert renderer.captures == 1
    assert [step for step, _shape in renderer.pushed] == [0, 1, 2]
    assert (tmp_path / "anchors/000_dining_table.png").is_file()
    payload = json.loads((tmp_path / "teleop.json").read_text())
    assert payload["anchors"][0]["root_pos_world"] == [0.01, 0.0, 0.0]


def test_keyboard_teleop_stops_a_command_after_hold_timeout(tmp_path):
    physics = FakePhysics()
    result = KeyboardTeleopRunner(
        physics,
        output_dir=tmp_path,
        capture_interval_s=0.02,
        command_hold_s=0.001,
        realtime=False,
    ).run(
        _episode(),
        FakeKeyboard([("w",), (), ("x",)], delays=(0.0, 0.005, 0.0)),
    )

    assert result.control_steps == 2
    assert physics.commands[0].vx == 0.5
    assert physics.commands[1].stop is True
