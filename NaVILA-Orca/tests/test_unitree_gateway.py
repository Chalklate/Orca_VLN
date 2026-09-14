import json
from types import SimpleNamespace

from PIL import Image
import pytest

from navila_orca.contracts import VelocityCommand
from navila_orca.hardware.unitree_gateway import (
    HardwareDecisionRecorder,
    JpegFrameHistory,
    SafeMotionExecutor,
    SafetyLimits,
    UnitreeSportController,
)


class FakeController:
    def __init__(self, *, fail_move=False):
        self.moves = []
        self.stops = 0
        self.stands = 0
        self.damps = 0
        self.fail_move = fail_move

    def move(self, vx, vy, wz):
        self.moves.append((vx, vy, wz))
        if self.fail_move:
            raise RuntimeError("move failed")

    def stop_move(self):
        self.stops += 1

    def balance_stand(self):
        self.stands += 1

    def damp(self):
        self.damps += 1


def test_safety_limits_clamp_every_motion_dimension_and_duration():
    limits = SafetyLimits(
        max_forward_mps=0.2,
        max_lateral_mps=0.1,
        max_yaw_rps=0.3,
        max_duration_s=0.5,
    )
    bounded = limits.apply(VelocityCommand(0.5, -0.4, 0.8, 1.5))
    assert (bounded.vx, bounded.vy, bounded.wz, bounded.duration_s) == pytest.approx(
        (0.2, -0.1, 0.3, 0.5)
    )


def test_executor_always_stops_after_success():
    controller = FakeController()
    executor = SafeMotionExecutor(controller, command_hz=200.0)
    bounded = executor.execute(VelocityCommand(0.1, 0.0, 0.0, 0.02))
    assert bounded.vx == pytest.approx(0.1)
    assert controller.moves
    assert controller.stops == 1


def test_executor_always_stops_after_move_failure():
    controller = FakeController(fail_move=True)
    executor = SafeMotionExecutor(controller)
    with pytest.raises(RuntimeError, match="move failed"):
        executor.execute(VelocityCommand(0.1, 0.0, 0.0, 0.1))
    assert controller.stops == 1


def test_jpeg_history_matches_eight_frame_sampling_shape():
    history = JpegFrameHistory()
    for value in range(10):
        history.append(Image.new("RGB", (10, 6), (value, value, value)))
    images, metadata = history.sample()
    assert len(images) == len(metadata) == 8
    assert [item.sequence for item in metadata if item is not None] == [
        0,
        1,
        2,
        3,
        5,
        6,
        7,
        9,
    ]


def test_short_history_is_left_padded():
    history = JpegFrameHistory()
    history.append(Image.new("RGB", (10, 6), "white"))
    images, metadata = history.sample()
    assert len(images) == 8
    assert [item is None for item in metadata] == [True] * 7 + [False]


def test_hardware_recorder_uses_existing_review_schema(tmp_path):
    history = JpegFrameHistory()
    for value in range(8):
        history.append(Image.new("RGB", (10, 6), (value, value, value)))
    images, metadata = history.sample()
    recorder = HardwareDecisionRecorder(
        tmp_path / "decision_samples",
        robot_model="a2",
        episode_id="physical-1",
        scene_id="site-a",
    )
    command = VelocityCommand(0.5, 0.0, 0.0, 0.5)
    bounded = SafetyLimits().apply(command)
    record = recorder.record(
        decision=1,
        instruction="Walk toward the chair.",
        images=images,
        history_frames=metadata,
        baseline_output="move forward 25 cm",
        parsed_command=command,
        bounded_command=bounded,
        executed=False,
    )
    assert record["review_status"] == "unreviewed"
    assert len(record["image_files"]) == 8
    manifest = tmp_path / "decision_samples" / "manifest.jsonl"
    assert json.loads(manifest.read_text())["source"] == "unitree_hardware_decision_sample"


@pytest.mark.parametrize(
    ("robot_model", "module_name"),
    [
        ("a2", "unitree_sdk2py.a2.sport.sport_client"),
        ("go2", "unitree_sdk2py.go2.sport.sport_client"),
    ],
)
def test_unitree_controller_selects_robot_module_and_initializes_channel(
    monkeypatch, robot_model, module_name
):
    calls = []

    class FakeSportClient:
        def SetTimeout(self, timeout):
            calls.append(("timeout", timeout))

        def Init(self):
            calls.append(("init",))

        def Move(self, vx, vy, wz):
            calls.append(("move", vx, vy, wz))
            return 0

        def StopMove(self):
            calls.append(("stop",))
            return 0

        def BalanceStand(self):
            return 0

        def Damp(self):
            return 0

    def fake_import(name):
        if name == "unitree_sdk2py.core.channel":
            return SimpleNamespace(
                ChannelFactoryInitialize=lambda domain, interface: calls.append(
                    ("channel", domain, interface)
                )
            )
        assert name == module_name
        return SimpleNamespace(SportClient=FakeSportClient)

    monkeypatch.setattr(
        "navila_orca.hardware.unitree_gateway.import_module", fake_import
    )
    controller = UnitreeSportController(
        robot_model=robot_model, network_interface="br0", timeout_s=3.0
    )
    controller.move(0.1, 0.0, -0.2)
    controller.stop_move()
    assert calls == [
        ("channel", 0, "br0"),
        ("timeout", 3.0),
        ("init",),
        ("move", 0.1, 0.0, -0.2),
        ("stop",),
    ]
