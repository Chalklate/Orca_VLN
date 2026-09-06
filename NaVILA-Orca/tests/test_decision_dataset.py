import json

import numpy as np
from PIL import Image

from navila_orca.contracts import RobotState, VelocityCommand
from navila_orca.decision_dataset import DecisionDatasetRecorder


def _state() -> RobotState:
    zeros3 = np.zeros(3)
    zeros12 = np.zeros(12)
    return RobotState(
        step_id=7,
        sim_time_s=0.14,
        root_pos_world=zeros3,
        root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        body_ang_vel=zeros3,
        base_rpy=zeros3,
        joint_pos=zeros12,
        joint_vel=zeros12,
        last_raw_action=zeros12,
    )


def test_recorder_writes_eight_images_metadata_and_manifest(tmp_path):
    recorder = DecisionDatasetRecorder(
        tmp_path / "decision_samples",
        episode_id="episode-1",
        scene_id="scene-1",
    )
    record = recorder.record(
        decision=1,
        instruction="Move to the table.",
        images=[Image.new("RGB", (8, 6), index) for index in range(8)],
        frame_step_ids=list(range(8)),
        state=_state(),
        baseline_output="move forward 25 cm",
        command=VelocityCommand(0.5, 0.0, 0.0, 0.5),
    )

    sample_dir = tmp_path / "decision_samples" / "decision_0001"
    assert record["target_action"] is None
    assert len(list(sample_dir.glob("frame_*.jpg"))) == 8
    assert json.loads((sample_dir / "metadata.json").read_text())["decision"] == 1
    assert len(
        (tmp_path / "decision_samples" / "manifest.jsonl")
        .read_text()
        .splitlines()
    ) == 1
