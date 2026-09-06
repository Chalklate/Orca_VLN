from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _exporter_module():
    script = Path(__file__).resolve().parents[1] / "scripts/export_vln_sft_records.py"
    spec = importlib.util.spec_from_file_location("export_vln_sft_records", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_record_marks_unreviewed_rollout(tmp_path):
    run_dir = tmp_path / "run"
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True)
    (run_dir / "measurements.json").write_text(
        json.dumps(
            {
                "episode": {
                    "episode_id": "warehouse-01",
                    "scene_id": "warehouse",
                    "instruction": "Move to the blue barrel.",
                },
                "vlm_outputs": ["move forward 25 cm", "stop"],
                "runtime": {
                    "frames_directory": str(frames_dir),
                    "frame_files": ["000.jpg"],
                },
            }
        ),
        encoding="utf-8",
    )

    record = _exporter_module().export_record(run_dir)

    assert record["review_status"] == "unreviewed"
    assert record["target_action"] is None
    assert record["baseline_actions"] == ["move forward 25 cm", "stop"]
    assert record["image_paths"] == [str((frames_dir / "000.jpg").resolve())]


def test_export_record_accepts_reviewed_label(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "measurements.json").write_text(
        json.dumps(
            {
                "episode": {"instruction": "Turn left."},
                "vlm_outputs": ["turn left 15 degrees"],
                "runtime": {"frames_directory": str(run_dir), "frame_files": []},
            }
        ),
        encoding="utf-8",
    )

    record = _exporter_module().export_record(run_dir, label=" stop ")

    assert record["review_status"] == "reviewed"
    assert record["target_action"] == "stop"


def test_export_decision_records_only_reviewed_samples(tmp_path):
    sample_root = tmp_path / "decision_samples"
    sample_dir = sample_root / "decision_0001"
    sample_dir.mkdir(parents=True)
    images = [f"decision_0001/frame_{index:03d}.jpg" for index in range(8)]
    (sample_root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "record_version": 1,
                "review_status": "reviewed",
                "episode_id": "episode-1",
                "scene_id": "scene-1",
                "decision": 1,
                "instruction": "Move to the table.",
                "image_files": images,
                "frame_step_ids": list(range(8)),
                "baseline_output": "move forward 25 cm",
                "baseline_command": {},
                "target_action": "turn left 15 degrees",
                "reviewer": "human",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = _exporter_module().export_decision_records(tmp_path)

    assert len(records) == 1
    assert records[0]["target_action"] == "turn left 15 degrees"
    assert len(records[0]["image_paths"]) == 8
