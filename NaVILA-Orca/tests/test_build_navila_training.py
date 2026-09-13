from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_navila_training.py"
SPEC = importlib.util.spec_from_file_location("build_navila_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_manifest(tmp_path: Path, *, target: str = "turn right 45 degrees") -> Path:
    sample_root = tmp_path / "run" / "decision_samples"
    sample = sample_root / "decision_0001"
    sample.mkdir(parents=True)
    image_files = []
    for index in range(8):
        image = sample / f"frame_{index:03d}.png"
        Image.new("RGB", (16, 16), (index, 1, 2)).save(image)
        image_files.append(f"decision_0001/{image.name}")
    record = {
        "review_status": "reviewed",
        "episode_id": "episode-a",
        "scene_id": "scene-a",
        "decision": 1,
        "instruction": "Go to the table.",
        "image_files": image_files,
        "baseline_output": "The next action is move forward 75 cm.",
        "target_action": target,
    }
    manifest = sample_root / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


def test_report_deduplicates_identical_histories(tmp_path: Path) -> None:
    first = _write_manifest(tmp_path / "first")
    second = _write_manifest(tmp_path / "second")
    report, records = MODULE._report_and_records([first, second], accept_baseline=False)

    assert len(records) == 1
    assert report["duplicate_exact_frame_histories"] == 1
    assert report["baseline_target_disagreements"]


def test_baseline_is_not_used_without_explicit_opt_in(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["review_status"] = "unreviewed"
    payload["target_action"] = None
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report, records = MODULE._report_and_records([manifest], accept_baseline=False)

    assert records == []
    assert report["skipped_unreviewed"] == 1
