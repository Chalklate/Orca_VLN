from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_navila_lora_dataset.py"
SPEC = importlib.util.spec_from_file_location("export_navila_lora_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_reviewed_jsonl(tmp_path: Path, *, reviewed: bool = True) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    image_paths: list[str] = []
    for index in range(8):
        path = source_dir / f"frame_{index:03d}.png"
        Image.new("RGB", (32, 24), (index, 10, 20)).save(path)
        image_paths.append(str(path))
    record = {
        "record_version": 2,
        "review_status": "reviewed" if reviewed else "unreviewed",
        "episode_id": "episode-a",
        "scene_id": "scene-a",
        "decision": 2,
        "instruction": "Go to the dining table and inspect it.",
        "image_paths": image_paths,
        "baseline_output": "The next action is move forward 75 cm.",
        "target_action": "turn right 45 degrees" if reviewed else None,
    }
    jsonl = tmp_path / "reviewed.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return jsonl


def test_packages_eight_images_and_canonical_action(tmp_path: Path) -> None:
    input_path = _write_reviewed_jsonl(tmp_path)
    output_dir = tmp_path / "navila"

    info = MODULE.package_dataset([input_path], output_dir)

    assert info["num_records"] == 1
    record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    assert len(record["images"]) == 8
    assert record["conversations"][0]["value"].count("<image>") == 8
    assert "Imagine you are a robot programmed for navigation tasks." in record[
        "conversations"
    ][0]["value"]
    assert 'Your assigned task is: "Go to the dining table and inspect it."' in record[
        "conversations"
    ][0]["value"]
    assert record["conversations"][1]["value"] == "The next action is turn right 45 degree."
    assert all((output_dir / path).is_file() for path in record["images"])
    assert all(not Path(path).is_absolute() for path in record["images"])


def test_unreviewed_records_are_not_training_data(tmp_path: Path) -> None:
    input_path = _write_reviewed_jsonl(tmp_path, reviewed=False)
    with pytest.raises(ValueError, match="no reviewed records"):
        MODULE.package_dataset([input_path], tmp_path / "navila")


def test_reviewed_record_without_target_is_skipped(tmp_path: Path) -> None:
    input_path = _write_reviewed_jsonl(tmp_path, reviewed=False)
    record = json.loads(input_path.read_text(encoding="utf-8"))
    record["review_status"] = "reviewed"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no reviewed records"):
        MODULE.package_dataset([input_path], tmp_path / "navila")


def test_accepts_reviewed_decision_manifest_image_files(tmp_path: Path) -> None:
    input_path = _write_reviewed_jsonl(tmp_path)
    source_record = json.loads(input_path.read_text(encoding="utf-8"))
    manifest_dir = tmp_path / "run" / "decision_samples"
    sample_dir = manifest_dir / "decision_0002"
    sample_dir.mkdir(parents=True)
    manifest_paths: list[str] = []
    for index, absolute_path in enumerate(source_record["image_paths"]):
        destination = sample_dir / f"frame_{index:03d}.png"
        destination.write_bytes(Path(absolute_path).read_bytes())
        manifest_paths.append(f"decision_0002/frame_{index:03d}.png")
    source_record.pop("image_paths")
    source_record["image_files"] = manifest_paths
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text(json.dumps(source_record) + "\n", encoding="utf-8")

    info = MODULE.package_dataset([manifest], tmp_path / "navila")

    assert info["num_records"] == 1


def test_nonempty_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    input_path = _write_reviewed_jsonl(tmp_path)
    output_dir = tmp_path / "navila"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        MODULE.package_dataset([input_path], output_dir)
