"""Reviewable per-decision dataset artifacts for NaVILA collection runs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .contracts import RobotState, VelocityCommand


class DecisionDatasetRecorder:
    """Persist the exact eight images and metadata for every VLM decision.

    The recorder intentionally leaves ``target_action`` empty.  A human
    reviewer fills that field later using the canonical action vocabulary.
    Paths in the manifest are relative to ``output_root`` so the whole
    ``decision_samples`` directory can be copied to a training machine.
    """

    RECORD_VERSION = 1

    def __init__(
        self,
        output_root: Path,
        *,
        episode_id: str,
        scene_id: str,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_root / "manifest.jsonl"
        self.episode_id = str(episode_id)
        self.scene_id = str(scene_id)
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        decision: int,
        instruction: str,
        images: Sequence[Image.Image],
        frame_step_ids: Sequence[int | None],
        state: RobotState,
        baseline_output: str,
        command: VelocityCommand,
    ) -> dict[str, Any]:
        """Save one exact model input and append its unreviewed record."""

        if len(images) != 8:
            raise ValueError(f"a decision sample requires exactly 8 images, got {len(images)}")
        if len(frame_step_ids) != len(images):
            raise ValueError("frame_step_ids must match the image count")
        if decision <= 0:
            raise ValueError("decision numbers are one-based")

        sample_name = f"decision_{decision:04d}"
        sample_dir = self.output_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_files: list[str] = []
        for index, image in enumerate(images):
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image), mode="RGB")
            path = sample_dir / f"frame_{index:03d}.jpg"
            image.convert("RGB").save(path, format="JPEG", quality=95)
            image_files.append(str(path.relative_to(self.output_root)))

        record = {
            "record_version": self.RECORD_VERSION,
            "source": "orca_vln_decision_sample",
            "review_status": "unreviewed",
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "decision": int(decision),
            "instruction": str(instruction).strip(),
            "image_files": image_files,
            "frame_step_ids": [
                None if step_id is None else int(step_id)
                for step_id in frame_step_ids
            ],
            "decision_state": _state_dict(state),
            "baseline_output": str(baseline_output),
            "baseline_command": asdict(command),
            "target_action": None,
            "reviewer": None,
        }
        metadata_path = sample_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with self.manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        return record


def _state_dict(state: RobotState) -> dict[str, Any]:
    return {
        "step_id": int(state.step_id),
        "sim_time_s": float(state.sim_time_s),
        "root_pos_world": state.root_pos_world.tolist(),
        "root_quat_wxyz": state.root_quat_wxyz.tolist(),
        "base_rpy": state.base_rpy.tolist(),
    }
