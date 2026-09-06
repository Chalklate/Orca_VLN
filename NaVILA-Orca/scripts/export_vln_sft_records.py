#!/usr/bin/env python3
"""Export one reviewable high-level VLN record from a saved baseline rollout.

The emitted JSONL is intentionally model-agnostic.  Review and relabel it
before adapting it to NaVILA's SFT/LoRA training format; it is not a claim that
raw baseline actions are ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export either a legacy single-run VLN record or reviewed "
            "per-decision samples from an Orca_VLN run."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "run directory containing measurements.json; with "
            "--decision-samples, it must contain decision_samples/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSONL file to append with model-agnostic records",
    )
    parser.add_argument(
        "--label",
        help=(
            "legacy single-run label; for --decision-samples, label first with "
            "scripts/label_decision_samples.py"
        ),
    )
    parser.add_argument(
        "--decision-samples",
        action="store_true",
        help="export the reviewed per-decision manifest produced by --collect-decisions",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="include unreviewed decision samples when using --decision-samples",
    )
    return parser


def _load_measurements(run_dir: Path) -> dict[str, Any]:
    path = run_dir.expanduser().resolve() / "measurements.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def export_record(
    run_dir: Path, *, label: str | None = None
) -> dict[str, Any]:
    """Create a portable high-level record from an Orca_VLN result directory."""

    payload = _load_measurements(run_dir)
    episode = payload.get("episode")
    runtime = payload.get("runtime")
    actions = payload.get("vlm_outputs")
    if not isinstance(episode, dict) or not isinstance(runtime, dict):
        raise ValueError("measurements.json is missing episode or runtime metadata")
    if not isinstance(actions, list) or not all(isinstance(item, str) for item in actions):
        raise ValueError("measurements.json is missing textual vlm_outputs")
    instruction = str(episode.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("episode instruction is empty")

    frames_dir = Path(str(runtime.get("frames_directory", ""))).expanduser()
    frame_files = runtime.get("frame_files", [])
    if not isinstance(frame_files, list) or not all(
        isinstance(item, str) for item in frame_files
    ):
        raise ValueError("runtime frame_files must be a list of strings")
    image_paths = [str((frames_dir / name).resolve()) for name in frame_files]
    return {
        "record_version": 1,
        "source": "orca_vln_rollout",
        "review_status": "reviewed" if label else "unreviewed",
        "episode_id": str(episode.get("episode_id", "")),
        "scene_id": str(episode.get("scene_id", "")),
        "instruction": instruction,
        "image_paths": image_paths,
        "baseline_actions": actions,
        "target_action": label.strip() if label else None,
        "note": "Review image/action alignment before converting this record to a NaVILA SFT or LoRA dataset.",
    }


def export_decision_records(
    run_dir: Path, *, include_unreviewed: bool = False
) -> list[dict[str, Any]]:
    """Export reviewed per-decision records from a collection run."""

    run_dir = run_dir.expanduser().resolve()
    sample_root = run_dir / "decision_samples"
    manifest_path = sample_root / "manifest.jsonl"
    if not manifest_path.is_file() and run_dir.name == "decision_samples":
        sample_root = run_dir
        manifest_path = sample_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"decision sample manifest not found under {run_dir}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid decision manifest line {line_number}: {exc}") from exc
        if not isinstance(source, dict):
            raise ValueError(f"decision manifest line {line_number} is not an object")
        reviewed = source.get("review_status") == "reviewed"
        if not include_unreviewed and not reviewed:
            continue
        image_files = source.get("image_files")
        if not isinstance(image_files, list) or len(image_files) != 8:
            raise ValueError(
                f"decision {source.get('decision')} must contain exactly eight image_files"
            )
        image_paths = [str((sample_root / str(path)).resolve()) for path in image_files]
        records.append(
            {
                "record_version": 2,
                "source": "orca_vln_decision_sample",
                "review_status": source.get("review_status", "unreviewed"),
                "episode_id": str(source.get("episode_id", "")),
                "scene_id": str(source.get("scene_id", "")),
                "decision": int(source.get("decision", 0)),
                "instruction": str(source.get("instruction", "")),
                "image_paths": image_paths,
                "frame_step_ids": source.get("frame_step_ids", []),
                "baseline_output": str(source.get("baseline_output", "")),
                "baseline_command": source.get("baseline_command"),
                "target_action": source.get("target_action"),
                "reviewer": source.get("reviewer"),
                "note": (
                    "Model-agnostic reviewed record. Convert this to the exact "
                    "NaVILA release training format before SFT/LoRA."
                ),
            }
        )
    return records


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.decision_samples:
        if args.label is not None:
            raise ValueError("--label is only valid for legacy single-run export")
        records = export_decision_records(
            args.run_dir, include_unreviewed=args.include_unreviewed
        )
        with output.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} decision records to {output}")
    else:
        record = export_record(args.run_dir, label=args.label)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
