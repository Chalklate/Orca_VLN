#!/usr/bin/env python3
"""Package reviewed Orca_VLN decision records for NaVILA LoRA training.

The Orca_VLN exporter deliberately produces a model-agnostic JSONL file. This
script is the release-specific finalization step: it copies the referenced
frames into a portable dataset directory and writes one NaVILA-style
``images`` + ``conversations`` record per reviewed decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


FRAME_COUNT = 8
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_STOP_RE = re.compile(r"\bstop\b", re.IGNORECASE)
_MOVE_RE = re.compile(
    r"\bmove\s+forward(?:\s+by)?\s+(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>cm|centimet(?:er|re)s?)\b",
    re.IGNORECASE,
)
_TURN_RE = re.compile(
    r"\bturn\s+(?P<direction>left|right)(?:\s+by)?\s+"
    r"(?P<amount>\d+(?:\.\d+)?)\s*(?:-|\s)?degrees?\b",
    re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package reviewed Orca_VLN decision JSONL and its eight-frame "
            "samples as a portable NaVILA LoRA dataset."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "one or more model-agnostic JSONL files, or reviewed "
            "decision_samples/manifest.jsonl files"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory containing train.jsonl and copied images",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help=(
            "hold out complete episodes for validation; default 0 keeps all "
            "reviewed records in train.jsonl"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="deterministic seed used when selecting validation episodes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty output directory",
    )
    return parser


def _safe_name(value: str, fallback: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip()).strip("._-")
    return cleaned[:80] or fallback


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"input JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"record in {path}:{line_number} is not an object")
        record["_source_file"] = str(path)
        record["_source_line"] = line_number
        records.append(record)
    return records


def _resolve_image(path_value: Any, source_file: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("each image_paths entry must be a non-empty string")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = source_file.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    return path


def _canonical_action(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reviewed record has no target_action")
    matches: list[tuple[str, re.Match[str]]] = []
    matches.extend(("stop", match) for match in _STOP_RE.finditer(value))
    matches.extend(("move", match) for match in _MOVE_RE.finditer(value))
    matches.extend(("turn", match) for match in _TURN_RE.finditer(value))
    matches.sort(key=lambda item: item[1].start())
    if not matches:
        raise ValueError(f"target_action is not canonical: {value!r}")
    if len(matches) != 1:
        raise ValueError(f"target_action contains multiple actions: {value!r}")

    kind, match = matches[0]
    if kind == "stop":
        return "The next action is stop."

    if kind == "move":
        amount = float(match.group("amount"))
        if amount not in {25.0, 50.0, 75.0}:
            raise ValueError(f"unsupported forward amount in target_action: {value!r}")
        return f"The next action is move forward {int(amount)} cm."

    if kind == "turn":
        amount = float(match.group("amount"))
        if amount not in {15.0, 30.0, 45.0}:
            raise ValueError(f"unsupported turn amount in target_action: {value!r}")
        direction = match.group("direction").lower()
        return f"The next action is turn {direction} {int(amount)} degree."

    # The match list above is exhaustive, but keep this guard in case the
    # vocabulary is extended without updating the canonical formatter.
    raise ValueError(f"unsupported target_action: {value!r}")


def _decision_key(record: dict[str, Any], image_paths: Iterable[Path]) -> str:
    episode = str(record.get("episode_id", "episode"))
    scene = str(record.get("scene_id", "scene"))
    decision = int(record.get("decision", 0))
    digest_input = "\0".join(str(path) for path in image_paths)
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
    return (
        f"{_safe_name(scene, 'scene')}_"
        f"{_safe_name(episode, 'episode')}_d{decision:04d}_{digest}"
    )


def _validate_record(record: dict[str, Any], source_file: Path) -> tuple[list[Path], str]:
    if record.get("review_status") != "reviewed":
        raise ValueError(
            f"{source_file}:{record.get('_source_line')} is not reviewed; "
            "only human-reviewed records may be sent to training"
        )
    instruction = str(record.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"{source_file}:{record.get('_source_line')} has no instruction")
    image_values = record.get("image_paths")
    if image_values is None:
        image_values = record.get("image_files")
    if not isinstance(image_values, list) or len(image_values) != FRAME_COUNT:
        raise ValueError(
            f"{source_file}:{record.get('_source_line')} must contain exactly "
            f"{FRAME_COUNT} image_paths/image_files"
        )
    image_paths = [_resolve_image(value, source_file) for value in image_values]
    return image_paths, _canonical_action(record.get("target_action"))


def _navila_prompt(instruction: str) -> str:
    image_tokens = "<image>\n" * (FRAME_COUNT - 1)
    return (
        "Imagine you are a robot programmed for navigation tasks. You have "
        f"been given historical observations {image_tokens}and the current "
        f'observation <image>. Your assigned task is: "{instruction.strip()}". '
        "Analyze this series of images to decide the next action: turn left "
        "or right by a specific degree, move forward a specific distance, or "
        "stop when the task is complete."
    )


def _copy_sample_images(
    image_paths: list[Path], *, dataset_root: Path, sample_id: str
) -> list[str]:
    sample_dir = dataset_root / "images" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)
    relative_paths: list[str] = []
    for index, source in enumerate(image_paths):
        suffix = source.suffix.lower() or ".jpg"
        destination = sample_dir / f"frame_{index:03d}{suffix}"
        shutil.copy2(source, destination)
        relative_paths.append(destination.relative_to(dataset_root).as_posix())
    return relative_paths


def _split_records(
    records: list[dict[str, Any]], *, fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if fraction <= 0.0:
        return records, []
    if fraction >= 1.0:
        raise ValueError("--validation-fraction must be less than 1")

    groups = sorted({str(record["metadata"]["episode_id"]) for record in records})
    if len(groups) < 2:
        return records, []

    import random

    randomizer = random.Random(seed)
    randomizer.shuffle(groups)
    validation_count = max(1, int(round(len(groups) * fraction)))
    validation_groups = set(groups[:validation_count])
    train = [
        record
        for record in records
        if record["metadata"]["episode_id"] not in validation_groups
    ]
    validation = [
        record
        for record in records
        if record["metadata"]["episode_id"] in validation_groups
    ]
    return train, validation


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def package_dataset(
    inputs: Iterable[Path],
    output_dir: Path,
    *,
    validation_fraction: float = 0.0,
    seed: int = 17,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a portable NaVILA dataset bundle from reviewed JSONL files."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(
                f"output directory is not empty: {output_dir}; use --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records: list[dict[str, Any]] = []
    for input_path in inputs:
        source_records.extend(_read_jsonl(input_path))

    converted: list[dict[str, Any]] = []
    seen_targets: dict[str, str] = {}
    skipped_unreviewed = 0
    skipped_missing_target = 0
    for record in source_records:
        source_file = Path(str(record.pop("_source_file")))
        source_line = record.pop("_source_line")
        if record.get("review_status") != "reviewed":
            skipped_unreviewed += 1
            continue
        # Older reviewer output could mark a record reviewed while leaving
        # target_action null. Treat that inconsistent state as excluded
        # training data rather than guessing that the baseline was right.
        if record.get("target_action") is None:
            skipped_missing_target += 1
            print(
                "warning: skipping reviewed record with no target_action: "
                f"{source_file}:{source_line} decision={record.get('decision')}"
            )
            continue
        image_paths, target = _validate_record(record, source_file)
        key = _decision_key(record, image_paths)
        previous_target = seen_targets.get(key)
        if previous_target is not None:
            if previous_target != target:
                raise ValueError(
                    f"duplicate decision {key} has conflicting reviewed actions: "
                    f"{previous_target!r} versus {target!r}"
                )
            continue
        seen_targets[key] = target
        relative_images = _copy_sample_images(
            image_paths, dataset_root=output_dir, sample_id=key
        )
        instruction = str(record["instruction"]).strip()
        converted.append(
            {
                "id": key,
                "images": relative_images,
                "conversations": [
                    {"from": "human", "value": _navila_prompt(instruction)},
                    {"from": "gpt", "value": target},
                ],
                "metadata": {
                    "scene_id": str(record.get("scene_id", "")),
                    "episode_id": str(record.get("episode_id", "")),
                    "decision": int(record.get("decision", 0)),
                    "source_file": str(source_file),
                    "source_line": source_line,
                    "baseline_output": str(record.get("baseline_output", "")),
                },
            }
        )

    if not converted:
        raise ValueError("no reviewed records were found in the input JSONL files")

    train, validation = _split_records(
        converted, fraction=validation_fraction, seed=seed
    )
    _write_jsonl(output_dir / "train.jsonl", train)
    if validation:
        _write_jsonl(output_dir / "validation.jsonl", validation)

    info = {
        "format": "navila_llava_conversations",
        "num_frames": FRAME_COUNT,
        "num_records": len(converted),
        "num_train_records": len(train),
        "num_validation_records": len(validation),
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_missing_target": skipped_missing_target,
        "deduplicated_records": (
            len(source_records)
            - skipped_unreviewed
            - skipped_missing_target
            - len(converted)
        ),
        "image_root": str(output_dir),
        "train_annotations": "train.jsonl",
        "validation_annotations": "validation.jsonl" if validation else None,
        "prompt_contract": (
            "the current Orca_VLN navila_vlm_server navigation prompt with "
            "eight <image> tokens and the exact instruction"
        ),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return info


def main() -> int:
    args = _parser().parse_args()
    info = package_dataset(
        args.inputs,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
