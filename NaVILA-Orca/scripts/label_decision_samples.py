#!/usr/bin/env python3
"""Review and label per-decision NaVILA collection samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navila_orca.actions import parse_velocity_command  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label exact eight-frame NaVILA decision samples."
    )
    parser.add_argument(
        "run_or_samples_dir",
        type=Path,
        help="run directory or its decision_samples directory",
    )
    parser.add_argument("--decision", type=int, help="one-based decision number")
    parser.add_argument("--label", help="correct canonical action for --decision")
    parser.add_argument(
        "--unreview",
        action="store_true",
        help="clear the target label and return --decision to unreviewed status",
    )
    parser.add_argument(
        "--reviewer",
        default="human",
        help="reviewer name stored in the sample metadata",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="prompt for a target action for every unreviewed decision",
    )
    parser.add_argument(
        "--accept-baseline",
        action="store_true",
        help=(
            "mark every unlabelled decision as reviewed using its existing "
            "baseline_output; use only after human verification"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list decisions and current labels without changing files",
    )
    return parser


def _sample_root(value: Path) -> Path:
    path = value.expanduser().resolve()
    if (path / "manifest.jsonl").is_file():
        return path
    root = path / "decision_samples"
    if not (root / "manifest.jsonl").is_file():
        raise ValueError(f"decision sample manifest not found under {path}")
    return root


def _load_records(sample_root: Path) -> list[dict[str, Any]]:
    manifest = sample_root / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        records.append(record)
    return records


def _validate_label(label: str) -> str:
    label = label.strip()
    if not label:
        raise ValueError("label must not be empty")
    parse_velocity_command(label)
    return label


def _apply_label(record: dict[str, Any], label: str, reviewer: str) -> None:
    record["target_action"] = _validate_label(label)
    record["review_status"] = "reviewed"
    record["reviewer"] = reviewer


def _clear_label(record: dict[str, Any]) -> None:
    record["target_action"] = None
    record["review_status"] = "unreviewed"
    record["reviewer"] = None


def _write_records(sample_root: Path, records: list[dict[str, Any]]) -> None:
    manifest = sample_root / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(manifest)
    for record in records:
        decision = int(record["decision"])
        metadata = sample_root / f"decision_{decision:04d}" / "metadata.json"
        metadata.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _describe(record: dict[str, Any]) -> str:
    return (
        f"decision={record.get('decision')} "
        f"status={record.get('review_status', 'unreviewed')} "
        f"baseline={record.get('baseline_output', '')!r} "
        f"target={record.get('target_action')!r} "
        f"frames={record.get('image_files', [])}"
    )


def main() -> int:
    args = _parser().parse_args()
    sample_root = _sample_root(args.run_or_samples_dir)
    records = _load_records(sample_root)
    if args.list:
        for record in records:
            print(_describe(record))
        return 0

    if args.label is not None and args.decision is None:
        raise ValueError("--label requires --decision")
    if args.unreview and args.decision is None:
        raise ValueError("--unreview requires --decision")
    if args.unreview and args.label is not None:
        raise ValueError("--unreview cannot be combined with --label")
    if args.unreview and args.interactive:
        raise ValueError("--unreview cannot be combined with --interactive")
    if args.accept_baseline and (
        args.decision is not None
        or args.label is not None
        or args.unreview
        or args.interactive
    ):
        raise ValueError(
            "--accept-baseline cannot be combined with --decision, --label, "
            "--unreview, or --interactive"
        )
    if args.interactive and (args.label is not None or args.decision is not None):
        raise ValueError("--interactive cannot be combined with --decision or --label")
    if not args.interactive and not args.accept_baseline and args.decision is None:
        raise ValueError("provide --decision/--label or use --interactive")

    by_decision = {int(record["decision"]): record for record in records}
    changed = False
    if args.decision is not None:
        record = by_decision.get(args.decision)
        if record is None:
            raise ValueError(f"decision {args.decision} is not in {sample_root}")
        if args.unreview:
            _clear_label(record)
        elif args.label is None:
            raise ValueError("--decision requires --label")
        else:
            _apply_label(record, args.label, args.reviewer)
        print(_describe(record))
        changed = True
    else:
        if args.accept_baseline:
            accepted_count = 0
            for record in records:
                if record.get("target_action") is not None:
                    continue
                baseline = str(record.get("baseline_output", "")).strip()
                if not baseline:
                    raise ValueError(
                        f"decision {record.get('decision')} has no baseline_output"
                    )
                _apply_label(record, baseline, args.reviewer)
                changed = True
                accepted_count += 1
            print(f"accepted baseline actions for {accepted_count} records")
        else:
            for record in records:
                if record.get("review_status") == "reviewed":
                    continue
                print(_describe(record))
                label = input("target action (blank=skip, q=quit): ").strip()
                if label.lower() == "q":
                    break
                if not label:
                    continue
                _apply_label(record, label, args.reviewer)
                changed = True

    if changed:
        _write_records(sample_root, records)
        print(f"updated {sample_root / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
