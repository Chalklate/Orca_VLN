#!/usr/bin/env python3
"""Build a checked NaVILA training bundle from Orca_VLN decision manifests.

This command is intentionally conservative. It only packages reviewed target
actions, removes exact duplicate eight-frame histories, and writes a report
before training. It does not infer ground truth from a baseline VLM response
unless ``--accept-baseline`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_COUNT = 8
EXPORTER_PATH = PROJECT_ROOT / "scripts" / "export_navila_lora_dataset.py"


def _load_exporter() -> Any:
    spec = importlib.util.spec_from_file_location("navila_lora_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dataset exporter: {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate, and package Orca_VLN decision samples for NaVILA."
    )
    parser.add_argument(
        "manifests",
        nargs="*",
        type=Path,
        help="decision_samples/manifest.jsonl files; defaults to all training runs",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training_runs",
        help="root searched when manifests are omitted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training_records" / "navila_orca_built",
        help="portable dataset directory to create",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="episode-level validation fraction; requires at least two episodes",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--accept-baseline",
        action="store_true",
        help=(
            "use each unlabelled baseline output as its target; only use after "
            "human verification because this cannot correct baseline mistakes"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the report without copying images",
    )
    return parser


def _manifest_paths(manifests: Iterable[Path], runs_root: Path) -> list[Path]:
    explicit = [path.expanduser().resolve() for path in manifests]
    if explicit:
        paths = explicit
    else:
        paths = sorted((runs_root.expanduser().resolve()).glob("*/decision_samples/manifest.jsonl"))
    if not paths:
        raise ValueError("no decision manifests found")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("manifest does not exist: " + ", ".join(missing))
    return paths


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"manifest line is not an object: {path}:{line_number}")
        record["_source_file"] = str(path)
        record["_source_line"] = line_number
        records.append(record)
    return records


def _resolve_images(record: dict[str, Any], source_file: Path) -> list[Path]:
    image_values = record.get("image_files")
    if not isinstance(image_values, list) or len(image_values) != FRAME_COUNT:
        raise ValueError(
            f"{source_file}:{record['_source_line']} must contain exactly eight image_files"
        )
    paths: list[Path] = []
    for value in image_values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"empty image path in {source_file}:{record['_source_line']}")
        image = (source_file.parent / value).expanduser().resolve()
        if not image.is_file():
            raise ValueError(f"image does not exist: {image}")
        paths.append(image)
    return paths


def _sample_digest(images: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(hashlib.sha256(image.read_bytes()).digest())
    return digest.hexdigest()


def _canonical_action(exporter: Any, value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return exporter._canonical_action(value)
    except ValueError:
        return None


def _report_and_records(manifests: list[Path], *, accept_baseline: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exporter = _load_exporter()
    source_lines = 0
    skipped_unreviewed = 0
    skipped_missing_target = 0
    accepted_baseline = 0
    duplicate_records = 0
    records: list[dict[str, Any]] = []
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    baseline_disagreements: list[dict[str, Any]] = []
    invalid_baselines: list[dict[str, Any]] = []

    for manifest in manifests:
        for record in _read_manifest(manifest):
            source_lines += 1
            status = record.get("review_status")
            target = record.get("target_action")
            if status != "reviewed" or not isinstance(target, str) or not target.strip():
                if accept_baseline and isinstance(record.get("baseline_output"), str):
                    baseline = record["baseline_output"].strip()
                    if _canonical_action(exporter, baseline) is not None:
                        target = baseline
                        accepted_baseline += 1
                    else:
                        if status != "reviewed":
                            skipped_unreviewed += 1
                        else:
                            skipped_missing_target += 1
                        invalid_baselines.append(
                            {"source": f"{manifest}:{record['_source_line']}", "baseline": baseline}
                        )
                        continue
                else:
                    if status != "reviewed":
                        skipped_unreviewed += 1
                    else:
                        skipped_missing_target += 1
                    continue

            images = _resolve_images(record, manifest)
            canonical_target = _canonical_action(exporter, target)
            if canonical_target is None:
                raise ValueError(
                    f"non-canonical target at {manifest}:{record['_source_line']}: {target!r}"
                )
            digest = _sample_digest(images)
            previous = seen.get(digest)
            if previous is not None:
                duplicate_records += 1
                if previous[0] != canonical_target:
                    raise ValueError(
                        "duplicate eight-frame history has conflicting targets: "
                        f"{previous[1]['_source_file']}:{previous[1]['_source_line']} "
                        f"versus {manifest}:{record['_source_line']}"
                    )
                continue

            source_line = int(record["_source_line"])
            record = dict(record)
            record.pop("_source_file", None)
            record.pop("_source_line", None)
            record["review_status"] = "reviewed"
            record["target_action"] = target
            record["image_paths"] = [str(image) for image in images]
            record["_source_file"] = str(manifest)
            record["_source_line"] = source_line
            seen[digest] = (canonical_target, record)
            records.append(record)

            baseline = _canonical_action(exporter, record.get("baseline_output"))
            if baseline is None:
                invalid_baselines.append(
                    {
                        "source": f"{manifest}:{record['_source_line']}",
                        "baseline": record.get("baseline_output"),
                    }
                )
            elif baseline != canonical_target:
                baseline_disagreements.append(
                    {
                        "source": f"{manifest}:{record['_source_line']}",
                        "baseline": baseline,
                        "target": canonical_target,
                    }
                )

    episodes = sorted({str(record.get("episode_id", "")) for record in records})
    scenes = sorted({str(record.get("scene_id", "")) for record in records})
    labels = Counter(_canonical_action(exporter, record.get("target_action")) for record in records)
    report = {
        "manifests": [str(path) for path in manifests],
        "source_manifest_lines": source_lines,
        "num_unique_records": len(records),
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_missing_target": skipped_missing_target,
        "accepted_baseline_targets": accepted_baseline,
        "duplicate_exact_frame_histories": duplicate_records,
        "episodes": episodes,
        "scenes": scenes,
        "label_distribution": dict(sorted(labels.items())),
        "baseline_target_disagreements": baseline_disagreements,
        "invalid_or_missing_baselines": invalid_baselines,
        "warnings": [],
    }
    if len(episodes) < 2:
        report["warnings"].append(
            "all usable records come from one episode; this is not a generalization set"
        )
    if len(records) < 32:
        report["warnings"].append(
            f"only {len(records)} unique records; collect more viewpoints and episodes before training"
        )
    if baseline_disagreements:
        report["warnings"].append(
            f"{len(baseline_disagreements)} unique records disagree with the baseline; verify each target"
        )
    return report, records


def _write_aggregate(records: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _restore_provenance(output_dir: Path, records: list[dict[str, Any]], exporter: Any) -> None:
    provenance: dict[str, dict[str, Any]] = {}
    for record in records:
        images = [Path(value) for value in record["image_paths"]]
        key = exporter._decision_key(record, images)
        provenance[key] = {
            "source_file": str(record["_source_file"]),
            "source_line": int(record["_source_line"]),
        }
    for annotation_name in ("train.jsonl", "validation.jsonl"):
        annotation = output_dir / annotation_name
        if not annotation.is_file():
            continue
        rewritten: list[str] = []
        for line in annotation.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            metadata = payload.setdefault("metadata", {})
            metadata.update(provenance.get(payload.get("id"), {}))
            rewritten.append(json.dumps(payload, ensure_ascii=False))
        annotation.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifests = _manifest_paths(args.manifests, args.runs_root)
    report, records = _report_and_records(
        manifests, accept_baseline=bool(args.accept_baseline)
    )
    if args.validation_fraction > 0 and len(report["episodes"]) < 2:
        raise ValueError(
            "--validation-fraction requires at least two distinct episodes; "
            "collect another start/layout instead of validating on the same episode"
        )

    output_dir = args.output_dir.expanduser().resolve()
    report["output_dir"] = str(output_dir)
    if args.dry_run:
        return report

    with tempfile.TemporaryDirectory(prefix="navila-orca-build-") as temp_dir:
        aggregate = Path(temp_dir) / "reviewed.jsonl"
        _write_aggregate(records, aggregate)
        exporter = _load_exporter()
        package_info = exporter.package_dataset(
            [aggregate],
            output_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    _restore_provenance(output_dir, records, exporter)
    report["package"] = package_info
    (output_dir / "collection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = _parser().parse_args()
    report = build(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
