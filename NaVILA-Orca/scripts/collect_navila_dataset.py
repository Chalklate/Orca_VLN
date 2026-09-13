#!/usr/bin/env python3
"""Run a batch of OrcaLab decision-collection jobs.

The job file describes instructions and output names; OrcaLab, its current
layout, and the NaVILA server must already be running. This automates the
repeatable launch/recording work but deliberately does not invent labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_orcalab_scene_locomotion.sh"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect exact eight-frame NaVILA decisions for many jobs."
    )
    parser.add_argument(
        "job_file",
        type=Path,
        help="JSON array/object or JSONL file describing collection jobs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="new directory containing one run directory per job",
    )
    parser.add_argument("--vlm-host", default="127.0.0.1")
    parser.add_argument("--vlm-port", type=int, default=54321)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run later jobs after one job fails",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"job file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"job file is empty: {path}")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(decoded, dict):
        decoded = decoded.get("jobs")
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("job file must contain a non-empty JSON array or {\"jobs\": [...]}")
    jobs: list[dict[str, Any]] = []
    for index, job in enumerate(decoded, 1):
        if not isinstance(job, dict):
            raise ValueError(f"job {index} is not an object")
        name = str(job.get("name", "")).strip()
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"job {index} needs a safe one-component name")
        if "instruction" in job and "instruction_file" in job:
            raise ValueError(f"job {name} cannot contain both instruction and instruction_file")
        max_decisions = int(job.get("max_decisions", 8))
        if max_decisions <= 0:
            raise ValueError(f"job {name} max_decisions must be positive")
        jobs.append({**job, "name": name, "max_decisions": max_decisions})
    return jobs


def _path(value: Any, *, base: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def _command(job: dict[str, Any], output: Path, *, vlm_host: str, vlm_port: int) -> list[str]:
    command = [
        str(RUNNER),
        "--output",
        str(output),
        "--collect-decisions",
        "--max-decisions",
        str(job["max_decisions"]),
        "--image-interval",
        str(float(job.get("image_interval", 0.2))),
        "--vlm-host",
        vlm_host,
        "--vlm-port",
        str(vlm_port),
        "--no-live-monitor",
    ]
    if job.get("scenario") is not None:
        command.extend(["--scenario", _path(job["scenario"], base=PROJECT_ROOT)])
    if job.get("instruction") is not None:
        command.extend(["--instruction", str(job["instruction"])])
    elif job.get("instruction_file") is not None:
        command.extend(["--instruction-file", _path(job["instruction_file"], base=PROJECT_ROOT)])
    elif job.get("waypoint_instruction_file") is not None:
        command.extend(
            [
                "--waypoint-instruction-file",
                _path(job["waypoint_instruction_file"], base=PROJECT_ROOT),
            ]
        )
    return command


def main() -> int:
    args = _parser().parse_args()
    jobs = _load_jobs(args.job_file)
    if args.output_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = PROJECT_ROOT / "outputs" / "training_runs" / f"batch-{stamp}"
    else:
        output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for job in jobs:
        output = output_root / job["name"]
        command = _command(job, output, vlm_host=args.vlm_host, vlm_port=args.vlm_port)
        print("COLLECT", job["name"], shlex.join(command), flush=True)
        result: dict[str, Any] = {"name": job["name"], "output": str(output), "command": command}
        if args.dry_run:
            result["status"] = "dry_run"
            results.append(result)
            continue
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        result["returncode"] = completed.returncode
        manifest = output / "decision_samples" / "manifest.jsonl"
        result["status"] = "ok" if completed.returncode == 0 and manifest.is_file() else "failed"
        result["manifest"] = str(manifest)
        results.append(result)
        if result["status"] != "ok" and not args.continue_on_error:
            break

    index = {
        "job_file": str(args.job_file.expanduser().resolve()),
        "output_root": str(output_root),
        "results": results,
        "note": (
            "Decision samples remain unreviewed. The current runner reuses the live authored "
            "layout and does not apply scenario start_position transforms."
        ),
    }
    (output_root / "collection_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0 if all(result["status"] in {"ok", "dry_run"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
