"""Build and query a lightweight semantic map from teleop anchors.

This is intentionally a topological/pose-tagged map, not SLAM.  Teleoperation
already records the information we need for a repeatable search routine:
semantic labels, world poses, and one or more camera views at each location.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any


VIEW_ORDER = ("front", "left", "rear", "right")
_VIEW_ALIASES = {"back": "rear"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LOCATION_STOPWORDS = {
    "a",
    "accessible",
    "and",
    "area",
    "at",
    "beside",
    "clear",
    "floor",
    "for",
    "go",
    "in",
    "nearby",
    "of",
    "on",
    "room",
    "side",
    "surfaces",
    "the",
    "to",
}


def _normalise(value: str) -> str:
    return "_".join(_TOKEN_RE.findall(str(value).lower()))


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(value).lower())
        if token not in _LOCATION_STOPWORDS
    }


def split_anchor_label(label: str) -> tuple[str, str | None]:
    """Return ``(location_id, view)`` for labels such as ``table_left``."""

    normalised = _normalise(label)
    for suffix in (*VIEW_ORDER, "back"):
        marker = f"_{suffix}"
        if normalised.endswith(marker) and len(normalised) > len(marker):
            return normalised[: -len(marker)], _VIEW_ALIASES.get(suffix, suffix)
    return normalised, None


def _finite_vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != size:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _circular_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sine = sum(math.sin(value) for value in values)
    cosine = sum(math.cos(value) for value in values)
    if abs(sine) + abs(cosine) <= 1.0e-12:
        return None
    return math.atan2(sine, cosine)


def build_semantic_map(
    teleop_payload: Mapping[str, Any],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Convert a version-1 ``teleop.json`` payload into a map payload."""

    if teleop_payload.get("version") != 1:
        raise ValueError("unsupported teleop payload version")
    raw_anchors = teleop_payload.get("anchors")
    if not isinstance(raw_anchors, list):
        raise ValueError("teleop payload must contain an anchors list")

    grouped: dict[str, dict[str, Any]] = {}
    for anchor_index, raw_anchor in enumerate(raw_anchors):
        if not isinstance(raw_anchor, Mapping):
            raise ValueError(f"anchor {anchor_index} must be an object")
        label = str(raw_anchor.get("label", "")).strip()
        location_id, view = split_anchor_label(label)
        if not location_id:
            continue
        location = grouped.setdefault(
            location_id,
            {
                "id": location_id,
                "display_name": location_id.replace("_", " "),
                "anchor_ids": [],
                "views": {},
                "_positions": [],
                "_yaws": [],
            },
        )
        anchor_id = f"anchor_{anchor_index:03d}"
        position = _finite_vector(raw_anchor.get("root_pos_world"), 3)
        rpy = _finite_vector(raw_anchor.get("base_rpy"), 3)
        view_record: dict[str, Any] = {
            "view": view,
            "anchor_id": anchor_id,
            "label": label,
            "step_id": raw_anchor.get("step_id"),
            "sim_time_s": raw_anchor.get("sim_time_s"),
            "image_path": raw_anchor.get("image_path"),
            "root_pos_world": position,
            "base_rpy": rpy,
        }
        location["anchor_ids"].append(anchor_id)
        if position is not None:
            location["_positions"].append(position)
        if rpy is not None:
            location["_yaws"].append(rpy[2])
        if view is None:
            location.setdefault("unoriented", []).append(view_record)
        else:
            # Keep the first anchor for a direction.  Repeated labels are
            # preserved in anchor_ids but do not make the scan routine repeat.
            location["views"].setdefault(view, view_record)

    locations: dict[str, Any] = {}
    for location_id, location in grouped.items():
        positions = location.pop("_positions")
        yaws = location.pop("_yaws")
        location["center_position"] = (
            [sum(axis) / len(positions) for axis in zip(*positions)]
            if positions
            else None
        )
        location["center_yaw"] = _circular_mean(yaws)
        ordered_views: list[dict[str, Any]] = []
        for view in VIEW_ORDER:
            if view in location["views"]:
                ordered_views.append(location["views"][view])
        for view, record in location["views"].items():
            if view not in VIEW_ORDER:
                ordered_views.append(record)
        ordered_views.extend(location.pop("unoriented", []))
        location["views"] = ordered_views
        locations[location_id] = location

    return {
        "version": 1,
        "map_type": "teleop_semantic_topological",
        "source_teleop": source,
        "anchor_count": len(raw_anchors),
        "location_count": len(locations),
        "view_order": list(VIEW_ORDER),
        "locations": locations,
        "route_order": list(locations),
    }


def load_semantic_map(path: str | Path) -> dict[str, Any]:
    map_path = Path(path).expanduser().resolve()
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("locations"), dict):
        raise ValueError(f"unsupported semantic map: {map_path}")
    return payload


def build_semantic_map_from_file(
    teleop_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source_path = Path(teleop_path).expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    semantic_map = build_semantic_map(payload, source=str(source_path))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(semantic_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return semantic_map


def resolve_location(
    search_text: str,
    semantic_map: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Resolve a human search-location phrase to one map location.

    This deliberately fails closed on weak or ambiguous matches.  An
    unresolved location remains a normal text waypoint rather than silently
    sending the robot to the wrong room.
    """

    locations = semantic_map.get("locations", {})
    if not isinstance(locations, Mapping):
        return None
    query = _normalise(search_text)
    query_tokens = _tokens(search_text)
    candidates: list[tuple[tuple[int, int, int], str, dict[str, Any]]] = []
    for location_id, raw_location in locations.items():
        if not isinstance(raw_location, dict):
            continue
        aliases = [str(location_id), str(raw_location.get("display_name", ""))]
        aliases.extend(str(value) for value in raw_location.get("aliases", []))
        normalised_aliases = {_normalise(alias) for alias in aliases if alias}
        if query in normalised_aliases:
            return str(location_id), raw_location
        location_tokens = set().union(*(_tokens(alias) for alias in aliases))
        overlap = len(query_tokens & location_tokens)
        if overlap < 2:
            continue
        # Prefer higher overlap, then fewer unmatched location tokens.  The
        # final term lets "living room table" beat "living room coffee table".
        score = (
            overlap,
            -len(location_tokens - query_tokens),
            -len(query_tokens - location_tokens),
        )
        candidates.append((score, str(location_id), raw_location))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1], candidates[0][2]


def ordered_views(location: Mapping[str, Any]) -> list[dict[str, Any]]:
    views = location.get("views", [])
    if not isinstance(views, list):
        return []
    return [view for view in views if isinstance(view, dict)]
