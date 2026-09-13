"""Build and query a lightweight semantic map from teleop anchors.

This is intentionally a topological/pose-tagged map, not SLAM.  Teleoperation
already records the information we need for a repeatable search routine:
semantic labels, world poses, and one or more camera views at each location.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import heapq
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


def _command_summary(
    raw_events: object,
    *,
    start_step: int | None,
    end_step: int | None,
) -> dict[str, Any]:
    """Summarize a demonstrated teleop segment without replaying motor keys.

    Terminal key-repeat can emit the same event many times at one physics step.
    Those duplicates are removed for reporting.  The result is provenance for
    a route demonstration, not an executable open-loop command sequence.
    """

    if not isinstance(raw_events, list) or start_step is None or end_step is None:
        return {"raw_event_count": 0, "unique_event_count": 0, "key_counts": {}}

    selected: list[tuple[int, str, tuple[float, ...]]] = []
    raw_count = 0
    seen: set[tuple[int, str, tuple[float, ...]]] = set()
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        try:
            step_id = int(raw_event.get("step_id"))
        except (TypeError, ValueError):
            continue
        if not start_step < step_id <= end_step:
            continue
        key = str(raw_event.get("key", "")).strip().lower()
        velocity = _finite_vector(raw_event.get("velocity"), 3)
        if not key or velocity is None:
            continue
        raw_count += 1
        event = (step_id, key, tuple(velocity))
        if event in seen:
            continue
        seen.add(event)
        selected.append(event)

    key_counts = Counter(key for _, key, _ in selected)
    return {
        "raw_event_count": raw_count,
        "unique_event_count": len(selected),
        "key_counts": dict(sorted(key_counts.items())),
    }


def _demonstrated_routes(
    anchor_sequence: Sequence[Mapping[str, Any]],
    locations: Mapping[str, Mapping[str, Any]],
    raw_events: object,
) -> list[dict[str, Any]]:
    """Build directed route demonstrations between consecutive locations."""

    groups: list[list[Mapping[str, Any]]] = []
    for anchor in anchor_sequence:
        if not groups or groups[-1][0]["location_id"] != anchor["location_id"]:
            groups.append([anchor])
        else:
            groups[-1].append(anchor)

    pair_counts: Counter[tuple[str, str]] = Counter()
    routes: list[dict[str, Any]] = []
    for source_group, destination_group in zip(groups, groups[1:]):
        source = source_group[-1]
        destination = destination_group[0]
        source_id = str(source["location_id"])
        destination_id = str(destination["location_id"])
        if source_id == destination_id:
            continue
        pair = (source_id, destination_id)
        pair_counts[pair] += 1
        route_id = f"{source_id}__to__{destination_id}__demo_{pair_counts[pair]:03d}"
        start_position = _finite_vector(source.get("root_pos_world"), 3)
        end_position = _finite_vector(destination.get("root_pos_world"), 3)
        endpoint_distance = None
        if start_position is not None and end_position is not None:
            endpoint_distance = math.dist(start_position[:2], end_position[:2])
        try:
            start_step = int(source.get("step_id"))
            end_step = int(destination.get("step_id"))
        except (TypeError, ValueError):
            start_step = None
            end_step = None
        try:
            duration_s = max(
                0.0,
                float(destination.get("sim_time_s")) - float(source.get("sim_time_s")),
            )
        except (TypeError, ValueError):
            duration_s = None

        source_name = str(locations[source_id].get("display_name", source_id))
        destination_name = str(
            locations[destination_id].get("display_name", destination_id)
        )
        routes.append(
            {
                "id": route_id,
                "from_location": source_id,
                "to_location": destination_id,
                "status": "human_demonstrated",
                "direction": "recorded_only",
                "instruction_status": "generated_unverified",
                "instruction": (
                    f"Leave the {source_name} area and navigate to the adjacent "
                    f"{destination_name} area. Use the visible {destination_name} "
                    "as the arrival landmark. Approach through clear floor space "
                    "and stop when it is clearly visible."
                ),
                "departure": {
                    "anchor_id": source.get("anchor_id"),
                    "view": source.get("view"),
                    "image_path": source.get("image_path"),
                    "step_id": start_step,
                },
                "arrival": {
                    "anchor_id": destination.get("anchor_id"),
                    "view": destination.get("view"),
                    "image_path": destination.get("image_path"),
                    "step_id": end_step,
                },
                "demonstration": {
                    "duration_s": duration_s,
                    "endpoint_distance_m": endpoint_distance,
                    "command_summary": _command_summary(
                        raw_events,
                        start_step=start_step,
                        end_step=end_step,
                    ),
                },
            }
        )
    return routes


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
    anchor_sequence: list[dict[str, Any]] = []
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
        anchor_sequence.append({"location_id": location_id, **view_record})
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

    routes = _demonstrated_routes(
        anchor_sequence,
        locations,
        teleop_payload.get("command_events"),
    )
    return {
        "version": 1,
        "map_type": "teleop_semantic_topological",
        "source_teleop": source,
        "anchor_count": len(raw_anchors),
        "location_count": len(locations),
        "route_count": len(routes),
        "view_order": list(VIEW_ORDER),
        "locations": locations,
        "route_order": list(locations),
        "routes": routes,
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


def build_semantic_map_from_files(
    teleop_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge independent teleop sessions into one semantic map.

    Anchors and routes are namespaced by session so repeated ``anchor_000``
    identifiers cannot collide. Relative image paths are resolved against the
    teleop file that owns them. Routes are only inferred within a session;
    merely merging two disconnected captures never invents a cross-session
    traversal.
    """

    sources: list[Path] = []
    seen_sources: set[Path] = set()
    for raw_path in teleop_paths:
        source = Path(raw_path).expanduser().resolve()
        if source in seen_sources:
            continue
        seen_sources.add(source)
        sources.append(source)
    if not sources:
        raise ValueError("at least one teleop JSON path is required")

    locations: dict[str, dict[str, Any]] = {}
    route_order: list[str] = []
    routes: list[dict[str, Any]] = []
    anchor_count = 0
    for session_index, source in enumerate(sources):
        payload = json.loads(source.read_text(encoding="utf-8"))
        session_map = build_semantic_map(payload, source=str(source))
        namespace = f"session_{session_index:03d}_{source.parent.name}"
        anchor_count += int(session_map.get("anchor_count", 0))

        session_locations = session_map.get("locations", {})
        if not isinstance(session_locations, Mapping):
            raise ValueError(f"teleop session produced invalid locations: {source}")
        for location_id in session_map.get("route_order", []):
            raw_location = session_locations.get(location_id)
            if not isinstance(raw_location, Mapping):
                continue
            if location_id not in route_order:
                route_order.append(str(location_id))
            destination = locations.setdefault(
                str(location_id),
                {
                    "id": str(location_id),
                    "display_name": raw_location.get(
                        "display_name", str(location_id).replace("_", " ")
                    ),
                    "aliases": list(raw_location.get("aliases", [])),
                    "anchor_ids": [],
                    "views": [],
                },
            )
            for raw_view in raw_location.get("views", []):
                if not isinstance(raw_view, Mapping):
                    continue
                view = dict(raw_view)
                original_anchor_id = str(view.get("anchor_id", "anchor"))
                view["anchor_id"] = f"{namespace}::{original_anchor_id}"
                view["session"] = namespace
                image_path = view.get("image_path")
                if image_path:
                    resolved_image = Path(str(image_path)).expanduser()
                    if not resolved_image.is_absolute():
                        resolved_image = source.parent / resolved_image
                    view["image_path"] = str(resolved_image.resolve())
                destination["anchor_ids"].append(view["anchor_id"])
                destination["views"].append(view)

        for raw_route in session_map.get("routes", []):
            if not isinstance(raw_route, Mapping):
                continue
            route = dict(raw_route)
            original_route_id = str(route.get("id", "route"))
            route["id"] = f"{namespace}::{original_route_id}"
            route["session"] = namespace
            for endpoint_name in ("departure", "arrival"):
                raw_endpoint = route.get(endpoint_name, {})
                if not isinstance(raw_endpoint, Mapping):
                    continue
                endpoint = dict(raw_endpoint)
                endpoint_anchor = str(endpoint.get("anchor_id", "anchor"))
                endpoint["anchor_id"] = f"{namespace}::{endpoint_anchor}"
                image_path = endpoint.get("image_path")
                if image_path:
                    resolved_image = Path(str(image_path)).expanduser()
                    if not resolved_image.is_absolute():
                        resolved_image = source.parent / resolved_image
                    endpoint["image_path"] = str(resolved_image.resolve())
                route[endpoint_name] = endpoint
            routes.append(route)

    for location in locations.values():
        positions = [
            position
            for view in location["views"]
            if (position := _finite_vector(view.get("root_pos_world"), 3)) is not None
        ]
        yaws = [
            rpy[2]
            for view in location["views"]
            if (rpy := _finite_vector(view.get("base_rpy"), 3)) is not None
        ]
        location["center_position"] = (
            [sum(values) / len(values) for values in zip(*positions)]
            if positions
            else None
        )
        location["center_yaw"] = _circular_mean(yaws)

    semantic_map = {
        "version": 1,
        "map_type": "teleop_semantic_topological",
        "source_teleop": str(sources[0]),
        "source_teleops": [str(source) for source in sources],
        "anchor_count": anchor_count,
        "location_count": len(locations),
        "route_count": len(routes),
        "view_order": list(VIEW_ORDER),
        "locations": locations,
        "route_order": route_order,
        "routes": routes,
    }
    destination_path = Path(output_path).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
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


def _reverse_route(
    route: Mapping[str, Any],
    locations: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicitly unverified reverse of one demonstrated edge."""

    source_id = str(route.get("to_location", ""))
    destination_id = str(route.get("from_location", ""))
    source = locations.get(source_id, {})
    destination = locations.get(destination_id, {})
    source_name = (
        str(source.get("display_name", source_id))
        if isinstance(source, Mapping)
        else source_id
    )
    destination_name = (
        str(destination.get("display_name", destination_id))
        if isinstance(destination, Mapping)
        else destination_id
    )
    return {
        "id": f"{route.get('id', 'route')}__reverse_unverified",
        "from_location": source_id,
        "to_location": destination_id,
        "status": "derived_reverse_unverified",
        "direction": "reverse_of_recording",
        "instruction_status": "generated_unverified",
        "instruction": (
            f"Leave the {source_name} area and navigate to the adjacent "
            f"{destination_name} area. Use the visible {destination_name} as the "
            "arrival landmark. Approach through clear floor space and stop when "
            "it is clearly visible."
        ),
        "departure": dict(route.get("arrival", {})),
        "arrival": dict(route.get("departure", {})),
        "demonstration": dict(route.get("demonstration", {})),
        "derived_from": route.get("id"),
    }


def plan_topological_route(
    semantic_map: Mapping[str, Any],
    from_location: str,
    to_location: str,
    *,
    allow_unverified_reverse: bool = False,
) -> dict[str, Any]:
    """Find a route through demonstrated semantic locations.

    Recorded edges are directed.  Reverse traversal is excluded unless the
    caller explicitly opts into a generated, unverified reverse instruction.
    No coordinate or open-loop motor replay is returned.
    """

    source = resolve_location(from_location, semantic_map)
    if source is None:
        raise ValueError(f"start location is not in the semantic map: {from_location}")
    destination = resolve_location(to_location, semantic_map)
    if destination is None:
        raise ValueError(f"destination is not in the semantic map: {to_location}")
    source_id = source[0]
    destination_id = destination[0]
    if source_id == destination_id:
        return {
            "from_location": source_id,
            "to_location": destination_id,
            "locations": [source_id],
            "edges": [],
            "uses_unverified_reverse": False,
            "instructions": [],
        }

    locations = semantic_map.get("locations", {})
    if not isinstance(locations, Mapping):
        raise ValueError("semantic map locations are invalid")
    raw_routes = semantic_map.get("routes", [])
    if not isinstance(raw_routes, list):
        raise ValueError("semantic map routes are invalid")

    edges: list[dict[str, Any]] = [
        dict(route)
        for route in raw_routes
        if isinstance(route, Mapping)
        and route.get("from_location") in locations
        and route.get("to_location") in locations
    ]
    if allow_unverified_reverse:
        edges.extend(_reverse_route(route, locations) for route in edges.copy())

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        adjacency.setdefault(str(edge["from_location"]), []).append(edge)
    for outgoing in adjacency.values():
        outgoing.sort(key=lambda edge: str(edge.get("id", "")))

    queue: list[tuple[float, int, str, list[str], list[dict[str, Any]]]] = [
        (0.0, 0, source_id, [source_id], [])
    ]
    best_cost: dict[str, float] = {source_id: 0.0}
    serial = 0
    while queue:
        cost, _, location_id, path, route_path = heapq.heappop(queue)
        if cost > best_cost.get(location_id, math.inf):
            continue
        if location_id == destination_id:
            return {
                "from_location": source_id,
                "to_location": destination_id,
                "locations": path,
                "edges": route_path,
                "uses_unverified_reverse": any(
                    edge.get("status") == "derived_reverse_unverified"
                    for edge in route_path
                ),
                "instructions": [str(edge["instruction"]) for edge in route_path],
            }
        for edge in adjacency.get(location_id, []):
            next_id = str(edge["to_location"])
            demonstration = edge.get("demonstration", {})
            duration = (
                demonstration.get("duration_s")
                if isinstance(demonstration, Mapping)
                else None
            )
            try:
                weight = max(float(duration), 1.0)
            except (TypeError, ValueError):
                weight = 1.0
            next_cost = cost + weight
            if next_cost >= best_cost.get(next_id, math.inf):
                continue
            best_cost[next_id] = next_cost
            serial += 1
            heapq.heappush(
                queue,
                (next_cost, serial, next_id, [*path, next_id], [*route_path, edge]),
            )

    policy = (
        "including unverified reverse edges"
        if allow_unverified_reverse
        else "using demonstrated directions only"
    )
    raise ValueError(
        f"no route from {source_id} to {destination_id} {policy}"
    )
