"""Mission routing and item memory for the Memory Guide MVP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from .semantic_map import load_semantic_map, ordered_views, resolve_location


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = PROJECT_ROOT / "assets" / "memory_guide_catalog.json"
DEFAULT_INVENTORY = PROJECT_ROOT / "outputs" / "memory_guide" / "inventory.json"
DEFAULT_PLAN = PROJECT_ROOT / "outputs" / "memory_guide" / "latest_plan.json"
DEFAULT_WAYPOINTS = PROJECT_ROOT / "outputs" / "memory_guide" / "latest_waypoints.txt"
DEFAULT_SEMANTIC_MAP = PROJECT_ROOT / "outputs" / "memory_guide" / "semantic_map.json"

_SPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^a-z0-9\s-]")


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Structured user request produced before invoking the navigation VLM."""

    intent: str
    target: str | None
    original: str
    router: str = "deterministic"
    confidence: float | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalise(text: str) -> str:
    return _SPACE_RE.sub(" ", _PUNCTUATION_RE.sub(" ", text.lower())).strip()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalise_identifier(text: str) -> str:
    """Turn an open-vocabulary item name into a stable inventory key."""

    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _unsupported_request_error() -> ValueError:
    return ValueError(
        "unsupported request; try 'Where are my glasses?', "
        "'Patrol the home', or 'Guide me to the bathroom'"
    )


def _parse_router_decision(
    query: str,
    catalog: Mapping[str, Any],
    decision: Any,
    *,
    router: str,
) -> ParsedQuery:
    """Validate a provider decision against the local catalog and query text."""

    original = str(query).strip()
    intent = str(decision.intent)
    target = decision.target_id
    confidence = float(decision.confidence)
    if intent == "find_item":
        # Prefer an exact known alias from the original request. This prevents
        # a model typo from replacing a catalog item that is plainly present.
        known_item = _resolve_alias(original, catalog.get("items", {}), within_text=True)
        target_id = known_item or _resolve_alias(
            str(target or ""), catalog.get("items", {})
        )
        if target_id is None:
            target_id = _normalise_identifier(str(target or ""))
        if not target_id:
            raise _unsupported_request_error()
        return ParsedQuery("find_item", target_id, original, router, confidence)

    if intent == "navigate_place":
        target_id = _resolve_alias(str(target or ""), catalog.get("places", {}))
        if target_id is None:
            target_id = _resolve_alias(
                original, catalog.get("places", {}), within_text=True
            )
        if target_id is None:
            raise _unsupported_request_error()
        return ParsedQuery("navigate_place", target_id, original, router, confidence)

    if intent == "patrol":
        return ParsedQuery("patrol", None, original, router, confidence)
    if intent == "leaving_checklist":
        return ParsedQuery("leaving_checklist", None, original, router, confidence)
    raise _unsupported_request_error()


def parse_query(
    query: str,
    catalog: Mapping[str, Any],
    *,
    llm_mode: str = "deterministic",
    bedrock_router: Any | None = None,
    openai_router: Any | None = None,
) -> ParsedQuery:
    """Classify a resident request and resolve known item/place aliases.

    ``bedrock`` and ``openai`` modes use an LLM only for intent and target
    extraction. The deterministic planner remains responsible for inventory
    freshness, map resolution, waypoint generation, and all navigation actions.
    """

    original = str(query).strip()
    if not original:
        raise ValueError("query must not be empty")
    if llm_mode == "bedrock":
        if bedrock_router is None:
            from .bedrock_router import BedrockQueryRouter

            bedrock_router = BedrockQueryRouter.from_environment()
        decision = bedrock_router.route(original, catalog)
        return _parse_router_decision(
            original,
            catalog,
            decision,
            router="bedrock:nova-micro",
        )
    if llm_mode == "openai":
        if openai_router is None:
            from .openai_router import OpenAIQueryRouter

            openai_router = OpenAIQueryRouter.from_environment()
        decision = openai_router.route(original, catalog)
        return _parse_router_decision(
            original,
            catalog,
            decision,
            router=str(
                getattr(openai_router, "router_name", "openai:gpt-5.6-luna")
            ),
        )
    if llm_mode != "deterministic":
        raise ValueError(f"unsupported query-router mode: {llm_mode}")
    normalised = _normalise(original)

    if any(phrase in normalised for phrase in ("what do i need", "checklist", "have everything")):
        return ParsedQuery("leaving_checklist", None, original)
    if any(phrase in normalised for phrase in ("patrol", "look around", "scan the house", "scan the home")):
        return ParsedQuery("patrol", None, original)

    place_prefixes = (
        "take me to ",
        "guide me to ",
        "lead me to ",
        "go to ",
        "navigate to ",
    )
    for prefix in place_prefixes:
        if normalised.startswith(prefix):
            candidate = normalised.removeprefix(prefix).strip()
            place = _resolve_alias(candidate, catalog.get("places", {}))
            if place is not None:
                return ParsedQuery("navigate_place", place, original)

    item = _resolve_alias(normalised, catalog.get("items", {}), within_text=True)
    find_language = any(
        phrase in normalised
        for phrase in (
            "where is",
            "where are",
            "find",
            "locate",
            "look for",
            "show me",
            "take me to",
            "guide me to",
            "lead me to",
        )
    )
    if item is not None and find_language:
        return ParsedQuery("find_item", item, original)

    # Preserve a useful target for an unregistered "where is my X" request.
    unknown_match = re.search(r"\bwhere (?:is|are) (?:my |the )?(.+)$", normalised)
    if unknown_match:
        return ParsedQuery("find_item", unknown_match.group(1).strip(), original)

    raise _unsupported_request_error()


def _resolve_alias(
    text: str,
    entries: Mapping[str, Any],
    *,
    within_text: bool = False,
) -> str | None:
    normalised = _normalise(text)
    matches: list[tuple[int, str]] = []
    for entry_id, entry in entries.items():
        aliases = [entry_id.replace("_", " "), *entry.get("aliases", [])]
        for alias in aliases:
            candidate = _normalise(str(alias))
            if not candidate:
                continue
            matched = (
                re.search(rf"\b{re.escape(candidate)}\b", normalised) is not None
                if within_text
                else candidate == normalised.removeprefix("the ")
            )
            if matched:
                matches.append((len(candidate), entry_id))
    if not matches:
        return None
    matches.sort(key=lambda match: (-match[0], match[1]))
    return matches[0][1]


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"memory-guide catalog does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("items"), dict):
        raise ValueError(f"unsupported memory-guide catalog: {path}")
    return payload


def load_inventory(path: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "observations": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("observations"), dict):
        raise ValueError(f"unsupported memory-guide inventory: {path}")
    return payload


def save_inventory(inventory: Mapping[str, Any], path: Path = DEFAULT_INVENTORY) -> None:
    """Atomically persist inventory so an interrupted write does not erase memory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def remember_item(
    inventory: dict[str, Any],
    *,
    item_id: str,
    location: str,
    room: str | None = None,
    confidence: float = 1.0,
    evidence_frame: str | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    location = str(location).strip()
    if not location:
        raise ValueError("location must not be empty")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    timestamp = _parse_timestamp(observed_at) if observed_at else _now_utc()
    observation = {
        "location": location,
        "room": str(room).strip() if room else None,
        "confidence": float(confidence),
        "observed_at": timestamp.isoformat().replace("+00:00", "Z"),
        "evidence_frame": str(evidence_frame) if evidence_frame else None,
    }
    inventory.setdefault("version", 1)
    inventory.setdefault("observations", {})[item_id] = observation
    return observation


def observation_score(
    observation: Mapping[str, Any],
    *,
    now: datetime | None = None,
    half_life_hours: float = 24.0,
) -> tuple[float, float]:
    """Return effective confidence and observation age in hours."""

    if half_life_hours <= 0.0:
        raise ValueError("half_life_hours must be positive")
    observed_at = _parse_timestamp(str(observation["observed_at"]))
    current = (now or _now_utc()).astimezone(timezone.utc)
    age_hours = max(0.0, (current - observed_at).total_seconds() / 3600.0)
    stored_confidence = float(observation.get("confidence", 0.0))
    effective = stored_confidence * math.pow(0.5, age_hours / half_life_hours)
    return effective, age_hours


def _display_name(item_id: str, catalog: Mapping[str, Any]) -> str:
    entry = catalog.get("items", {}).get(item_id, {})
    return str(entry.get("display_name", item_id.replace("_", " ")))


def _patrol_waypoints(item_id: str, catalog: Mapping[str, Any]) -> list[str]:
    entry = catalog.get("items", {}).get(item_id, {})
    locations = entry.get("search_locations") or catalog.get("default_search_locations", [])
    display_name = _display_name(item_id, catalog)
    return [
        (
            f"Go to {location}. Visually inspect the accessible surfaces and nearby floor "
            f"for the {display_name}. Stop after inspecting this search location."
        )
        for location in locations
    ]


def _map_waypoints(
    locations: Sequence[str],
    *,
    display_name: str,
    semantic_map: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Keep one navigation waypoint per search location.

    The semantic map may contain several recorded camera views for a location,
    but those views describe the coverage expected *inside* that location. They
    must not become separate outer navigation stages: doing so causes a patrol
    over a few locations to turn into a long list of waypoint prompts and
    repeatedly resets the VLM's route context.
    """

    waypoints: list[str] = []
    inspection_steps: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for search_location in locations:
        resolved = resolve_location(search_location, semantic_map)
        if resolved is None:
            unresolved.append(str(search_location))
            waypoints.append(
                (
                    f"Go to {search_location}. Visually inspect the accessible surfaces "
                    f"and nearby floor for the {display_name}. Stop after inspecting "
                    "this search location."
                )
            )
            inspection_steps.append(
                {
                    "search_location": str(search_location),
                    "map_location": None,
                    "view": None,
                    "coverage": "text_fallback",
                }
            )
            continue

        location_id, location = resolved
        views = ordered_views(location)
        if not views:
            views = [{"view": None, "anchor_id": None}]

        view_names = [str(view.get("view") or "current") for view in views]
        if len(view_names) == 1:
            coverage_hint = f"the recorded {view_names[0]} view"
        else:
            coverage_hint = (
                "the recorded "
                + ", ".join(view_names[:-1])
                + f", and {view_names[-1]} views"
            )
        waypoints.append(
            (
                f"Go to the recorded {location['display_name']} search location "
                f"for {search_location} and settle in clear floor space. Perform a "
                f"complete visual sweep using {coverage_hint}. Visually inspect "
                f"all accessible surfaces and the nearby floor for the {display_name}. "
                "Stop after this search location has been fully inspected; do not "
                "advance to another search location."
            )
        )
        inspection_steps.append(
            {
                "search_location": str(search_location),
                "map_location": location_id,
                "views": [
                    {
                        "view": str(view.get("view") or "current"),
                        "anchor_id": view.get("anchor_id"),
                        "image_path": view.get("image_path"),
                        "pose": {
                            "root_pos_world": view.get("root_pos_world"),
                            "base_rpy": view.get("base_rpy"),
                        },
                    }
                    for view in views
                ],
                "coverage": "recorded_location_all_views",
            }
        )
    return waypoints, inspection_steps, unresolved


def plan_query(
    query: str,
    *,
    catalog: Mapping[str, Any],
    inventory: Mapping[str, Any],
    confidence_threshold: float = 0.65,
    max_age_hours: float = 24.0,
    now: datetime | None = None,
    semantic_map: Mapping[str, Any] | None = None,
    llm_mode: str = "deterministic",
    bedrock_router: Any | None = None,
    bedrock_region: str | None = None,
    bedrock_model_id: str | None = None,
    bedrock_profile: str | None = None,
    openai_router: Any | None = None,
    openai_model_id: str | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    """Route a natural request to direct guidance or a staged patrol."""

    if llm_mode == "bedrock" and bedrock_router is None:
        from .bedrock_router import BedrockQueryRouter

        bedrock_router = BedrockQueryRouter.from_environment(
            region=bedrock_region,
            model_id=bedrock_model_id,
            profile=bedrock_profile,
        )
    if llm_mode == "openai" and openai_router is None:
        from .openai_router import OpenAIQueryRouter

        openai_router = OpenAIQueryRouter.from_environment(
            model_id=openai_model_id,
            base_url=openai_base_url,
        )
    parsed = parse_query(
        query,
        catalog,
        llm_mode=llm_mode,
        bedrock_router=bedrock_router,
        openai_router=openai_router,
    )
    plan: dict[str, Any] = {
        "version": 1,
        "query": parsed.original,
        "intent": parsed.intent,
        "target": parsed.target,
        "query_router": parsed.router,
    }
    if parsed.confidence is not None:
        plan["query_confidence"] = round(parsed.confidence, 4)

    if parsed.intent == "navigate_place":
        place = catalog["places"][parsed.target]
        plan.update(
            mode="direct_place",
            reason="known semantic destination",
            waypoints=[str(place["navigation_instruction"])],
        )
        return plan

    if parsed.intent == "patrol":
        plan.update(
            mode="patrol",
            reason="resident requested a general patrol",
            waypoints=list(catalog.get("patrol_waypoints", [])),
        )
        return plan

    if parsed.intent == "leaving_checklist":
        checklist = list(catalog.get("leaving_checklist", []))
        missing = [item for item in checklist if item not in inventory.get("observations", {})]
        plan.update(
            mode="checklist",
            reason="check essential items before leaving",
            checklist=checklist,
            missing_items=missing,
            waypoints=[
                waypoint
                for item in missing
                for waypoint in _patrol_waypoints(item, catalog)[:1]
            ],
        )
        return plan

    assert parsed.intent == "find_item" and parsed.target is not None
    observation = inventory.get("observations", {}).get(parsed.target)
    if observation is not None:
        effective_confidence, age_hours = observation_score(observation, now=now)
        reliable = (
            effective_confidence >= confidence_threshold and age_hours <= max_age_hours
        )
        plan["memory"] = {
            **observation,
            "effective_confidence": round(effective_confidence, 4),
            "age_hours": round(age_hours, 3),
            "reliable": reliable,
        }
        if reliable:
            display_name = _display_name(parsed.target, catalog)
            waypoints = [
                (
                    f"Go to {observation['location']}. Look for the {display_name}, "
                    "approach only through clear floor space, and stop about one meter away."
                )
            ]
            plan.update(
                mode="direct_item",
                reason="recent, confident inventory observation",
                waypoints=waypoints,
            )
            if semantic_map is not None:
                mapped_waypoints, steps, unresolved = _map_waypoints(
                    [str(observation["location"])],
                    display_name=display_name,
                    semantic_map=semantic_map,
                )
                if steps and any(step["map_location"] for step in steps):
                    plan["waypoints"] = mapped_waypoints
                    plan["inspection_policy"] = "recorded_views_requires_visual_verification"
                    plan["inspection_steps"] = steps
                    plan["unresolved_locations"] = unresolved
                    plan["semantic_map_used"] = True
            return plan

    locations = catalog.get("default_search_locations", [])
    plan.update(
        mode="patrol_item",
        reason=(
            "inventory observation is stale or uncertain"
            if observation is not None
            else "item is not yet in memory"
        ),
        waypoints=_patrol_waypoints(parsed.target, catalog),
    )
    if semantic_map is not None:
        entry = catalog.get("items", {}).get(parsed.target, {})
        mapped_locations = entry.get("search_locations") or locations
        mapped_waypoints, steps, unresolved = _map_waypoints(
            [str(location) for location in mapped_locations],
            display_name=_display_name(parsed.target, catalog),
            semantic_map=semantic_map,
        )
        if steps and any(step["map_location"] for step in steps):
            plan["waypoints"] = mapped_waypoints
            plan["inspection_policy"] = "recorded_views_requires_visual_verification"
            plan["inspection_steps"] = steps
            plan["unresolved_locations"] = unresolved
            plan["semantic_map_used"] = True
    return plan


def write_plan(plan: Mapping[str, Any], plan_path: Path, waypoint_path: Path) -> None:
    waypoints = [str(value).strip() for value in plan.get("waypoints", []) if str(value).strip()]
    if not waypoints:
        raise ValueError("mission plan did not produce any navigation waypoints")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    waypoint_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    waypoint_path.write_text("\n".join(waypoints) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory Guide mission planner")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="route a resident query into navigation waypoints")
    plan.add_argument("--query", required=True)
    plan.add_argument("--confidence-threshold", type=float, default=0.65)
    plan.add_argument("--max-age-hours", type=float, default=24.0)
    plan.add_argument(
        "--llm-mode",
        choices=("deterministic", "bedrock", "openai"),
        default=os.environ.get("NAVILA_MEMORY_LLM_MODE", "deterministic"),
        help="query router: deterministic rules, Bedrock Nova Micro, or OpenAI",
    )
    plan.add_argument(
        "--bedrock-region",
        default=os.environ.get("NAVILA_BEDROCK_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "ap-southeast-1",
    )
    plan.add_argument(
        "--bedrock-model-id",
        default=os.environ.get("NAVILA_BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0"),
    )
    plan.add_argument(
        "--bedrock-profile",
        default=os.environ.get(
            "NAVILA_BEDROCK_PROFILE", "683803166476_hack2026_IsbUsersPS"
        ),
        help="boto3 profile for Bedrock credentials",
    )
    plan.add_argument(
        "--openai-model-id",
        default=os.environ.get("NAVILA_OPENAI_MODEL", "gpt-5.6-luna"),
        help="OpenAI model ID",
    )
    plan.add_argument(
        "--openai-base-url",
        default=os.environ.get("NAVILA_OPENAI_BASE_URL"),
        help="optional OpenAI-compatible API base URL",
    )
    plan.add_argument(
        "--semantic-map",
        type=Path,
        help="pose-tagged teleop semantic map used to annotate locations with recorded views",
    )
    plan.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN)
    plan.add_argument("--waypoint-output", type=Path, default=DEFAULT_WAYPOINTS)

    remember = subparsers.add_parser("remember", help="record the last-seen location of an item")
    remember.add_argument("--item", required=True)
    remember.add_argument("--location", required=True)
    remember.add_argument("--room")
    remember.add_argument("--confidence", type=float, default=1.0)
    remember.add_argument("--evidence-frame")
    remember.add_argument("--observed-at")

    map_build = subparsers.add_parser(
        "map-build", help="convert a teleop.json collection into a semantic map"
    )
    map_build.add_argument("--teleop-json", type=Path, required=True)
    map_build.add_argument("--output", type=Path, default=DEFAULT_SEMANTIC_MAP)

    subparsers.add_parser("list", help="show remembered item observations")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        inventory = load_inventory(args.inventory)
        if args.command == "plan":
            if not 0.0 <= args.confidence_threshold <= 1.0:
                raise ValueError("confidence threshold must be between 0 and 1")
            if args.max_age_hours <= 0.0:
                raise ValueError("maximum age must be positive")
            plan = plan_query(
                args.query,
                catalog=catalog,
                inventory=inventory,
                confidence_threshold=args.confidence_threshold,
                max_age_hours=args.max_age_hours,
                semantic_map=(
                    load_semantic_map(args.semantic_map)
                    if args.semantic_map is not None
                    else None
                ),
                llm_mode=args.llm_mode,
                bedrock_region=args.bedrock_region,
                bedrock_model_id=args.bedrock_model_id,
                bedrock_profile=args.bedrock_profile,
                openai_model_id=args.openai_model_id,
                openai_base_url=args.openai_base_url,
            )
            write_plan(plan, args.plan_output, args.waypoint_output)
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        if args.command == "map-build":
            from .semantic_map import build_semantic_map_from_file

            semantic_map = build_semantic_map_from_file(args.teleop_json, args.output)
            print(
                json.dumps(
                    {
                        "output": str(Path(args.output).expanduser().resolve()),
                        "anchor_count": semantic_map["anchor_count"],
                        "location_count": semantic_map["location_count"],
                        "locations": semantic_map["route_order"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "remember":
            item_id = _resolve_alias(args.item, catalog.get("items", {}))
            if item_id is None:
                item_id = _normalise(args.item).replace(" ", "_")
            observation = remember_item(
                inventory,
                item_id=item_id,
                location=args.location,
                room=args.room,
                confidence=args.confidence,
                evidence_frame=args.evidence_frame,
                observed_at=args.observed_at,
            )
            save_inventory(inventory, args.inventory)
            print(json.dumps({"item": item_id, "observation": observation}, indent=2))
            return 0
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"memory-guide error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
