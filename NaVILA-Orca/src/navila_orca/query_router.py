"""Provider-neutral types and schema for Memory Guide query routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


SUPPORTED_INTENTS = (
    "find_item",
    "navigate_place",
    "patrol",
    "leaving_checklist",
    "unsupported",
)
SUPPORTED_TARGET_TYPES = ("item", "place", "none")


@dataclass(frozen=True)
class RouteDecision:
    """Validated semantic result returned by an LLM provider."""

    intent: str
    target_type: str
    target_id: str | None
    confidence: float
    rationale: str | None = None


def catalog_context(catalog: Mapping[str, Any]) -> str:
    """Create a compact catalog allow-list for a router prompt."""

    items: list[dict[str, Any]] = []
    for item_id, item in catalog.get("items", {}).items():
        items.append(
            {
                "id": str(item_id),
                "display_name": str(item.get("display_name", item_id)),
                "aliases": [str(alias) for alias in item.get("aliases", [])],
            }
        )
    places: list[dict[str, Any]] = []
    for place_id, place in catalog.get("places", {}).items():
        places.append(
            {
                "id": str(place_id),
                "aliases": [str(alias) for alias in place.get("aliases", [])],
            }
        )
    return json.dumps(
        {"items": items, "places": places},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def route_schema() -> dict[str, Any]:
    """Return the strict JSON schema shared by Bedrock and OpenAI."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(SUPPORTED_INTENTS),
                "description": "The single Memory Guide intent.",
            },
            "target_type": {
                "type": "string",
                "enum": list(SUPPORTED_TARGET_TYPES),
                "description": "Whether target_id is an item, place, or absent.",
            },
            "target_id": {
                "type": "string",
                "description": (
                    "Catalog ID when known. For an unknown item, return a short "
                    "lowercase noun such as bread. Return an empty string when absent."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the intent and target extraction.",
            },
            "rationale": {
                "type": "string",
                "description": "A short explanation for logs; never used for navigation.",
            },
        },
        "required": ["intent", "target_type", "target_id", "confidence", "rationale"],
    }


ROUTER_SYSTEM_PROMPT = (
    "You are the Memory Guide query router for an indoor robot. "
    "Classify the resident's request and extract one target. "
    "You are not a navigation planner: never return coordinates, poses, "
    "motor actions, or generated waypoints. Use the exact catalog ID when "
    "the target is listed. For an item that is not listed, return a short "
    "lowercase noun as target_id so the local planner can search for it. "
    "For an unknown destination, use unsupported."
)


def router_user_prompt(query: str, catalog: Mapping[str, Any]) -> str:
    return (
        "Catalog allow-list:\n"
        f"{catalog_context(catalog)}\n\n"
        f"Resident request: {str(query).strip()}"
    )


def validate_route_decision(
    payload: Mapping[str, Any],
    *,
    error_type: type[ValueError] = ValueError,
    provider_name: str = "LLM",
) -> RouteDecision:
    """Validate provider output before it reaches the deterministic planner."""

    intent = payload.get("intent")
    target_type = payload.get("target_type")
    target_id = payload.get("target_id")
    confidence = payload.get("confidence")
    rationale = payload.get("rationale")
    if intent not in SUPPORTED_INTENTS:
        raise error_type(f"{provider_name} returned unsupported intent: {intent!r}")
    if target_type not in SUPPORTED_TARGET_TYPES:
        raise error_type(
            f"{provider_name} returned unsupported target type: {target_type!r}"
        )
    expected_target_type = {
        "find_item": "item",
        "navigate_place": "place",
        "patrol": "none",
        "leaving_checklist": "none",
        "unsupported": "none",
    }[intent]
    if target_type != expected_target_type:
        raise error_type(
            f"{provider_name} returned target type {target_type!r} for intent "
            f"{intent!r}; expected {expected_target_type!r}"
        )
    if not isinstance(target_id, str):
        raise error_type(f"{provider_name} returned a non-string target_id")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{provider_name} returned an invalid confidence") from exc
    if not 0.0 <= confidence <= 1.0:
        raise error_type(f"{provider_name} confidence must be between 0 and 1")
    if target_type == "none":
        normalized_target: str | None = None
    elif target_id.strip():
        normalized_target = target_id.strip()
    else:
        raise error_type(f"{provider_name} returned an empty target for {intent}")
    return RouteDecision(
        intent=intent,
        target_type=target_type,
        target_id=normalized_target,
        confidence=confidence,
        rationale=str(rationale).strip() if rationale is not None else None,
    )
