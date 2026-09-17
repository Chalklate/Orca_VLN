"""OpenAI Responses API query routing for the Memory Guide."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from .frames import encode_jpeg_base64
from .query_router import (
    ROUTER_SYSTEM_PROMPT,
    RouteDecision,
    route_schema,
    router_user_prompt,
    validate_route_decision,
)


DEFAULT_MODEL_ID = "gpt-5.6-luna"
SCHEMA_NAME = "memory_guide_route"
LANDMARK_SCAN_SCHEMA_NAME = "memory_guide_landmark_scan"
LANDMARK_RANKING_SCHEMA_NAME = "memory_guide_landmark_ranking"


class OpenAIRouterError(ValueError):
    """Raised when the OpenAI router cannot produce a safe route decision."""


class OpenAIQueryRouter:
    """Route Memory Guide text through the OpenAI Responses API."""

    def __init__(self, client: Any, *, model_id: str = DEFAULT_MODEL_ID) -> None:
        model_id = str(model_id).strip()
        if not model_id:
            raise OpenAIRouterError("OpenAI model ID must not be empty")
        self.client = client
        self.model_id = model_id

    @property
    def router_name(self) -> str:
        return f"openai:{self.model_id}"

    @classmethod
    def from_environment(
        cls,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "OpenAIQueryRouter":
        """Build an OpenAI client from environment credentials and settings."""

        selected_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not selected_api_key:
            raise OpenAIRouterError(
                "OpenAI mode requires OPENAI_API_KEY; keep the key in the "
                "environment rather than passing it on the command line"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIRouterError(
                "OpenAI mode needs the openai package; install the optional "
                "dependency with python -m pip install -e '.[openai]'"
            ) from exc

        selected_base_url = base_url or os.environ.get("NAVILA_OPENAI_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": selected_api_key}
        if selected_base_url:
            client_kwargs["base_url"] = selected_base_url
        try:
            client = OpenAI(**client_kwargs)
        except Exception as exc:
            raise OpenAIRouterError(f"could not configure OpenAI client: {exc}") from exc
        return cls(
            client,
            model_id=(
                model_id
                or os.environ.get("NAVILA_OPENAI_MODEL")
                or DEFAULT_MODEL_ID
            ),
        )

    def route(self, query: str, catalog: Mapping[str, Any]) -> RouteDecision:
        original = str(query).strip()
        if not original:
            raise OpenAIRouterError("query must not be empty")

        request = {
            "model": self.model_id,
            "instructions": ROUTER_SYSTEM_PROMPT,
            "input": router_user_prompt(original, catalog),
            "max_output_tokens": 256,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": SCHEMA_NAME,
                    "strict": True,
                    "schema": route_schema(),
                }
            },
        }
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            raise OpenAIRouterError(
                f"OpenAI {self.model_id} query routing failed: {exc}"
            ) from exc

        raw_text = getattr(response, "output_text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise OpenAIRouterError("OpenAI response did not contain structured output")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OpenAIRouterError(
                "OpenAI structured output was not valid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise OpenAIRouterError("OpenAI structured output was not an object")
        return validate_route_decision(
            payload,
            error_type=OpenAIRouterError,
            provider_name="OpenAI",
        )

    @staticmethod
    def _landmark_candidates(landmark_map: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw_landmarks = landmark_map.get("landmarks", [])
        if not isinstance(raw_landmarks, list):
            raise OpenAIRouterError("landmark map contains an invalid landmarks list")
        candidates: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_landmarks):
            if not isinstance(raw, Mapping):
                continue
            landmark_id = str(raw.get("id", "")).strip()
            name = str(raw.get("name", "")).strip()
            if not landmark_id or not name:
                continue
            candidates.append(
                {
                    "id": landmark_id,
                    "name": name,
                    "kind": str(raw.get("kind", "other")),
                    "description": str(raw.get("description", "")),
                    "search_surface": bool(raw.get("search_surface", False)),
                    "view_index": raw.get("view_index", index),
                }
            )
        return candidates

    @staticmethod
    def _decode_structured_response(response: Any, *, purpose: str) -> Mapping[str, Any]:
        raw_text = getattr(response, "output_text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise OpenAIRouterError(
                f"OpenAI response did not contain structured output for {purpose}"
            )
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OpenAIRouterError(
                f"OpenAI structured output for {purpose} was not valid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise OpenAIRouterError(
                f"OpenAI structured output for {purpose} was not an object"
            )
        return payload

    def rank_landmarks(
        self,
        target: str,
        landmark_map: Mapping[str, Any],
        *,
        max_landmarks: int = 6,
    ) -> list[str]:
        """Rank scanned landmarks as search locations for an item.

        The model may choose only IDs from the local scan.  It cannot invent
        coordinates or motion commands; the caller turns the selected visual
        references into conservative text waypoints.
        """

        if max_landmarks <= 0:
            raise OpenAIRouterError("max_landmarks must be positive")
        candidates = self._landmark_candidates(landmark_map)
        if not candidates:
            return []
        max_landmarks = min(int(max_landmarks), len(candidates))
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "landmark_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": max_landmarks,
                },
                "rationale": {"type": "string"},
            },
            "required": ["landmark_ids", "rationale"],
        }
        instructions = (
            "You are the search-location ranker for a mobile indoor robot. "
            "Given an item and a visual landmark scan, choose the most useful "
            "landmarks to inspect in order. Prefer tables, desks, counters, "
            "shelves, bags, and other accessible surfaces; then choose broad "
            "coverage landmarks such as aisles or room areas. Use only the exact "
            "landmark IDs provided. Never return coordinates, poses, distances, "
            "motor actions, or invented landmarks. Return strict JSON matching "
            "the supplied schema."
        )
        user_prompt = (
            f"Item to find: {str(target).strip()}\n"
            "Scanned landmarks (allow-list):\n"
            f"{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}"
        )
        request = {
            "model": self.model_id,
            "instructions": instructions,
            "input": user_prompt,
            "max_output_tokens": 512,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": LANDMARK_RANKING_SCHEMA_NAME,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            raise OpenAIRouterError(
                f"OpenAI {self.model_id} landmark ranking failed: {exc}"
            ) from exc
        payload = self._decode_structured_response(response, purpose="landmark ranking")
        raw_ids = payload.get("landmark_ids")
        if not isinstance(raw_ids, list):
            raise OpenAIRouterError("OpenAI landmark ranking returned invalid IDs")
        allowed = {candidate["id"] for candidate in candidates}
        selected: list[str] = []
        for raw_id in raw_ids:
            landmark_id = str(raw_id).strip()
            if landmark_id in allowed and landmark_id not in selected:
                selected.append(landmark_id)
        if not selected:
            raise OpenAIRouterError(
                "OpenAI landmark ranking returned no valid scanned landmark IDs"
            )
        return selected[:max_landmarks]

    def describe_landmarks(
        self,
        images: Sequence[Any],
        *,
        view_indices: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract searchable visual landmarks from a small scan image set."""

        if not images:
            raise OpenAIRouterError("landmark scan requires at least one image")
        indices = list(view_indices or range(len(images)))
        if len(indices) != len(images):
            raise OpenAIRouterError("view_indices must match the image count")
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Identify stable, useful visual landmarks in these numbered "
                    "views of one indoor room. Include furniture, doors, stages, "
                    "tables, desks, counters, shelves, aisles, and other visually "
                    "recognisable areas. Do not include transient people or tiny "
                    "unreliable details. A landmark can appear in more than one "
                    "view, but report its clearest view. Return concise descriptions "
                    "and mark whether it is a plausible surface/area for searching "
                    "for a misplaced item. Return strict JSON only.\n\n"
                    "View indices are the integers printed immediately before each "
                    "image."
                ),
            }
        ]
        for view_index, image in zip(indices, images):
            content.append(
                {
                    "type": "input_text",
                    "text": f"View index: {int(view_index)}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        "data:image/jpeg;base64," + encode_jpeg_base64(image)
                    ),
                }
            )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "landmarks": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string"},
                            "view_index": {"type": "integer"},
                            "description": {"type": "string"},
                            "search_surface": {"type": "boolean"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": [
                            "name",
                            "kind",
                            "view_index",
                            "description",
                            "search_surface",
                            "confidence",
                        ],
                    },
                }
            },
            "required": ["landmarks"],
        }
        request = {
            "model": self.model_id,
            "instructions": (
                "You are a visual landmark extractor. Ground every result in the "
                "provided images, use the provided view index, and never invent "
                "coordinates or objects hidden from view. Return strict JSON "
                "matching the supplied schema."
            ),
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 2048,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": LANDMARK_SCAN_SCHEMA_NAME,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            raise OpenAIRouterError(
                f"OpenAI {self.model_id} landmark extraction failed: {exc}"
            ) from exc
        payload = self._decode_structured_response(response, purpose="landmark extraction")
        raw_landmarks = payload.get("landmarks")
        if not isinstance(raw_landmarks, list):
            raise OpenAIRouterError("OpenAI landmark extraction returned invalid landmarks")
        valid_indices = set(indices)
        result: list[dict[str, Any]] = []
        for raw in raw_landmarks:
            if not isinstance(raw, Mapping):
                continue
            name = str(raw.get("name", "")).strip()
            description = str(raw.get("description", "")).strip()
            try:
                view_index = int(raw.get("view_index"))
                confidence = float(raw.get("confidence"))
            except (TypeError, ValueError):
                continue
            if (
                not name
                or view_index not in valid_indices
                or not 0.0 <= confidence <= 1.0
            ):
                continue
            result.append(
                {
                    "name": name,
                    "kind": str(raw.get("kind", "other")).strip() or "other",
                    "view_index": view_index,
                    "description": description,
                    "search_surface": bool(raw.get("search_surface", False)),
                    "confidence": confidence,
                }
            )
        return result
