"""OpenAI Responses API query routing for the Memory Guide."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
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
LANDMARK_SEER_SCHEMA_NAME = "memory_guide_landmark_seer"


class OpenAIRouterError(ValueError):
    """Raised when the OpenAI router cannot produce a safe route decision."""


@dataclass(frozen=True)
class LandmarkSeerResult:
    """Structured visual reacquisition result for one live camera image."""

    target_visible: bool
    confidence: float
    relative_position: str
    bearing_degrees: float
    distance_state: str
    safe_to_advance: bool
    rationale: str


@dataclass(frozen=True)
class GoalSeerResult:
    """Structured landmark-and-item verification for one live camera image."""

    landmark_visible: bool
    landmark_confidence: float
    landmark_relative_position: str
    landmark_bearing_degrees: float
    landmark_distance_state: str
    safe_to_advance: bool
    item_visible: bool
    item_confidence: float
    item_relative_position: str
    item_bearing_degrees: float
    rationale: str


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
            status = getattr(response, "status", None)
            incomplete = getattr(response, "incomplete_details", None)
            refusal = getattr(response, "refusal", None)
            diagnostics = " ".join(
                part
                for part in (
                    f"status={status!r}" if status is not None else "",
                    f"incomplete_details={incomplete!r}" if incomplete is not None else "",
                    f"refusal={refusal!r}" if refusal is not None else "",
                )
                if part
            )
            raise OpenAIRouterError(
                f"OpenAI response did not contain structured output for {purpose}"
                + (f" ({diagnostics})" if diagnostics else "")
            )
        raw_text = raw_text.strip()
        candidates = [raw_text]
        if raw_text.startswith("```"):
            fenced_lines = raw_text.splitlines()
            if fenced_lines and fenced_lines[0].lstrip().startswith("```"):
                fenced_lines = fenced_lines[1:]
            if fenced_lines and fenced_lines[-1].strip().startswith("```"):
                fenced_lines = fenced_lines[:-1]
            candidates.append("\n".join(fenced_lines).strip())
        object_start = raw_text.find("{")
        object_end = raw_text.rfind("}")
        if object_start >= 0 and object_end > object_start:
            candidates.append(raw_text[object_start : object_end + 1])
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                return payload
        try:
            json.loads(raw_text)
        except json.JSONDecodeError as exc:
            preview = raw_text[:240].replace("\n", "\\n")
            raise OpenAIRouterError(
                f"OpenAI structured output for {purpose} was not valid JSON; "
                f"preview={preview!r}"
            ) from exc
        raise OpenAIRouterError(
            f"OpenAI structured output for {purpose} was not an object"
        )

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

    def see_landmark(
        self,
        *,
        target_name: str,
        target_description: str,
        reference_image: Any,
        live_image: Any,
    ) -> LandmarkSeerResult:
        """Compare a scan reference with the current robot camera image.

        This is deliberately a perception query, not a motion planner.  The
        caller remains responsible for converting a positive result into a
        bounded action and for stopping when the target is not visible.
        """

        target_name = str(target_name).strip()
        target_description = str(target_description).strip()
        if not target_name:
            raise OpenAIRouterError("landmark seer target name must not be empty")
        if reference_image is None or live_image is None:
            raise OpenAIRouterError("landmark seer requires reference and live images")

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Compare the reference image of a stable indoor landmark with "
                    "the current robot-camera image. Decide only whether the same "
                    "landmark is visibly present in the current image. Do not infer "
                    "that it is present from the text description. If it is absent, "
                    "set target_visible=false, relative_position=not_visible, and "
                    "bearing_degrees=0. If it is visible, estimate its horizontal "
                    "bearing in the camera image: negative means left, positive means "
                    "right, and zero means centered. Return concise strict JSON."
                ),
            },
            {
                "type": "input_text",
                "text": (
                    f"Landmark name: {target_name}\n"
                    f"Landmark description: {target_description or 'none'}\n"
                    "Image A is the scan reference. Image B is the live robot image."
                ),
            },
            {
                "type": "input_text",
                "text": "Image A: scan reference",
            },
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + encode_jpeg_base64(reference_image),
            },
            {
                "type": "input_text",
                "text": "Image B: current live camera image",
            },
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + encode_jpeg_base64(live_image),
            },
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_visible": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "relative_position": {
                    "type": "string",
                    "enum": ["left", "center", "right", "not_visible", "unknown"],
                },
                "bearing_degrees": {"type": "number", "minimum": -90, "maximum": 90},
                "distance_state": {
                    "type": "string",
                    "enum": ["far", "approach", "near", "too_close", "unknown"],
                },
                "safe_to_advance": {"type": "boolean"},
                "rationale": {"type": "string", "maxLength": 160},
            },
            "required": [
                "target_visible",
                "confidence",
                "relative_position",
                "bearing_degrees",
                "distance_state",
                "safe_to_advance",
                "rationale",
            ],
        }
        request = {
            "model": self.model_id,
            "instructions": (
                "You are a conservative visual landmark verifier for a mobile robot. "
                "Use only evidence in the two supplied images. A partial or ambiguous "
                "match is not enough for target_visible=true. Never output motor "
                "commands, poses, or invented objects. Estimate qualitative distance "
                "from framing only: far, approach, near, too_close, or unknown. Set "
                "safe_to_advance=false when the landmark is near/too_close, fills the "
                "close foreground, is substantially cropped, or distance is unclear. "
                "A visible landmark is not automatically safe to approach. Return "
                "strict JSON matching the supplied schema."
            ),
            "input": [{"role": "user", "content": content}],
            # Responses max_output_tokens includes reasoning tokens.  Vision
            # calls can otherwise be cut off before the small JSON object is
            # emitted, leaving output_text with an unterminated prefix.
            "max_output_tokens": 1024,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": LANDMARK_SEER_SCHEMA_NAME,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            raise OpenAIRouterError(
                f"OpenAI {self.model_id} landmark seer failed: {exc}"
            ) from exc
        payload = self._decode_structured_response(response, purpose="landmark seer")
        try:
            target_visible = payload["target_visible"]
            confidence = float(payload["confidence"])
            relative_position = str(payload["relative_position"])
            bearing_degrees = float(payload["bearing_degrees"])
            distance_state = str(payload["distance_state"])
            safe_to_advance = payload["safe_to_advance"]
            rationale = str(payload["rationale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenAIRouterError(
                "OpenAI landmark seer returned invalid structured fields"
            ) from exc
        if not isinstance(target_visible, bool):
            raise OpenAIRouterError("OpenAI landmark seer returned invalid visibility")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise OpenAIRouterError("OpenAI landmark seer returned invalid confidence")
        if relative_position not in {"left", "center", "right", "not_visible", "unknown"}:
            raise OpenAIRouterError("OpenAI landmark seer returned invalid position")
        if not math.isfinite(bearing_degrees) or not -90.0 <= bearing_degrees <= 90.0:
            raise OpenAIRouterError("OpenAI landmark seer returned invalid bearing")
        if distance_state not in {"far", "approach", "near", "too_close", "unknown"}:
            raise OpenAIRouterError("OpenAI landmark seer returned invalid distance")
        if not isinstance(safe_to_advance, bool):
            raise OpenAIRouterError("OpenAI landmark seer returned invalid advance flag")
        if not target_visible:
            relative_position = "not_visible"
            bearing_degrees = 0.0
            distance_state = "unknown"
            safe_to_advance = False
        if distance_state in {"near", "too_close", "unknown"}:
            safe_to_advance = False
        return LandmarkSeerResult(
            target_visible=target_visible,
            confidence=confidence,
            relative_position=relative_position,
            bearing_degrees=bearing_degrees,
            distance_state=distance_state,
            safe_to_advance=safe_to_advance,
            rationale=rationale.strip(),
        )

    def see_goal(
        self,
        *,
        item_name: str,
        item_description: str,
        landmark_name: str,
        landmark_description: str,
        reference_image: Any,
        live_image: Any,
    ) -> GoalSeerResult:
        """Verify the current landmark and requested item in the live image.

        The landmark reference is used only to reacquire the search location.
        The item must be visible in the live image before this result can be
        treated as a successful find.  This is perception and state estimation;
        it deliberately does not return motor commands.
        """

        item_name = str(item_name).strip()
        item_description = str(item_description).strip()
        landmark_name = str(landmark_name).strip()
        landmark_description = str(landmark_description).strip()
        if not item_name:
            raise OpenAIRouterError("goal seer item name must not be empty")
        if not landmark_name:
            raise OpenAIRouterError("goal seer landmark name must not be empty")
        if reference_image is None or live_image is None:
            raise OpenAIRouterError("goal seer requires reference and live images")

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Compare the stable-landmark reference image with the current "
                    "robot-camera image. Verify two things independently: whether "
                    "the same landmark is visible, and whether the requested item "
                    "is clearly visible in the live image. The item may be absent "
                    "from the reference image; never infer that the item is present "
                    "from the text or reference. Only set item_visible=true when "
                    "the item is visually identifiable in the live image, preferably "
                    "on an accessible surface or nearby floor in this search area. "
                    "Estimate landmark bearing with negative=left and positive=right. "
                    "Estimate landmark distance from framing only. Set safe_to_advance "
                    "false when the landmark is near, too close, cropped, or its "
                    "distance is unclear. Return concise strict JSON."
                ),
            },
            {
                "type": "input_text",
                "text": (
                    f"Requested item: {item_name}\n"
                    f"Item description: {item_description or 'none'}\n"
                    f"Landmark name: {landmark_name}\n"
                    f"Landmark description: {landmark_description or 'none'}\n"
                    "Image A is the scan reference for the landmark. Image B is the "
                    "current live robot image."
                ),
            },
            {"type": "input_text", "text": "Image A: landmark scan reference"},
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + encode_jpeg_base64(reference_image),
            },
            {"type": "input_text", "text": "Image B: current live camera image"},
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + encode_jpeg_base64(live_image),
            },
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "landmark_visible": {"type": "boolean"},
                "landmark_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "landmark_relative_position": {
                    "type": "string",
                    "enum": ["left", "center", "right", "not_visible", "unknown"],
                },
                "landmark_bearing_degrees": {
                    "type": "number",
                    "minimum": -90,
                    "maximum": 90,
                },
                "landmark_distance_state": {
                    "type": "string",
                    "enum": ["far", "approach", "near", "too_close", "unknown"],
                },
                "safe_to_advance": {"type": "boolean"},
                "item_visible": {"type": "boolean"},
                "item_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "item_relative_position": {
                    "type": "string",
                    "enum": ["left", "center", "right", "not_visible", "unknown"],
                },
                "item_bearing_degrees": {
                    "type": "number",
                    "minimum": -90,
                    "maximum": 90,
                },
                "rationale": {"type": "string", "maxLength": 180},
            },
            "required": [
                "landmark_visible",
                "landmark_confidence",
                "landmark_relative_position",
                "landmark_bearing_degrees",
                "landmark_distance_state",
                "safe_to_advance",
                "item_visible",
                "item_confidence",
                "item_relative_position",
                "item_bearing_degrees",
                "rationale",
            ],
        }
        request = {
            "model": self.model_id,
            "instructions": (
                "You are the goal verifier for a mobile indoor robot. Use only the "
                "two supplied images. Never output motor commands, coordinates, or "
                "invented objects. A visible landmark is not automatically safe to "
                "approach, and an item is not found merely because the request names "
                "it. Return strict JSON matching the supplied schema."
            ),
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 1536,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "memory_guide_goal_seer",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            raise OpenAIRouterError(
                f"OpenAI {self.model_id} goal seer failed: {exc}"
            ) from exc
        payload = self._decode_structured_response(response, purpose="goal seer")
        try:
            landmark_visible = payload["landmark_visible"]
            landmark_confidence = float(payload["landmark_confidence"])
            landmark_relative_position = str(payload["landmark_relative_position"])
            landmark_bearing_degrees = float(payload["landmark_bearing_degrees"])
            landmark_distance_state = str(payload["landmark_distance_state"])
            safe_to_advance = payload["safe_to_advance"]
            item_visible = payload["item_visible"]
            item_confidence = float(payload["item_confidence"])
            item_relative_position = str(payload["item_relative_position"])
            item_bearing_degrees = float(payload["item_bearing_degrees"])
            rationale = str(payload["rationale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenAIRouterError(
                "OpenAI goal seer returned invalid structured fields"
            ) from exc

        if not isinstance(landmark_visible, bool) or not isinstance(item_visible, bool):
            raise OpenAIRouterError("OpenAI goal seer returned invalid visibility")
        for value, label in (
            (landmark_confidence, "landmark confidence"),
            (item_confidence, "item confidence"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise OpenAIRouterError(f"OpenAI goal seer returned invalid {label}")
        valid_positions = {"left", "center", "right", "not_visible", "unknown"}
        if landmark_relative_position not in valid_positions:
            raise OpenAIRouterError("OpenAI goal seer returned invalid landmark position")
        if item_relative_position not in valid_positions:
            raise OpenAIRouterError("OpenAI goal seer returned invalid item position")
        if not math.isfinite(landmark_bearing_degrees) or not -90.0 <= landmark_bearing_degrees <= 90.0:
            raise OpenAIRouterError("OpenAI goal seer returned invalid landmark bearing")
        if not math.isfinite(item_bearing_degrees) or not -90.0 <= item_bearing_degrees <= 90.0:
            raise OpenAIRouterError("OpenAI goal seer returned invalid item bearing")
        if landmark_distance_state not in {"far", "approach", "near", "too_close", "unknown"}:
            raise OpenAIRouterError("OpenAI goal seer returned invalid landmark distance")
        if not isinstance(safe_to_advance, bool):
            raise OpenAIRouterError("OpenAI goal seer returned invalid advance flag")
        if not landmark_visible:
            landmark_relative_position = "not_visible"
            landmark_bearing_degrees = 0.0
            landmark_distance_state = "unknown"
            safe_to_advance = False
        if not item_visible:
            item_relative_position = "not_visible"
            item_bearing_degrees = 0.0
        if landmark_distance_state in {"near", "too_close", "unknown"}:
            safe_to_advance = False
        return GoalSeerResult(
            landmark_visible=landmark_visible,
            landmark_confidence=landmark_confidence,
            landmark_relative_position=landmark_relative_position,
            landmark_bearing_degrees=landmark_bearing_degrees,
            landmark_distance_state=landmark_distance_state,
            safe_to_advance=safe_to_advance,
            item_visible=item_visible,
            item_confidence=item_confidence,
            item_relative_position=item_relative_position,
            item_bearing_degrees=item_bearing_degrees,
            rationale=rationale.strip(),
        )

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
