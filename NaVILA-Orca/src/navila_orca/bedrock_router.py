"""Optional Amazon Bedrock query routing for the Memory Guide.

Nova Micro is used here as a small text-only classifier/entity normalizer. For
Claude Haiku 4.5, the router uses Bedrock Converse structured JSON output.
Neither provider plans robot motion or receives permission to invent poses or
semantic-map entries. The caller still validates the returned target against
the local catalog before constructing a navigation plan.

The Bedrock client is imported lazily so the normal deterministic Memory Guide
installation does not need boto3.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Mapping

from .query_router import (
    ROUTER_SYSTEM_PROMPT,
    RouteDecision,
    route_schema,
    router_user_prompt,
    validate_route_decision,
)


DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_PROFILE = "683803166476_hack2026_IsbUsersPS"
DEFAULT_REGION = "us-east-1"
TOOL_NAME = "route_memory_query"


class BedrockRouterError(ValueError):
    """Raised when the Bedrock router cannot produce a safe route decision."""


BedrockDecision = RouteDecision


class BedrockQueryRouter:
    """Route Memory Guide text through Amazon Nova Micro."""

    def __init__(self, client: Any, *, model_id: str = DEFAULT_MODEL_ID) -> None:
        model_id = str(model_id).strip()
        if not model_id:
            raise BedrockRouterError("Bedrock model ID must not be empty")
        self.client = client
        self.model_id = model_id

    @staticmethod
    def _debug_enabled() -> bool:
        return os.environ.get("NAVILA_BEDROCK_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @classmethod
    def _debug(cls, label: str, value: Any) -> None:
        """Print safe, opt-in routing diagnostics to stderr."""

        if not cls._debug_enabled():
            return
        try:
            rendered = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = repr(value)
        print(f"[bedrock-debug] {label}: {rendered}", file=sys.stderr)

    def _uses_structured_output(self) -> bool:
        """Return whether this model supports Bedrock Converse JSON output."""

        return "anthropic.claude-haiku-4-5" in self.model_id.lower()

    @classmethod
    def from_environment(
        cls,
        *,
        region: str | None = None,
        model_id: str | None = None,
        profile: str | None = None,
    ) -> "BedrockQueryRouter":
        """Build a boto3 Bedrock Runtime client from normal AWS configuration."""

        try:
            import boto3
        except ImportError as exc:
            raise BedrockRouterError(
                "Bedrock mode needs boto3; install the optional dependency with "
                "python -m pip install -e '.[bedrock]'"
            ) from exc

        selected_region = (
            region
            or os.environ.get("NAVILA_BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_REGION
        )
        selected_profile = (
            profile
            or os.environ.get("NAVILA_BEDROCK_PROFILE")
            or DEFAULT_PROFILE
        )
        session_kwargs: dict[str, Any] = {}
        if selected_profile:
            session_kwargs["profile_name"] = selected_profile
        if selected_region:
            session_kwargs["region_name"] = selected_region

        try:
            session = boto3.Session(**session_kwargs)
            resolved_region = session.region_name
            if not resolved_region:
                raise BedrockRouterError(
                    "Bedrock region is unavailable; set --bedrock-region, "
                    "NAVILA_BEDROCK_REGION, or AWS_DEFAULT_REGION"
                )
            client = session.client("bedrock-runtime", region_name=resolved_region)
        except BedrockRouterError:
            raise
        except Exception as exc:
            raise BedrockRouterError(f"could not configure Bedrock Runtime: {exc}") from exc

        return cls(
            client,
            model_id=(
                model_id
                or os.environ.get("NAVILA_BEDROCK_MODEL_ID")
                or DEFAULT_MODEL_ID
            ),
        )

    def route(self, query: str, catalog: Mapping[str, Any]) -> BedrockDecision:
        original = str(query).strip()
        if not original:
            raise BedrockRouterError("query must not be empty")

        structured_output = self._uses_structured_output()
        system_prompt = ROUTER_SYSTEM_PROMPT
        if structured_output:
            system_prompt += (
                " Return exactly one JSON object matching the supplied schema. "
                "Do not include markdown or explanatory text."
            )
        else:
            system_prompt += " Always call the provided tool exactly once."
        user_prompt = router_user_prompt(original, catalog)
        inference_config: dict[str, Any] = {
            "maxTokens": 160,
            "temperature": 0.0,
        }
        # Anthropic Claude rejects requests that specify both temperature and
        # topP. Nova's existing tool-call path keeps topP for its decoder.
        if not structured_output:
            inference_config["topP"] = 0.9

        request = {
            "modelId": self.model_id,
            "system": [{"text": system_prompt}],
            "messages": [
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            "inferenceConfig": inference_config,
        }
        if structured_output:
            request["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": "memory_guide_route",
                            "description": "Return the validated Memory Guide route fields.",
                            "schema": json.dumps(
                                route_schema(),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    },
                }
            }
        else:
            tool_spec: dict[str, Any] = {
                "name": TOOL_NAME,
                "description": "Return the validated Memory Guide route fields.",
                "inputSchema": {"json": route_schema()},
            }
            request["toolConfig"] = {
                "tools": [{"toolSpec": tool_spec}],
                "toolChoice": {"tool": {"name": TOOL_NAME}},
            }
        self._debug(
            "request",
            {
                "model_id": self.model_id,
                "mode": "converse_json_schema" if structured_output else "tool_call",
                "tool_choice": request.get("toolConfig", {}).get("toolChoice"),
                "strict_tool_schema": request.get("toolConfig", {})
                .get("tools", [{}])[0]
                .get("toolSpec", {})
                .get("strict", False),
                "structured_output": structured_output,
                "query": original,
            },
        )
        try:
            response = self.client.converse(**request)
        except Exception as exc:
            raise BedrockRouterError(
                f"Bedrock model {self.model_id} query routing failed: {exc}"
            ) from exc

        response_output = response.get("output", {}) if isinstance(response, Mapping) else {}
        response_message = (
            response_output.get("message", {})
            if isinstance(response_output, Mapping)
            else {}
        )
        self._debug(
            "response",
            {
                "stop_reason": response.get("stopReason")
                if isinstance(response, Mapping)
                else None,
                "content": response_message.get("content")
                if isinstance(response_message, Mapping)
                else None,
                "usage": response.get("usage") if isinstance(response, Mapping) else None,
            },
        )
        if structured_output:
            payload = self._extract_structured_output(response)
            self._debug("structured_output_payload", payload)
        else:
            payload = self._extract_tool_input(response)
            self._debug("tool_input", payload)
        repaired_payload = self._repair_tool_input(payload, original)
        self._debug("repaired_tool_input", repaired_payload)
        return self._validate_decision(repaired_payload)

    @staticmethod
    def _extract_tool_input(response: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            content = response["output"]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BedrockRouterError("Bedrock response did not contain a model message") from exc
        if not isinstance(content, list):
            raise BedrockRouterError("Bedrock response message content was not a list")

        for block in content:
            if isinstance(block, Mapping) and isinstance(block.get("toolUse"), Mapping):
                tool_use = block["toolUse"]
                if tool_use.get("name") != TOOL_NAME:
                    continue
                input_payload = tool_use.get("input")
                if isinstance(input_payload, Mapping):
                    return input_payload
                raise BedrockRouterError("Bedrock tool call input was not an object")

        # Tolerate a proxy that strips tool metadata while still requiring JSON
        # and applying the same validation below.
        for block in content:
            if not isinstance(block, Mapping) or not isinstance(block.get("text"), str):
                continue
            try:
                decoded = json.loads(block["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, Mapping):
                return decoded
        raise BedrockRouterError("Bedrock response did not contain the routing tool call")

    @staticmethod
    def _extract_structured_output(response: Mapping[str, Any]) -> Mapping[str, Any]:
        """Extract the JSON object returned by Converse outputConfig."""

        try:
            content = response["output"]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BedrockRouterError(
                "Bedrock structured response did not contain a model message"
            ) from exc
        if not isinstance(content, list):
            raise BedrockRouterError(
                "Bedrock structured response content was not a list"
            )

        for block in content:
            if not isinstance(block, Mapping) or not isinstance(block.get("text"), str):
                continue
            try:
                decoded = json.loads(block["text"])
            except json.JSONDecodeError as exc:
                raise BedrockRouterError(
                    "Bedrock structured response was not valid JSON"
                ) from exc
            if isinstance(decoded, Mapping):
                return decoded
        raise BedrockRouterError(
            "Bedrock structured response did not contain a JSON object"
        )

    @staticmethod
    def _validate_decision(payload: Mapping[str, Any]) -> BedrockDecision:
        return validate_route_decision(
            payload,
            error_type=BedrockRouterError,
            provider_name="Bedrock",
        )

    @staticmethod
    def _repair_tool_input(
        payload: Mapping[str, Any], query: str
    ) -> Mapping[str, Any]:
        """Repair fields that non-strict Bedrock tool calls may omit.

        Nova's non-strict tool output sometimes contains the discriminating
        target fields but omits redundant metadata. Deriving the intent from
        ``target_type`` is deterministic and keeps the provider from inventing
        navigation semantics. A missing confidence is represented as zero,
        meaning unknown—not high confidence.
        """

        repaired = dict(payload)
        target_type = repaired.get("target_type")
        target_id = repaired.get("target_id")
        if repaired.get("intent") is None:
            if target_type == "item" and isinstance(target_id, str) and target_id.strip():
                repaired["intent"] = "find_item"
            elif target_type == "place" and isinstance(target_id, str) and target_id.strip():
                repaired["intent"] = "navigate_place"
            elif target_type == "none":
                normalized_query = str(query).lower()
                if any(
                    phrase in normalized_query
                    for phrase in ("patrol", "look around", "scan the house", "scan the home")
                ):
                    repaired["intent"] = "patrol"
                elif any(
                    phrase in normalized_query
                    for phrase in ("what do i need", "checklist", "have everything")
                ):
                    repaired["intent"] = "leaving_checklist"
                else:
                    repaired["intent"] = "unsupported"
        if repaired.get("confidence") is None:
            repaired["confidence"] = 0.0
        if repaired.get("rationale") is None:
            repaired["rationale"] = "Bedrock omitted rationale; route fields were repaired locally."
        return repaired
