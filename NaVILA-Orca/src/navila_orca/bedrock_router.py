"""Optional Amazon Bedrock query routing for the Memory Guide.

Nova Micro is used here as a small text-only classifier/entity normalizer. It
does not plan robot motion and it never receives permission to invent poses or
semantic-map entries. The caller still validates the returned target against
the local catalog before constructing a navigation plan.

The Bedrock client is imported lazily so the normal deterministic Memory Guide
installation does not need boto3.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from .query_router import (
    ROUTER_SYSTEM_PROMPT,
    RouteDecision,
    route_schema,
    router_user_prompt,
    validate_route_decision,
)


DEFAULT_MODEL_ID = "amazon.nova-micro-v1:0"
DEFAULT_PROFILE = "683803166476_hack2026_IsbUsersPS"
DEFAULT_REGION = "ap-southeast-1"
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

        system_prompt = ROUTER_SYSTEM_PROMPT + " Always call the provided tool exactly once."
        user_prompt = router_user_prompt(original, catalog)
        request = {
            "modelId": self.model_id,
            "system": [{"text": system_prompt}],
            "messages": [
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            "inferenceConfig": {
                "maxTokens": 160,
                "temperature": 0.0,
                "topP": 0.9,
            },
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": TOOL_NAME,
                            "description": "Return the validated Memory Guide route fields.",
                            "inputSchema": {"json": route_schema()},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": TOOL_NAME}},
            },
        }
        try:
            response = self.client.converse(**request)
        except Exception as exc:
            raise BedrockRouterError(
                f"Bedrock Nova Micro query routing failed: {exc}"
            ) from exc

        payload = self._extract_tool_input(response)
        return self._validate_decision(payload)

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
    def _validate_decision(payload: Mapping[str, Any]) -> BedrockDecision:
        return validate_route_decision(
            payload,
            error_type=BedrockRouterError,
            provider_name="Bedrock",
        )
