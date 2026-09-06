"""OpenAI Responses API query routing for the Memory Guide."""

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


DEFAULT_MODEL_ID = "gpt-5.6-luna"
SCHEMA_NAME = "memory_guide_route"


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
