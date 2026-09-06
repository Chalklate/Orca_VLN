import json

from navila_orca.memory_guide import load_catalog, parse_query, plan_query
from navila_orca.openai_router import OpenAIQueryRouter


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.request = None

    def create(self, **request):
        self.request = request
        return type("FakeResponse", (), {"output_text": json.dumps(self.payload)})()


class FakeOpenAIClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def test_openai_router_uses_strict_responses_schema_for_open_vocabulary_item():
    client = FakeOpenAIClient(
        {
            "intent": "find_item",
            "target_type": "item",
            "target_id": "bread",
            "confidence": 0.98,
            "rationale": "The resident is asking for an item location.",
        }
    )
    router = OpenAIQueryRouter(client)
    parsed = parse_query(
        "Where did I put my bread?",
        load_catalog(),
        llm_mode="openai",
        openai_router=router,
    )

    request = client.responses.request
    assert parsed.intent == "find_item"
    assert parsed.target == "bread"
    assert parsed.router == "openai:gpt-5.6-luna"
    assert request["model"] == "gpt-5.6-luna"
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True


def test_openai_router_feeds_existing_deterministic_planner():
    client = FakeOpenAIClient(
        {
            "intent": "find_item",
            "target_type": "item",
            "target_id": "bread",
            "confidence": 0.9,
            "rationale": "item lookup",
        }
    )
    plan = plan_query(
        "Where did I put my bread?",
        catalog=load_catalog(),
        inventory={"version": 1, "observations": {}},
        llm_mode="openai",
        openai_router=OpenAIQueryRouter(client),
    )

    assert plan["query_router"] == "openai:gpt-5.6-luna"
    assert plan["target"] == "bread"
    assert plan["mode"] == "patrol_item"
    assert plan["waypoints"]
