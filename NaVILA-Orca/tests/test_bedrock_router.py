import json

from navila_orca.bedrock_router import BedrockQueryRouter, TOOL_NAME
from navila_orca.memory_guide import load_catalog, parse_query, plan_query


class FakeBedrockClient:
    def __init__(self, decision, *, structured=False):
        self.decision = decision
        self.structured = structured
        self.request = None

    def converse(self, **request):
        self.request = request
        if self.structured:
            return {
                "output": {
                    "message": {
                        "content": [{"text": json.dumps(self.decision)}]
                    }
                }
            }
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": TOOL_NAME,
                                "toolUseId": "tool-1",
                                "input": self.decision,
                            }
                        }
                    ]
                }
            }
        }


def test_nova_micro_router_extracts_open_vocabulary_item(catalog=None):
    catalog = catalog or load_catalog()
    client = FakeBedrockClient(
        {
            "intent": "find_item",
            "target_type": "item",
            "target_id": "bread",
            "confidence": 0.97,
            "rationale": "The resident is asking for an item location.",
        }
    )
    router = BedrockQueryRouter(client, model_id="amazon.nova-micro-v1:0")

    parsed = parse_query(
        "Where did I put my bread?",
        catalog,
        llm_mode="bedrock",
        bedrock_router=router,
    )

    assert parsed.intent == "find_item"
    assert parsed.target == "bread"
    assert parsed.router == "bedrock:amazon.nova-micro-v1:0"
    assert parsed.confidence == 0.97
    assert client.request["modelId"] == "amazon.nova-micro-v1:0"
    assert client.request["toolConfig"]["toolChoice"] == {
        "tool": {"name": TOOL_NAME}
    }
    assert "strict" not in client.request["toolConfig"]["tools"][0]["toolSpec"]


def test_claude_haiku_router_uses_structured_output(catalog=None):
    catalog = catalog or load_catalog()
    client = FakeBedrockClient(
        {
            "intent": "find_item",
            "target_type": "item",
            "target_id": "bread",
            "confidence": 0.97,
            "rationale": "item lookup",
        },
        structured=True,
    )
    router = BedrockQueryRouter(
        client,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    decision = router.route("Where is my bread?", catalog)
    assert decision.intent == "find_item"
    assert decision.target_id == "bread"
    assert "toolConfig" not in client.request
    assert client.request["inferenceConfig"] == {
        "maxTokens": 160,
        "temperature": 0.0,
    }
    schema = client.request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
    assert schema["name"] == "memory_guide_route"
    assert json.loads(schema["schema"])["required"]


def test_bedrock_repairs_omitted_redundant_route_fields(catalog=None):
    catalog = catalog or load_catalog()
    client = FakeBedrockClient(
        {
            "target_type": "item",
            "target_id": "bread",
        }
    )
    decision = BedrockQueryRouter(
        client, model_id="amazon.nova-micro-v1:0"
    ).route("Where is my bread?", catalog)
    assert decision.intent == "find_item"
    assert decision.target_type == "item"
    assert decision.target_id == "bread"
    assert decision.confidence == 0.0


def test_bedrock_target_still_uses_deterministic_planner(catalog=None):
    catalog = catalog or load_catalog()
    client = FakeBedrockClient(
        {
            "intent": "find_item",
            "target_type": "item",
            "target_id": "bread",
            "confidence": 0.91,
            "rationale": "item lookup",
        }
    )
    plan = plan_query(
        "Where did I put my bread?",
        catalog=catalog,
        inventory={"version": 1, "observations": {}},
        llm_mode="bedrock",
        bedrock_router=BedrockQueryRouter(
            client, model_id="amazon.nova-micro-v1:0"
        ),
    )

    assert plan["query_router"] == "bedrock:amazon.nova-micro-v1:0"
    assert plan["target"] == "bread"
    assert plan["mode"] == "patrol_item"
    assert plan["waypoints"]
