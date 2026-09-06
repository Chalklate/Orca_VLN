from navila_orca.bedrock_router import BedrockQueryRouter, TOOL_NAME
from navila_orca.memory_guide import load_catalog, parse_query, plan_query


class FakeBedrockClient:
    def __init__(self, decision):
        self.decision = decision
        self.request = None

    def converse(self, **request):
        self.request = request
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
    router = BedrockQueryRouter(client)

    parsed = parse_query(
        "Where did I put my bread?",
        catalog,
        llm_mode="bedrock",
        bedrock_router=router,
    )

    assert parsed.intent == "find_item"
    assert parsed.target == "bread"
    assert parsed.router == "bedrock:nova-micro"
    assert parsed.confidence == 0.97
    assert client.request["modelId"] == "amazon.nova-micro-v1:0"
    assert client.request["toolConfig"]["toolChoice"] == {
        "tool": {"name": TOOL_NAME}
    }


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
        bedrock_router=BedrockQueryRouter(client),
    )

    assert plan["query_router"] == "bedrock:nova-micro"
    assert plan["target"] == "bread"
    assert plan["mode"] == "patrol_item"
    assert plan["waypoints"]
