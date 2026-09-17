import json

from PIL import Image

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


def test_openai_router_landmark_seer_returns_conservative_structured_visibility():
    client = FakeOpenAIClient(
        {
            "target_visible": True,
            "confidence": 0.92,
            "relative_position": "left",
            "bearing_degrees": -24.0,
            "distance_state": "approach",
            "safe_to_advance": True,
            "rationale": "The same gray case is visible on the left side.",
        }
    )
    router = OpenAIQueryRouter(client)
    result = router.see_landmark(
        target_name="Large gray equipment case",
        target_description="Large open gray hard case",
        reference_image=Image.new("RGB", (12, 8), "gray"),
        live_image=Image.new("RGB", (12, 8), "gray"),
    )

    request = client.responses.request
    assert result.target_visible is True
    assert result.confidence == 0.92
    assert result.relative_position == "left"
    assert result.bearing_degrees == -24.0
    assert result.distance_state == "approach"
    assert result.safe_to_advance is True
    assert request["text"]["format"]["name"] == "memory_guide_landmark_seer"
    assert request["text"]["format"]["strict"] is True
    assert sum(
        item["type"] == "input_image"
        for item in request["input"][0]["content"]
    ) == 2


def test_openai_router_landmark_seer_clears_bearing_when_target_is_absent():
    client = FakeOpenAIClient(
        {
            "target_visible": False,
            "confidence": 0.99,
            "relative_position": "unknown",
            "bearing_degrees": 12.0,
            "distance_state": "unknown",
            "safe_to_advance": True,
            "rationale": "The landmark is not in the live image.",
        }
    )
    result = OpenAIQueryRouter(client).see_landmark(
        target_name="case",
        target_description="gray equipment case",
        reference_image=Image.new("RGB", (4, 4), "black"),
        live_image=Image.new("RGB", (4, 4), "white"),
    )

    assert result.target_visible is False
    assert result.relative_position == "not_visible"
    assert result.bearing_degrees == 0.0
    assert result.distance_state == "unknown"
    assert result.safe_to_advance is False


def test_openai_router_goal_seer_verifies_item_separately_from_landmark():
    client = FakeOpenAIClient(
        {
            "landmark_visible": True,
            "landmark_confidence": 0.98,
            "landmark_relative_position": "center",
            "landmark_bearing_degrees": 2.0,
            "landmark_distance_state": "near",
            "safe_to_advance": False,
            "item_visible": True,
            "item_confidence": 0.91,
            "item_relative_position": "right",
            "item_bearing_degrees": 18.0,
            "item_distance_state": "near",
            "item_accessible": True,
            "item_safe_to_advance": False,
            "rationale": "The white bottle is visible on the case.",
        }
    )
    result = OpenAIQueryRouter(client).see_goal(
        item_name="white water bottle",
        item_description="white reusable bottle",
        landmark_name="Gray wheeled equipment case",
        landmark_description="large gray wheeled case",
        reference_image=Image.new("RGB", (12, 8), "gray"),
        live_image=Image.new("RGB", (12, 8), "white"),
    )

    request = client.responses.request
    assert result.item_visible is True
    assert result.item_confidence == 0.91
    assert result.item_relative_position == "right"
    assert result.item_distance_state == "near"
    assert result.item_accessible is True
    assert result.item_safe_to_advance is False
    assert result.safe_to_advance is False
    assert request["text"]["format"]["name"] == "memory_guide_goal_seer"
    assert request["text"]["format"]["strict"] is True
    prompt_text = request["input"][0]["content"][1]["text"]
    assert "white water bottle" in prompt_text
    assert "Gray wheeled equipment case" in prompt_text


def test_openai_router_goal_seer_keeps_a_far_item_as_unfound():
    client = FakeOpenAIClient(
        {
            "landmark_visible": True,
            "landmark_confidence": 0.98,
            "landmark_relative_position": "center",
            "landmark_bearing_degrees": 0.0,
            "landmark_distance_state": "approach",
            "safe_to_advance": True,
            "item_visible": True,
            "item_confidence": 0.95,
            "item_relative_position": "center",
            "item_bearing_degrees": 0.0,
            "item_distance_state": "far",
            "item_accessible": True,
            "item_safe_to_advance": True,
            "rationale": "The bottle is visible on the far side of the case.",
        }
    )

    result = OpenAIQueryRouter(client).see_goal(
        item_name="water bottle",
        item_description="white reusable bottle",
        landmark_name="equipment case",
        landmark_description="large gray case",
        reference_image=Image.new("RGB", (4, 4), "gray"),
        live_image=Image.new("RGB", (4, 4), "white"),
    )

    assert result.item_visible is True
    assert result.item_distance_state == "far"
    assert result.item_accessible is True
    assert result.item_safe_to_advance is True
    required = client.responses.request["text"]["format"]["schema"]["required"]
    assert "item_distance_state" in required
    assert "item_accessible" in required
    assert "item_safe_to_advance" in required


def test_openai_router_accepts_fenced_structured_output():
    response = type(
        "FakeResponse",
        (),
        {
            "output_text": (
                '```json\n{"landmark_ids":["landmark_1"],'
                '"rationale":"surface"}\n```'
            )
        },
    )()
    payload = OpenAIQueryRouter._decode_structured_response(
        response, purpose="test"
    )

    assert payload == {"landmark_ids": ["landmark_1"], "rationale": "surface"}
