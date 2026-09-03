from datetime import datetime, timedelta, timezone
import json

import pytest

from navila_orca.memory_guide import (
    load_catalog,
    load_inventory,
    main,
    observation_score,
    parse_query,
    plan_query,
    remember_item,
    save_inventory,
)


@pytest.fixture
def catalog():
    return load_catalog()


def test_query_resolves_find_item_alias(catalog):
    parsed = parse_query("Could you find my spectacles?", catalog)
    assert parsed.intent == "find_item"
    assert parsed.target == "glasses"


def test_query_resolves_known_destination_before_item(catalog):
    parsed = parse_query("Guide me to the bathroom", catalog)
    assert parsed.intent == "navigate_place"
    assert parsed.target == "bathroom"


def test_recent_confident_memory_routes_directly(catalog):
    inventory = {"version": 1, "observations": {}}
    remember_item(
        inventory,
        item_id="glasses",
        location="the kitchen-facing side of the dining table",
        confidence=0.95,
        observed_at="2026-09-03T10:00:00Z",
    )
    plan = plan_query(
        "Where are my glasses?",
        catalog=catalog,
        inventory=inventory,
        now=datetime(2026, 9, 3, 10, 30, tzinfo=timezone.utc),
    )
    assert plan["mode"] == "direct_item"
    assert plan["memory"]["reliable"] is True
    assert len(plan["waypoints"]) == 1
    assert "dining table" in plan["waypoints"][0]


def test_unknown_item_routes_to_catalog_patrol(catalog):
    plan = plan_query(
        "Where is my phone?",
        catalog=catalog,
        inventory={"version": 1, "observations": {}},
    )
    assert plan["mode"] == "patrol_item"
    assert plan["reason"] == "item is not yet in memory"
    assert len(plan["waypoints"]) == 3


def test_stale_observation_routes_to_patrol(catalog):
    inventory = {
        "version": 1,
        "observations": {
            "keys": {
                "location": "the entryway tray",
                "room": "entryway",
                "confidence": 1.0,
                "observed_at": "2026-09-01T10:00:00Z",
                "evidence_frame": None,
            }
        },
    }
    plan = plan_query(
        "Find my keys",
        catalog=catalog,
        inventory=inventory,
        now=datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc),
    )
    assert plan["mode"] == "patrol_item"
    assert plan["memory"]["reliable"] is False


def test_inventory_round_trip_is_persistent(tmp_path):
    path = tmp_path / "inventory.json"
    inventory = load_inventory(path)
    remember_item(inventory, item_id="wallet", location="the entryway tray")
    save_inventory(inventory, path)
    assert load_inventory(path)["observations"]["wallet"]["location"] == (
        "the entryway tray"
    )


def test_observation_confidence_decays_by_half_after_one_day():
    now = datetime(2026, 9, 3, 10, tzinfo=timezone.utc)
    observation = {
        "confidence": 0.8,
        "observed_at": (now - timedelta(hours=24)).isoformat(),
    }
    confidence, age_hours = observation_score(observation, now=now)
    assert confidence == pytest.approx(0.4)
    assert age_hours == pytest.approx(24.0)


def test_cli_plan_writes_json_and_waypoints(tmp_path, capsys):
    inventory = tmp_path / "inventory.json"
    plan_path = tmp_path / "plan.json"
    waypoint_path = tmp_path / "waypoints.txt"
    result = main(
        [
            "--inventory",
            str(inventory),
            "plan",
            "--query",
            "Where is my phone?",
            "--plan-output",
            str(plan_path),
            "--waypoint-output",
            str(waypoint_path),
        ]
    )
    assert result == 0
    assert json.loads(plan_path.read_text(encoding="utf-8"))["mode"] == "patrol_item"
    assert len(waypoint_path.read_text(encoding="utf-8").splitlines()) == 3
    assert '"target": "phone"' in capsys.readouterr().out


def test_cli_remember_then_plan_routes_directly(tmp_path):
    inventory = tmp_path / "inventory.json"
    assert main(
        [
            "--inventory",
            str(inventory),
            "remember",
            "--item",
            "glasses",
            "--location",
            "the dining table",
            "--confidence",
            "1.0",
        ]
    ) == 0
    assert load_inventory(inventory)["observations"]["glasses"]["location"] == (
        "the dining table"
    )