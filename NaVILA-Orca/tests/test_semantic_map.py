import json

from navila_orca.memory_guide import load_catalog, plan_query
from navila_orca.semantic_map import (
    build_semantic_map,
    build_semantic_map_from_file,
    build_semantic_map_from_files,
    plan_topological_route,
    resolve_location,
)


def _teleop_payload():
    return {
        "version": 1,
        "anchors": [
            {
                "label": "dining_table_front",
                "step_id": 1,
                "sim_time_s": 1.0,
                "root_pos_world": [1.0, 2.0, 0.0],
                "base_rpy": [0.0, 0.0, 0.0],
                "image_path": "anchors/000_dining_table_front.png",
            },
            {
                "label": "dining_table_left",
                "step_id": 2,
                "sim_time_s": 2.0,
                "root_pos_world": [1.1, 2.0, 0.0],
                "base_rpy": [0.0, 0.0, 1.57],
                "image_path": "anchors/001_dining_table_left.png",
            },
            {
                "label": "dining_table_rear",
                "step_id": 3,
                "sim_time_s": 3.0,
                "root_pos_world": [1.0, 2.1, 0.0],
                "base_rpy": [0.0, 0.0, 3.14],
                "image_path": "anchors/002_dining_table_rear.png",
            },
            {
                "label": "dining_table_right",
                "step_id": 4,
                "sim_time_s": 4.0,
                "root_pos_world": [0.9, 2.0, 0.0],
                "base_rpy": [0.0, 0.0, -1.57],
                "image_path": "anchors/003_dining_table_right.png",
            },
        ],
    }


def _route_teleop_payload():
    payload = _teleop_payload()
    payload["anchors"].extend(
        [
            {
                "label": "kitchen_counter_front",
                "step_id": 20,
                "sim_time_s": 8.0,
                "root_pos_world": [4.0, 2.0, 0.0],
                "base_rpy": [0.0, 0.0, 0.0],
                "image_path": "anchors/004_kitchen_counter_front.png",
            },
            {
                "label": "kitchen_counter_left",
                "step_id": 21,
                "sim_time_s": 9.0,
                "root_pos_world": [4.0, 2.0, 0.0],
                "base_rpy": [0.0, 0.0, 1.57],
                "image_path": "anchors/005_kitchen_counter_left.png",
            },
        ]
    )
    payload["command_events"] = [
        {"key": "d", "step_id": 5, "velocity": [0.0, 0.0, -0.8]},
        {"key": "d", "step_id": 5, "velocity": [0.0, 0.0, -0.8]},
        {"key": "w", "step_id": 6, "velocity": [0.5, 0.0, 0.0]},
        {"key": "w", "step_id": 7, "velocity": [0.5, 0.0, 0.0]},
    ]
    return payload


def test_build_map_groups_views_and_preserves_pose_metadata():
    semantic_map = build_semantic_map(_teleop_payload(), source="teleop.json")

    assert semantic_map["location_count"] == 1
    location = semantic_map["locations"]["dining_table"]
    assert [view["view"] for view in location["views"]] == [
        "front",
        "left",
        "rear",
        "right",
    ]
    assert location["views"][1]["base_rpy"][-1] == 1.57
    assert location["center_position"] == [1.0, 2.025, 0.0]


def test_map_location_resolution_fails_closed_on_weak_match():
    semantic_map = build_semantic_map(_teleop_payload())
    resolved = resolve_location("the dining table", semantic_map)
    assert resolved is not None
    assert resolved[0] == "dining_table"
    assert resolve_location("the entryway tray", semantic_map) is None


def test_semantic_map_keeps_one_patrol_waypoint_per_location():
    semantic_map = build_semantic_map(_teleop_payload())
    plan = plan_query(
        "Where is my bread?",
        catalog=load_catalog(),
        inventory={"version": 1, "observations": {}},
        semantic_map=semantic_map,
    )

    assert plan["semantic_map_used"] is True
    assert plan["inspection_policy"] == "recorded_views_requires_visual_verification"
    assert len(plan["waypoints"]) == 4
    assert len(plan["inspection_steps"]) == 4
    assert plan["inspection_steps"][0]["map_location"] == "dining_table"
    assert len(plan["inspection_steps"][0]["views"]) == 4
    assert "complete visual sweep" in plan["waypoints"][0]
    assert "Stop after this search location has been fully inspected" in plan["waypoints"][0]


def test_map_file_round_trip(tmp_path):
    teleop_path = tmp_path / "teleop.json"
    output_path = tmp_path / "semantic_map.json"
    teleop_path.write_text(json.dumps(_teleop_payload()), encoding="utf-8")

    result = build_semantic_map_from_file(teleop_path, output_path)

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8"))["version"] == 1
    assert result["source_teleop"] == str(teleop_path.resolve())


def test_map_builds_directed_route_from_consecutive_location_groups():
    semantic_map = build_semantic_map(_route_teleop_payload())

    assert semantic_map["route_count"] == 1
    route = semantic_map["routes"][0]
    assert route["from_location"] == "dining_table"
    assert route["to_location"] == "kitchen_counter"
    assert route["status"] == "human_demonstrated"
    assert route["instruction_status"] == "generated_unverified"
    assert route["departure"]["view"] == "right"
    assert route["arrival"]["view"] == "front"
    assert route["demonstration"]["command_summary"] == {
        "raw_event_count": 4,
        "unique_event_count": 3,
        "key_counts": {"d": 1, "w": 2},
    }


def test_route_planner_requires_opt_in_for_unverified_reverse():
    semantic_map = build_semantic_map(_route_teleop_payload())

    forward = plan_topological_route(
        semantic_map, "the dining table", "the kitchen counter"
    )
    assert forward["locations"] == ["dining_table", "kitchen_counter"]
    assert forward["uses_unverified_reverse"] is False

    try:
        plan_topological_route(
            semantic_map, "the kitchen counter", "the dining table"
        )
    except ValueError as exc:
        assert "using demonstrated directions only" in str(exc)
    else:
        raise AssertionError("reverse traversal should fail closed by default")

    reverse = plan_topological_route(
        semantic_map,
        "the kitchen counter",
        "the dining table",
        allow_unverified_reverse=True,
    )
    assert reverse["locations"] == ["kitchen_counter", "dining_table"]
    assert reverse["uses_unverified_reverse"] is True


def test_plan_from_known_location_reorders_searches_along_forward_routes():
    semantic_map = build_semantic_map(_route_teleop_payload())
    catalog = load_catalog()
    catalog["default_search_locations"] = [
        "the kitchen counter",
        "the dining table",
    ]

    plan = plan_query(
        "Where is my bread?",
        catalog=catalog,
        inventory={"version": 1, "observations": {}},
        semantic_map=semantic_map,
        current_location="the dining table",
    )

    assert [step["map_location"] for step in plan["inspection_steps"]] == [
        "dining_table",
        "kitchen_counter",
    ]
    assert [role["role"] for role in plan["waypoint_roles"]] == [
        "inspection",
        "route",
        "inspection",
    ]
    assert plan["topological_route"]["start_location"] == "dining_table"
    assert plan["topological_route"]["instruction_status"] == "generated_unverified"


def test_plan_from_known_location_fails_on_missing_route_direction():
    semantic_map = build_semantic_map(_route_teleop_payload())
    catalog = load_catalog()
    catalog["default_search_locations"] = ["the dining table"]

    try:
        plan_query(
            "Where is my bread?",
            catalog=catalog,
            inventory={"version": 1, "observations": {}},
            semantic_map=semantic_map,
            current_location="the kitchen counter",
        )
    except ValueError as exc:
        assert "capture the missing direction" in str(exc)
    else:
        raise AssertionError("mission planning should reject an undemonstrated reverse")


def test_multiple_teleop_sessions_merge_without_inventing_cross_session_route(tmp_path):
    route_session = tmp_path / "route" / "teleop.json"
    spawn_session = tmp_path / "spawn" / "teleop.json"
    route_session.parent.mkdir()
    spawn_session.parent.mkdir()
    route_session.write_text(json.dumps(_route_teleop_payload()), encoding="utf-8")
    spawn_payload = {
        "version": 1,
        "anchors": [
            {
                "label": "spawnpoint_front",
                "step_id": 1,
                "sim_time_s": 0.1,
                "root_pos_world": [0.0, 0.0, 0.0],
                "base_rpy": [0.0, 0.0, 0.0],
                "image_path": "anchors/000_spawnpoint_front.png",
            }
        ],
    }
    spawn_session.write_text(json.dumps(spawn_payload), encoding="utf-8")

    semantic_map = build_semantic_map_from_files(
        [route_session, spawn_session], tmp_path / "semantic_map.json"
    )

    assert semantic_map["anchor_count"] == 7
    assert semantic_map["location_count"] == 3
    assert semantic_map["route_count"] == 1
    assert semantic_map["source_teleops"] == [
        str(route_session.resolve()),
        str(spawn_session.resolve()),
    ]
    spawn_view = semantic_map["locations"]["spawnpoint"]["views"][0]
    assert spawn_view["anchor_id"].startswith("session_001_spawn::")
    assert spawn_view["image_path"] == str(
        (spawn_session.parent / "anchors/000_spawnpoint_front.png").resolve()
    )
    try:
        plan_topological_route(semantic_map, "spawnpoint", "dining table")
    except ValueError as exc:
        assert "no route" in str(exc)
    else:
        raise AssertionError("merging disconnected sessions must not invent a route")
