import json

from navila_orca.memory_guide import load_catalog, plan_query
from navila_orca.semantic_map import (
    build_semantic_map,
    build_semantic_map_from_file,
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
