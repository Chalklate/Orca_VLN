import importlib.util
from pathlib import Path

from navila_orca.openai_router import GoalSeerResult


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_unitree_navila.py"
SPEC = importlib.util.spec_from_file_location("run_unitree_navila", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_visible_item_on_near_search_surface_counts_as_found():
    result = GoalSeerResult(
        landmark_visible=True,
        landmark_confidence=0.95,
        landmark_relative_position="center",
        landmark_bearing_degrees=0.0,
        landmark_distance_state="near",
        safe_to_advance=False,
        item_visible=True,
        item_confidence=0.96,
        item_relative_position="right",
        item_bearing_degrees=12.0,
        item_distance_state="near",
        item_accessible=False,
        item_safe_to_advance=False,
        rationale="Visible on the nearby table.",
    )

    assert MODULE._goal_item_found(result, minimum_confidence=0.70) is True
