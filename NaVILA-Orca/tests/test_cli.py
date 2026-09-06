import numpy as np

from navila_orca.cli import (
    ScriptedVLMClient,
    _build_parser,
    _make_instruction_provider,
    _procedural_rgb,
    _resolve_instruction,
    _resolve_waypoint_instructions,
    main,
)
from navila_orca.contracts import RobotState


def test_procedural_renderer_handles_negative_world_coordinates():
    state = RobotState(
        step_id=3,
        sim_time_s=0.06,
        root_pos_world=np.array([-0.2, 0.0, 0.3]),
        root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        body_ang_vel=np.zeros(3),
        base_rpy=np.array([0.0, 0.0, -0.2]),
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        last_raw_action=np.zeros(12),
    )
    image = _procedural_rgb(state, np.zeros((1, 19)))
    assert image.shape == (512, 512, 3)
    assert image.dtype == np.uint8


def test_scripted_vlm_fails_safe_to_stop_after_sequence():
    client = ScriptedVLMClient(["move forward 25 cm"])
    assert client.infer([], "x") == "move forward 25 cm"
    assert client.infer([], "x") == "stop"


def test_scene_ready_is_fail_closed_before_backend_start(capsys):
    assert main(["run", "--scene-ready"]) == 1
    assert "flat physics" in capsys.readouterr().err


def test_orcalab_run_defaults_preserve_and_align_existing_scene():
    args = _build_parser().parse_args(["run", "--render-backend", "orcalab"])
    assert args.publish_scene is False
    assert args.no_publish is False
    assert args.robot_actor_name == "auto"
    assert args.anchor_existing_scene is True
    assert args.scene_profile == "orca-train"
    assert args.strict_scene_alignment is True
    assert args.manual_xml_override is True
    assert args.randomized_play is False
    assert args.warmup_steps == 100


def test_go2_warmup_steps_can_be_overridden():
    args = _build_parser().parse_args(["run", "--warmup-steps", "0"])
    assert args.warmup_steps == 0


def test_keyboard_teleop_parser_has_safe_defaults():
    args = _build_parser().parse_args(["teleop", "--no-preview"])
    assert args.func.__name__ == "_teleop"
    assert args.forward_speed == 0.5
    assert args.command_hold == 0.35
    assert args.render_timeout == 10.0
    assert args.max_control_steps == 0
    assert args.no_preview is True


def test_instruction_file_overrides_dataset_prompt(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("  Walk to the cabinet and stop.\n", encoding="utf-8")
    args = _build_parser().parse_args(["run", "--instruction-file", str(prompt_path)])
    assert _resolve_instruction(args, "dataset prompt") == (
        "Walk to the cabinet and stop."
    )


def test_instruction_file_provider_reloads_edits(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Walk to the cabinet.\n", encoding="utf-8")
    args = _build_parser().parse_args(["run", "--instruction-file", str(prompt_path)])
    provider = _make_instruction_provider(args)

    assert provider is not None
    assert provider() == "Walk to the cabinet."
    prompt_path.write_text("Turn right at the cabinet.\n", encoding="utf-8")
    assert provider() == "Turn right at the cabinet."


def test_waypoint_file_loads_one_stage_per_nonempty_line(tmp_path):
    prompt_path = tmp_path / "waypoints.txt"
    prompt_path.write_text(
        "# staged scene\nReach orange and stop.\n\nReach blue and stop.\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        ["run", "--waypoint-instruction-file", str(prompt_path)]
    )
    stages = _resolve_waypoint_instructions(args)
    assert stages == ("Reach orange and stop.", "Reach blue and stop.")
    assert _resolve_instruction(
        args, "dataset prompt", waypoint_instructions=stages
    ) == (
        "Waypoint 1: Reach orange and stop. "
        "Waypoint 2: Reach blue and stop."
    )
