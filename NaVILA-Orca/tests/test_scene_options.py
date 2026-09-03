from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip("mujoco")

from navila_orca.orcalab_runtime.scene_options import (  # noqa: E402
    assert_flat_ground_options,
    patch_scene_xml_options,
    scene_xml_contract,
)


GROUND_CONTACT = (
    'friction="1 0.005 0.0001" solref="0.02 1" '
    'solimp="0.9 0.95 0.001 0.5 2" contype="1" '
    'conaffinity="1" condim="3"'
)


def _compile(worldbody: str):
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{worldbody}</worldbody></mujoco>"
    )


def test_flat_ground_validator_accepts_plane() -> None:
    model = _compile(f'<geom name="floor" type="plane" size="0 0 0.1" {GROUND_CONTACT}/>')
    ground = assert_flat_ground_options(model)
    assert ground[0]["name"] == "floor"
    assert ground[0]["type"] == "plane"


def test_flat_ground_validator_accepts_large_fixed_level_box() -> None:
    model = _compile(
        f'<body name="scene"><geom name="floor_slab" type="box" '
        f'size="6 10 0.1" pos="0 0 -0.1" {GROUND_CONTACT}/></body>'
    )
    ground = assert_flat_ground_options(model)
    assert ground[0]["name"] == "floor_slab"
    assert ground[0]["type"] == "box_slab"
    assert ground[0]["size"] == pytest.approx([6.0, 10.0, 0.1])


@pytest.mark.parametrize(
    "worldbody",
    [
        f'<geom name="small_box" type="box" size="1 1 0.1" {GROUND_CONTACT}/>',
        (
            f'<geom name="tilted_slab" type="box" size="6 10 0.1" '
            f'quat="0.965925826 0.258819045 0 0" {GROUND_CONTACT}/>'
        ),
        (
            f'<body name="moving_floor"><freejoint/><geom name="moving_slab" '
            f'type="box" size="6 10 0.1" {GROUND_CONTACT}/></body>'
        ),
    ],
)
def test_flat_ground_validator_rejects_unsafe_box_candidates(worldbody: str) -> None:
    model = _compile(worldbody)
    with pytest.raises(RuntimeError, match="no compatible flat ground geom"):
        assert_flat_ground_options(model)


def test_patch_scene_xml_options_aligns_large_floor_slab(tmp_path) -> None:
    source = tmp_path / "source.xml"
    output = tmp_path / "aligned.xml"
    source.write_text(
        "<mujoco><worldbody>"
        '<geom name="floor_slab" type="box" size="6 10 0.1" '
        'friction="0.5 0.1 0.1" contype="0" conaffinity="0" condim="1"/>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )

    patch_scene_xml_options(source, output, profile="orca-train")

    geom = ET.parse(output).getroot().find(".//geom")
    assert geom is not None
    assert geom.get("friction") == "1 0.0050000000000000001 0.0001"
    assert geom.get("contype") == "1"
    assert geom.get("conaffinity") == "1"
    assert geom.get("condim") == "3"
    assert scene_xml_contract(output)["ground_geoms"][0]["type"] == "box"
    assert assert_flat_ground_options(mujoco.MjModel.from_xml_path(str(output)))