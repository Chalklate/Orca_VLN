from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import navila_orca.render.orca_camera as orca_camera_module
from navila_orca.render.orca_camera import (
    OrcaEgoCameraFollower,
    OrcaGrpcPngCamera,
    _OrcaRuntime,
    _read_complete_png,
    compose_camera_pose,
    rotate_vector_wxyz,
)


def test_compose_camera_pose_matches_orca_forward_mount_at_identity() -> None:
    position, quat = compose_camera_pose([1.0, 2.0, 0.4], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(position, [1.1, 2.0, 1.4], atol=1.0e-12)
    np.testing.assert_allclose(
        quat,
        [
            np.cos(np.deg2rad(10.0)) * np.sqrt(0.5),
            -np.sin(np.deg2rad(10.0)) * np.sqrt(0.5),
            np.sin(np.deg2rad(10.0)) * np.sqrt(0.5),
            -np.cos(np.deg2rad(10.0)) * np.sqrt(0.5),
        ],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        rotate_vector_wxyz(quat, [0.0, 1.0, 0.0]),
        [np.cos(np.deg2rad(20.0)), 0.0, -np.sin(np.deg2rad(20.0))],
        atol=1.0e-12,
    )


def test_compose_camera_pose_rotates_mount_with_go2_base() -> None:
    root_quat = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    position, quat = compose_camera_pose([0.0, 0.0, 0.4], root_quat)
    np.testing.assert_allclose(position, [0.0, 0.1, 1.4], atol=1.0e-12)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_complete_png_is_decoded_only_after_iend_arrives(
    tmp_path, monkeypatch
) -> None:
    complete_path = tmp_path / "complete.png"
    pending_path = tmp_path / "pending.png"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(complete_path)
    complete_png = complete_path.read_bytes()
    pending_path.write_bytes(complete_png[:-12])

    image_open_calls = 0
    real_image_open = Image.open

    def counted_image_open(*args, **kwargs):
        nonlocal image_open_calls
        image_open_calls += 1
        return real_image_open(*args, **kwargs)

    def finish_write(_seconds):
        pending_path.write_bytes(complete_png)

    monkeypatch.setattr(orca_camera_module.Image, "open", counted_image_open)
    monkeypatch.setattr(orca_camera_module.time, "sleep", finish_write)

    rgb = _read_complete_png(str(pending_path), timeout_s=1.0)

    assert image_open_calls == 1
    assert rgb.shape == (3, 4, 3)
    assert np.all(rgb == [10, 20, 30])


def test_compose_camera_pose_can_reject_base_roll_from_camera_orientation() -> None:
    roll_30 = [np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)), 0.0, 0.0]
    position, quat = compose_camera_pose(
        [0.0, 0.0, 0.4],
        roll_30,
        stabilize_horizon=True,
    )

    # The mount position remains physically attached and therefore moves with
    # base roll, while image orientation is identical to a level, zero-yaw base.
    assert not np.allclose(position, [0.1, 0.0, 1.4])
    np.testing.assert_allclose(
        quat,
        [
            np.cos(np.deg2rad(10.0)) * np.sqrt(0.5),
            -np.sin(np.deg2rad(10.0)) * np.sqrt(0.5),
            np.sin(np.deg2rad(10.0)) * np.sqrt(0.5),
            -np.cos(np.deg2rad(10.0)) * np.sqrt(0.5),
        ],
        atol=1.0e-12,
    )


class _Path:
    def __init__(self, value: str = "/"):
        self.value = value

    @classmethod
    def root_path(cls):
        return cls("/")

    def string(self):
        return self.value


class _Transform:
    def __init__(self, position, rotation, scale):
        self.position = np.asarray(position)
        self.rotation = np.asarray(rotation)
        self.scale = scale


class _Actor:
    def __init__(self, name, asset_path):
        self.name = name
        self.asset_path = asset_path
        self.transform = None


@dataclass
class _AddRequest:
    actor: _Actor
    parent_path: _Path


@dataclass
class _Key:
    actor_path: _Path
    group_prefix: str
    property_name: str
    property_type: object


class _Property:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def value_type(self):
        return object()


class _Group:
    prefix = "/AgentCamera:Camera"

    def __init__(self):
        self.properties = [
            _Property(name)
            for name in (
                "IsRecording",
                "Width",
                "Height",
                "RandomObjectColor",
                "ColorCamera",
                "DepthCamera",
                "NormalCamera",
                "ObjectColorCamera",
                "UseNvEnc",
                "NvencGpuIndex",
                "ColorPort",
                "DepthPort",
            )
        ]


class _EditService:
    def __init__(self):
        self.address = None
        self.exists = False
        self.add_calls = []
        self.property_calls = []
        self.transform_calls = []
        self.values = {
            "IsRecording": True,
            "Width": 1080,
            "Height": 1080,
            "ColorCamera": True,
            "ColorPort": 1,
        }
        self.closed = False

    def init_grpc(self, address):
        self.address = address

    async def aloha(self):
        return True

    async def get_property_groups(self, _path):
        if not self.exists:
            raise RuntimeError("not found")
        return [_Group()]

    async def add_actor_batch(self, requests, stop_on_error):
        self.add_calls.append((requests, stop_on_error))
        self.exists = True
        return True, [""]

    async def set_properties(self, keys, values):
        names = [key.property_name for key in keys]
        self.property_calls.append((names, list(values)))
        self.values.update(zip(names, values))

    async def get_properties(self, keys):
        return [self.values[key.property_name] for key in keys]

    async def set_actor_transform_batch(self, paths, transforms):
        self.transform_calls.append((paths, transforms))

    async def get_camera_data_png(self, camera_name, output_dir, index):
        color_dir = Path(output_dir) / "color"
        color_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 6), (10, 20, 30, 255)).save(
            color_dir / f"{camera_name}_color_{index}.png"
        )
        return SimpleNamespace(has_color=True, transform="camera-transform")

    async def destroy_grpc(self):
        self.closed = True


def test_follower_provisions_rgb_only_camera_and_updates_root_actor() -> None:
    service = _EditService()
    runtime = _OrcaRuntime(
        service_factory=lambda: service,
        asset_actor_type=_Actor,
        transform_type=_Transform,
        path_type=_Path,
        add_actor_request_type=_AddRequest,
        property_key_type=_Key,
    )
    loop = asyncio.new_event_loop()
    follower = OrcaEgoCameraFollower(
        edit_address="127.0.0.1:50151",
        color_port=7070,
        event_loop=loop,
        runtime_factory=lambda: runtime,
    )
    try:
        follower.start()
        assert service.add_calls[0][1] is True
        request = service.add_calls[0][0][0]
        assert request.actor.name == "navila_ego"
        assert request.actor.asset_path == "prefabs/agentcamera"
        assert request.parent_path.string() == "/"
        assert service.property_calls[0] == (["IsRecording"], [False])
        assert service.property_calls[-1] == (["IsRecording"], [True])
        configured = dict(
            zip(service.property_calls[1][0], service.property_calls[1][1])
        )
        assert configured["Width"] == 512
        assert configured["Height"] == 512
        assert configured["ColorPort"] == 7070
        assert configured["ColorCamera"] is True
        assert configured["DepthCamera"] is False

        position, _quat = follower.update([1.0, 2.0, 0.4], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(position, [1.1, 2.0, 1.4])
        paths, transforms = service.transform_calls[-1]
        assert paths[0].string() == "/navila_ego"
        np.testing.assert_allclose(transforms[0].position, position)
    finally:
        follower.close()
        loop.close()
    assert service.closed


def test_grpc_png_camera_pulls_a_fresh_rgb_frame(tmp_path) -> None:
    service = _EditService()
    runtime = _OrcaRuntime(
        service_factory=lambda: service,
        asset_actor_type=_Actor,
        transform_type=_Transform,
        path_type=_Path,
        add_actor_request_type=_AddRequest,
        property_key_type=_Key,
    )
    camera = OrcaGrpcPngCamera(
        "navila_ego",
        7070,
        edit_address="127.0.0.1:50151",
        output_dir=str(tmp_path),
        runtime_factory=lambda: runtime,
    )
    camera.start()
    try:
        rgb, frame_index = camera.get_frame(format="rgb24")
        assert rgb.shape == (6, 8, 3)
        assert rgb.dtype == np.uint8
        assert np.all(rgb == [10, 20, 30])
        assert frame_index == 0
        assert camera.is_first_frame_received()
        assert camera.last_transform == "camera-transform"
    finally:
        camera.stop()
    assert service.closed
