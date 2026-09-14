# Unitree physical deployment

This is the minimal physical-robot path. OrcaLab is not installed on the
robot. The robot captures camera frames and executes bounded high-level motion;
the NVIDIA workstation runs NaVILA inference.

## Confirmed hackathon interface

The supplied hackathon slides describe a Unitree A2 with:

- code executed over SSH on the onboard computer;
- `unitree_sdk2py` initialized with `ChannelFactoryInitialize(0, "br0")`;
- `unitree_sdk2py.a2.sport.sport_client.SportClient` for motion;
- an H.264 RTP multicast camera at `230.1.1.1:1720` on `br0`;
- OpenCV built with GStreamer in the provided `robot-env` virtual environment.

Do not replace the robot-provided OpenCV with the PyPI `opencv-python` wheel:
that wheel may not have the required GStreamer support.

## 1. Workstation

Start the existing NaVILA server on the A4500. Bind it to an address reachable
from the robot (prefer the workstation's Tailscale address):

```bash
cd /home/kohming/Orca_VLN/NaVILA-Orca
NAVVLM_HOST=<A4500_TAILSCALE_IP> ./scripts/start_navvlm_server.sh
```

Validate it from another machine before using the robot:

```bash
python3 scripts/check_navvlm_endpoint.py --host <A4500_TAILSCALE_IP>
```

## 2. Copy the lightweight client to the robot

The minimum required project content is `src/navila_orca`,
`scripts/run_unitree_navila.py`, and `scripts/run_unitree_navila.sh`. The robot
environment needs Pillow plus its preinstalled `unitree_sdk2py`, OpenCV, and
GStreamer stack. It does not need CUDA, NaVILA weights, OrcaLab, or MJLab.

## 3. Camera-only check

On the A2 onboard computer, validate the multicast stream without requiring
the model server or initializing SportClient:

```bash
cd <copied-NaVILA-Orca-directory>
source <robot-env>/.venv/bin/activate
./scripts/run_unitree_navila.sh \
  --robot-model a2 \
  --network-interface br0 \
  --instruction camera-check \
  --camera-check-output /tmp/unitree-camera-check.jpg
```

Copy or view `/tmp/unitree-camera-check.jpg` and verify orientation and colour.

## 4. Camera plus inference smoke test (motors disabled)

On the A2 onboard computer:

```bash
cd <copied-NaVILA-Orca-directory>
source <robot-env>/.venv/bin/activate
./scripts/run_unitree_navila.sh \
  --robot-model a2 \
  --network-interface br0 \
  --vlm-host <A4500_TAILSCALE_IP> \
  --instruction "Walk toward the brown chair and stop next to it." \
  --max-decisions 1 \
  --scene-id site-a
```

This is a dry run. It captures eight images, queries NaVILA once, prints the
bounded command, and records the decision without touching SportClient.

## 5. First bounded movement

Clear the area, keep the physical remote in hand, have a spotter next to the
robot, and test Unitree's own motion example first. Then arm this client:

```bash
./scripts/run_unitree_navila.sh \
  --robot-model a2 \
  --network-interface br0 \
  --vlm-host <A4500_TAILSCALE_IP> \
  --instruction "Walk toward the brown chair and stop next to it." \
  --execute-actions \
  --max-decisions 1 \
  --max-forward-mps 0.10 \
  --max-yaw-rps 0.20 \
  --max-action-seconds 0.50 \
  --scene-id site-a
```

`--execute-actions` is the only switch that initializes SportClient. Commands
are resent at 50 Hz for a bounded duration and `StopMove()` is called after
every action, on parsing failures, on Ctrl-C/SIGTERM, and during final cleanup.
The default limits are deliberately slower than NaVILA's canonical command.

Do not use `--balance-stand` until the event operator confirms that automatic
standing is desired. The default assumes the robot has already been placed in
the correct balanced state using the approved Unitree procedure.

## 6. Longer run and dataset

After a one-action test succeeds, increase `--max-decisions` gradually. Every
decision is written beneath `outputs/unitree/<timestamp>/decision_samples` in
the same review schema as simulator decisions. It can be labelled with
`scripts/label_decision_samples.py` and exported with
`scripts/export_navila_lora_dataset.py`.

The physical site's semantic map and item-memory integration remain separate
follow-up work. First prove camera, remote inference, StopMove, and one bounded
action end to end.
