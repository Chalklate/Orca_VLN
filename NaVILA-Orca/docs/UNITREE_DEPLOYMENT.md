# Unitree physical deployment

This is the minimal physical-robot path. OrcaLab is not installed on the
laptop. The laptop connects to the Go2 over a local Ethernet or Wi-Fi network,
captures camera frames, and executes bounded high-level motion; a remote NVIDIA
machine runs NaVILA inference. The remote machine can be the existing AWS host
reached through an SSM port-forward, or the A4500 workstation reached through
Tailscale.

## Target interface

The current target is a Go2 reachable from the laptop over a local network. A
direct Ethernet connection is preferred for reliability, but Wi-Fi works when
the access point permits DDS discovery and peer-to-peer client traffic. The
client uses:

- `ChannelFactoryInitialize(0, <ethernet-interface>)`;
- `unitree_sdk2py.go2.sport.sport_client.SportClient` for motion;
- `unitree_sdk2py.go2.video.video_client.VideoClient` for JPEG camera frames.

The older A2 multicast/GStreamer adapter remains available with
`--robot-model a2`, but it is not the default.

### Selecting Ethernet or Wi-Fi

`--network-interface` takes the laptop interface name, not the robot IP. To
find the interface selected by Linux for a known robot address:

```bash
DOG_IP=<robot-ip>
ip route get "${DOG_IP}"
```

Look for `dev <interface>` in the result. Typical Wi-Fi names begin with
`wlp`, while USB Ethernet names often begin with `enx`. Confirm basic unicast
reachability:

```bash
ping -c 3 "${DOG_IP}"
```

Successful ping is necessary but not sufficient: Unitree SDK2 uses DDS, whose
discovery traffic may be blocked by guest Wi-Fi, AP/client isolation, VLANs,
or multicast filtering. The camera-only test in section 3 is the definitive
read-only check. If it times out over Wi-Fi, use direct Ethernet or a private
router rather than proceeding to motion testing.

## 1. Choose the remote inference connection

### Option A: existing AWS SSM port-forward

This is the simplest option if the existing AWS NaVILA service is running. The
laptop needs Internet access, AWS CLI, the Session Manager plugin, and an
authenticated `navila` AWS profile. It does not need an NVIDIA GPU.

Verify the laptop setup before travelling:

```bash
aws --version
session-manager-plugin --version
aws sts get-caller-identity --profile navila --region ap-northeast-1
```

Start the tunnel in a dedicated laptop terminal and leave it running:

```bash
cd <copied-NaVILA-Orca-directory>
./scripts/start_navila_aws_tunnel.sh
```

The helper is the physical-deployment equivalent of the AWS section in
`memory_guide_dev.sh`. It forwards laptop address `127.0.0.1:54321` to port
`54321` on SSM instance `i-066515f762428ba55` in `ap-northeast-1`. Override
the defaults only if the AWS deployment changes:

```bash
NAVILA_AWS_INSTANCE_ID=<instance-id> \
NAVILA_AWS_PROFILE=<profile> \
NAVILA_AWS_REGION=<region> \
NAVVLM_PORT=54321 \
./scripts/start_navila_aws_tunnel.sh
```

In a second terminal, verify the model endpoint through the tunnel:

```bash
python3 scripts/check_navvlm_endpoint.py --host 127.0.0.1 --port 54321
```

The remote NaVILA server must already be listening on port `54321`; the tunnel
does not start the server. If local port `54321` is occupied, launch the helper
with `NAVVLM_PORT=154321` and use `--vlm-port 154321` in every command below.

Do not source `modal.env` for this tunnel. Its `AWS_PROFILE` and region are for
Bedrock and are different from the existing NaVILA SSM profile. The tunnel
helper passes the `navila` profile explicitly.

The SSM tunnel carries only NaVILA TCP requests. Unitree DDS, camera data, and
motion commands remain on the local interface selected with
`--network-interface`. The laptop can therefore use Ethernet for the robot and
Wi-Fi or a phone hotspot for AWS. If the robot and laptop share Internet-enabled
Wi-Fi, the same Wi-Fi interface may carry both DDS and the AWS tunnel.

For all commands below, the AWS endpoint is:

```text
--vlm-host 127.0.0.1 --vlm-port 54321
```

### Option B: Tailscale to the A4500 workstation

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

For all commands below, the Tailscale endpoint is:

```text
--vlm-host <A4500_TAILSCALE_IP> --vlm-port 54321
```

## 2. Copy the lightweight client to the laptop

The minimum required project content is `src/navila_orca`,
`scripts/run_unitree_navila.py`, and `scripts/run_unitree_navila.sh`. The robot
environment needs Pillow, NumPy, CycloneDDS 0.10.2, and `unitree_sdk2py`. It
does not need CUDA, NaVILA weights, OrcaLab, ROS 2, or MJLab.

To rebuild the lightweight laptop archive on the development PC:

```bash
./scripts/build_unitree_deployment_bundle.sh
```

Copy `outputs/deployment/unitree-navila-client.tar.gz` to the laptop and
extract it there. The archive includes the AWS tunnel helper and this guide.

## 3. Camera-only check

Connect the Go2 and laptop using direct Ethernet or the same Wi-Fi network.
Find the correct interface as described above, then validate the SDK camera
without requiring the model server or initializing SportClient:

```bash
cd <copied-NaVILA-Orca-directory>
conda activate unitree-go2-py310
./scripts/run_unitree_navila.sh \
  --robot-model go2 \
  --network-interface <ethernet-interface> \
  --instruction camera-check \
  --camera-check-output /tmp/unitree-camera-check.jpg
```

Copy or view `/tmp/unitree-camera-check.jpg` and verify orientation and colour.

## 4. Camera plus inference smoke test (motors disabled)

On the laptop:

```bash
cd <copied-NaVILA-Orca-directory>
conda activate unitree-go2-py310
./scripts/run_unitree_navila.sh \
  --robot-model go2 \
  --network-interface <ethernet-interface> \
  --vlm-host 127.0.0.1 \
  --vlm-port 54321 \
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
  --robot-model go2 \
  --network-interface <ethernet-interface> \
  --vlm-host 127.0.0.1 \
  --vlm-port 54321 \
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
