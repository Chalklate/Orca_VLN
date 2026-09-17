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

For a Go2 run, `--robot-model go2` selects both the Go2 SDK2 VideoClient and
Go2 SDK2 SportClient. The A2 GStreamer pipeline and A2 SDK module are not used.
A2 names elsewhere in the repository belong to simulator/training
configurations or the optional legacy adapter; they do not change a Go2 run.

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
`assets/memory_guide_catalog.json`, `scripts/run_unitree_navila.py`,
`scripts/run_unitree_navila.sh`, and `scripts/run_unitree_memory_guide.sh`. The
robot environment needs Pillow, NumPy, CycloneDDS 0.10.2, and
`unitree_sdk2py`. It does not need CUDA, NaVILA weights, OrcaLab, ROS 2, or
MJLab.

If the robot will use Luna query parsing or the landmark scan, install the
OpenAI client into the same interpreter that runs the launcher:

```bash
python3 -m pip install 'openai>=1.0.0,<2'
```

To rebuild the lightweight laptop archive on the development PC:

```bash
./scripts/build_unitree_deployment_bundle.sh
```

Copy `outputs/deployment/unitree-navila-client.tar.gz` to the laptop and
extract it there. The archive includes the AWS tunnel helper and this guide.

The archive does not vendor Unitree's SDK2 Python checkout. If SDK2 is present
as source rather than installed into the selected interpreter, point the
launcher at its repository root before running the client:

```bash
export UNITREE_PYTHON=/usr/bin/python3
export UNITREE_SDK2_ROOT=/home/unitree/unitree_sdk2_python
```

The root must contain `unitree_sdk2py/`. Use the same `UNITREE_PYTHON` that can
import both `unitree_sdk2py.core.channel` and
`unitree_sdk2py.go2.video.video_client`.

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
  --image-brightness 1.25 \
  --scene-id site-a
```

This is a dry run. It captures eight images, brightens them by 1.25x before
sending them to NaVILA, queries NaVILA once, prints the bounded command, and
records the decision without touching SportClient. Use `1.0` to disable the
adjustment; values around `1.15`–`1.35` are a reasonable starting range, but
large values can clip highlights.

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

The Go2 VideoClient adapter retries a malformed or failed JPEG sample three
times, and the capture worker tolerates three consecutive failed capture
cycles. A persistent camera failure still fails closed and final cleanup calls
`StopMove()`. Tune this with `--camera-frame-retries`,
`--camera-error-retries`, and `--camera-rpc-timeout`.

## 6. Longer run and dataset

After a one-action test succeeds, increase `--max-decisions` gradually. Every
decision is written beneath `outputs/unitree/<timestamp>/decision_samples` in
the same review schema as simulator decisions. It can be labelled with
`scripts/label_decision_samples.py` and exported with
`scripts/export_navila_lora_dataset.py`.

## 7. Typed Memory Guide search

The lightweight client includes the deployable part of the “Where is my bread?”
flow. A typed query is routed locally, expanded into one waypoint per search
location, and executed sequentially. For an unknown item such as bread, the
default catalog produces four waypoints. The VLM receives explicit
short approach instructions. Without the seer, the next waypoint is not
started until the current one returns a `stop` action. With `--landmark-seer`,
the goal verifier can instead end the waypoint after a bounded inspection, and
it ends the whole mission when the requested item is confirmed.

Run it from the Go2 SSH terminal after validating the single-waypoint path:

```bash
./scripts/run_unitree_memory_guide.sh \
  --query "Where is my bread?" \
  --robot-model go2 \
  --network-interface eth10 \
  --vlm-host <model-pc-tailscale-ip> \
  --vlm-port 54321 \
  --execute-actions \
  --max-decisions 8 \
  --max-forward-mps 0.20 \
  --max-yaw-rps 0.35 \
  --max-action-seconds 0.75 \
  --scene-id physical-site
```

`--max-decisions` is per waypoint for this launcher. The default is 8, so a
four-waypoint patrol can make at most 32 VLM decisions. If a waypoint does not
produce `stop` within its limit, the mission fails closed and does not silently
advance. Frames are cleared between waypoints so the next decision is not
dominated by the previous search location.

Generate and inspect the plan without connecting to the dog:

```bash
./scripts/run_unitree_memory_guide.sh \
  --query "Where is my bread?" \
  --plan-only
```

The plan and waypoint files are written under `outputs/memory_guide/`. A recent
manual observation can be stored for direct routing on the next query:

```bash
PYTHONPATH=src python3 -m navila_orca.memory_guide \
  --inventory outputs/memory_guide/inventory.json \
  remember --item bread --location "the kitchen counter" --confidence 0.9
```

Without `--landmark-seer`, this physical MVP does not automatically confirm an
item: a VLM `stop` only marks a completed inspection waypoint. With the seer
enabled, Luna verifies the requested item in the live image, calls the direct
Go2 `StopMove()` path on confirmation, and terminates the search. If the
landmark is reached but the item is not visible, it stops and performs a
bounded in-place inspection sweep before moving to the next candidate.
Automatic inventory updates and physical map routing remain separate additions.

## 8. Luna query parsing and a quick site scan

The physical Memory Guide can use OpenAI for intent/target parsing and for
ranking site-specific visual landmarks. Keep the API key out of scripts and
Git. Create a private environment file on whichever machine runs the
Memory-Guide wrapper (the laptop in the laptop-to-Go2 deployment):

```bash
cat > "$HOME/navila-secrets.env" <<'EOF'
export OPENAI_API_KEY='paste-key-here'
EOF
chmod 600 "$HOME/navila-secrets.env"
source "$HOME/navila-secrets.env"
```

Run a slow in-place eight-view scan before operating in a new room. It sends
the captured views to the configured OpenAI model and writes
`outputs/memory_guide/latest_landmark_map.json`:

```bash
source "$HOME/navila-secrets.env"
python3 scripts/run_unitree_landmark_scan.py \
  --robot-model go2 \
  --network-interface eth10 \
  --scan-views 8 \
  --scan-yaw-rps 0.20 \
  --scan-direction left \
  --image-brightness 1.20 \
  --execute-actions
```

The scan is intentionally an open-loop visual sweep. It has no odometry
correction, metric coordinates, or obstacle-clearance guarantees; use a
spotter and clear the area. The result is a list of visual references such as
a table, door, lectern, aisle, or stage. It is useful for generating plausible
search locations in an unfamiliar office or auditorium, but it cannot
guarantee that the robot can safely reach each reference.

If the scan was run on the Go2, copy the resulting JSON to the laptop that
will run the query wrapper. The wrapper automatically uses this standard path:

```bash
mkdir -p outputs/memory_guide
scp unitree@192.168.1.119:/home/unitree/Desktop/dethread/unitree-navila-client/outputs/memory_guide/latest_landmark_map.json \
  outputs/memory_guide/latest_landmark_map.json
```

The file must be the JSON file written by the scan, not a terminal log that
also contains the JSON. Set `NAVILA_LANDMARK_MAP` or pass `--landmark-map` when
using a different path. If the standard map is absent, the wrapper fails
closed instead of silently using the old dining-table/entryway catalog.

Then route a query through Luna and use the scan map to choose up to six search
locations:

```bash
source "$HOME/navila-secrets.env"
./scripts/run_unitree_memory_guide.sh \
  --query "Where is my bag?" \
  --robot-model go2 \
  --network-interface <laptop-ethernet-interface> \
  --vlm-host <model-pc-tailscale-ip> \
  --vlm-port 54321 \
  --llm-mode openai \
  --openai-model-id gpt-5.6-luna \
  --max-landmark-waypoints 6 \
  --landmark-seer \
  --execute-actions \
  --max-decisions 8 \
  --max-forward-mps 0.20 \
  --max-yaw-rps 0.35 \
  --max-action-seconds 0.75 \
  --image-brightness 1.20 \
  --scene-id auditorium
```

Luna first returns structured intent, target, and landmark IDs. With
`--landmark-seer`, the same OpenAI client then compares each selected
landmark's scan reference image with the live Go2 image and also checks for the
requested item. If the landmark is not visible, the client rotates in place in
bounded increments and checks again; forward motion is not permitted during
this acquisition phase. NaVILA is called only for short approach suggestions.
The goal seer is checked before approach, periodically during approach, and
throughout the final in-place inspection. Seeing the item is not success: the
seer classifies it as `far`, `approach`, or `near`, and requires it to be
accessible before declaring a find. While the item is visible but still far or
approaching, the client replaces the waypoint text with a short item-approach
instruction so NaVILA closes the gap instead of continuing to navigate toward
the landmark. A confirmed item produces a direct `StopMove()` and
`goal-found ...; search complete`; NaVILA does not need to emit the textual
`stop` action for the mission to finish. If the item is visible but the seer
does not consider forward motion safe, the client holds position and performs
the bounded inspection sweep. The seer is intentionally opt-in because each
check uploads images and can trigger physical in-place turns. If the Go2 has no
Internet route to the OpenAI API, use a reachable OpenAI-compatible
`--openai-base-url` or omit `--landmark-seer`.

The seer needs the scan reference JPEGs as well as the JSON map. Run the scan
on the same laptop that will run the search, or copy the scan image directory
and update the paths in the map before starting the mission. A JSON-only copy
is sufficient for landmark ranking but not for visual reacquisition.
