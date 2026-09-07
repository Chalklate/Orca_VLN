# Orca_VLN

An agentic visual-language navigation proof of concept for a Unitree Go2 in
OrcaLab. A language query is converted into a small search plan; NaVILA uses
eight ego-camera frames to choose the next movement; the navigation runner
executes that movement in MuJoCo/MJLab and mirrors the pose into OrcaLab.

```text
query → Memory Guide plan → eight RGB frames → NaVILA action
      → Go2 locomotion → OrcaLab scene/camera → next observation
```

## Requirements

- Ubuntu 22.04/24.04, Git, Conda, and an NVIDIA driver for `nvidia-smi`.
- OrcaLab 26.7.1 with access to the `VLN_Presentation` and `unitree_robots`
  assets.
- An NVIDIA GPU with enough memory for the selected NaVILA checkpoint. A
  separate GPU server can be used for NaVILA inference.

Model weights, OrcaLab subscriptions, credentials, and generated run outputs
are intentionally not part of the repository.

## Install and verify

From the repository root:

```bash
./NaVILA-Orca/scripts/setup_all.sh
./NaVILA-Orca/scripts/doctor.sh
```

The setup creates two project-local environments under `.conda/envs/` and
downloads the reviewed NaVILA checkpoint. Launchers resolve these environments
themselves; shell activation is optional.

## Recommended local run

This is the simplest complete development loop. Run the first command in one
terminal and keep it open:

```bash
cd /path/to/Orca_VLN/NaVILA-Orca

NAVILA_SERVER_MODE=local \
./scripts/memory_guide_dev.sh start
```

The supervisor starts OrcaLab, the external simulation, and the local NaVILA
server. In a second terminal, run a query:

```bash
cd /path/to/Orca_VLN/NaVILA-Orca
./scripts/memory_guide_dev.sh run --query "Where are my glasses?"
```

Stop the stack with:

```bash
./scripts/memory_guide_dev.sh stop
```

### Local NaVILA LoRA adapter

An unmerged LoRA adapter needs both the adapter directory and its matching base
checkpoint. For the adapter layout used in this project:

```bash
cd /path/to/Orca_VLN/NaVILA-Orca

NAVILA_SERVER_MODE=local \
NAVVLM_MODEL_PATH=/path/to/Orca_VLN/models/orca_navila_lora \
NAVVLM_MODEL_BASE=/path/to/Orca_VLN/models/navila-llama3-8b-8f \
./scripts/memory_guide_dev.sh start
```

If the local LoRA server reports a `bitsandbytes` CUDA 12.8 error, this
unquantized inference server does not need bitsandbytes:

```bash
/path/to/Orca_VLN/.conda/envs/navila/bin/python \
  -m pip uninstall -y bitsandbytes
```

For a remote GPU server, copy the adapter and base checkpoint to that server,
run `scripts/start_navvlm_server.sh` there with the same two model variables,
and forward its loopback port to the client. See
[`NaVILA-Orca/docs/REMOTE_INFERENCE.md`](NaVILA-Orca/docs/REMOTE_INFERENCE.md).

## Direct scene locomotion

To bypass the Memory Guide and give NaVILA one instruction directly:

```bash
cd /path/to/Orca_VLN/NaVILA-Orca
./scripts/run_orcalab_scene_locomotion.sh \
  --instruction "Walk to the dining table, inspect it, and stop."
```

The runner uses the existing OrcaLab layout, the Go2 policy checkpoint, the
`mujococamera1080` RGB camera, and TCP NaVILA at `127.0.0.1:54321`. It records
measurements, motion chunks, camera frames, and optional decision samples under
`outputs/`.

## Teleoperation and data collection

Record pose-tagged views for the semantic map with:

```bash
./scripts/memory_guide_dev.sh teleop --capture-interval 0.4
```

Use `W/S` to move, `A/D` to turn, `Q/E` to strafe, `Space` to brake, `M` to
label a view, and `X` or `Esc` to exit. Completed teleoperation data is read
from `outputs/memory_guide/latest-teleop/teleop.json`.

## Useful scripts

| Script | Purpose |
| --- | --- |
| `setup_all.sh` | Install both pinned environments and the base model |
| `doctor.sh` | Verify drivers, environments, packages, and model files |
| `memory_guide_dev.sh` | Start/stop the complete stack, run queries, and teleoperate |
| `run_orcalab_memory_guide.sh` | Build a query plan and launch waypoint locomotion |
| `run_orcalab_scene_locomotion.sh` | Run direct closed-loop NaVILA navigation |
| `start_navvlm_server.sh` | Start the local or remote NaVILA TCP service |
| `export_navila_lora_dataset.py` | Package reviewed frames/actions for LoRA training |
| `label_decision_samples.py` | Review and correct recorded NaVILA actions |

## Optional query-planning credentials

The default Memory Guide planner is deterministic and requires no API key.
OpenAI, Bedrock, voice, and managed AWS modes read credentials from the shell
environment or a local untracked `.env` file. Never commit those values. The
managed AWS access procedure is documented in
[`NaVILA-Orca/docs/ACCESS_GUIDE.md`](NaVILA-Orca/docs/ACCESS_GUIDE.md).

## Tests and artifacts

Run the installation check and unit tests with:

```bash
./NaVILA-Orca/scripts/doctor.sh
cd NaVILA-Orca
pytest -q
```

The main runtime boundary is `src/navila_orca/runner.py`; OrcaLab transport is
under `src/navila_orca/render/`, planning is under `memory_guide.py`, and the
shell scripts provide reproducible setup and launch commands. Evaluation and
demo results are written to `outputs/`, which is ignored by Git.
