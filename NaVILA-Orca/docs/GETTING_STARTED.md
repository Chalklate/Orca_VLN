<p align="right"><sub><strong>English</strong> · <a href="GETTING_STARTED_zh.md">中文</a></sub></p>

# Getting started: run a navigation instruction in OrcaLab

The goal is not merely to launch a model. Observe the full robot decision chain: **what it sees, what the language model says, how the quadruped executes, and what changes in the scene**.

With Option A, run OrcaLab, the NaVILA server, and the navigation process in
terminals 1, 2, and 3. Option B keeps the OrcaLab and navigation terminals on
the client and moves the NaVILA service to a remote inference server. Option C
also keeps OrcaLab and navigation on the participant's machine, but reaches an
organizer-managed NaVILA service through an AWS SSM port-forward. Keeping the
layers separate makes failures easy to isolate.

## Choose a deployment option

Choose exactly one deployment option before installing:

| Deployment option | Layout | Follow this guide |
| --- | --- | --- |
| **Option A (default) — single-host deployment** | OrcaLab, NaVILA, and navigation run on one machine | Continue with [Option A installation](#option-a-single-host) below |
| **Option B — remote inference** | OrcaLab and navigation run on the client; NaVILA runs on a separate GPU server | Follow the [remote inference guide](REMOTE_INFERENCE.md) |
| **Option C — managed remote inference (AWS SSM)** | OrcaLab and navigation run on the participant's machine; NaVILA runs on an organizer-managed AWS instance | Follow the [managed access guide](ACCESS_GUIDE.md) |

The installation and first-run sequence below describes Option A. Option B
uses the same scene and navigation behavior, but its per-machine installation,
service startup, SSH tunnel, and end-to-end NaVILA protocol check are documented
only in the dedicated remote guide. Option C uses that same client-side scene
and navigation behavior, while its temporary SSO credentials, AWS SSM tunnel,
health check, and client startup are documented only in the managed access
guide. The connectivity checks do not run model inference.

## 1. Goal and success criteria

The default instruction is stored in
[`prompts/orcalab_scene_locomotion.txt`](../prompts/orcalab_scene_locomotion.txt):

> Walk toward the red waste bin and pass close by it without stopping. Continue toward the blue barrels and pass them. Then turn right and follow the open aisle beside the white safety fence toward the red fire extinguisher. Keep outside the fenced work cell and avoid the boxes. When the white industrial robotic arm mounted on a gray pedestal is visible, approach the open floor directly in front of the pedestal. Stop about 1.5 meters away from the arm.

A successful run is more than an error-free terminal. It should satisfy all of the following:

- OrcaLab has the `VLN_Presentation` scene open, with one complete Go2, the red waste bin, blue barrels, the red fire extinguisher, and the white industrial robotic arm on its gray pedestal.
- Images from `mujococamera1080` change as the Go2 moves.
- The NaVILA server receives eight images plus the task text and returns a parseable action.
- Go2 moves stably; `outputs/scene_locomotion_smoke/` contains result JSON and RGB frames after the run.

## 2. Four roles in the system

| Layer | Input | Output | Does not handle |
| --- | --- | --- | --- |
| NaVILA | 8 RGB frames + natural language | textual action | joint control or collision solving |
| Navigation loop | textual action + current state | velocity target and duration | visual-language inference |
| Go2 locomotion | velocity target | 12 joint actions | semantic meaning such as “blue barrel” |
| OrcaLab | Go2 pose | scene RGB and visualization | low-level gait training or solving |

For example, when NaVILA returns `turn left 15 degrees`, the navigation loop converts it to a fixed yaw velocity for 0.5 seconds. The Go2 policy executes continuously at 50 Hz, then the OrcaLab camera returns the next image. This is the high-level VLM / low-level control boundary.

<a id="option-a-single-host"></a>

## 3. Option A: install on one host

Install [Miniconda or Anaconda](https://docs.anaconda.com/miniconda/install/), Git, and an NVIDIA GPU of at least RTX 4090 class first. Do not proceed until `nvidia-smi` succeeds. Then clone this repository and run exactly these commands:

```bash
git clone https://github.com/openverse-orca/Orca_VLN.git
cd Orca_VLN
```

Install Python 3.12 OrcaLab, Python 3.10 NaVILA, and the reviewed checkpoint:

```bash
./NaVILA-Orca/scripts/setup_all.sh
```

Run the independent installation check. Stop unless every check passes:

```bash
./NaVILA-Orca/scripts/doctor.sh
```

The installer creates two isolated prefixes under `Orca_VLN/.conda/envs/`:
`orcalab` and `navila`. It pins the package versions, the NaVILA source commit,
the Transformers commit, and the FlashAttention wheel hash. It also runs
`pip check` on the OrcaLab environment and verifies imports in both
environments. OrcaLab's verified native viewport and scene pak are prepared
during setup instead of being installed inside the first GUI process.

The scripts resolve these prefixes from their own location. Do not activate
either environment and do not export a repository-root variable. This remains
true if another Conda environment is active in the terminal.

Verify or repair individual layers at any time:

```bash
./NaVILA-Orca/scripts/setup_orcalab_env.sh --verify
./NaVILA-Orca/scripts/setup_navila_env.sh --verify
./NaVILA-Orca/scripts/doctor.sh
```

`setup_all.sh` downloads a large model checkpoint. Use `--skip-model` only
when preparing the environments offline, then run
`./NaVILA-Orca/scripts/download_navila_model.sh` before starting the service.

## 4. Option A: first run

<a id="scene-setup"></a>

### Step 1: open the default scene

In terminal 1:

```bash
./NaVILA-Orca/scripts/start_orcalab_gui.sh
```

In the GUI:

1. In the OrcaLab asset browser, subscribe to `VLN_Presentation` and
   `unitree_robots`; wait until both subscriptions are current.
2. Select the `VLN_Presentation` scene.
3. Choose **File → Open Layout → `NaVILA-Orca/factory.json`**.
4. Confirm the scene tree contains exactly one complete Go2 actor.
5. Confirm that the red waste bin, blue barrels, red fire extinguisher, and
   white industrial robotic arm are visible in the intended order.

`VLN_Presentation` supplies the factory scene; `factory.json` stores the actor
layout layered onto it. The layout references `vln_presentation` assets for the
factory props and `unitree_robots` for Go2, so importing it before those
subscriptions finish produces missing actors.

The launcher opens the normal OrcaLab editor and does not force a scene,
layout, full-screen view, or external simulation. The navigation command in
terminal 3 applies and verifies the `orca-train` profile after the selected
scene is running.

### Step 2: start NaVILA

In terminal 2:

```bash
./NaVILA-Orca/scripts/start_navvlm_server.sh
```

Keep this terminal running after it reports listening on `127.0.0.1:54321`.
The server adapter is included at `scripts/navila_vlm_server.py`; users do not
need to find or export a server script. A missing or partial model is rejected
before model loading with the exact recovery command.

<a id="run-navigation"></a>

### Step 3: run navigation

Before using terminal 3, keep OrcaLab open and start its external simulation:
**Run → Start Simulation → No Simulation Program → Start**. Wait until the
simulation is running; terminal 3 connects to that live OrcaLab session and
does not launch it itself.

In terminal 3:

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh
```

This command uses the default prompt quoted above. To make the active prompt
explicit for a run, use:

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh \
  --instruction "Walk toward the red waste bin and pass close by it without stopping. Continue toward the blue barrels and pass them. Then turn right and follow the open aisle beside the white safety fence toward the red fire extinguisher. Keep outside the fenced work cell and avoid the boxes. When the white industrial robotic arm mounted on a gray pedestal is visible, approach the open floor directly in front of the pedestal. Stop about 1.5 meters away from the arm."
```

Important defaults:

| CLI argument | Default behavior | Why it matters |
| --- | --- | --- |
| `--robot-actor-name auto` | requires exactly one complete Go2 in the scene | prevents controlling the wrong actor |
| `--camera-asset-path prefabs/mujococamera1080` | creates once and captures PNG continuously | uses robot ego view, not the viewport |
| default camera mount `0.1 0 0.6` | approximately 1 m above the floor in the standing scene | gives the VLM a slightly higher view of tables and counters |
| `--warmup-steps 100` | executes 100 zero-velocity policy steps before motion | stabilizes policy state before VLM commands |
| `--scene-profile orca-train` | 200 Hz physics and 50 Hz control | makes action distance reproducible by tick |

## 5. Read the outputs

Results are written to `outputs/scene_locomotion_smoke/`. Every run saves at least:

- RGB frames for checking camera placement and image changes during motion.
- Run JSON containing the instruction, parsed action, timing, and trajectory.
- A scene-alignment file for investigating coordinate or actor issues in the OrcaLab combined XML.

Maintain an experiment table for each run: instruction, first model action, final position, whether it passed the red bin and blue barrels, followed the safety fence to the red fire extinguisher, stopped at the white industrial arm, unexpected turns, and screenshot filename. Do not record only “pass/fail”.

## 6. Three progressive tasks

### Task 1: reproduce the baseline

Keep all defaults and run the case twice. Compare action sequences and final trajectories. Discuss whether model inference is deterministic and whether simulation initialization is repeatable.

### Task 2: language ablation

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh \
  --instruction 'Pass the red bin, then turn right and stop at the blue barrel.'
```

Then try “turn left first, then approach the blue barrel.” Record whether wording changes the returned action. This is not a language-model trivia test; it asks whether language, images, and geometry jointly influence the decision.

### Task 3: camera ablation

Raise the camera slightly:

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh \
  --camera-mount-position 0.35 0 0.58
```

Compare RGB frames and NaVILA actions. Changing camera position changes the VLM observation, not the physical controller; explain any behavioral change as a change in visual information.

## 7. Integrate a custom Go2 policy (advanced)

The default checkpoint is sufficient to reproduce the VLN baseline. To use a custom low-level policy, choose [OrcaLocomotion](https://github.com/openverse-orca/OrcaLocomotion), IsaacLab, or another training platform; training platform choice is not restricted by this project.

MJLab in Orca_VLN only runs the baseline and writes an alignment report. Custom integration must align Go2 joint order/sign, root pose, action order, control frequency, and the `vx / vy / wz / duration` velocity-command interface. See [Low-level locomotion](LOW_LEVEL_LOCOMOTION.md) for direct loading and adapter paths, and [VLN fine-tuning](VLN_FINE_TUNING.md) for high-level NaVILA SFT/LoRA.

## 8. Common failures: locate the failing layer first

| Symptom | Check first | Common cause |
| --- | --- | --- |
| `Actor does not exist` | OrcaLab scene tree | setting not imported, Go2 deleted, or actor name mismatch |
| `Failed to initialize NVML: Driver/library version mismatch` | Host NVIDIA driver | Userspace driver was updated while an older kernel module remains loaded; keep `.conda/`, reboot once, then run `nvidia-smi` and `setup_all.sh` |
| Qt cannot load the `xcb` platform plugin | Ubuntu system libraries | Rerun `setup_all.sh`, or run `setup_system_deps.sh` directly to install the required Qt/XCB packages |
| `libOpenGL.so.0: undefined symbol: _glapi_tls_Current` | An OrcaLab OpenGL front end is mixed with a different GLVND dispatcher | Pull the latest branch and rerun `setup_orcalab_env.sh`; the project binds the ELF-declared `libGL.so.1` or `libOpenGL.so.0` ABI to the matching host library |
| OrcaLab installs `orcalab-pyside` and asks for a restart | incomplete old setup | Pull the latest repository and rerun `setup_orcalab_env.sh`; Doctor verifies the native viewport, `patchelf`, and its environment-specific RPATH |
| `No module named 'deepspeed'` | NaVILA environment | Rerun `setup_navila_env.sh`; Doctor now validates the real model-builder import |
| zero or multiple Go2 actors | current scene | no complete Go2 or setting imported more than once |
| missing camera properties | `orca-lab` / `orca-gym` versions | not on 26.7.1 or using old `agentcamera` |
| VLM cannot connect | terminal 2 and port 54321 | NaVILA server is not running or port differs; for Option B, follow the remote guide's end-to-end check; for Option C, verify the SSM tunnel and managed-service health check |
| model cannot load | `NAVVLM_MODEL_PATH` | wrong directory or incomplete NaVILA environment |
| Go2 shakes or falls | checkpoint, warmup, scene start pose | incompatible checkpoint, penetration at start, or unstable policy state |

Always debug in this order: scene/actor → camera → NaVILA server → action text → Go2 policy. This avoids misdiagnosing a connectivity error as a navigation-model failure.
