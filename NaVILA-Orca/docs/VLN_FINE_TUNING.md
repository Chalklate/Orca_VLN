<p align="right"><sub><strong>English</strong> · <a href="VLN_FINE_TUNING_zh.md">中文</a></sub></p>

# High-level VLN adaptation: SFT and LoRA

This track changes the decision model, not the walking controller. Keep the Go2 locomotion checkpoint fixed while collecting and evaluating high-level navigation data. The stable interface is the canonical action vocabulary described in [Architecture](ARCHITECTURE.md).

## Start with a baseline, not training

Reproduce the default case exactly as described in
[Getting started](GETTING_STARTED.md), inspect the RGB frames, and review the
action sequence. Do not change the camera convention, scene layout, or action
interface during collection. The collection flag below only records the
existing model inputs and does not change the command vocabulary or controller.

## Collect reviewable examples

The repository now includes a model-agnostic collection and review path. NaVILA
training releases can still differ in image history, prompt, token, and label
formats, so the exported JSONL remains an intermediate review format rather
than a direct training file.

Run the planner once, then collect several decisions from one simulator
process and starting pose:

```bash
./scripts/run_orcalab_scene_locomotion.sh \
  --waypoint-instruction-file outputs/memory_guide/latest_waypoints.txt \
  --output outputs/training_runs/bread_start_from_table \
  --collect-decisions \
  --image-interval 0.2 \
  --max-decisions 8
```

`--collect-decisions` warms up eight real camera frames before the first VLM
request and records the exact eight images used for every subsequent request.
Each run contains `decision_samples/decision_XXXX/` and a JSONL manifest. The
warmup is stationary; it does not create a new movement label.

Review a single decision with the canonical action vocabulary:

```bash
python scripts/label_decision_samples.py \
  outputs/training_runs/bread_start_from_table \
  --decision 1 \
  --label "turn right 15 degrees"
```

Or label all unreviewed decisions interactively:

```bash
python scripts/label_decision_samples.py \
  outputs/training_runs/bread_start_from_table --interactive
```

Export only reviewed records for later conversion to the exact NaVILA release
format:

```bash
python scripts/export_vln_sft_records.py \
  outputs/training_runs/bread_start_from_table \
  --decision-samples \
  --output outputs/training_records/reviewed.jsonl
```

Existing rollout directories without `decision_samples/` remain valid raw
rollout evidence, but they cannot be retroactively reconstructed into exact
eight-frame decision samples. Keep them for debugging; collect new samples
with `--collect-decisions` for training.

Teams should prepare data against the exact NaVILA training release they use
and retain at least:

- the original instruction and consecutive ego-view images;
- scene, episode, and timestep identifiers;
- the baseline output and a human-reviewed target from the canonical action
  vocabulary;
- explicit `unreviewed` / `reviewed` status and reviewer provenance.

Baseline actions are review candidates, **not ground truth**. Check image,
instruction, and target-action alignment record by record before converting
the data into the selected training release's official template.

## SFT direction

Use SFT when the desired improvement is mostly semantic or task-specific:

- factory terminology: red bins, blue oil barrels, robotic arms, inspection points;
- a clearer stop policy near a target or hazard;
- staged routes such as “reach shelf A, then inspect the aisle”;
- action wording that remains inside the supported command vocabulary.

A practical first dataset is small and curated: collect a fixed set of scene states, write one unambiguous instruction for each, review the expected next action, then hold out different camera positions and object placements for evaluation. Keep the assistant target to one action; the runtime already turns that action into a fixed motion chunk.

## LoRA direction

Use LoRA when a full NaVILA fine-tune is unnecessary. In the NaVILA training environment:

1. Start from the provided NaVILA checkpoint.
2. Freeze the base model and train adapters on the reviewed navigation records.
3. Keep the image-history length and prompt format aligned with deployment.
4. Merge or load the adapter according to the selected NaVILA release.
5. Re-run the same fixed Orca_VLN episodes before changing scene assets.

The exact target modules, image processor, and launch command are determined by the NaVILA release used by the organizer. The competition baseline intentionally does not hard-code them.

## Evaluation checklist

Report more than whether the robot eventually moves:

- valid-action rate: output parses into exactly one permitted action;
- instruction adherence: correct turn/forward/stop decision for a reviewed state;
- visual grounding: behavior changes when the target object or camera view changes;
- closed-loop outcome: final distance to goal, path trace, and saved RGB evidence;
- regression: baseline factory runs still work with the adapted model.

If the model proposes an invalid action, fix the high-level data, prompt, or model output—not the Go2 policy.
