# What Hugging Face provides, and what our dataset actually is

Written 2026-09-04. Companion to `CONTRACT.md` (which states the rules) — this one explains *why* the pieces
are what they are, for someone meeting the stack for the first time.

## Part 1 — What Hugging Face provides

Four distinct things, easily conflated because they share one brand and one auth token.

### 1. The pretrained model we fine-tune: `lerobot/pi05_base`

Hugging Face's PyTorch port of Physical Intelligence's **π0.5**, a vision-language-action (VLA) model. It takes
camera images, a robot state vector and a language instruction, and predicts a chunk of future actions.

**This is the expensive part we did not pay for.** It arrives already knowing what objects look like, what
"graspable" means, and roughly how manipulation unfolds — learned from a large multi-robot corpus. Our 50
episodes only teach it *this* arm, *this* workspace and *this* task.

Two consequences worth internalising:

- **~$4 of training was enough.** Fine-tuning 3000 steps took 68 minutes on one H100. Training a comparable
  model from scratch is not a thing a project this size does.
- **It generalises past its fine-tuning data.** Trained only on a red block — with a blue block visible in every
  frame as a distractor it was taught to ignore — it picks up a blue block anyway. The priors carry "cube on
  table"; colour was never the discriminating feature. Expect colour variation to need no new data and *shape*
  variation to need it.

`--policy.path=lerobot/pi05_base` in `training/cloud/train.sh` is the only place this is named.

### 2. The LeRobot library (v0.6.1) — the framework, not just a model host

- the **dataset format** (`LeRobotDataset`, see Part 2)
- the **`lerobot-train` CLI** that runs the fine-tune
- the **`PI05Policy`** class `inference/run_policy.py` loads to run on the robot
- video encoding/decoding, normalisation, the pre/post-processors saved alongside a checkpoint

### 3. Hub storage for our own artifacts

| | |
|---|---|
| dataset | `royc1981/piper_block_bowl` (private) |
| trained model | `royc1981/lerobot-pi05-base-50` (private) |

This is what makes cloud training possible at all: the rented H100 pulls the dataset from the Hub and pushes the
checkpoint back — verified by size **and** sha256 before the instance is allowed to terminate. Neither artifact
passes through a laptop. It is also the only real backup: a local dataset was destroyed once in this project,
and was unrecoverable precisely because it had never been pushed.

### 4. Auth and transfer

The `hf` CLI and `huggingface_hub`, driven by a write token. `tools/push_dataset.py` and
`training/pull_checkpoint.sh` wrap them with confirmation and verification. Token handling is in `RESTORE.md`.

### What Hugging Face is NOT doing here

No hosted inference, no hosted training, no data labelling. Training runs on a **Lambda Cloud** H100 we rent
directly; inference runs on the **Jetson Thor** on the robot. HF is the model zoo, the library, and the artifact
store.

## Part 2 — The dataset we built

### The format: `LeRobotDataset` v3.0

Not one file — a directory that separates cheap numeric columns from expensive pixels:

```
data/piper_block_bowl/
  meta/info.json          schema, fps, episode and frame counts
  meta/tasks.parquet      the language instruction(s)
  meta/stats.json         normalisation statistics (see the quantile note below)
  meta/episodes/          per-episode index: length, where its rows and video live
  data/chunk-000/*.parquet    per-frame numeric columns
  videos/observation.images.<cam>/chunk-000/*.mp4   one video per camera per episode
```

Frames are stored as **video**, not PNGs. 20228 frames × 2 cameras as stills would be tens of GB; as MP4 the
whole dataset is 463 MB. Decoding is `pyav` (the `torchcodec` default is unavailable on this platform).

### What one frame contains

| field | type | meaning |
|---|---|---|
| `observation.images.top` | video 480×640×3 | overhead camera — where the block and bowl are |
| `observation.images.wrist` | video 480×640×3 | gripper camera — close-in view for the grasp |
| `observation.state` | float32[7] | `j1..j6` in **degrees** + gripper opening in **mm**, as MEASURED |
| `action` | float32[7] | joints at the **next** frame + the gripper commanded **now** |
| `task` | string | `"put the block in the bowl"` |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | | bookkeeping |

Ours: **50 episodes, 20228 frames, 15 fps, robot_type `piper`**.

### Three details that caused real debugging

**1. `action` is offset by one frame from `observation.state`.** `action[t]` holds the joint angles measured at
`t+1`. That one-frame lag is what makes it a *control target* — "given what you see now, where should the arm be
next" — rather than a redundant copy of the state. `collect/record.py`'s `Episode` class implements it by
holding each frame back until the following one arrives. `inspect_dataset.py` verifies
`action[t][:6] == state[t+1][:6]` exactly; a non-zero result means the dataset teaches the wrong thing while
looking perfectly valid.

**2. π0.5 expects different camera names than we use.** It wants `base_0_rgb` and `left_wrist_0_rgb`; we store
`top` and `wrist`. Training bridges this with a CLI flag rather than renaming our data:

```
--rename_map='{"observation.images.top": "observation.images.base_0_rgb",
               "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

Get this wrong and training runs happily, having wired the wrong camera to the wrong input.

**3. π0.5 normalises on QUANTILES, not mean/std.** `meta/stats.json` must contain `q01` and `q99` per feature.
LeRobot writes them automatically, but a hand-built or converted dataset may not have them — and the failure
surfaces only once training has started. `inspect_dataset.py --strict` checks for them.

### Why the state is joint angles rather than end-effector pose

7-D joint space (`j1..j6` + gripper mm) is what the arm actually reports and accepts. Cartesian pose would need
an IK round-trip at every step, and this arm's firmware IK **refuses unreachable targets silently** — a failure
mode that already cost this project a frozen session. Joint space has no such ambiguity.

### Validate before spending money on it

```bash
python tools/inspect_dataset.py data/piper_block_bowl --strict
```

Schema, units (degrees not radians, joints inside limits, a gripper action the arm can actually reach), one
decoded video frame per parquet row, quantiles present. Exits non-zero on failure.

**It cannot see whether the episodes show the task being done.** A dataset where the arm repeatedly grasped the
*bowl* passed every numeric check. Always look at the contact sheet it writes.
