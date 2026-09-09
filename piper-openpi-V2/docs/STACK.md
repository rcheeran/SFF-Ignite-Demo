# What Hugging Face provides, and what our dataset actually is

Written 2026-09-04. Companion to `CONTRACT.md` (which states the rules) — this one explains *why* the pieces
are what they are, for someone meeting the stack for the first time.

- **Part 0** — the vocabulary: model, checkpoint, open weights, framework, norm stats
- **Part 1** — what Hugging Face supplies, and how the LeRobot π0.5 port differs from Physical Intelligence's own
- **Part 2** — the dataset we built, and the three details in it that caused real debugging

## Part 0 — The vocabulary, in terms of this project

Five words that get used interchangeably and are not the same thing. Each is anchored to a real artifact here.

### Model

The **architecture plus the learned parameters** — the thing that turns inputs into outputs. Ours is π0.5:
two camera images + a 7-number arm state + a sentence go in, 50 future actions come out.

Concretely, our fine-tuned model is **4.14 billion parameters** (3.61 B stored as bfloat16, 0.53 B as float32).
"Fine-tuning" changed the *values* of those parameters; it did not change the architecture, which is why
`lerobot/pi05_base` and our `lerobot-pi05-base-50` load with identical code.

A model with no parameter values is just a shape — it produces noise. Which is why the file matters:

### Checkpoint

**A saved snapshot of a model's parameters at one moment in training** — the model frozen to disk so it can be
moved, shared, or resumed. Ours:

```
outputs/lerobot-pi05-base-50/pretrained_model/
  model.safetensors        9.35 GB   the 4.14 B parameter values
  config.json                        the architecture, so the values can be rebuilt into a model
  train_config.json                  what produced it (dataset, steps, batch size, LR)
  policy_preprocessor.json  + .safetensors    normalisation to apply to inputs
  policy_postprocessor.json + .safetensors    un-normalisation to apply to outputs
  train.log                          the run that made it
```

Note that a checkpoint is **more than weights**. Without `config.json` you cannot reconstruct the architecture;
without the pre/post-processors the numbers going in and out are on the wrong scale and the arm moves wrongly.
Losing any of them makes the 9.35 GB useless — one reason `pull_checkpoint.sh` verifies size *and* sha256.

"Checkpoint" also implies *a point in time*. Training wrote one every 500 steps; we deployed
`checkpoints/last/`. If a run diverges you can go back to an earlier one.

### Open weights (≠ open source)

π0.5's **parameters are downloadable** — that is what makes any of this affordable, since we get the result of
a very large training run for the cost of a download. But downloadable weights are not the same as open source:

- `lerobot/pi05_base` carries the **Gemma licence** (it inherits Google's Gemma as its language backbone), not
  MIT or Apache. It has terms — read them before anything commercial.
- The **training data is not public**. We can use and fine-tune the model; we cannot reproduce it.

So: *open weights* means you can download and adapt the parameters. *Open source* would mean the code, data and
recipe are all public and re-runnable. The **code** around it (openpi, LeRobot) is open source; the model is
open-weights. The distinction matters the moment someone asks whether this can ship in a product.

### Framework

**The software that defines, trains and runs models.** Several are in play, at different layers:

| framework | role here |
|---|---|
| **PyTorch** (2.9.1+cu130) | the numerical engine — tensors, autograd, GPU kernels |
| **JAX** | Google's alternative to PyTorch; π0.5's *reference* implementation is JAX-first, which is the root of the openpi/LeRobot split below |
| **LeRobot** (0.6.1) | robotics layer on top of PyTorch — dataset format, `lerobot-train`, the policy classes |
| **transformers** | supplies the vision/language backbones π0.5 is built from |

The framework is not the model. The same π0.5 weights can be executed by PyTorch or JAX — which is exactly what
the two distributions do.

### Normalisation statistics ("norm stats")

**The numbers that convert your robot's real-world units into the range the network expects, and back again.**

The network works best on inputs roughly centred on zero with a spread near one. Our joints run 0–180° and the
gripper 0–70 mm. Norm stats are the per-element scale and offset that map between the two — and they must be
*the same numbers* at training and inference, or the policy is fluent in units the robot does not speak.

They live in the dataset at `meta/stats.json`, computed from the data itself, and are baked into the checkpoint
as the pre/post-processors listed above. Ours, for `action`:

```
              j1       j2       j3       j4       j5       j6     grip
min        -63.2     -0.1   -136.5    -35.2    -52.6    -65.3      0.0
q01        -26.6      0.0   -101.6    -12.4      3.6    -26.3     -0.0
mean         2.3     66.5    -51.3      3.5     44.2     -3.9     20.3
q99         33.1    127.3      0.0     19.7     70.1     17.7     69.4
max         67.5    147.7      0.1     46.9     70.2     59.5     70.0
```

**π0.5 normalises on `q01`/`q99` — the 1st and 99th percentiles — not mean/std and not min/max.** Look at `j3`:
`min` is −136.5 but `q01` is −101.6. Some single frame reached 35° beyond where the joint normally travels. Had
min/max set the scale, that one outlier would have compressed every ordinary movement into a narrower band.
Quantiles discard the extreme 1% at each end, so the scale reflects the motion that actually matters.

The practical failure this causes: a dataset without `q01`/`q99` in `stats.json` trains **fine for a while and
then fails**, because the error only appears when normalisation is applied. `inspect_dataset.py --strict`
checks for them up front, which is cheaper than discovering it on a rented GPU.

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

#### How the LeRobot port differs from Physical Intelligence's own π0.5

Same model, two distributions. PI's reference implementation lives in
[openpi](https://github.com/Physical-Intelligence/openpi) and is **JAX-first**; LeRobot ships a PyTorch port
that announces itself on load as *"a direct port of the OpenPI implementation… follows the original OpenPI
structure for compatibility"*, and its config file cites openpi line by line — `chunk_size` is openpi's
`action_horizon`, `optimizer_lr: 2.5e-5` is openpi's `CosineDecaySchedule: peak_lr`, `tokenizer_max_length: 200`
matches openpi's `__post_init__`. So the **architecture and hyperparameters are meant to match**; what differs
is everything around them.

**This project ran both.** V1 (`~/Documents/piper-openpi`, now legacy) used openpi directly; V2 uses LeRobot.
The difference in what you have to do:

| | openpi (PI's own) | LeRobot port (what we use) |
|---|---|---|
| framework | JAX first; PyTorch path is secondary | PyTorch native |
| getting the weights | download a **JAX** checkpoint from GCS, then run `convert_jax_model_to_pytorch.py` | `model.safetensors` pulled straight from the Hub |
| making it importable | **patch site-packages** — copy `transformers_replace/*` over the installed `transformers`, or it raises | plain `pip install lerobot` |
| defining your robot | register a config in `config.py` + write a policy class mapping your cameras/state | none — a `--rename_map` CLI flag |
| normalisation stats | run `scripts/compute_norm_stats.py`, writes to `assets/` | already in the dataset's `meta/stats.json` |
| dataset | LeRobot format either way | LeRobot format, native |
| training entry point | `scripts/train_pytorch.py <config>` | `lerobot-train --policy.path=...` |

**Why we moved to LeRobot.** V1's route was a five-step bootstrap where each step could fail on this hardware —
JAX installed purely so `openpi.models.gemma` can import it, a `transformers` patch that a routine `pip install`
silently undoes, and a JAX→PyTorch conversion of a 12 GB checkpoint before a single training step. The LeRobot
port removes all of it. `training/cloud/train.sh` is one `lerobot-train` invocation.

**The practical consequence: advice written for openpi does not apply here.** No `config.py` to edit, no
`compute_norm_stats.py`, no `transformers_replace`, no checkpoint conversion. Searching for π0.5 fine-tuning
guidance mostly surfaces openpi instructions, and following them here wastes hours.

**What we have not verified:** that the two produce numerically identical outputs from identical inputs. We
never ran openpi and LeRobot side by side on the same checkpoint. The port is stated to be direct and the
hyperparameters match, but treat "identical" as a claim from the library, not something this project measured.

**Shared regardless of port** (these are π0.5 properties, not packaging):

- **normalisation on QUANTILES** (`q01`/`q99`), not mean/std — `NormalizationMode.QUANTILES` for state and action
- **state and action padded to 32 dimensions** (`max_state_dim`/`max_action_dim`); our 7-D vectors are padded
- **50-action chunks** predicted per forward pass, via flow matching (10 inference steps)
- **camera slots** named `base_0_rgb` / `left_wrist_0_rgb`, with missing slots padded and masked

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
