# The stack: what we built, and the words for it

Written 2026-09-04. For someone meeting this project for the first time. `CONTRACT.md` states the rules and
`README.md` is the runbook — this explains what the system *is* and why each piece is there.

1. [What we did](#1-what-we-did)
2. [The components](#2-the-components)
3. [Collecting the data](#3-collecting-the-data)
4. [The dataset](#4-the-dataset)
5. [Training](#5-training)
6. [Inference — running it on the robot](#6-inference--running-it-on-the-robot)
7. [Results, and what we have not measured](#7-results-and-what-we-have-not-measured)
8. [Glossary](#8-glossary)

---

## 1. What we did

We taught a robot arm to pick up a block and place it in a bowl, by demonstration, and it now does it on its own.

The route: a person **hand-guides** the arm through the task a few dozen times → those demonstrations become a
**dataset** → the dataset **fine-tunes a pretrained robotics foundation model** on a rented cloud GPU → the
resulting **policy** runs on the computer bolted to the robot and drives the arm in real time.

Nothing here trains a model from scratch. The heavy lifting was done by π0.5, a model already trained on a large
multi-robot corpus; our 50 demonstrations only teach it *this* arm, *this* workspace and *this* task. That is why
the training bill was about **$4**.

---

## 2. The components

### Hardware

| | |
|---|---|
| **AgileX Piper** | 6-DOF robot arm with a parallel-jaw gripper. Joint limits `[±150, 0–180, −170–0, ±100, ±70, ±120]°`, gripper 0–70 mm. Commanded over **CAN** at 1 Mbps. |
| **Jetson AGX Thor** | The computer on the robot. Runs the policy locally — no cloud call in the control loop. Blackwell-class GPU (`sm_110`), which drives several toolchain quirks. |
| **USB-CAN adapter** | Links Thor to the arm. Thor's kernel has no in-tree `gs_usb` driver, so `prepare-jetson/setup.sh` builds it; without that the adapter never appears as a network interface. |
| **Top camera** | USB webcam on a stand, wide view of the whole table — 640×480. Sees where the block and bowl are. |
| **Wrist camera** | Intel RealSense D405 on the gripper — 640×480. Close-in view for the grasp itself. |
| **Quest 3 headset** | VR teleoperation. Built and working, but **superseded** by drag-teach (§3) because it was too hard to drive consistently. |

### Software

| | |
|---|---|
| **piper_sdk** | AgileX's CAN library. Everything in `piper/robot.py` wraps it. |
| **PyTorch 2.9.1+cu130** | Numerical engine. Must be the aarch64 SBSA CUDA-13 build — the only one supporting Thor's `sm_110`. |
| **LeRobot 0.6.1** | Hugging Face's robotics framework: the dataset format, the `lerobot-train` command, and the policy class we load at inference. |
| **π0.5 (`lerobot/pi05_base`)** | The pretrained model we fine-tune. See §5. |
| **Hugging Face Hub** | Stores our dataset and trained model. The rented GPU pulls from it and pushes back; nothing large goes via a laptop. |
| **Lambda Cloud** | Where training runs — one H100 rented by the hour. |

---

## 3. Collecting the data

We tried two methods and kept the second.

**VR teleoperation** (`collect/record.py`) — drive the arm from a Quest 3 controller while cameras record. It
works, and it is still in the repo, but driving smoothly enough to produce good demonstrations proved slow and
frustrating.

**Drag-teach** (`collect/teach.py`) — what we actually used. Two phases per episode:

1. **Teach.** Press the button on the arm; the joints go limp. Guide it by hand through the pick and place,
   opening and closing the gripper with your fingers. Joint angles and gripper opening are sampled at 15 Hz.
2. **Replay.** The arm re-torques and executes that trajectory **on its own** while both cameras record.

The second phase is the point. Recording the teaching directly would produce footage of a *human hand* moving the
arm — the policy would learn to imitate a scene that never occurs at inference. The replay shows the robot doing
the job alone, which is what it must reproduce.

A useful side effect: a saved trajectory can be re-recorded later (`--play`) under different lighting, or after
a bug fix, without teaching it again.

---

## 4. The dataset

`royc1981/piper_block_bowl` — **50 episodes, 20228 frames, 15 fps**, in **LeRobotDataset v3.0** format: parquet
files for the numbers, MP4 for the pixels (as stills it would be tens of GB; as video, 463 MB).

Each frame holds:

| field | type | meaning |
|---|---|---|
| `observation.images.top` | video 480×640×3 | overhead camera |
| `observation.images.wrist` | video 480×640×3 | gripper camera |
| `observation.state` | float32[7] | `j1..j6` **degrees** + gripper **mm**, as MEASURED |
| `action` | float32[7] | joints at the **next** frame + gripper commanded **now** |
| `task` | string | `"put the block in the bowl"` |

### Three details that cost real debugging time

**Action is offset by one frame from state.** `action[t]` holds the joints measured at `t+1`. That lag is what
makes it a control target — "given what you see now, where should the arm be next" — rather than a copy of the
state. `inspect_dataset.py` asserts `action[t][:6] == state[t+1][:6]` exactly.

**π0.5 expects different camera names.** It wants `base_0_rgb` and `left_wrist_0_rgb`; we store `top` and
`wrist`. Training bridges this with `--rename_map` rather than renaming our data. Get it wrong and training runs
happily with the wrong camera wired to the wrong input.

**State is joint angles, not end-effector pose.** Joint space is what the arm reports and accepts. Cartesian
pose would need an inverse-kinematics round trip every step, and this firmware **refuses unreachable targets
silently** — a failure that already cost one frozen session.

### Validate before spending money on it

```bash
python tools/inspect_dataset.py data/piper_block_bowl --strict
```

Checks schema, units, video/row alignment and normalisation stats; exits non-zero on failure. It **cannot** see
whether the episodes show the task being done — a dataset where the arm repeatedly grasped the *bowl* passed
every numeric check. Always look at the contact sheet it writes.

---

## 5. Training

### The model: π0.5

A **vision-language-action** model: two camera images + the 7-number arm state + a sentence go in; 50 future
actions come out. Ours is `lerobot/pi05_base`, **4.14 billion parameters**.

We **fine-tune** rather than train: the pretrained weights already encode what objects look like and what
"graspable" means, from a large multi-robot corpus. Our 50 episodes adjust them to this arm and task.

### Which distribution, and why it matters

π0.5 comes in two forms, and **advice written for one does not apply to the other**:

| | openpi (Physical Intelligence's own) | LeRobot port (**what we use**) |
|---|---|---|
| framework | JAX first; PyTorch secondary | PyTorch native |
| getting weights | download a JAX checkpoint, then convert it | `model.safetensors` from the Hub |
| install | patch `transformers` in site-packages, or it raises | `pip install lerobot` |
| defining your robot | register a config + write a policy class | a `--rename_map` CLI flag |
| norm stats | run `compute_norm_stats.py` | already in the dataset |
| entry point | `scripts/train_pytorch.py <config>` | `lerobot-train --policy.path=...` |

This project ran **both** — V1 (`~/Documents/piper-openpi`, legacy) on openpi, V2 on LeRobot — so that table is
from use, not from reading. LeRobot's config cites openpi line by line (`chunk_size` is openpi's
`action_horizon`, `optimizer_lr: 2.5e-5` its `CosineDecaySchedule: peak_lr`), so the architecture and
hyperparameters are meant to match. We never ran them side by side, so treat "identical outputs" as the
library's claim, not our measurement.

### The run

```bash
bash training/cloud/launch.sh royc1981/piper_block_bowl royc1981/lerobot-pi05-base-50
```

That rents a 1× H100, installs the stack, pulls the dataset from the Hub, and runs one `lerobot-train`:

```
--policy.path=lerobot/pi05_base   --policy.dtype=bfloat16
--policy.gradient_checkpointing=true
--rename_map='{"observation.images.top": "observation.images.base_0_rgb", ...}'
--batch_size=16   --steps=3000   --save_freq=500
```

| | |
|---|---|
| steps | 3000 × batch 16 = **2.4 epochs** |
| time / cost | **68 minutes, ~$4** |
| loss | 0.319 → **0.037** |
| result | `royc1981/lerobot-pi05-base-50`, 9.35 GB |

The instance verifies the upload by size **and** sha256, then terminates itself — but deliberately **not** after
a failure, so a broken run never destroys the checkpoint you paid for.

---

## 6. Inference — running it on the robot

```bash
python -m inference.run_policy --checkpoint outputs/lerobot-pi05-base-50/pretrained_model \
                               --n-action-steps 30 --max-steps 450
```

Runs **entirely on the Jetson** — no network in the control loop. The loop ticks at **15 Hz**, matching the
recording rate.

**The model is slower than the loop.** One forward pass takes ~527 ms — about 8 ticks. It survives this because
π0.5 predicts **50 actions at once**: the loop executes from the current chunk while a worker thread computes
the next one from a fresh observation, handing over before the current chunk runs out. Measured: **0 late ticks**
in 900.

Three things were needed to make it work on this hardware, each documented in `CONTRACT.md`:

- **`torch.compile` is off by default.** Triton bundles a CUDA 12.8 `ptxas` with no `sm_110a`, so its default
  compilation path fails on Thor. Eager mode costs little (~527 ms, 0 late ticks). The system CUDA 13.2
  `ptxas` does recognise `sm_110a` and can be selected for experimental compiled-mode testing; details are below.
- **`--n-action-steps 30`, not the default 10.** Every teach episode opens with ~2.8 s of stationary arm, so
  chunk actions 0–25 are settling. At 10 the loop only ever executes that prefix, re-predicts from an unchanged
  state, and **stalls forever looking like a broken policy**.
- **Chunk-boundary blending.** A fresh chunk starts where the *model* thinks the arm is, not where the last
  action put it — measured 10.6° mean jump at a boundary against 1.08° elsewhere, which is felt as jitter. The
  discontinuity is now decayed over 6 ticks.

Safety runs underneath all of it: per-tick step limits, a lead clamp, joint limits, a tool-tip floor, and a
firmware-fault stop that rests the arm. `--shadow` runs the whole loop read-only, printing what it *would* send.

### Why `torch.compile` is disabled on Thor

PyTorch can execute a model in two ways:

- **Eager mode** runs each PyTorch operation immediately using CUDA kernels shipped with PyTorch.
- **Compiled mode** captures groups of operations with TorchDynamo, optimises and fuses them with
  TorchInductor, then asks Triton to generate specialised GPU kernels. Triton's `ptxas` assembler turns those
  generated kernels into code for the target NVIDIA GPU.

The compiled path is roughly:

```text
PyTorch model -> TorchDynamo -> TorchInductor -> Triton -> ptxas -> GPU kernel
```

Thor's Blackwell GPU is an `sm_110`/`sm_110a` target. The installed `torch 2.9.1+cu130` aarch64 SBSA wheel
supports `sm_110`, so its prebuilt CUDA kernels run correctly on Thor. The problem is narrower: the current
Triton package bundles a CUDA 12.8 `ptxas` that predates `sm_110a`. When LeRobot's π0.5 sampler triggers
compilation, that assembler rejects the target:

```text
ptxas fatal : Value 'sm_110a' is not defined for option 'gpu-name'
```

This does **not** mean CUDA or the Thor GPU is unavailable, and eager mode does **not** fall back to the CPU.
Inference still runs on the GPU; it simply uses PyTorch's existing CUDA kernels instead of generating and
fusing new Triton kernels. Updating only the NVIDIA driver would not fix the mismatch because Triton would
still select its bundled userspace assembler.

[`inference/run_policy.py`](../inference/run_policy.py) therefore disables TorchDynamo unless compilation is
explicitly requested:

```python
if os.environ.get("PIPER_COMPILE") != "1":
    torch._dynamo.config.disable = True
```

Leave `PIPER_COMPILE` unset for normal dry-run, shadow, simulation, and live operation. Setting it to `0` also
keeps compilation disabled; only the exact value `1` opts into the compiled path. The policy performs a
blank-observation warm-up while it is constructed, so a compiler failure normally occurs at startup rather
than after the control loop starts.

The eager model takes roughly 420–527 ms to predict a 50-action chunk. That is slower than one 15 Hz control
tick, but the worker thread computes the next chunk while the loop consumes the current one. With
`--n-action-steps 30`, this has produced zero late ticks in the measured runs, so compilation is not required
to meet the current control-loop timing.

#### Testing with the system CUDA 13.2 `ptxas`

The Thor has a newer system assembler at `/usr/local/cuda/bin/ptxas`. Unlike Triton's bundled CUDA 12.8
assembler, it recognises `sm_110a`:

```bash
/usr/local/cuda/bin/ptxas --version
/usr/local/cuda/bin/ptxas --help | grep sm_110a
```

Triton 3.5.1 supports the `TRITON_PTXAS_PATH` environment variable, so use it to select the system assembler
without replacing files inside the Python environment or independently upgrading the PyTorch-pinned Triton
package:

```bash
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas python -c \
  "from triton import knobs; print(knobs.nvidia.ptxas.path, knobs.nvidia.ptxas.version)"
# /usr/local/cuda/bin/ptxas 13.2
```

This fixes the known architecture-recognition failure. It does not prove that every kernel Triton 3.5.1 emits
is valid or performant on Thor, so qualify the actual π0.5 workload without hardware first. Use a fresh
TorchInductor cache on a filesystem with enough free space:

```bash
CKPT=/mnt/k3s-data/clouduser/models/lerobot-pi05-base-50/pretrained_model
mkdir -p /mnt/k3s-data/clouduser/torchinductor-cache

TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
TORCHINDUCTOR_CACHE_DIR=/mnt/k3s-data/clouduser/torchinductor-cache \
PIPER_COMPILE=1 \
python -m inference.run_policy \
  --checkpoint "$CKPT" \
  --dry-run --max-steps 3
```

If the checkpoint lives elsewhere, set `CKPT` to its absolute `pretrained_model` directory. A missing local
path is interpreted as a Hugging Face repository ID and produces an unrelated `HFValidationError`.

The test must load the model, complete warm-up, and repeatedly produce valid actions without a Triton,
`ptxas`, PTX instruction, or runtime error. Then compare a longer compiled dry run with the eager baseline;
compilation is useful only if steady-state latency or resource use improves. Progress through simulation and
shadow mode before a supervised live test. Keep eager mode as the default until that qualification is complete.

If the system-assembler override exposes a different Triton code-generation error, do not spoof an older GPU
with `TRITON_OVERRIDE_ARCH`. Test a mutually compatible PyTorch, Triton, CUDA, and Jetson runtime combination
in a separate virtual environment instead of upgrading Triton underneath the working PyTorch installation.

---

## 7. Results, and what we have not measured

The policy picks up the block and places it in the bowl.

**It generalises past its training data.** Trained only on a red block — with a blue block visible in every frame
as a distractor it was taught to ignore — it picks up a **blue** block too. The pretrained priors carry "cube on
table"; colour was never the discriminating feature. Expect colour variation to need no new data, and *shape*
variation to need it.

**We have not measured a success rate.** It has worked on individual attempts; nobody has run 20 trials and
counted. That is the honest gauge of readiness and the obvious next step.

Also open: only 7 of 50 grasps are near the centre of the workspace, so that band is thin; and the multi-task
prompt idea (`"pick the red cube"` vs `"the blue cube"`) is harder than it looks, because the policy currently
grabs the salient cube regardless of colour — the language only becomes load-bearing with both cubes present.

---

## 8. Glossary

**Model** — an architecture plus its learned parameters; the thing that turns inputs into outputs. Ours is π0.5
at 4.14 B parameters. Fine-tuning changes the parameter *values*, not the architecture, which is why the base
model and our fine-tuned one load with identical code.

**Parameters / weights** — the billions of numbers inside the model that training adjusts. Ours are stored as
3.61 B in bfloat16 and 0.53 B in float32.

**Checkpoint** — a saved snapshot of a model's parameters at one moment in training, so it can be moved or
resumed. **More than weights**: ours is `model.safetensors` (the values) **plus** `config.json` (the
architecture, without which the values cannot be rebuilt into a model) **plus** the pre/post-processors (without
which inputs and outputs are on the wrong scale and the arm moves wrongly). Training wrote one every 500 steps;
we deployed the last.

**Open weights** — the parameters are downloadable, which is why this project was affordable. **Not the same as
open source**: `pi05_base` carries the **Gemma licence** (it inherits Google's Gemma as its language backbone),
and the training data is not public. You can download and adapt it; you cannot reproduce it. The surrounding
*code* (openpi, LeRobot) is open source. The distinction matters the moment someone asks about shipping it.

**Framework** — the software that defines, trains and runs models. **PyTorch** is the numerical engine;
**JAX** is Google's alternative and π0.5's reference implementation, which is why two distributions exist;
**LeRobot** is the robotics layer on top of PyTorch; **transformers** supplies the vision/language backbones.
The framework is not the model — the same weights can run under either.

**Fine-tuning** — continuing training on a pretrained model with your own, much smaller dataset, to adapt it to
a specific robot and task rather than teach it from nothing.

**VLA (vision-language-action)** — a model taking images *and* a language instruction and emitting robot
actions. The language is why the same model can in principle do several tasks.

**Policy** — the trained model in its role as a controller: observation in, action out.

**Episode** — one demonstration, start to finish. Ours run 19–35 s, ~400 frames.

**Action chunk** — π0.5 predicts **50 future actions per forward pass**, not one. This is what lets a 527 ms
model drive a 15 Hz loop, and it is also the source of the boundary jitter in §6.

**Normalisation statistics ("norm stats")** — the per-element numbers converting real-world units into the range
the network expects, and back. Joints run 0–180° and the gripper 0–70 mm; the network wants values near zero
with spread near one. They live in the dataset's `meta/stats.json` and are baked into the checkpoint, and they
**must match between training and inference** or the policy speaks units the robot does not.

π0.5 normalises on **quantiles** (`q01`/`q99` — the 1st and 99th percentiles), not mean/std and not min/max.
Why, from our own data: joint 3's `min` is **−136.5°** but its `q01` is **−101.6°**. One frame reached 35°
beyond where that joint normally travels. Had min/max set the scale, that single outlier would have compressed
every ordinary movement into a narrower band. Quantiles discard the extreme 1% at each end.

The failure this causes is nasty: a dataset without `q01`/`q99` trains *fine for a while* and then fails, because
the error only surfaces when normalisation is applied. `inspect_dataset.py --strict` checks up front, which is
cheaper than finding out on a rented GPU.

**Teleoperation** — driving the robot remotely in real time (here, a Quest 3 controller). Distinct from
drag-teach, where you move the arm with your hands.

**Drag-teach** — putting the arm in a limp mode and physically guiding it. Also called kinesthetic teaching.

**Epoch** — one pass through the whole dataset. Our 3000 steps × batch 16 over 20228 frames is 2.4 epochs.
