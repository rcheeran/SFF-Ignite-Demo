---
name: project_piper_groot_training
description: "Fine-tuning Isaac GR00T N1.7 for the real AgileX Piper arm — env setup, gotchas, dataset format"
metadata: 
  node_type: memory
  type: project
  originSessionId: 54b08491-128c-4bce-85b2-a18b244c58f4
---

Goal: give the Piper its own VLA by fine-tuning `nvidia/GR00T-N1.7-3B` on scripted pick-place demos
collected on the real arm, then serving it back through `piper-real-toolset` for closed-loop rollout.
Plan file: `/home/rog01/.claude/plans/stateless-baking-sundae.md`. Related: [[project_asimov_robotics]].

**Env (Phase 1, DONE):** repo cloned to `Documents/Isaac-GR00T`, uv venv at `.venv` (Python 3.12,
torch 2.9.0+cu128, CUDA works on the 5080 despite the host's CUDA-13.2 driver). `uv` at `~/.local/bin`.
- Install = `uv sync --python 3.12` (README). GPU deps (flash-attn URL wheel) included; no LoRA exists.
- **Two local pyproject.toml edits were required** to make `uv sync` work on this x86_64-only box:
  (1) `required-environments` trimmed to x86_64 only; (2) removed the `[tool.uv.sources]` torchcodec
  aarch64 path wheel (it's a git-lfs pointer → "invalid zip"). torchcodec 0.8.0 then resolves from PyPI.
  Deleted `uv.lock` (saved as `uv.lock.orig`) to force a fresh lock.
- git-lfs installed standalone to `~/.local/bin/git-lfs` (no sudo). Used `git lfs pull --include=...`
  to fetch `demo_data/cube_to_bowl_5` (the reference SO100 dataset — 5 episodes, all meta files).
- **HF token gotcha:** `Documents/HG-TOKEN.txt` has the real `hf_...` token on LINE 1 (37 chars) plus a
  stray label line below — use `head -1 | tr -d '[:space:]'`. Auth user `royc1981`; both gated repos
  `nvidia/Cosmos-Reason2-2B` and `nvidia/GR00T-N1.7-3B` are accessible. Persisted via `hf auth login --token`.
- ffmpeg CLI absent but FFmpeg-6 shared libs (`libavcodec.so.60` etc.) are on the system → torchcodec
  decode works with no install. sudo still needs the user.

**Dataset = GR00T-flavored LeRobot v2** (mirror `demo_data/cube_to_bowl_5`). Piper is **7-dim**
(6 joints + 1 gripper) vs SO100's 6-dim. Layout: `data/chunk-000/episode_000000.parquet`
(cols: observation.state[7], action[7], timestamp, frame_index, episode_index, index, task_index),
`videos/chunk-000/observation.images.wrist/episode_000000.mp4` (encode **H.264**, not demo's av1),
`meta/{info.json,modality.json,tasks.jsonl,episodes.jsonl}`. modality.json: single_arm start0 end6,
gripper start6 end7, video key `wrist`, annotation `human.task_description`→original_key `task_index`.
**stats.json / relative_stats.json are auto-generated** by `gr00t/data/dataset/factory.py`
(`generate_stats`/`generate_rel_stats`) on first load — don't hand-write them.

**Modality config** = copy `examples/SO100/so100_config.py`, drop the `front` video key, register under
`EmbodimentTag.NEW_EMBODIMENT`. Action space: single_arm RELATIVE (joint), gripper ABSOLUTE. Action
horizon reduced 16→8 in `piper_modality.py` (memory). All pipeline code lives in `Documents/piper-groot/`
with a README runbook; **whole pipeline validated end-to-end on a mock dataset** (record→build→train→
serve→rollout→open_loop_eval all green).

**Serve/rollout contract (verified):** server = `deploy/serve_piper.sh` (run_gr00t_server, ZMQ :5555,
reads modality from checkpoint). Client `deploy/rollout_client.py` runs in the HW venv with a minimal
msgpack_numpy ZMQ client (no full gr00t import). Wire: send `{"endpoint":"get_action","data":
{"observation":obs,"options":None}}`; response is `(action, info)`. Observation is NESTED:
`{"video":{"wrist":uint8 (B,T,H,W,C)}, "state":{"single_arm":f32 (B,T,6),"gripper":f32 (B,T,1)},
"language":{"annotation.human.task_description":[[task]]}}`. Action returns keys `single_arm` (B,H,6)
+ `gripper` (B,H,1), absolute joint deg + 0..1. HW venv got `pyzmq msgpack msgpack-numpy`.

**Fine-tune** (no LoRA): use `Documents/piper-groot/finetune_piper.sh` (wraps `launch_finetune.py`).
Env knobs it exposes: DATASET, OUTPUT, MAX_STEPS, SAVE_STEPS, BATCH, GRAD_ACCUM, EDGE (image
downscale), TUNE_DIFFUSION/TUNE_PROJECTOR, GR00T_OPTIM, GR00T_LOAD_BF16, GR00T_TRAINABLE_FP32,
GR00T_GRAD_CKPT. Baseline sanity: run the as-shipped SO100 command on `demo_data/cube_to_bowl_5`.
**Verify offline with `gr00t/eval/open_loop_eval.py`** (MSE/MAE vs GT, plots) — no arm needed.
Full finetune flag list at `Documents/piper-groot/_finetune_help.txt`.

**Blockers hit + fixes (all real, durable):**
- deepspeed import crashed on `CUDA_HOME does not exist` (no CUDA toolkit installed). gr00t only uses
  deepspeed when num_gpus>1, so **uninstalled deepspeed** — single-GPU trains fine without it.
- torchcodec couldn't load libav (`libavdevice.so.60` missing on Ubuntu 24.04 base). Fix: userspace
  **micromamba** env `~/.local/ffmpeg-env` with conda-forge `ffmpeg<8`; put its `lib` on `LD_LIBRARY_PATH`
  (baked into finetune_piper.sh / serve_piper.sh). Encoding uses `imageio-ffmpeg` (added to gr00t venv).
- Made env-overridable in `launch_finetune.py` (local edits): `optim`, `gradient_checkpointing`,
  `load_bf16`, `backbone_trainable_params_fp32`.
- `bitsandbytes` installed for 8-bit optimizers.

**16 GB VRAM reality (RTX 5080):** GR00T-N1.7-3B is 3.14B params; tuning projector+diffusion head =
**1.62B trainable**, a fixed ~13.2 GB floor (params+grads+diffusion activations) that image size /
horizon / 8-bit/paged optimizer barely move. Full-policy fine-tune **OOMs by ~700 MB** — the ~1 GB the
desktop GUI holds is the difference. So for the REAL policy run either (a) free desktop VRAM (train from
a TTY / stop gdm) so the diffusion-head config fits, or (b) CPU-offload (deepspeed/FSDP ZeRO, needs a
CUDA toolkit via micromamba). Pipeline mechanics (train→save→serve→rollout) were validated with a
**projector-only** fit (`TUNE_DIFFUSION=0`), which fits 16 GB comfortably.
