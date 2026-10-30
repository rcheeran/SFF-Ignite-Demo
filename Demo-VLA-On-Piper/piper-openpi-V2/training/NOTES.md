# LeRobot / pi05 on the Thor — what is installed, what works, what bit us

## Environment (`source ~/lerobot-venv/bin/activate`)

| package | version | note |
|---|---|---|
| lerobot | 0.6.1 | `pip install "lerobot[pi,av-dep]==0.6.1"` + `datasets pandas pyarrow jsonlines accelerate` (the `dataset`/`training` extras minus torchcodec/wandb) |
| torch / torchvision | 2.9.1+cu130 / 0.24.1 | installed FIRST from `https://download.pytorch.org/whl/cu130` so pip never swaps in the CPU aarch64 wheel; `torch.cuda.is_available()` True, `torch.version.cuda` 13.0 |
| transformers | 5.5.4 | pulled by the `pi` extra |
| av (PyAV) | 15.1.0 | wheel bundles ffmpeg 7.1 with libsvtav1, libx264/x265, nvenc, dav1d — no system ffmpeg needed |
| torchcodec | not installed | lerobot pins `>=0.11` on aarch64 but 0.11 needs torch 2.11; lerobot falls back to pyav decoding automatically (`--dataset.video_backend=pyav`) |
| accelerate | 1.14.0 | `lerobot-train` needs it; wandb is imported lazily, not installed |
| hardware | piper_sdk 0.6.2, pyrealsense2 2.58.3, python-can 4.6.1, scipy 1.18.1, aiohttp 3.14.3, opencv-python 4.13 (pinned <4.14 next to lerobot's opencv-python-headless) |
| sim | mujoco (aarch64 wheel, EGL offscreen rendering on the Thor) — see README 'Simulation' |
| dev | pytest |

`pip check` only complains that `nvidia-cusparselt-cu13` is "not supported on this platform" — harmless, CUDA works.

## Dataset (see `training/make_fake_dataset.py`, the reference for `collect/record.py`)

`python -m training.make_fake_dataset` → `data/fake_block_bowl` (3 eps × 60 frames, 10 s wall, 672 KB; features/names come from `piper.config`).
Facts the recorder must follow:
- features: `observation.images.{top,wrist}` `{"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channels"]}`,
  `observation.state` / `action` `{"dtype": "float32", "shape": (7,), "names": [...]}`.
- `LeRobotDataset.create(repo_id, fps=15, features=..., root=..., robot_type="piper", use_videos=True, image_writer_threads=4)`
- `add_frame({...every feature..., "task": "put the block in the bowl"})` — images uint8 HWC **RGB** (cv2 gives BGR),
  numeric values `np.ndarray` with the exact dtype/shape. Never pass `timestamp`/`frame_index`: add_frame computes them from fps.
- `save_episode()` encodes + writes parquet and clears the buffer; `clear_episode_buffer()` to discard; `finalize()` once at the end (mandatory).
- `push_to_hub(private=..., tags=[...])` after finalize; needs `HF_TOKEN`.
- Read back: `LeRobotDataset(repo_id, root=root)`; `ds[i]` gives float32 CHW images in [0,1], stats include q01/q99 (pi05 needs QUANTILES).
- Resume: `LeRobotDataset.resume(repo_id, root=..., image_writer_threads=4)` needs an explicit root, loads only meta/ and
  appends new data/meta/video files (file-001...). Episode metadata is buffered in memory (`metadata_buffer_size`, default
  10 episodes) and only flushed by finalize(): a SIGKILL / power loss after a saved episode would leave info.json counting
  episodes whose meta/episodes rows were never written → the dataset does not load and cannot be resumed. `collect/record.py`
  defers Ctrl-C around save_episode()/finalize() and keeps the buffer at 1 episode for this reason.
- Noise: `SVT_LOG=1` silences the SVT-AV1 banner, `av.logging.set_libav_level(av.logging.ERROR)` the mp4 muxer, and
  `datasets.disable_progress_bar()` must be called AFTER resume() because `lerobot.utils.utils.SuppressProgressBars`
  re-enables bars on exit.

## Training (local fallback — real runs go to the cloud, see "Cloud" below)

Base model: `--policy.path=lerobot/pi05_base` straight from the Hub (the PaliGemma license is accepted on the account).
`prepare_pi05_base.py` is gone and `models/pi05_base` (14 GB) is unreferenced — `rm -rf models/pi05_base` when disk matters.

```
bash training/train.sh smoke [data/fake_block_bowl]    # 500 steps, batch 8, one checkpoint at 500 → outputs/pi05_piper_smoke
bash training/train.sh full  <hf_user>/piper_block_bowl # 3000 steps, batch 16 → outputs/pi05_piper_local
```
`training/train.sh` is a 15-line wrapper over `training/cloud/train.sh` — ONE flag set for the cloud and the Thor
(`OUT=outputs/pi05_piper_smoke COMPILE=false bash training/cloud/train.sh data/fake_block_bowl - 500 8`), i.e.:
```
lerobot-train --dataset.repo_id=local/fake_block_bowl --dataset.root=data/fake_block_bowl --dataset.video_backend=pyav \
  --policy.path=lerobot/pi05_base --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.compile_model=false --policy.push_to_hub=false \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --num_workers=4 --log_freq=10 --wandb.enable=false --batch_size=8 --steps=500 --save_freq=500 \
  --output_dir=outputs/pi05_piper_smoke --job_name=pi05_piper_smoke
```
The exact command is echoed at the top of `~/train.log`. Checkpoints land in `outputs/<run>/checkpoints/<step>/pretrained_model/`
(`config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json` + normalizer stats) with a
`checkpoints/last` symlink; lerobot always saves after the final step.

Measured on the Thor, 20-step run on the fake dataset (batch 8, bf16, gradient checkpointing, no torch.compile):
- **5.0 s/step** steady state (`updt_s` mean 5.02, max 5.23 over steps 2–19; data loading 0.007 s/step with 4 pyav workers).
  Batch 16 is untested here; expect ~9–10 s/step, i.e. the old openpi number.
- **peak GPU allocation 37.1 GB** (`torch.cuda.max_memory_allocated`); host `free` shows 54 GB used while training and
  66 GB while writing the checkpoint (unified memory, so those are the numbers that matter on the Thor).
- startup ~3 min (14.5 GB safetensors load + bf16 cast) before step 1; first step pays ~5 s of dataloader spawn.
- a checkpoint is **23 GB** (9.4 GB bf16 model + 15 GB AdamW state) and takes ~30 s to write; 5 of them for the
  500-step smoke, so keep an eye on disk if `save_freq` is lowered.
- 500-step smoke ≈ 3 min load + 500 × 5.0 s + 1 × 30 s ≈ **45 min**; 3000 steps at batch 16 ≈ 8–9 h locally.
- the trained loss on the synthetic data drops 0.87 → ~0.1 in 20 steps (sanity only, the data is fake).

## Inference (`inference/run_policy.py`)

```python
from inference.run_policy import Pi05Runner, ActionStream
runner = Pi05Runner(ckpt, task="put the block in the bowl")       # ~2.5 min load + 0.85 s CUDA warm-up
chunk = runner.predict_chunk(top_bgr, wrist_bgr, state7)          # float32[50, 7]: next 50 frames of j1..j6 deg, gripper mm (~500 ms)
stream = ActionStream(runner.predict_chunk, n_action_steps=10)    # worker thread; per tick: action, hold = stream.next((top, wrist, state7), t)
stream.close()                                                    # at exit (a live CUDA thread at interpreter exit aborts the process)
```
Under the hood: `PreTrainedConfig.from_pretrained(ckpt)` → `PI05Policy.from_pretrained(ckpt, config=cfg)`,
`make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": "cuda"}})`,
`prepare_observation_for_inference(obs, device, task)` → preprocessor → `policy.predict_action_chunk(batch)` →
the postprocessor on the whole (50, 7) chunk (unnormalize is per element; identical to per-action postprocessing).
- Measured on the Thor with the loop running (bf16 smoke checkpoint, 2×480×640 frames, 10 flow steps): model 498 ms mean /
  516 max per chunk in the worker thread, loop work 0.8 ms/tick, tick period 66.7 ms (15.0 Hz), 0 stalls over 150 ticks,
  GPU peak 8.9 GB (37 GB was training).
- Chunk prefetch: the next chunk is requested when `PREFETCH_AT` = 9 of the 10 executed actions remain and takes over
  aligned by the number of actions executed since its observation; plan age 0.6 s at hand-over, 1.2 s just before the
  next one. A late model keeps serving the live 50-step chunk (`late` counter); only an exhausted chunk holds the last
  action. Not using lerobot's async inference / RTC: a gRPC client/server split plus a policy-config change for a
  single-machine deploy.
- The checkpoint's preprocessor carries the camera rename (`top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`) and the
  dataset's q01/q99 normalization stats; actions come out in joint degrees / gripper mm.

## Pitfalls hit

1. **torch wheel**: `pip install lerobot` on aarch64 resolves a CPU-only torch. Install `torch==2.9.1 torchvision==0.24.1`
   from the cu130 index first; lerobot's `torch<2.12` constraint then keeps it.
2. **torchcodec**: no compatible aarch64 wheel for torch 2.9.1 (lerobot wants `>=0.11` = torch 2.11). Skipped; pyav backend
   works for encode (libsvtav1, default crf 30, g 2) and decode. Expect a one-line "falling back to pyav" warning.
3. **Gated tokenizer — OBSOLETE 2026-08-21, license accepted, `lerobot/pi05_base` loads as-is (see Cloud)**:
   pi05's preprocessor loads `google/paligemma-3b-pt-224` (gated, the HF account got 403 on it).
   The public `unsloth/gemma-2b` tokenizer is the identical SentencePiece model for all text ids (< 256000; checked
   id-for-id against the big_vision `paligemma_tokenizer.model` openpi uses); PaliGemma only adds `<loc>/<seg>` ids
   ≥ 256000 which pi05 prompts never emit. `prepare_pi05_base.py` rewrites the tokenizer name in a local copy of
   the base model. Accept the license at https://huggingface.co/google/paligemma-3b-pt-224 to use `lerobot/pi05_base` as-is.
   Note `--policy.type=pi05` (from scratch, no `--policy.path`) hardcodes the gated name in `processor_pi05.py` and
   fails with a 403 until then — always go through `--policy.path=lerobot/pi05_base`.
4. **CLI key is `--policy.path`**, not `--policy.pretrained_path` (draccus path arg). pi05_base's config says
   `device: mps`, so `--policy.device=cuda` is mandatory.
5. **Camera names**: pi05_base declares `observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}` and
   `make_policy` refuses datasets whose camera set is neither a subset nor superset — hence `--rename_map`. The rename
   is stored in the checkpoint's preprocessor, so inference feeds the contract names (`top`/`wrist`). The missing
   third camera is padded with -1 and masked, as pi05 does natively.
6. `--output_dir` must not exist (no implicit resume); `--resume=true --config_path=<ckpt>/train_config.json` resumes.
7. opencv: `opencv-python` unpinned pulls 5.0 beside lerobot's `opencv-python-headless<4.14` (same `cv2/` dir); pin `<4.14`.

## Cloud (Lambda) — `training/cloud/`

Training runs on an x86 Lambda GPU instance: `setup.sh` + `train.sh` + `hub_check.py` run there, `launch.sh` drives them
from the Jetson over ssh/tmux, `training/pull_checkpoint.sh` brings the model back to `outputs/pi05_piper/pretrained_model`.
Runbook with costs: `training/cloud/README.md`. `training/train.sh full` writes `outputs/pi05_piper_local` so it cannot
collide with the pulled checkpoint.

2026-08-22 (no instance):
- `launch.sh` pre-flight = `python -m training.cloud.hub_check dataset REPO STEPS BATCH` (token via whoami, meta/info.json vs
  the piper.config contract, epochs / est. hours / $ from `H100_STEP_S`, `CLOUD_OVERHEAD_MIN`, `H100_USD_PER_H`); `--check`
  stops there. Verified against `royc1981/piper_fake_block_bowl` (PASS) and a missing repo / bad token (exit 1, no ssh).
- `cloud/train.sh`: `--policy.push_to_hub=false`; the checkpoint goes up with `hf upload` (3 attempts, 60/120 s backoff),
  each verified by `hub_check.py model` (size + sha256 on the Hub); DONE only after verification; `train.log` is uploaded
  next to the model; `AUTO_TERMINATE=1` (from `launch.sh` via `~/train.env`) terminates only after DONE.
- `pull_checkpoint.sh` checks the Hub before downloading and verifies size + sha256 after (13 s for 14.5 GB on the Thor).
- `status.sh` parses the tqdm `N/M [elapsed<remaining` and `loss:` lines from `~/train.log` over ssh.
- Optional API path: `training/cloud/lambda_api.py` (launch/list/terminate; shapes from https://cloud.lambda.ai/api/v1/openapi.json,
  Bearer auth, NOT exercised — no key) and `terminate.sh`.
- 3-step validation of the shared `train.sh` on the Thor: exit 0, first step 11.8 s then 5–7 s/step with the GPU shared,
  checkpoint 23 GB.

Verified from the Jetson on 2026-08-21 (no instance launched):
- Gated tokenizer: with the license accepted, `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")` loads with
  the account token (`GemmaTokenizer`, 257153 ids, 6.5 s). Pitfall 3 and `prepare_pi05_base.py` are obsolete.
- Exact cloud command on the Thor, 2 steps: `--dataset.repo_id=royc1981/piper_fake_block_bowl` read from the Hub (no
  `--dataset.root`; lands in `~/.cache/huggingface/lerobot/hub/datasets--<user>--<name>/`), `--policy.path=lerobot/pi05_base`
  with its original preprocessor, `--policy.private=true`, the rename map: exit 0, loss 0.66 → 0.31, 37.1 GB peak,
  4.9 s/step at batch 8. `lerobot/pi05_base` in `~/.cache/huggingface/hub` (14.5 GB, sha256 identical to `models/pi05_base`).
- x86_64 / Python 3.12 dependency set resolved offline with uv → `training/cloud/requirements.txt`: torch 2.11.0+cu128,
  torchvision 0.26.0+cu128, torchcodec 0.11.1+cu128, transformers 5.5.4, lerobot 0.6.1 (106 pins). PyPI's default
  torch 2.11 wheel is a CUDA 13 build (driver ≥ 580), hence the cu128 index. lerobot 0.6.1 needs Python ≥ 3.12 and
  Lambda images may ship 3.10, so `setup.sh` builds the venv with `uv venv --python 3.12`.
- Checkpoint round-trip: `hf upload <repo> checkpoints/last/pretrained_model . --private` (the command `train.sh` runs)
  works, but the Jetson WiFi uploads at ~0.6 MB/s (9.4 GB ≈ 4.5 h), so `royc1981/pi05_piper_smoke_test` holds only the
  config/processor files + a README. `pull_checkpoint.sh` exits 1 with a message when `model.safetensors` is missing and
  runs the 3-step dry-run when it is present. Hub download is ~6.3 MB/s → ~25 min per 9.4 GB checkpoint.
- Not verified (needs a real instance): `setup.sh` on a Lambda image, `--policy.compile_model=true` on x86
  (`COMPILE=false bash train.sh …` turns it off), the H100 step time (estimate ~1 s/step at batch 16 → ~1–1.5 h).
