# Handoff — state of the system and everything that remains (2026-09-04)

New to the stack? **`docs/STACK.md`** — what we built, the components, training, deployment, and a glossary.

For any agent continuing this work: read `README.md` (runbook), `docs/CONTRACT.md` (hardware facts, dataset/teleop
contracts, SDK gotchas, **and the deploy gotchas at the end — read those before debugging a "broken" policy**),
`training/NOTES.md` (LeRobot/pi05 + cloud), then this file. Env: `source ~/lerobot-venv/bin/activate` for
everything. HF user `royc1981`, token `~/Documents/HG-TOKEN.txt` (the Lambda API key lives in the same file).
`bash tools/selftest.sh` proves the stack without hardware (137 pytest tests + replay + fake recording).

## THE LOOP IS CLOSED (2026-09-04)

Hand-guided demonstrations → LeRobot dataset → pi0.5 fine-tune on a Lambda H100 → **a policy driving the real arm
and completing the task**. Every stage has been run end to end on hardware, not just in tests.

| | |
|---|---|
| dataset | `royc1981/piper_block_bowl` — **50 episodes, 20228 frames**, 50/50 grasp→carry→release, contract-clean, on the Hub (private) |
| model | `royc1981/lerobot-pi05-base-50` — `lerobot/pi05_base`, 3000 steps, batch 16, 2.4 epochs, loss **0.319 → 0.037**, 1 h 08 m, ~$3.75 on 1× H100 PCIe |
| local checkpoint | `outputs/lerobot-pi05-base-50/pretrained_model` (9.35 GB, sha256-verified against the Hub) |
| deploy | works: `python -m inference.run_policy --checkpoint outputs/lerobot-pi05-base-50/pretrained_model --n-action-steps 30 --max-steps 450` |

**It generalises across cube colour.** Trained only on a red cube — with a blue cube present in *every* frame as an
ignored distractor sitting in the red bowl — it picks a **blue** cube too. This was predicted NOT to work; the
pretrained `pi05_base` priors carry "graspable cube on table" and the 50 episodes only fine-tune motion and
workspace. **Do not assume this policy is as narrow as its dataset looks.** Colour variation may need no new data;
shape/size variation (soda can, non-cubic) probably does.

## What was built in this pass (2026-09-01 → 09-04)

- **`collect/teach.py` — drag-teach collection, the headline addition.** Quest teleop was too hard to drive
  consistently, so: hand-guide the arm, then the arm **replays the motion alone** at 15 fps while the cameras
  record. The frames therefore show the robot doing the job, not a hand in shot. It imports `FEATURES`,
  `open_dataset`, `Episode`, `no_interrupt` from `collect/record.py`, so teach- and teleop-collected episodes are
  interchangeable in one dataset. `--play latest` re-records a saved trajectory without re-teaching;
  `--record NAME` teaches without recording; `--test` writes to `data/test`. Trajectories land in
  `data/trajectories/` with automatic timestamp names.
- **`tools/inspect_dataset.py --strict`** — turns the summary into a pass/fail gate (non-zero exit): feature
  schema, units (degrees not radians, joint limits, commandable gripper range), one video frame per parquet row,
  q01/q99 in stats.json. Channel order is deliberately NOT asserted (a red-vs-blue test is worthless on a scene
  holding both) — **look at the contact sheet**.
- **`tools/push_dataset.py`** — publishing split out of the recorders, because whether episodes are worth
  publishing is judged *after* looking at them. Summarises and confirms before uploading; refuses `data/test`.
- **`tools/teach_probe.py`** — measures which SDK command actually leaves teach mode (four inferences were wrong).
- **`tools/placement_guide.py` / `tools/camera_server.py`** — live top-camera view with the placement area and past
  grasp positions drawn on it (`http://<jetson>:8093`). No ffmpeg (not installed, not needed — MJPEG direct).
- **`piper/robot.py`**: `in_teach()`, `wait_teach()`, `teach_stop()`, `gripper_enable()`, `_enable_motors()`.
- **`inference/run_policy.py`**: dynamo disabled on Thor; chunk-boundary blending (`--blend-ticks`).
- **`training/cloud/`**: `lambda_api.py` now works against the live API (it never had before); `launch.sh --wait`.

## Scars — mistakes made here, so they are not repeated

- **A dataset was destroyed.** A `--fake` smoke test run with `--overwrite` and a default `--name` wiped 2 real
  teleop episodes that had never been pushed. Guards added: `--name` has no default, `--overwrite` prompts and
  requires typing the dataset name, and `--fake --overwrite` over real episodes is refused outright.
  **Push after every session** — a Hub copy is the only thing that makes a loss recoverable.
- **The Lambda API key leaked** into stdout and the launch log: `launch.sh` printed `${TRAIN_ENV:-...}` and
  `TRAIN_ENV` holds the raw key. Fixed to print a description. **The key should be rotated.**
- **Four wrong inferences about teach mode** before measuring it (`MotionCtrl_1`'s grag_teach byte, a second
  button press, `ModeCtrl` streamed — which is what AgileX's own playPos uses). The answer was streaming
  `MotionCtrl_2`. When SDK behaviour is uncertain, write a probe rather than reason from documentation.
- **Two false alarms from my own dataset checks** (episodes 11 and 19 flagged as "never releases" when the frames
  showed success). Heuristics over gripper traces need a *sustained* hold and a release judged against the held
  width, not a fixed opening.

## What remains

1. **Trim the stationary prefix from the dataset and retrain.** Every teach episode opens with ~2.8 s of
   stationary arm (median 42 frames; 50/50 episodes >10), which is why deploy needs `--n-action-steps 30` instead
   of the default 10. Trimming teaches "at start pose → go" and restores 0.67 s re-planning. Do this **before**
   collecting the next batch, so new episodes do not inherit it.
2. **Rotate the Lambda API key** (see Scars).
3. **More coverage where the base model's priors do not help**: shape/size (soda can), the centre band (only 7 of
   50 grasps have |y| < 50 mm), lighting.
4. **Multi-task prompts** (`"pick the red cube and place in the red bowl"`, etc.) — wanted originally, and
   `collect.teach --task` already sets the string per episode. Harder than it looks: since the policy appears to
   grab the salient cube regardless of colour, the prompt only becomes load-bearing if **both** cubes are on the
   table at once so the language is the only disambiguator.
5. The `auto_data/` scripted-collection package still exists and works, but drag-teach superseded it.
