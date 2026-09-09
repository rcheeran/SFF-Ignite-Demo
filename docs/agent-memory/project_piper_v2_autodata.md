---
name: project_piper_v2_autodata
description: piper-openpi-V2 (LeRobot pi0.5 pipeline) - working end to end as of 2026-09-04; drag-teach collection, cloud training, deploy
metadata:
  node_type: memory
  type: project
  originSessionId: acabfc4d-beea-4306-a43a-cbd8b34f1fb9
  modified: 2026-09-04T19:07:27.920Z
---

**`~/Documents/piper-openpi-V2` is the ACTIVE project** and the full loop WORKS as of 2026-09-04: drag-teach
collection -> 50-episode dataset -> pi0.5 fine-tune on a Lambda H100 -> policy driving the real arm.
Env: `source ~/lerobot-venv/bin/activate`. **Its own docs are authoritative and now cover almost everything**:
`README.md` (runbook), `docs/CONTRACT.md` (hardware/dataset/SDK facts + the deploy gotchas), `docs/HANDOFF.md`,
`training/NOTES.md`. Read those first - this file only holds state and pointers.

**Not openpi.** Training is `lerobot-train --policy.path=lerobot/pi05_base`, so no config.py, no
`compute_norm_stats.py`, no JAX->PyTorch conversion. Advice written for the openpi repo does NOT apply.
V1 (`~/Documents/piper-openpi`) is legacy; its `cloud/lambda_train.sh` is the old openpi path and contains **no
Lambda credentials** (it runs ON a hand-launched instance).

**State at 2026-09-04:**
- dataset `royc1981/piper_block_bowl` - 50 episodes, 20228 frames, all 50 grasp->carry->release, contract-clean
  (`python tools/inspect_dataset.py data/piper_block_bowl --strict`). Pushed to the Hub (private).
- model `royc1981/lerobot-pi05-base-50` - 3000 steps, batch 16, 2.4 epochs, loss 0.319 -> 0.037, 1 h 08 m,
  ~$3.75 on a 1x H100 PCIe. Local copy `outputs/lerobot-pi05-base-50/pretrained_model`.
- **It runs on the real arm** with `--n-action-steps 30` (see below).
- **It generalises across cube COLOUR**: trained only on a red cube (with a blue cube present in every frame as
  an ignored distractor sitting in the red bowl), it picks a BLUE cube too. I predicted it would not - the
  pretrained pi05_base priors carry "graspable cube on table" and the 50 episodes only fine-tune motion and
  workspace, so do not assume this policy is as narrow as its dataset looks. Colour variation may not need new
  episodes; SHAPE/size variation probably does.

**Hard-won facts NOT to rediscover** (the rest are in docs/CONTRACT.md):
- **Deploy needs `--n-action-steps 30`, not the default 10.** Teach episodes start with ~2.8 s of stationary arm
  (median 42 frames; 50/50 episodes >10), so chunk actions 0-25 are settling. At 10 the loop executes only that
  prefix, re-predicts from an unchanged state, and **stalls forever looking like a broken policy**. Verify a
  suspect policy by replaying TRAINING frames through it (it reproduced actions to 0.3-6.2 deg) and by printing
  the whole 50-action chunk, not `chunk[0]`. Data-side fix for next time: trim the stationary prefix.
- **Thor inference cannot torch.compile**: Triton's bundled CUDA 12.8 `ptxas` has no `sm_110a`. run_policy.py
  disables dynamo; eager is ~527 ms/call with real cameras, 0 late ticks at 15 Hz.
- **Teach mode is the PHYSICAL button only.** Exit is NOT the button and NOT `ModeCtrl` (what AgileX's playPos
  uses - it leaves ctrl_mode stuck at 2): stream `MotionCtrl_2(0x01,0x01,...)` until ctrl_mode reads 1.
  Measured with `python -m tools.teach_probe`. `MasterSlaveConfig(0xFA)` needs a power cycle to leave - useless.
- **Lambda API**: key lives in `~/Documents/HG-TOKEN.txt` (`secret_...`), read by `training/cloud/lambda.env` at
  source time. Cloudflare 403s urllib's default User-Agent (error 1010) - `lambda_api.py` sends a curl UA.
  H100 capacity comes and goes; `launch.sh --wait N` polls without billing. `AUTO_TERMINATE=1` shuts down only
  after a size+sha256-verified upload, never after a failure.
- **`--overwrite` guards exist because I destroyed a dataset with it** (a `--fake` smoke test with a default
  `--name` wiped 2 real teleop episodes that had never been pushed). `--name` now has no default, `--overwrite`
  prompts, and `--fake --overwrite` over real episodes is refused. **Push after every session.**

**User preference:** they run cost-incurring and hardware-moving commands themselves unless they say otherwise;
give the command in a bash block rather than executing it.
