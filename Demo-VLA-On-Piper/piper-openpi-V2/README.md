# Piper runbook

Quest 3 teleop → LeRobot dataset → pi0.5 fine-tune (Lambda H100) → deploy on the Jetson — with a MuJoCo digital twin
(`--sim`) to run any of it without the robot.
Contract (hardware facts, dataset/teleop contracts, SDK gotchas): `docs/CONTRACT.md`.
New here? **`docs/STACK.md`** — what we built, each component, how it was trained and deployed, plus a glossary. Training notes: `training/NOTES.md`.
Cloud: `training/cloud/README.md`. State of the system and what remains: `docs/HANDOFF.md`.

## Cheat sheet (everything, in order)

```bash
source ~/lerobot-venv/bin/activate && cd ~/productdaydemo/Demo-VLA-On-Piper/piper-openpi-V2
bash tools/selftest.sh                                   # no robot: 110+ tests, replay, fake recording  (~3 min)
bash ../../prepare-jetson/can_up.sh && python tools/check_arm.py  # robot on: CAN: OK / ARM: OK, J6 zero ~0 with the camera on top
python tools/zero_j6.py --stops                          # ONLY if the J6 zero line says so (done 2026-08-22)
python tools/stream_test.py                              # first motion of the day: home -> READY -> wiggle -> rest
python -m teleop.quest [--scale 0.25] [--sim]            # teleop only (practice); --sim = MuJoCo twin, no robot
python -m collect.record --repo-id royc1981/piper_block_bowl --episodes 10 [--sim]   # teleop + recording, UI on :8092
bash tools/review.sh data/piper_block_bowl 0             # replay episode 0 in rerun (URL for the laptop)
python -m collect.teach --episodes 10                     # no headset: drag teach + replay -> data/piper_block_bowl
python tools/inspect_dataset.py data/piper_block_bowl [--strict]   # lengths, steps/frame, lag, contact sheet (+gate)
python -m tools.push_dataset data/piper_block_bowl       # push to the Hub once you have judged the episodes
bash training/train.sh smoke data/piper_block_bowl       # Thor: 500 steps, ~45 min -> outputs/pi05_piper_smoke
bash training/cloud/launch.sh --check royc1981/piper_block_bowl royc1981/pi05_piper  # H100: pre-flight, then without --check
bash training/cloud/status.sh ; bash training/cloud/terminate.sh all
bash training/pull_checkpoint.sh royc1981/pi05_piper     # -> outputs/pi05_piper/pretrained_model
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --shadow        # no motion
python -m inference.run_policy --checkpoint outputs/<model>/pretrained_model --n-action-steps 30 --max-steps 450  # live
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --max-steps 450 # live
python tools/rest.py                                     # park + de-torque (--drop while holding a stuck arm)
```

Controller (right hand) — **A owns teleop, B owns episodes**:

| press | effect |
|---|---|
| **hold A 3 s** | ARM / DISARM teleop (two beeps / one beep; nothing moves while disarmed) |
| **tap A** | discard the running episode |
| **hold B 3 s** | START an episode |
| **tap B** | STOP and SAVE the episode |
| squeeze (grip) | clutch — hold to move; release to let go |
| trigger | gripper (pulled = closed) |
| thumbstick click | back to READY (not while recording) |

## Every session

Open one terminal. Only ever run **one** arm command at a time; every command below homes and de-torques the arm when it
exits (Ctrl-C included, crashes included).

```bash
source ~/lerobot-venv/bin/activate && cd ~/productdaydemo/Demo-VLA-On-Piper/piper-openpi-V2
bash ../../prepare-jetson/can_up.sh  # brings the USB-CAN up (down/up also resets a bus left in ERROR-PASSIVE by a crash)
python tools/check_arm.py       # read-only: CAN verdict, feedback Hz, joints, faults, FK check, camera snapshots -> data/snapshots/
```

`check_arm.py` must end with `CAN: OK` and `ARM: OK` (exit 1 otherwise, with the reason on one line). After a crash or an
unplug the bus sits in `ERROR-PASSIVE` until `can_up.sh` is run again. Read its **`J6 zero`** line: with the wrist
camera on top of the gripper J6 must read ≈ 0 — the motor's zero is the centre of the joint range and of every pose. If
it does not (camera upside down at 0, wrist on a hard stop), run `python tools/zero_j6.py --stops` once: it de-torques
J6 only, you turn the wrist by hand to camera-up centre, it writes the motor's zero and reads it back.

### Teleop

```bash
python -m teleop.quest --scale 0.25      # first flights: robot moves 25 % of the controller motion
python -m teleop.quest                   # 1:1
```

Quest (same WiFi): Quest Browser → `https://192.168.1.45:8443` → Advanced → proceed. The page must say **transport OK**
before you tap ENTER.
Controller: **hold A for 3 s to ARM** (two beeps; hold again to disarm). Then squeeze = clutch (hold to move),
trigger = gripper, thumbstick click = READY pose. Episodes are B (see the cheat sheet); plain teleop ignores them.
Your hand moves the robot's wrist centre and rotates the tool about it, so turning your hand is a wrist-only motion
(the base joints stay put); moving your hand moves the arm.
Leaving XR (headset off, EXIT XR) disarms. The wrist cannot follow every hand orientation (pitch J5 is limited to
±70°): when your hand asks for more, the target is projected onto the nearest reachable pose, the arm slides along
the limit and the headset says **J5 at limit — rotate back** (terminal: `limit=J5`). `--rot-scale 0.5` halves the
wrist demand if that happens a lot. If the arm still stops following, the headset shows **STUCK — RE-SQUEEZE** (the
firmware refused the target for 2 s) or the firmware's own verdict (e.g. **COLLISION_OCCURRED — RE-SQUEEZE**):
release the grip and squeeze again. The terminal shows `fw=<status>` whenever the firmware reports anything but NORMAL.

### Collect a dataset

```bash
python -m collect.record --repo-id royc1981/piper_block_bowl --episodes 10
```

This is teleop + recording in one process. Controller: **hold A 3 s = arm**, then per episode **hold B 3 s = start**,
**tap B = stop and save**, **tap A = discard**; disarming (hold A 3 s) discards too; thumbstick = READY (not while
recording). The laptop UI and the headset overlay show the B-hold progress ("starting episode 67 %").
No headset on your head needed: open **`http://192.168.1.45:8092`** on the laptop (any browser over WiFi) — both
cameras, REC state with seconds and frames, episode n of N, dataset name and path, last saved/discarded episode,
controller state (disarmed / armed / engaged), headset link (packets/s, age, tracked), arm joints/pose/gripper/J6/faults.
An existing `data/piper_block_bowl` is **resumed** (`--episodes` is the total); `--overwrite` starts over. Ctrl-C ends
the session: the running episode is dropped, the arm rests, the dataset is finalized. `saved 312 frames (20.8 s),
2 stuck` means the IK refused mid-episode — consider discarding such episodes. Push to the Hub at the end:
`--push` (`--private` for a private repo) or re-run the same command with `--push` once the target is reached.

#### Without the headset: drag teach

```bash
python -m collect.teach --episodes 10
```

There is no `--repo-id` here: publishing is `tools/push_dataset.py`, so this only needs to know which directory to
write to — `--name` (default `piper_block_bowl` → `data/piper_block_bowl`), or `--test` for the sandbox. The old
`--repo-id` is still accepted and its last path segment is used as `--name`.

Hand-guide the arm through the task, then the arm **replays it alone** at 15 fps while the cameras record — so the
frames show the robot doing the job, not a hand in shot. Per episode: **hold the arm and press the teach button on
it** (the joints go limp), Enter starts sampling, Enter stops, **keep holding** and Enter takes control back (the
arm stiffens where it stands — no second button press), then `Y` to replay and `Y` to keep. The gripper is
hand-operable in teach mode and its opening is sampled too. Writes the same dataset as `collect.record`, so teach
and teleop episodes mix freely in one dataset.

Teach mode is **entered** by the physical button only — there is no CAN command for it. `MotionCtrl_1`'s
`grag_teach_ctrl` byte looks like one but does nothing on this firmware, and `MasterSlaveConfig(0xFA)` does
de-torque the arm but needs a power cycle to leave. Entry is detected by polling `ctrl_mode == 2`. **Leaving** is
not the button either: `teach_stop()` *streams* `MotionCtrl_2(0x01, 0x01, ...)` until `ctrl_mode` reads 1, then
re-enables. Measured with `python -m tools.teach_probe` — `MotionCtrl_1(0,0,0x02)` and `ModeCtrl` streamed (what
AgileX's playPos uses) both leave `ctrl_mode` stuck at 2; `MotionCtrl_2` is the same CAN ID `0x151` but sends two
more fields and only the longer frame is accepted here. `EmergencyStop` is not needed. A single send is ignored
while teaching — it must be a loop. **The operator must be holding the arm** until it returns.

`tools/teach_probe.py` re-runs that comparison on demand (prints `ctrl_mode` after each candidate, then verifies
`JointCtrl` is actually obeyed) — use it if a firmware update changes this again.

Taught trajectories are saved under `data/trajectories/` with an automatic timestamp name (`--save-as` to name one
yourself) and can be re-recorded without teaching again (`--play latest`, or `--play demo_20260903_114230`),
rehearsed with nothing written (`--play latest --no-dataset`), or taught without recording (`--record pick_left`).
`--test` writes to `data/test`, which the push tool refuses to publish. If the replay reports a tracking lag above
8°, the arm is not keeping up: raise `--speed`. If the block slips out of the jaws, raise `--squeeze`.

### Push a dataset to the Hub

```bash
python -m tools.push_dataset data/piper_block_bowl
```

Separate from recording on purpose: whether episodes are worth publishing is judged *after* `inspect_dataset` and
the contact sheet, not chosen before the session starts. Prints a summary (episodes, frames, cameras, tasks, size,
target URL, public/private) and asks before uploading — `--yes` skips the prompt, `--dry-run` shows it and uploads
nothing, `--private` keeps the repo private, `--repo-id` overrides the default `royc1981/<directory name>`.
It refuses `data/test`, which exists precisely to hold recordings that are not trusted yet. The token comes from
`HF_TOKEN`, else the `hf_...` string in `~/Documents/HG-TOKEN.txt`.

Review an episode in rerun (web viewer served from the Jetson; open the printed URL on the laptop):

```bash
bash tools/review.sh data/piper_block_bowl 0
```

Judge the whole dataset before paying for training (episode lengths, joint steps per frame vs the 10° deploy clamp,
gripper range, action/state lag check, timestamp gaps, contact sheet `data/piper_block_bowl_contact.jpg`):

```bash
python tools/inspect_dataset.py data/piper_block_bowl
python tools/inspect_dataset.py data/piper_block_bowl --strict   # pass/fail gate, non-zero exit on failure
```

`--strict` adds what the summary alone cannot show and **exits non-zero** if any of it fails: the feature schema
pi05 is handed via `--rename_map`; units (degrees not radians, joints inside the arm's limits, a gripper action
the arm can actually reach — a hand-forced 92 mm jaw command looks like an ordinary dataset otherwise); one
decoded video frame per parquet row; and `q01`/`q99` in `stats.json`, since pi05 normalises on quantiles and
their absence only surfaces once training starts. Channel order is not asserted — a global red-vs-blue test is
worthless on a scene holding both red and blue objects — so **look at the contact sheet** and confirm the red
cube looks red.

Small-dataset loop (5–10 episodes): record → `inspect_dataset` → `bash training/train.sh smoke data/piper_block_bowl`
(Thor, ~45 min) or the cloud run below → `python -m inference.run_policy --checkpoint outputs/pi05_piper_smoke/checkpoints/last/pretrained_model --shadow` → live.

### Train (Lambda cloud)

Once: put either a hand-launched instance or an API key in `training/cloud/lambda.env`:

```bash
printf 'LAMBDA_HOST=ubuntu@<instance-ip>\nLAMBDA_KEY=~/.ssh/<lambda-key>\n' > training/cloud/lambda.env      # instance from the website
printf 'LAMBDA_API_KEY=<key>\nLAMBDA_KEY=~/.ssh/<lambda-key>\nAUTO_TERMINATE=1\n' > training/cloud/lambda.env   # launch.sh creates the 1x H100 itself
```

```bash
bash training/cloud/launch.sh --check royc1981/piper_block_bowl royc1981/pi05_piper   # free pre-flight: token, dataset contract, plan + cost
bash training/cloud/launch.sh royc1981/piper_block_bowl royc1981/pi05_piper           # setup, train in tmux, upload, verify
bash training/cloud/status.sh                                                         # last log lines + step / loss / ETA
bash training/cloud/terminate.sh all                                                  # when the log says DONE (billing runs until then)
```

3000 steps at batch 16 ≈ 1–1.5 h ≈ $4–6 (estimate; `launch.sh` prints the plan). The checkpoint is uploaded with three
retries and verified (size + sha256) before the log says `DONE`; nothing is terminated before that.

### Deploy

```bash
bash training/pull_checkpoint.sh royc1981/pi05_piper     # checks the Hub, downloads, verifies, 3 dry-run steps, prints the deploy command
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --dry-run --max-steps 60   # fake arm + synthetic frames: timing table
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --shadow                   # real cameras + real state, prints what it WOULD send, no motion
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --max-steps 450 [--episodes 2]   # live: READY -> 15 Hz joint streaming -> rest
```

Run `--shadow` first. Keys while running (type + Enter): `p` pause (hold the joints), `r` resume, `q` quit gracefully,
Enter = next episode with `--episodes`. Ctrl-C rests. One status line per second (chunk age, model latency, late / holds /
floor counts, J6, arm status); every tick is logged to `data/logs/policy_<stamp>_ep<n>.csv` for the post-mortem.

### End of session

```bash
python tools/rest.py
```

Arm goes to HOME slowly, pitches the wrist into its gravity-settle band (REST), de-torques every joint, CAN goes to
standby. Run this whenever the arm is left enabled. A joint the driver has cut (limp) is ignored when judging HOME; if
the joints that still hold torque cannot reach HOME the arm stays torqued — hold it and use `--drop` to de-torque
where it stands. `--measure` logs where the joints settle; `--soft` = the experimental MIT stiffness ramp (supervised
use only, see `docs/HANDOFF.md`).

## Safety procedure

- One controller at a time: every entry point takes `/tmp/piper_arm.lock` and aborts (naming the holder's PID) if
  another process is on `can_piper`. A lock whose holder is dead clears itself; `python tools/rest.py --force` is only
  for a live, hung holder.
- Nothing moves from the headset until you ARM (hold A 3 s). Disarm the same way; disarming, leaving XR, a lost
  controller, no arm feedback, a firmware fault and a control-loop crash all release the clutch.
- Every controller exits through `robot.rest()`: slow move to HOME → wrist to REST → de-torque → standby. If HOME is not
  reached (obstacle) it refuses to de-torque (the arm has no brakes) and tells you to clear and re-run `tools/rest.py`.
- Teleop clamps on every streamed target: workspace box, tool-tip floor, ≤15 mm / ≤8° per tick, lead clamp (the target
  never runs more than 30 mm / 20° ahead of the arm), stuck auto-declutch after 2 s, 0.5 s watchdog, J6 temperature pause
  at 46 °C, `--scale` motion cap.
- Policy deploy (JointCtrl, 15 Hz): ≤10° per tick, ≤20° joint lead clamp over the measured joints, tool-tip floor from
  forward kinematics (holds instead of descending), gripper 0–70 mm, firmware joint limits, J6 pause,
  `COLLISION_OCCURRED` / `EMERGENCY_STOP` → stop + rest. The model runs in a worker thread; the loop never waits for the GPU.
- Constants live in one place: `piper/config.py`.
- Before any motion: table clear around the arm, teach button OFF, e-stop released, hand near the e-stop on first runs.

## Testing without the robot

The whole stack has hardware-free modes; this is what to run before a hardware session (also `bash tools/selftest.sh`):

```bash
python -m pytest tests -q                              # 100 tests: clamps, lock, teleop tick + server + page, recorder, deploy loop, cloud scripts, sim (~100 s)
python -m teleop.quest --replay                        # teleop math with a synthetic controller (PASS/FAIL)
python -m teleop.quest --dry-run                       # serve the page, print targets, no robot (open it in the headset to check the overlay)
python -m collect.record --fake --episodes 2 --overwrite   # recorder without robot or headset -> data/fake_block_bowl
python tools/inspect_dataset.py data/fake_block_bowl
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --shadow --fake-arm --max-steps 150   # real cameras, fake arm
bash training/cloud/launch.sh --check royc1981/piper_block_bowl royc1981/pi05_piper                          # cloud pre-flight, no cost
bash training/train.sh smoke                           # the cloud run's exact flags on the Thor: 500 steps, batch 8, ~45 min, log ~/train.log
```

Hardware self-tests (robot on, user present):

```bash
python tools/stream_test.py                            # home -> READY -> ±20 mm / ±10° wiggle -> rest; prints lag, RMS, arm status + faults; log data/logs/stream_test.csv
python tools/stream_test.py --joints 0 40 -20 0 0 -165 --secs 6 --amp-mm 10 --amp-deg 5   # does the firmware IK accept targets from this pose?
python tools/rest.py --measure                         # where the joints settle after de-torque
curl -sk https://192.168.1.45:8443/status              # teleop status from any machine
```

## Simulation (digital twin)

`sim/` is a MuJoCo twin of the arm (DeepMind menagerie AgileX Piper model, table, bowl, block, both cameras rendered
offscreen). `--sim` swaps the hardware backend on any entry point: same clamps, same dataset format, same deploy loop,
and it never opens the CAN bus (the real arm can stay powered). Once per clone: `bash sim/fetch_assets.sh` (32 MB of
meshes). **Run it on the Jetson desktop: a live MuJoCo window of the twin opens by itself** (drag = orbit, scroll =
zoom; closing it keeps the twin running). Over plain ssh there is no display, so no window — watch the camera views
on a laptop via the recorder monitor instead (`http://192.168.1.45:8092`, `collect.record --sim`). `SIM_VIEWER=0`
suppresses the window.

```bash
python -m sim.piper_sim --demo --episodes 10      # HOME -> READY -> wiggle (lag, RMS, refused ticks) -> 10 scripted pick-and-place -> data/sim/demo_{top,wrist}.mp4
python -m sim.piper_sim --snap                    # both cameras at HOME and READY -> data/sim/snap_*.jpg (wrist view upright with J6 = 0)
python -m teleop.quest --sim                      # fly the headset against the twin (same page, no robot)
python -m collect.record --sim --scripted --episodes 10 --repo-id local/sim_block_bowl   # sim dataset from the scripted expert, no headset
python -m collect.record --sim --episodes 10 --repo-id local/sim_teleop                  # headset teleop recorded on the twin
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --sim      # deploy loop on the twin
```

What it is good for: proving the whole teleop → dataset → train → deploy loop without the robot, practising teleop,
and generating sim datasets. What it is not: the firmware's IK (the twin uses damped least squares with the same
joint limits and refuses the same way), calibrated speeds, or realistic images — a policy trained on real data will
not transfer to the twin's renders. Details and measured gaps: `docs/CONTRACT.md` → "Digital twin".

## Telemetry

Teleop prints one line per second: `[transport rate age] ctrl= sq= tr= A= | ARMED/DISARMED/ENGAGED | status [fw=<ArmStatus>] |
grip tgt=` and a second line with the measured pose, joints, gripper and J6 temperature. Startup prints joints before/after
enable (twitch), before/after home, and at READY. The ack / `/status` JSON carries `arm_status` (firmware verdict, 1 Hz,
null in `--dry-run`), `err_mm` / `err_deg` (target vs arm), `lead_limited`, `stuck`, `no_feedback`, `j6_margin_deg`.
The headset overlay shows the state banner, gripper bar, lag, wrist-limit warning, REC, transport rate, A-hold progress.

## Troubleshooting

| symptom | fix |
|---|---|
| `ABORT: another controller holds the arm (pid N)` | Ctrl-C that terminal; if it is hung, `python tools/rest.py --force` |
| `no feedback on can_piper after 2.0 s (bus ERROR-PASSIVE)` | arm powered? cable? `bash ../../prepare-jetson/can_up.sh` (down/up resets the bus), then `python tools/check_arm.py` |
| `CAN interface can_piper is down` | `bash ../../prepare-jetson/can_up.sh`; replug the USB-CAN adapter |
| `arm did not enable` | teach button ON or e-stop pressed |
| terminal says `NO PACKETS YET` | page transport failed: reload, confirm it shows transport OK before ENTER; `curl -sk https://192.168.1.45:8443/status` from a laptop |
| headset buttons do nothing | not armed — hold A 3 s until the double beep |
| arm does not move under teleop | it is at folded HOME; teleop must be at READY (thumbstick click), the firmware IK refuses targets from HOME |
| headset says `STUCK — RE-SQUEEZE` | the target sat at the lead clamp for 2 s (IK refused, or a collision / e-stop was reported): release and squeeze again; see `fw=` in the terminal |
| arm "freezes" mid-teleop | J6 ≥ 46 °C pause — wait for `hold: J6 cooled`; or controller lost tracking (watchdog) |
| terminal says `CONTROL FAULT: ...` | the control thread crashed; teleop/record stop streaming and rest the arm by themselves — read the traceback, fix, rerun |
| Quest page loads but no poses | ENTER was not tapped (WebXR needs a gesture), or the cert warning was not accepted |
| camera busy | `pkill -f dual_camera_stream` (done automatically before motion); only one process can hold a camera |
| `launch.sh`: `ERROR: dataset ... violates the contract` / `HF token rejected` / `RepositoryNotFoundError` | fix before launching (nothing was billed) |
| `UPLOAD FAILED 3x` in train.log | the checkpoint is on the instance: run the printed `hf upload ... && hub_check` line by hand; do not terminate |
| `pull_checkpoint.sh`: `has no model.safetensors` / `incomplete copy` | training did not finish or the upload was not verified: `bash training/cloud/status.sh`; re-run the pull |
| `model late and chunk exhausted: holding` | GPU slower than ~3 s per chunk (other GPU jobs? `nvidia-smi`); the arm holds its last target — press `q` |
| arm left torqued after a crash | `python tools/rest.py` |
| sim: `meshes missing` / model fails to load | `bash sim/fetch_assets.sh` |
| sim: `no frames from sim camera` / GL errors | EGL is required (`MUJOCO_GL=egl`, the default); `osmesa` works but at ~2.5 fps |
| sim: no window opens | run it from a terminal on the Jetson desktop (`echo $DISPLAY` must print `:1` or similar); ssh sessions have no display; `SIM_VIEWER=0` unset? |
