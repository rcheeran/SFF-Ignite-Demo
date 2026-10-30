# auto_data — scripted collection (no headset)

`python -m auto_data.auto_collect` records the same task as `collect/record.py`, but the motion is
scripted instead of flown from the Quest. Same dataset, same trainer:
`lerobot-train --policy.path=lerobot/pi05_base` consumes either without a flag change.

Use it when teleop is the bottleneck. Use `collect/record.py` when you want human-like variety,
recovery behaviour, or anything the script cannot do.

## Order of operations

```bash
source ~/lerobot-venv/bin/activate && cd ~/Documents/piper-openpi-V2
python -m auto_data.auto_collect --plan                # no hardware: waypoints + reachability + episode length
python -m auto_data.auto_collect --set-bowl 300 -90    # where the bowl is, in flange XY (once per setup)
python -m auto_data.auto_collect --probe               # ONE cycle on the real arm, no recording, gripper idle
python -m auto_data.auto_collect --episodes 10         # record (resumes an existing dataset)
python tools/inspect_dataset.py data/piper_block_bowl  # judge it BEFORE paying for training
```

Setting the bowl position: `--probe` drives the gripper to the stored bowl XY and prints where it
actually went. Watch it, adjust with `--set-bowl`, repeat until the tool sits over the bowl. There is
no camera calibration in this path on purpose — the robot never needs to *find* anything.

## What each cycle does

Recorded episode: over block → descend → close → lift → over bowl → release → retreat (~19 s, ~288 frames).
Then, **not recorded**: dig the block back out of the bowl and place it at a fresh random spot, so the
next episode starts from a new block position with nobody touching the table.

Success is gated on measured gripper width after the lift (16–45 mm = a block is really held). A failed
grasp is discarded with `clear_episode_buffer()` — it never reaches the dataset.

## Ported from V1's `piper-openpi/auto_collect.py`

| V1 | here | why |
|---|---|---|
| own `piper_safe.py` motion + limits | `piper.robot.Piper`, `connect_for_motion()` | V2's layer already has the lock, clamps, fault handling and `rest()` |
| own sampled `joint_table.json` | `piper.kinematics.ik` | real IK beats a 19-point lookup table, and it shares limits with the deploy clamp |
| raw `obs.jsonl` + JPEG frames | `LeRobotDataset` v3 (video) | this is the format `lerobot/pi05_base` trains on |
| state = `[x,y,z,rx,ry,rz,grip]` (pose) | state = `[j1..j6, grip_mm]` (joints) | the policy is deployed with `JointCtrl`, which never refuses a target |
| action = next absolute pose | action = joints at t+1 + gripper commanded at t | `collect/record.py`'s `Episode`, imported, so both paths lag identically |
| own camera threads | `piper.cameras.open_cameras()` | one implementation, already handles the D405 and the USB top cam |
| wrist pinned at J6 = −165 "camera up" | J6 free within factory ±120 | that was a workaround for a mis-taught J6 zero; the zero is fixed now |
| own MJPEG monitor on :8092 | none | `tools/review.sh` and `inspect_dataset.py` cover review |

## Geometry (measured, not guessed)

`piper.kinematics` says a straight-down tool is **unreachable** — J5 is limited to ±70°, and IK fails on
every table target. So grasps use a 20° pitch (`rpy [180, 20, 180]`), which reaches the whole sampling box
(200 spots × 2 heights, 0 failures) with ≥37° of joint margin. At that pitch the tool tip sits about
`flange_z − 113` mm, which is where `FLANGE_Z_GRASP/CARRY/DROP/DIG` come from.

Every waypoint is IK-checked and tip-floor-checked **before** the arm moves; an unreachable target aborts
instead of being streamed into a firmware that would silently drop it.

## Known weak point

Digging the block out of the bowl is the least reliable step — a block can sit anywhere in the bowl and the
script always reaches for its centre. The gripper-width check catches a miss, and collection stops and asks
for a human reset rather than recording episodes with no block on the table. If it misses often, widen the
bowl or use a flatter dish.
