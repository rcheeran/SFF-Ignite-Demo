"""Run a fine-tuned pi05 checkpoint on the Piper: two BGR frames + 7-D state -> joint targets (deg, gripper mm).

    source ~/lerobot-venv/bin/activate
    python -m inference.run_policy --checkpoint <ckpt> --dry-run             # stub arm + synthetic frames: timing, no hardware
    python -m inference.run_policy --checkpoint <ckpt> --shadow              # real cameras + read-only arm: prints actions, NO motion
    python -m inference.run_policy --checkpoint <ckpt> --shadow --fake-arm   # same with the fake arm (robot off)
    python -m inference.run_policy --checkpoint <ckpt> --sim                 # MuJoCo twin: same loop, no hardware
    python -m inference.run_policy --checkpoint <ckpt> [--max-steps 900] [--episodes 2]   # drive the arm from READY

Keys (type + Enter): p = pause (hold the joints), r = resume, q = quit gracefully; Ctrl-C also rests the arm.
The model (~455 ms) runs in a worker thread: the 15 Hz loop takes one action per tick from the live chunk and hands
the newest observation to the worker when PREFETCH_AT actions remain, so no tick ever waits for the GPU.
Every tick goes to data/logs/policy_<stamp>_ep<n>.csv; the terminal prints one status line per second.
"""
import argparse
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from piper import config as C
from piper.cameras import FakeCamera, open_cameras
from piper.robot import clamp_joints, connect_for_motion, connect_readonly

# lerobot's pi05 torch.compiles sample_actions, and on Thor that dies in Triton:
#   ptxas fatal : Value 'sm_110a' is not defined for option 'gpu-name'
# Triton bundles a CUDA 12.8 ptxas that predates Thor's sm_110a and there is no system toolkit to fall
# back to, so the compile can never succeed here - training/cloud/train.sh passes COMPILE=false on the
# Thor for the same reason. Eager mode is the only thing that runs; PIPER_COMPILE=1 re-enables it if a
# future Triton/ptxas learns sm_110a.
if os.environ.get("PIPER_COMPILE") != "1":
    torch._dynamo.config.disable = True


class Pi05Runner:
    """predict_chunk(top_bgr, wrist_bgr, state7) -> float32[50, 7] = j1..j6 deg, gripper mm for the next 50 frames."""

    def __init__(self, ckpt_dir: str, task: str = C.TASK, device: str = "cuda"):
        cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
        cfg.device = device
        self.policy = PI05Policy.from_pretrained(ckpt_dir, config=cfg).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=ckpt_dir, preprocessor_overrides={"device_processor": {"device": device}})
        self.device, self.task = torch.device(device), task
        t0 = time.perf_counter()
        blank = np.zeros((C.IMG_H, C.IMG_W, 3), np.uint8)
        self.predict_chunk(blank, blank, np.zeros(7, np.float32))
        print(f"model warm-up {(time.perf_counter() - t0) * 1000:.0f} ms")

    def predict_chunk(self, top_bgr: np.ndarray, wrist_bgr: np.ndarray, state7: np.ndarray) -> np.ndarray:
        obs = {"observation.images.top": cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB),
               "observation.images.wrist": cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB),
               "observation.state": np.asarray(state7, np.float32)}
        with torch.inference_mode():
            batch = self.pre(prepare_observation_for_inference(obs, self.device, self.task))
            chunk = self.policy.predict_action_chunk(batch)[0]
            return self.post(chunk).numpy().astype(np.float32)     # unnormalize is per element: the whole chunk at once


class ActionStream:
    """Double buffer between the tick loop and the model thread. The loop takes one action per tick from the live
    chunk; when PREFETCH_AT of its n actions remain, the newest observation goes to the worker; after n actions the
    prefetched chunk takes over, skipping as many actions as were executed since its observation (so the hand-over is
    continuous). A late model keeps serving the live chunk's remaining actions; only an exhausted chunk holds."""

    def __init__(self, predict_chunk, n_action_steps=C.N_ACTION_STEPS, blend_ticks=C.BLEND_TICKS):
        self._predict, self.n, self._blend = predict_chunk, n_action_steps, int(blend_ticks)
        self._cv = threading.Condition()
        self._obs = self._ready = self._error = None
        self._busy = self._stop = False
        self.reset()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def close(self):
        """Stop the worker (a running model call finishes first): a CUDA thread alive at exit aborts the process."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join()

    def reset(self):
        with self._cv:
            self._cv.wait_for(lambda: not self._busy)
            self._chunk, self._i, self._ready, self.last = np.zeros((0, 7), np.float32), 0, None, None
            self._off, self._off_left = np.zeros(6, np.float32), 0
            self.consumed, self.lat, self.holds, self.late, self.t_chunk = 0, [], 0, 0, time.monotonic()

    def next(self, obs: tuple, t: float) -> tuple[np.ndarray, bool]:
        """Action for this tick; obs = (top_bgr, wrist_bgr, state7) taken at time t. Second value: True on a hold."""
        with self._cv:
            if self.last is None:                                  # episode start: wait for the first chunk
                self._request(obs, t)
                self._cv.wait_for(lambda: self._ready is not None or self._error is not None)
            if self._error is not None:
                raise RuntimeError("model thread failed") from self._error
            if self._ready is not None and self._i >= min(self.n, len(self._chunk)):
                full, self.t_chunk, at = self._ready
                new = full[self.consumed - at:]
                # A fresh chunk starts where the MODEL thinks the arm should be, which is not where the last
                # action put it: measured over a live run, the action step is 10.6 deg mean / 45.9 max AT a
                # boundary against 1.08 deg elsewhere. Stepping straight into it is the jitter you feel (the
                # clamp caps it at 10 deg/tick = 150 deg/s, so it jerks rather than lurches). Carry the
                # discontinuity as an offset and decay it over `blend` ticks so the command stays continuous
                # and still converges onto the new plan. Joints only - offsetting the gripper would blur the
                # moment it closes.
                if self._blend and self.last is not None and len(new):
                    self._off, self._off_left = (self.last[:6] - new[0][:6]).astype(np.float32), self._blend
                self._chunk, self._i, self._ready = new, 0, None
            exhausted = self._i >= len(self._chunk)
            if (self.n - self._i <= C.PREFETCH_AT or exhausted) and not self._busy and self._ready is None:
                self._request(obs, t)                              # before consuming: this tick's action follows obs
            if exhausted:
                if not self.holds:
                    print("model late and chunk exhausted: holding the last action")
                self.holds += 1
            else:
                act = self._chunk[self._i].copy()
                if self._off_left > 0:
                    act[:6] += self._off * (self._off_left / self._blend)
                    self._off_left -= 1
                self.last, self._i, self.consumed = act, self._i + 1, self.consumed + 1
                self.late += self._i > self.n
            return self.last, exhausted

    def invalidate(self):
        """Drop the plan (after a pause): the next tick waits for a chunk from a fresh observation, like episode start."""
        with self._cv:
            self._cv.wait_for(lambda: not self._busy)
            self._chunk, self._ready, self.last = self._chunk[:0], None, None

    @property
    def model_ms(self) -> float:
        return self.lat[-1] if self.lat else 0.0

    def _request(self, obs, t):
        self._obs, self._busy = (obs, t, self.consumed), True
        self._cv.notify_all()

    def _work(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._obs is not None or self._stop)
                if self._stop:
                    return
                (obs, t, at), self._obs = self._obs, None
            t0 = time.perf_counter()
            try:
                full = self._predict(*obs)
            except Exception as e:                                  # surfaces in next(), which ends the run
                with self._cv:
                    self._error, self._busy = e, False
                    self._cv.notify_all()
                return
            with self._cv:
                self._ready, self._busy = (full, t, at), False
                self.lat.append((time.perf_counter() - t0) * 1000)
                self._cv.notify_all()


class Keys:
    """Terminal control, one key + Enter per line: p = pause (hold joints), r = resume, q = quit, Enter = next episode."""

    def __init__(self):
        self.paused = self.quit = False
        self.enter = threading.Event()

    def read_stdin(self):
        for line in sys.stdin:
            self.key(line.strip().lower())

    def key(self, k: str):
        if k == "p":
            self.paused = True
        elif k == "r":
            self.paused = False
        elif k == "q":
            self.quit = True
        elif k == "":
            self.enter.set()


def run_episode(stream: ActionStream, robot, cams: dict, keys: Keys, max_steps: int, shadow=False, ep=0) -> bool:
    """Stream one action per tick at C.FPS from the current pose. True when max_steps ran out, False on q / arm fault."""
    os.makedirs(C.LOG_DIR, exist_ok=True)
    path = os.path.join(C.LOG_DIR, time.strftime(f"policy_%Y%m%d_%H%M%S_ep{ep}.csv"))
    f = open(path, "w", newline="")
    log = csv.writer(f)
    log.writerow(["t", *("state_" + n for n in C.STATE_NAMES), *("action_" + n for n in C.STATE_NAMES),
                  *("sent_" + n for n in C.STATE_NAMES), "chunk_age_s", "model_ms", "work_ms", "flags"])
    stream.reset()
    prev, grip, status = robot.joints_deg(), float(robot.grip_mm()), robot.arm_status()
    action, temp_paused, stale, floor_holds, works, ok = np.full(7, np.nan, np.float32), False, False, 0, [], True
    verb = "shadow" if shadow else "sent"
    t0 = next_t = time.monotonic()
    for step in range(max_steps):
        now = time.monotonic()
        if step % C.FPS == 0:                           # 1 Hz, before sending: firmware status, J6 temperature
            s = robot.arm_status()
            if s != status:
                print(f"arm status {status} -> {s}")
                status = s
            if any(x in s for x in C.FAULT_STATUS):
                print(f"ARM FAULT {s}: stopping the stream")
                ok = False
                break
            temp = robot.j6_temp()
            if temp >= C.J6_TEMP_PAUSE_C and not temp_paused:
                temp_paused = True
                print(f"PAUSED: J6 {temp} C (resumes at {C.J6_TEMP_RESUME_C} C)")
            elif temp_paused and temp <= C.J6_TEMP_RESUME_C:
                temp_paused = False
                print("J6 cooled: resuming")
        state = robot.state7()
        meas = state[:6].astype(np.float64)
        flags = []
        if keys.paused or temp_paused:                  # hold: the stream does not advance, nothing is consumed
            joints, flags, stale = prev, ["paused"], True
        else:
            if stale:                                   # the plan predates a pause: re-observe before moving again
                stream.invalidate()
                stale = False
            action, hold = stream.next((cams["top"].read(), cams["wrist"].read(), state), now)
            joints, floor = clamp_joints(prev, meas, action[:6])     # step, lead, joint limits, tool-tip floor
            grip = float(np.clip(action[6], C.GRIP_CLOSED_MM, C.GRIP_OPEN_MM))
            if floor:
                floor_holds += 1
                flags.append("floor")
            if hold:
                flags.append("hold")
        if not shadow:
            robot.send_joints(joints, grip)
        prev = meas if shadow else joints
        works.append((time.monotonic() - now) * 1000)
        log.writerow([round(now - t0, 4), *np.round(state, 2), *np.round(action, 2), *np.round(joints, 2), round(grip, 1),
                      round(now - stream.t_chunk, 3), round(stream.model_ms), round(works[-1], 1), "+".join(flags)])
        f.flush()
        if step % C.FPS == 0:
            print(f"t={now - t0:5.1f}s {verb} j={np.round(joints, 1).tolist()} g={grip:.0f} | "
                  f"state j={np.round(meas, 1).tolist()} g={state[6]:.0f} | chunk age {now - stream.t_chunk:.2f}s "
                  f"model {stream.model_ms:.0f}ms late {stream.late} holds {stream.holds} floor {floor_holds} | "
                  f"work max {max(works[-C.FPS:]):.0f}ms | J6 {temp}C {s}"
                  + (" | PAUSED" if keys.paused or temp_paused else ""))
        next_t += 1.0 / C.FPS
        if next_t < time.monotonic():                   # a stall (the first chunk): re-anchor, never burst to catch up
            next_t = time.monotonic() + 1.0 / C.FPS
        time.sleep(max(0.0, next_t - time.monotonic()))
        if keys.quit:
            print("q: ending the episode")
            ok = False
            break
    f.close()
    w, lat = np.array(works or [0.0]), np.array(stream.lat or [0.0])
    rest = w[1:] if len(w) > 1 else w
    print(f"{len(works)} ticks in {time.monotonic() - t0:.1f} s: work mean {rest.mean():.1f} max {rest.max():.1f} ms "
          f"(tick 0 = {w[0]:.0f} ms incl. the first chunk), ticks over {1000 / C.FPS:.0f} ms: {int((rest > 1000 / C.FPS).sum())} "
          f"| model {len(stream.lat)} calls mean {lat.mean():.0f} max {lat.max():.0f} ms | late {stream.late} "
          f"holds {stream.holds} floor {floor_holds} | {path}")
    return ok


def deploy(stream: ActionStream, robot, cams: dict, keys: Keys, max_steps: int, episodes=1, shadow=False):
    """Episodes from READY until --max-steps or q. The arm always leaves through rest() (disconnect only in shadow)."""
    try:
        for ep in range(episodes):
            if ep:
                keys.enter.clear()                      # an Enter typed earlier (model load, last episode) does not count
                print("Enter = next episode, q = quit")
                while not keys.enter.wait(0.2):
                    if keys.quit:
                        return
                keys.enter.clear()
            if not shadow:
                robot.ready(C.GRIP_OPEN_MM)
            print(f"episode {ep}: {'shadow' if shadow else 'streaming'} from joints {np.round(robot.joints_deg(), 1).tolist()}")
            if not run_episode(stream, robot, cams, keys, max_steps, shadow, ep):
                break
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping stream, homing, de-torquing")
    finally:
        robot.disconnect() if shadow else robot.rest()
        for cam in cams.values():
            cam.close()
        stream.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help=".../checkpoints/<step>/pretrained_model")
    p.add_argument("--task", default=C.TASK)
    p.add_argument("--n-action-steps", type=int, default=C.N_ACTION_STEPS, help="actions executed per predicted chunk")
    p.add_argument("--blend-ticks", type=int, default=C.BLEND_TICKS,
                   help="ticks to decay the chunk-boundary discontinuity over (0 = off)")
    p.add_argument("--dry-run", action="store_true", help="fake arm + synthetic frames: timing table, no hardware")
    p.add_argument("--shadow", action="store_true", help="real cameras + read-only arm state, print actions, NO motion")
    p.add_argument("--fake-arm", action="store_true", help="FakeArm (piper/fake_arm.py) instead of the CAN bus")
    p.add_argument("--max-steps", "--steps", type=int, default=C.FPS * 60, help="ticks per episode (15 Hz)")
    p.add_argument("--episodes", type=int, default=1, help="episodes; Enter starts the next one")
    p.add_argument("--sim", action="store_true", help="MuJoCo twin: sim cameras + SimPiper, same loop (sim/)")
    args = p.parse_args()

    runner = Pi05Runner(args.checkpoint, args.task)
    if args.sim:
        from sim.cameras import open_sim_cameras
        from sim.piper_sim import connect_sim
    cams = ({n: FakeCamera(n) for n in C.CAMERA_NAMES} if args.dry_run
            else open_sim_cameras() if args.sim else open_cameras())
    if args.dry_run or args.fake_arm:
        from piper.fake_arm import FakeArm
        robot = FakeArm()
    elif args.sim:
        robot = connect_sim()
    else:
        robot = connect_readonly() if args.shadow else connect_for_motion()
    keys = Keys()
    threading.Thread(target=keys.read_stdin, daemon=True).start()
    deploy(ActionStream(runner.predict_chunk, args.n_action_steps, args.blend_ticks), robot, cams, keys, args.max_steps, args.episodes,
           args.shadow)
    print(f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
