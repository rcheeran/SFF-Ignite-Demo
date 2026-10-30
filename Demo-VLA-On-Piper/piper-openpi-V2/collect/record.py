"""VR teleop + LeRobotDataset recording (dataset contract: docs/CONTRACT.md).

    python -m collect.record --repo-id <user>/piper_block_bowl --episodes 10 [--push] [--private]
    python -m collect.record --fake --episodes 2 --overwrite     # no robot/headset: synthetic controller + cameras
    python -m collect.record --sim --scripted --episodes 10 --repo-id local/sim_block_bowl   # MuJoCo twin + scripted expert
    python -m collect.record --sim --episodes 10 --repo-id local/sim_teleop                  # headset teleop on the twin

Controller: A owns teleop (hold 3 s = arm / disarm, tap = discard the running episode), B owns episodes (hold 3 s
= start, tap = stop and save). Thumbstick click = READY (ignored while recording). Laptop UI: http://<jetson>:8092
(cameras + state, no headset on your head needed).
Each frame: observation = [j1..j6, gripper_measured_mm] + top/wrist images;
action = joints observed at the NEXT frame + the gripper mm commanded at this frame.
An existing --root is resumed and --episodes is the total; --overwrite deletes it first. Ctrl-C (or SIGTERM) ends
the session: the running episode is dropped, the arm rests, the dataset is finalized (and pushed with --push).
Every saved episode is durable on disk at once (finalize + reopen), so a kill or power loss loses only the running one.
"""
import argparse
import contextlib
import json
import os
import shutil
import signal
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import av
import cv2
import datasets
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from piper import config as C
from piper.cameras import FakeCamera, open_cameras
from piper.robot import connect_for_motion
from teleop.quest import QuestTeleop, serve, synthetic_feed

IMG = {"dtype": "video", "shape": (C.IMG_H, C.IMG_W, 3), "names": ["height", "width", "channels"]}
FEATURES = {**{f"observation.images.{n}": IMG for n in C.CAMERA_NAMES},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": C.STATE_NAMES},
            "action": {"dtype": "float32", "shape": (7,), "names": C.STATE_NAMES}}


def fake_state7(teleop):
    x, y, z = teleop.target_pose[:3]
    return np.array([np.degrees(np.arctan2(y, x)), x / 10, -z / 10, 46, -23.8, -165, teleop.grip_cmd_mm], np.float32)


class Episode:
    """One running episode. Frame t is written once frame t+1's joints are known (one-frame lag):
    action[t] = [joints observed at t+1, gripper commanded at t]; the last frame holds its own joints."""

    def __init__(self, add_frame):
        self.add_frame, self.pending, self.n = add_frame, None, 0
        self.t0, self.stuck, self.no_feedback = time.monotonic(), 0, 0   # quality counters: events, ticks

    def push(self, frame: dict, grip_cmd_mm: float):
        self._write(frame["observation.state"][:6])
        self.pending = frame, float(grip_cmd_mm)

    def close(self) -> int:
        if self.pending:
            self._write(self.pending[0]["observation.state"][:6])
        return self.n

    def _write(self, next_joints):
        if self.pending:
            frame, grip = self.pending
            self.add_frame({**frame, "action": np.append(next_joints, grip).astype(np.float32)})
            self.pending, self.n = None, self.n + 1

    def summary(self) -> str:
        bad = (f", {self.stuck} stuck" if self.stuck else "") + (
            f", {self.no_feedback / C.FPS:.1f} s no arm feedback" if self.no_feedback else "")
        return f"{self.n} frames ({self.n / C.FPS:.1f} s){bad}" + (" - consider B next time" if bad else "")


@contextlib.contextmanager
def no_interrupt():
    """Defer Ctrl-C / SIGTERM (forked encoder workers inherit it): a half-done save_episode()/finalize() =
    unloadable dataset. Raises KeyboardInterrupt on exit if either arrived."""
    hit, sigs = [], (signal.SIGINT, signal.SIGTERM)
    prev = [signal.signal(s, lambda *_: hit.append(1)) for s in sigs]
    try:
        yield
    finally:
        for s, h in zip(sigs, prev):
            signal.signal(s, h)
    if hit:
        raise KeyboardInterrupt


def open_dataset(repo_id: str, root: Path, overwrite=False) -> LeRobotDataset:
    """Resume the dataset at root if there is one, else create it (overwrite deletes it first)."""
    if overwrite and root.exists():
        shutil.rmtree(root)
    if (root / "meta" / "info.json").exists():
        ds = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4)
    elif root.exists():
        raise SystemExit(f"{root} exists but is not a dataset - pass --overwrite or choose another --root")
    else:
        ds = LeRobotDataset.create(repo_id=repo_id, fps=C.FPS, features=FEATURES, root=root,
                                   robot_type=C.ROBOT_TYPE, use_videos=True, image_writer_threads=4)
    datasets.disable_progress_bar()            # after resume: its metadata load re-enables the "Map:" bar
    return ds


PAGE = """<!doctype html><meta charset=utf-8><title>Piper recorder</title>
<style>body{margin:0;background:#111;color:#ddd;font:15px/1.5 sans-serif;display:flex;flex-wrap:wrap;gap:12px;padding:12px}
img{width:480px;height:360px;background:#222}#p{min-width:22em;background:#1b1b1b;border:1px solid #333;border-radius:6px;padding:10px 14px}
#rec{font-size:1.6em;font-weight:bold}.on{color:#f44}.ok{color:#6f6}.warn{color:#fc6}.bad{color:#f66}.dim{color:#888}
td{padding:1px 8px 1px 0;vertical-align:top}td:first-child{color:#888}</style>
<div><img src=/stream/top><br><span class=dim>top</span></div><div><img src=/stream/wrist><br><span class=dim>wrist</span></div>
<div id=p>loading...</div>
<script>
const cls=(ok,bad)=>ok?'ok':(bad?'bad':'warn');
async function tick(){try{const s=await (await fetch('/status.json')).json();
const rec=s.recording;const rows=[
['dataset',`${s.repo_id} &rarr; ${s.root}`],['episode',`${s.episode} of ${s.episodes} &middot; ${s.saved} saved`],
['recording',`<span id=rec class="${rec?'on':'dim'}">${rec?'&#9679; REC '+s.seconds.toFixed(1)+' s &middot; '+s.frames+' frames':'idle'}</span>`],
['last saved',s.last||'-'],
['controller',`<span class="${cls(s.engaged,!s.armed)}">${s.engaged?'ENGAGED (moving)':s.armed?'ARMED - squeeze to move':'DISARMED - hold A 3 s'}</span>`],
['headset',`<span class="${cls(s.tracked,!s.pkts)}">${s.pkts?s.transport+' '+s.rate_hz+' pkt/s, age '+s.pkt_age_s+' s, '+(s.tracked?'controller tracked':'controller NOT tracked'):'no packets yet'}</span>`],
['status',`${s.status}${s.limit&&s.limit.length?' &middot; <span class=warn>'+s.limit.join('+')+' at limit</span>':''}${s.stuck?' &middot; <span class=bad>STUCK</span>':''}`],
['arm',s.joints?`joints ${s.joints.join(', ')}<br>pose ${s.pose.join(', ')}<br>grip ${s.grip_mm} mm &middot; J6 ${s.j6_c} &deg;C &middot; fw ${s.arm_status||'-'}${s.disabled&&s.disabled.length?' &middot; <span class=bad>DISABLED '+s.disabled.join('+')+'</span>':''}${s.faults&&s.faults.length?' &middot; <span class=bad>'+s.faults.join(', ')+'</span>':''}`:'<span class=dim>no arm (fake)</span>'],
['buttons',`hold B 3 s start &middot; tap B stop + save &middot; tap A discard &middot; hold A 3 s arm/disarm${s.b_progress>0&&s.b_progress<1?' &middot; <span class=warn>starting episode '+Math.round(s.b_progress*100)+'%</span>':''}${s.arm_progress>0&&s.arm_progress<1?' &middot; <span class=warn>arming '+Math.round(s.arm_progress*100)+'%</span>':''}`]];
document.getElementById('p').innerHTML='<table>'+rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('')+'</table>';
}catch(e){document.getElementById('p').innerHTML='<span class=bad>recorder not responding</span>';}}
setInterval(tick,500);tick();
</script>"""


class Monitor:
    """Laptop UI on :8092 (any browser, WiFi): both cameras as MJPEG streams + the recorder/teleop state."""
    def __init__(self, cams, status, port=C.MONITOR_PORT):
        self.cams, self.status = cams, status            # status() -> dict, built by record()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/status.json":
                    body = json.dumps(status()).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif self.path.startswith("/stream/") and self.path[8:] in cams:
                    cam = cams[self.path[8:]]
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                    try:
                        while True:                      # 5 fps, 480x360, q60: ~0.2 MB/s per camera over WiFi
                            ok, jpg = cv2.imencode(".jpg", cv2.resize(cam.read(), (480, 360)), [cv2.IMWRITE_JPEG_QUALITY, 60])
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                            time.sleep(0.2)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    body = PAGE.encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        srv = socketserver.ThreadingTCPServer(("0.0.0.0", port), H, bind_and_activate=False)
        srv.daemon_threads = srv.allow_reuse_address = True      # must precede bind: restart within TIME_WAIT
        srv.server_bind(), srv.server_activate()
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"laptop UI: http://{C.TELEOP_URL.split('//')[1].split(':')[0]}:{port}/")


def record(ds: LeRobotDataset, args) -> LeRobotDataset:
    """Cameras + arm + headset loop until ds holds args.episodes episodes. The arm rests on every exit.
    Returns the live dataset object (every saved episode reopens it)."""
    if args.sim:
        from sim.cameras import open_sim_cameras
        from sim.piper_sim import connect_sim
    cams = ({n: FakeCamera(n) for n in C.CAMERA_NAMES} if args.fake and not args.real_cams
            else open_sim_cameras() if args.sim else open_cameras())
    robot = None if args.fake else (connect_sim() if args.sim else connect_for_motion())
    teleop = QuestTeleop(None if args.scripted else robot, scale=args.scale, armed=args.fake or args.scripted,
                         rot_scale=args.rot_scale).start()      # --scripted: the expert drives, teleop carries events + grip
    stop = threading.Event()
    if args.scripted:
        from sim.expert import run_episodes
        threading.Thread(target=run_episodes, args=(robot, args.episodes - ds.meta.total_episodes, args.seed, teleop),
                         daemon=True).start()
    elif args.fake:
        threading.Thread(target=synthetic_feed, args=(teleop, 1e9, stop), daemon=True).start()
    else:
        serve(teleop)
    ui = {"ep": None, "saved": ds.meta.total_episodes, "last": ""}

    def status():
        ep = ui["ep"]
        return {**teleop.snapshot(), "repo_id": args.repo_id, "root": str(ds.root), "episodes": args.episodes,
                "episode": ui["saved"], "saved": ui["saved"], "recording": ep is not None,
                "seconds": round(time.monotonic() - ep.t0, 1) if ep else 0.0, "frames": ep.n if ep else 0, "last": ui["last"]}
    mon = Monitor(cams, status)
    state = (lambda: fake_state7(teleop)) if args.fake else robot.state7

    def capture():
        return {**{f"observation.images.{n}": cv2.cvtColor(cams[n].read(), cv2.COLOR_BGR2RGB) for n in cams},
                "observation.state": state(), "task": args.task}

    ep, saved, next_t = None, ds.meta.total_episodes, time.monotonic()
    print(f"recording episodes {saved}..{args.episodes - 1} -> {ds.root}  (hold B 3 s = start, tap B = save, tap A = discard)")
    try:
        while saved < args.episodes and teleop.fault is None:
            while not teleop.events.empty():
                ev = teleop.events.get()
                if ev == "b_hold" and ep is None:
                    ep, next_t = Episode(ds.add_frame), time.monotonic()
                    print(f"episode {saved}: REC")
                elif ev == "b_tap" and ep:
                    if ep.close() >= args.min_frames:
                        with no_interrupt():        # durable before anything else can happen: only finalize()
                            ds.save_episode()       # writes the parquet footers (the metadata buffer size does not
                            ds.finalize()           # help), and only a reopen starts fresh files, so a kill or
                            ds = open_dataset(args.repo_id, ds.root)   # power loss loses just the running episode
                            ui["last"] = f"episode {saved}: {ep.summary()}"
                            print(f"episode {saved}: saved {ep.summary()}")
                            saved, ep = saved + 1, None      # before a deferred Ctrl-C: the episode is NOT dropped
                    else:
                        ds.clear_episode_buffer()
                        print(f"episode {saved}: too short ({ep.n} frames) - dropped")
                    ep = None
                elif ev in ("a_tap", "disarmed") and ep:
                    ds.clear_episode_buffer()
                    ui["last"] = f"episode {saved}: DISCARDED ({ep.n} frames)"
                    print(ui["last"])
                    ep = None
                elif ev not in ("a_tap", "b_hold", "b_tap", "thumb", "engaged", "released"):   # rare state changes
                    print("event:", ev)
                    if ev == "stuck" and ep:
                        ep.stuck += 1
            if args.fake and (ep is None or time.monotonic() - ep.t0 > args.fake_episode_s):
                teleop.events.put("b_tap" if ep else "b_hold")
            if ep:
                ep.push(capture(), teleop.grip_cmd_mm)
                ep.no_feedback += teleop.no_feedback
            n = ep.n if ep else 0
            ui["ep"], ui["saved"] = ep, saved
            teleop.rec_state = {"recording": ep is not None, "episode": saved, "frames": n}
            if ep and n and n % C.FPS == 0:                        # one line per second while recording
                s = teleop.snapshot()
                print(f"ep {saved}/{args.episodes} REC {n}  {teleop.status}  grip={teleop.grip_cmd_mm:.0f} | "
                      f"{s['transport']} {s['rate_hz']}/s err={s['err_mm']} mm")
            next_t += 1.0 / C.FPS
            time.sleep(max(0.0, next_t - time.monotonic()))
        if teleop.fault:
            print(f"control loop fault: {teleop.fault} - stopping")
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping" + (f", episode {saved} dropped ({ep.n} frames)" if ep else ""))
    finally:
        if ep:
            ds.clear_episode_buffer()
        stop.set()
        teleop.stop()
        for cam in cams.values():
            cam.close()                     # a live RealSense pipeline at interpreter exit aborts the process
        if robot is not None:
            robot.rest()
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/fake_block_bowl")
    ap.add_argument("--root", default=None, help="default data/<dataset name>; resumed if it exists")
    ap.add_argument("--episodes", type=int, default=10, help="total episodes the dataset should hold")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true", help="with --push: private Hub repo")
    ap.add_argument("--fake", action="store_true", help="no robot, no headset: synthetic controller + cameras")
    ap.add_argument("--real-cams", action="store_true", help="with --fake: use the real cameras")
    ap.add_argument("--sim", action="store_true", help="MuJoCo twin instead of the arm and the real cameras (sim/)")
    ap.add_argument("--scripted", action="store_true", help="with --sim: the scripted expert drives the episodes, no headset")
    ap.add_argument("--seed", type=int, default=0, help="with --scripted: block-spawn seed")
    ap.add_argument("--task", default=C.TASK)
    ap.add_argument("--min-frames", type=int, default=C.FPS)
    ap.add_argument("--fake-episode-s", type=float, default=3.0)
    ap.add_argument("--overwrite", action="store_true", help="delete an existing --root first")
    ap.add_argument("--scale", type=float, default=C.TELEOP_SCALE, help="robot motion per controller motion")
    ap.add_argument("--rot-scale", type=float, default=None, help="rotation scale (default = --scale)")
    args = ap.parse_args()
    os.environ.setdefault("SVT_LOG", "1")      # else the AV1 encoder prints a 20-line banner per video
    av.logging.set_libav_level(av.logging.ERROR)   # and the mp4 muxer a line per concatenation
    signal.signal(signal.SIGTERM, signal.default_int_handler)   # pkill = Ctrl-C: drop the episode, rest, finalize
    root = Path(args.root or os.path.join(C.DATA_DIR, args.repo_id.split("/")[-1]))
    ds = open_dataset(args.repo_id, root, args.overwrite)
    if ds.meta.total_episodes:
        print(f"resuming: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
    try:
        if ds.meta.total_episodes < args.episodes:
            ds = record(ds, args)
        else:
            print(f"already {ds.meta.total_episodes} episodes (--episodes {args.episodes} is the total): nothing to record")
    finally:
        with no_interrupt():
            ds.finalize()
        print(f"{ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames -> {root}")
    if args.push and ds.meta.total_episodes:
        if "HF_TOKEN" not in os.environ:           # (setdefault would open the file even when the env is set)
            os.environ["HF_TOKEN"] = Path(C.HF_TOKEN_FILE).expanduser().read_text().strip()
        try:
            ds.push_to_hub(tags=["piper", "quest", "pi05"], private=args.private)
            print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")
        except KeyboardInterrupt:
            print("\nCtrl-C during push: the local dataset is complete - re-run the same command to retry")


if __name__ == "__main__":
    main()
