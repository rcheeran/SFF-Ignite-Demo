"""Scripted "put the block in the bowl" on the digital twin (privileged: it reads the block and bowl poses
from MuJoCo). Drives SimPiper with send_pose at 30 Hz through piper.robot.clamp_pose - exactly the clamps
teleop obeys - so the recorded joint trajectories look like teleop data.

    python -m sim.expert --episodes 10 --seed 0      success rate over seeded episodes (random block spawn)

run_episodes(sim, n, seed, teleop=None) also fires the recorder's A-events through teleop.events and mirrors the
gripper command into teleop.grip_cmd_mm: that is how `collect.record --sim --scripted` writes datasets.
"""
import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper.robot import clamp_pose, pose_to_xyz_rot, xyz_rot_to_pose
from sim import piper_sim
from sim.piper_sim import SimPiper

DOWN = (Rot.from_euler("y", 180, degrees=True), Rot.from_euler("x", 180, degrees=True))   # tool z down


def grasp_rot(sim, points, block_yaw):
    """Tool pointing down, fingers closing across the block faces. Of the 8 equivalent orientations (4 yaws x
    2 finger sides) take the one the IK reaches at every point of the carry (above the block AND over the
    bowl: J6 spans +-120 deg and must mirror the base rotation of the carry) with the least joint
    motion from where the arm is. Returns (rot, joints at points[0]) or None."""
    q0, best = sim.joints_deg(), None
    for k in range(4):
        for base in DOWN:
            rot = Rot.from_euler("z", block_yaw + 90.0 * k, degrees=True) * base
            qs = [piper_sim.ik(xyz, rot, q0, sim.limits, iters=60) for xyz in points]
            cost = sum(np.abs(q - q0).sum() for q, _ in qs)
            if all(ok for _, ok in qs) and (best is None or cost < best[0]):
                best = (cost, rot, qs[0][0])
    return None if best is None else best[1:]


def goto(sim, xyz, rot, grip, teleop=None, tol_mm=4.0, tol_deg=3.0, timeout=6.0, hold=0.0) -> bool:
    """Stream the waypoint at CONTROL_HZ through clamp_pose until the measured pose is within tolerance,
    then `hold` s more (gripper time). Returns False on timeout."""
    xyz = np.asarray(xyz, np.float64)
    cmd_xyz, cmd_rot = pose_to_xyz_rot(sim.pose())
    t0 = next_t = time.monotonic()
    reached = None
    while time.monotonic() - t0 < timeout:
        meas_xyz, meas_rot = pose_to_xyz_rot(sim.pose())
        cmd_xyz, cmd_rot, _ = clamp_pose(cmd_xyz, cmd_rot, xyz, rot, meas_xyz, meas_rot)
        cmd = xyz_rot_to_pose(cmd_xyz, cmd_rot)
        sim.send_pose(cmd_xyz, cmd[3:], grip)
        if teleop is not None:
            teleop.grip_cmd_mm, teleop.target_pose = grip, cmd
        close = (np.linalg.norm(meas_xyz - xyz) < tol_mm
                 and np.linalg.norm((meas_rot.inv() * rot).as_rotvec(degrees=True)) < tol_deg)
        reached = (reached or time.monotonic()) if close else None
        if reached is not None and time.monotonic() - reached >= hold:
            return True
        next_t += 1.0 / C.CONTROL_HZ
        time.sleep(max(0.0, next_t - time.monotonic()))
    return False


def success(sim) -> bool:
    """Block centre inside the bowl (within its floor radius, below the wall top)."""
    b, bowl = sim.block_pose()[0], sim.body_xyz("bowl")
    return bool(np.linalg.norm(b[:2] - bowl[:2]) < sim.bowl_r_mm - 5 and b[2] < 40)


def run_episode(sim: SimPiper, rng, teleop=None) -> bool:
    """Random block spawn -> READY -> [A] above the block (joint-space: streaming a vertical tool out from
    READY pitches it down while the flange is still near the base, past J5's limit), then Cartesian:
    descend, close, lift, over bowl, lower, open, up -> READY [A]."""
    lo, hi = C.SIM_BLOCK_SPAWN_MM
    sim.place_block(rng.uniform(lo, hi), rng.uniform(-45, 45))
    sim.ready(C.GRIP_OPEN_MM)
    time.sleep(0.3)
    (bx, by, _), byaw = sim.block_pose()
    bowl = sim.body_xyz("bowl")
    above, over = np.array([bx, by, C.SIM_LIFT_Z_MM]), np.array([bowl[0], bowl[1], C.SIM_LIFT_Z_MM])
    plan = grasp_rot(sim, (above, over), byaw)
    if plan is None:
        print(f"expert: no grasp orientation reaches both the block at {[round(bx), round(by)]} and the bowl")
        return False
    rot, q_above = plan
    grasp, release = [bx, by, C.SIM_GRASP_Z_MM], [bowl[0], bowl[1], C.SIM_RELEASE_Z_MM]
    o, c = C.GRIP_OPEN_MM, C.GRIP_CLOSED_MM
    if teleop is not None:
        teleop.events.put("b_hold")                  # = the operator holding B: episode starts
    sim.move_joints(q_above, 6.0, o, speed=C.SPEED_JOINT)
    for xyz, grip, hold in ((grasp, o, 0.2), (grasp, c, 0.8), (above, c, 0.0),
                            (over, c, 0.0), (release, c, 0.2), (release, o, 0.8), (over, o, 0.0)):
        goto(sim, xyz, rot, grip, teleop, hold=hold)
    ok = success(sim)
    sim.ready(C.GRIP_OPEN_MM)
    if teleop is not None:            # the episode ends back at READY, like a teleop one
        teleop.events.put("b_tap")                   # = a tap of B: stop and save
    return ok


def run_episodes(sim, n, seed=0, teleop=None) -> list:
    rng, results = np.random.default_rng(seed), []
    for i in range(n):
        t0 = time.monotonic()
        results.append(run_episode(sim, rng, teleop))
        print(f"expert episode {i}: {'SUCCESS' if results[-1] else 'FAIL'}  {time.monotonic() - t0:.0f} s  "
              f"block at {np.round(sim.block_pose()[0]).tolist()}  status {sim.arm_status()}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sim = SimPiper().connect()
    try:
        ok = run_episodes(sim, args.episodes, args.seed)
        print(f"EXPERT: {sum(ok)}/{len(ok)} succeeded (seed {args.seed})")
    finally:
        sim.rest()


if __name__ == "__main__":
    main()
