"""All hardware / teleop / dataset constants. Nothing else in the repo hard-codes these."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SNAP_DIR = os.path.join(DATA_DIR, "snapshots")   # tools/check_arm.py camera snapshots (gitignored, made on demand)
LOG_DIR = os.path.join(DATA_DIR, "logs")         # tools/stream_test.py csv logs

# ---- CAN / arm ---------------------------------------------------------------
CAN_IFACE = "can_piper"
CAN_BITRATE = 1000000
FEEDBACK_WAIT_S = 2.0    # connect() waits this long for the first joint feedback before giving up
CONTROL_HZ = 30
FPS = 15

SPEED_TELEOP = 30        # % — Cartesian streaming (teleop, self-test)
SPEED_JOINT = 30         # % — JointCtrl streaming (policy deploy)
SPEED_HOME = 15          # % — homing

HOME_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]           # folded, straight wrist, J6 at its factory mid-range (0),
                                                            # where the wrist camera sits on top. The old -119/-165 roll
                                                            # (plus a custom J6 limit) put the camera almost upside down
                                                            # and froze both live sessions - gone
REST_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 25.0, 0.0]          # HOME with the wrist pitched into the band where it
                                                            # settles under gravity (measured 10..31 deg with
                                                            # tools/rest.py --measure), so de-torquing drops nothing
SPEED_REST = 10
SOFT_DISABLE_S = 2.5     # MIT-mode stiffness ramp before de-torque (shoulder/elbow sag 2 deg smoothly, not as a drop)
SOFT_KP, SOFT_KD = 10.0, 3.0
READY_JOINTS_DEG = [0.0, 45.0, -25.0, 0.0, 40.0, 0.0]      # flange ~[128,0,232], gripper 50 deg down. The firmware
                                                            # IK refuses EndPoseCtrl from folded HOME and is ill-conditioned
                                                            # near J5=0 (wrist singularity), so J5 stays well pitched; and
                                                            # with the flange only 84 mm from the base axis (old READY
                                                            # [0,30,-15,0,40]) a +-15 mm wiggle pinned J6 at its -179 limit
                                                            # (the live-session freeze; 28 % refused ticks on the twin) -
                                                            # 127 mm out it is 0 % (sim sweep 2026-08-22, confirm live)
HOME_TOL_DEG = 2.0
HOME_TIMEOUT_S = 12.0

JOINT_LIMITS_DEG = [[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-120, 120]]   # factory limits (piper_sdk
                         # piper_param table). connect() reads the motors' stored limits and writes these back if a
                         # previous session changed them (the old custom J6 [-179,-60] is exactly such a change)
J6_TEMP_PAUSE_C = 46
J6_TEMP_RESUME_C = 41
FAULT_STATUS = ("EMERGENCY_STOP", "COLLISION_OCCURRED")   # firmware arm_status values that stop every controller:
                                                          # teleop drops the clutch, deploy stops streaming + rests

GRIP_OPEN_MM = 70.0
GRIP_CLOSED_MM = 0.0
GRIP_EFFORT = 1000       # 0.001 N/m

# ---- Cartesian safety (flange frame, mm / deg) -------------------------------
WORKSPACE_MIN_MM = [30.0, -400.0, 40.0]     # flange at HOME is x~46, y~-26, z~238 - must lie inside
WORKSPACE_MAX_MM = [500.0, 400.0, 480.0]
TOOL_LEN_MM = 120.0      # gripper tip lies this far past the flange along the tool z-axis
TIP_Z_FLOOR_MM = 9.0     # tool tip never commanded below this height above the table
STEP_MAX_MM = 15.0       # per 30 Hz tick
STEP_MAX_DEG = 8.0

# ---- Teleop ------------------------------------------------------------------
TELEOP_SCALE = 1.0       # robot mm per controller mm while clutched
TELEOP_WATCHDOG_S = 0.5
LEAD_MAX_MM = 30.0       # the commanded target may never lead the measured pose further than this
LEAD_MAX_DEG = 20.0      # (the firmware IK refuses silently; an unbounded target causes a violent catch-up)
STUCK_S = 2.0            # engaged but the arm not following for this long -> auto-declutch, re-squeeze
TELEOP_PIVOT_MM = 91.0   # controller rotations pivot the tool about the point this far behind the flange along tool z:
                         # the Piper's wrist centre (J4/J5/J6 axes meet there, DH d6), so a pure hand rotation is a
                         # wrist-only motion and the base joints stay put. 0 = pivot at the flange (old behaviour)
IK_LIMIT_MARGIN_DEG = 2.0    # reachable() projects targets to this far INSIDE the factory limits: a pose whose
                             # solution sits exactly on a limit is refused by the firmware (its own solution lands
                             # a hair past it - seen live 2026-08-30, J5 pinned at +70 while pitching down)
IK_JOINT_COST = [3.0, 3.0, 2.0, 1.0, 1.0, 1.0]   # piper/kinematics.ik: a degree of J1..J3 costs this much more than a
                         # degree of wrist; when a target cannot be reached exactly (joint limit, no solution) the
                         # projection leaves error rather than swinging the whole arm
TELEOP_ARM_HOLD_S = 3.0  # hold A this long to ARM / DISARM
TELEOP_EPISODE_HOLD_S = 3.0  # hold B this long to START an episode; a tap of B stops and saves it; a tap of A
                             # (any press shorter than TELEOP_ARM_HOLD_S) discards it
J6_WARN_DEG = 15.0       # headset warning when the wrist roll is this close to its firmware limit
TELEOP_PORT = 8443
TELEOP_CERT = os.path.join(ROOT, "teleop", "certs", "cert.pem")
TELEOP_KEY = os.path.join(ROOT, "teleop", "certs", "key.pem")
TELEOP_URL = "https://192.168.1.45:8443/"
STATIC_DIR = os.path.join(ROOT, "teleop", "static")
TRIGGER_DEADZONE = 0.05

# ---- Cameras -----------------------------------------------------------------
IMG_W, IMG_H = 640, 480
CAM_FPS = 30
TOP_CAM_DEV = "/dev/v4l/by-id/usb-Generic_USB_2.0_Camera_20220523-video-index0"       # Realtek: the WIDE view
                         # (whole table incl. the taped borders; chosen by Chris 2026-08-30, ~15 fps sustained)
ALT_TOP_CAM_DEV = "/dev/v4l/by-id/usb-WSD_usb_camera_202406051510-video-index0"        # HP 720p: tighter FOV, unused
WRIST_CAM_SERIAL = None  # RealSense D405; None = first device found. Mounted on TOP of the gripper at J6 = 0 (the
                         # centre of the wrist's mechanical range): the image is upright whenever J6 is near 0, and
                         # upside down at the old -165 "camera-up" roll (that comment had it backwards)

# ---- Dataset -----------------------------------------------------------------
TASK = "put the block in the bowl"
ROBOT_TYPE = "piper"
STATE_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper_mm"]
CAMERA_NAMES = ["top", "wrist"]
MONITOR_PORT = 8092
HF_TOKEN_FILE = "~/Documents/HG-TOKEN.txt"
HF_USER = "royc1981"                             # default owner when a repo id is not spelled out in full
HUB_TAGS = ["piper", "pi05"]                     # dataset tags on the Hub

# ---- Deploy (inference/run_policy.py, joint-space streaming at FPS) ---------
N_ACTION_STEPS = 10      # actions executed per predicted 50-step chunk before the prefetched chunk takes over
PREFETCH_AT = 9          # hand the newest observation to the model thread when this many of the N actions remain
                         # (model ~500 ms = 7.5 ticks with the loop running; 9 leaves ~100 ms before the chunk runs out)
BLEND_TICKS = 6          # ticks to decay a chunk-boundary jump over (0 = off); see run_policy.Chunk.next
MAX_JOINT_STEP_DEG = 10.0    # per 15 Hz tick; teleop data never exceeds a few degrees per frame
JOINT_LEAD_MAX_DEG = 20.0    # the joint target may never lead the measured joints further than this (a blocked
                             # arm otherwise accumulates a target far away and lurches when it frees)

# ---- Cloud training (training/cloud) -----------------------------------------
H100_USD_PER_H = 3.29    # Lambda 1x H100 PCIe, on-demand
H100_STEP_S = 1.0        # estimated s/step at batch 16 (Thor: 5 s/step at batch 8); unverified until the first run
CLOUD_OVERHEAD_MIN = 15  # setup + base-model load + checkpoint upload per run, for the cost estimate

# ---- Sim (MuJoCo digital twin, sim/) -----------------------------------------
SIM_GL = "egl"                    # MUJOCO_GL if unset: egl = GPU offscreen (2 ms/frame on the Thor); osmesa works too (40 ms)
SIM_SCENE = os.path.join(ROOT, "sim", "scene.xml")
SIM_JOINT_OFFSET_DEG = [0.0, 2.0, -2.0, 0.0, 0.0, 0.0]   # MJCF qpos = reported joints + this: the firmware/SDK FK carries
                                                         # a 2 deg J2/J3 zero offset ("dh_is_offset"); residual 10 -> 1.4 mm
SIM_JOINT_VMAX_DEG_S = 200.0      # joint speed at 100 % (~3.5 rad/s): JointCtrl moves slew at speed % of it (HOME -> READY at
                                  # 15 % ~2 s); EndPoseCtrl streaming follows the IK target at the full rate (real lag 67-200 ms)
SIM_GRIP_VMAX_MM_S = 140.0        # full 70 mm stroke in 0.5 s
SIM_LAG_S = 0.05                  # first-order lag on the joint target (the real arm trails its command by 67-200 ms)
SIM_IK_ITERS = 10                 # damped-least-squares iterations per send_pose tick (warm-started from the last target)
SIM_IK_DAMP = 0.2                 # lambda (mm / deg units): 2.0 stalled the rotation part, 0.2 converges in ~4 iterations
SIM_IK_ROT_WEIGHT = 3.0           # 1 deg of orientation error weighs like 3 mm
SIM_IK_TOL_MM, SIM_IK_TOL_DEG = 1.0, 0.5   # worse than this after the iterations = refused, like the firmware IK
SIM_TOOL_LEN_MM = 135.0           # MJCF finger tips past the flange (pads span 105..135 mm; the real TOOL_LEN_MM is 120)
SIM_J6_TEMP_C = 32
SIM_BOWL_XY_MM = [280.0, -150.0]  # bowl centre on the table (applied to the scene at load)
SIM_BLOCK_SPAWN_MM = [[220.0, 0.0], [320.0, 120.0]]   # [x, y] min / max of the block spawn box, yaw +-45 deg
SIM_GRASP_Z_MM = 142.0            # flange height for the grasp: tips 7 mm above the table, pads across the 30 mm cube
SIM_LIFT_Z_MM = 215.0             # flange height while carrying (block bottom ~80 mm, bowl wall 40 mm); a straight-down
                                  # tool above ~230 mm needs J5 past its 70 deg limit
SIM_RELEASE_Z_MM = 200.0
SIM_VIDEO_DIR = os.path.join(DATA_DIR, "sim")
