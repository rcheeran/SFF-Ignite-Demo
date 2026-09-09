---
name: project_asimov_robotics
description: Local Asimov SDK robotics setup — FR3 VLA sim (Docker) and the Piper MuJoCo replica
metadata: 
  node_type: memory
  type: project
  originSessionId: ec507e05-ee96-4ca5-aed5-bd4035b2455d
---

Two working robot-agent sims live under `/home/rog01/Documents`, both driven by a **local Ollama LLM** (`~/.local/ollama/bin/ollama serve`, model `qwen2.5:7b`, OpenAI API at :11434/v1 — used via `USE_LOCAL_LLM=true`):

1. **Microsoft Asimov FR3 VLA sim** — `Documents/Asimov`, sample `vla-fr3-single-arm-manipulator`. Docker stack (GR00T N1.7 + LIBERO on the RTX 5080). Toolset :8102 (compose in `robot-agent/samples/toolsets/vla-fr3-single-arm-sim-real-toolset`, bring up with `-f docker-compose.yml -f docker-compose.fr3.yml -f docker-compose.egl.yml`), agent host-run on :8203. Needs `HF_TOKEN` in the toolset `.env` for the gated Cosmos backbone.

2. **Piper-Asimov replica** — `Documents/piper-asimov` (I built this). Shared venv `piper-asimov/.venv` (mujoco + asimov + piper_sdk). Two variants:
   - **Sim** — `piper-sim-toolset` (:8210) + `piper-arm-agent` (:8211). AgileX Piper in MuJoCo, explicit kinematic control (no VLA exists for Piper), IK + weld-grasp, live viewer via `start_sim.sh` (main-thread viewer, server on bg thread).
   - **Real** — `piper-real-toolset` (:8212) + `piper-real-agent` (:8213). Maps Asimov to the PHYSICAL Piper over CAN via `piper_sdk`. `PIPER_MODE=mock|real`, motion OBSERVE-ONLY unless `PIPER_ALLOW_MOTION=1`. Real arm-motion path written to the `SFF-AgileX-Piper` scripts' API but NOT yet exercised on the moving arm. Real arm needs Slave mode + CAN bring-up per `Documents/SFF-AgileX-Piper/SETUP.md`.
   - **Camera** — Intel RealSense **D405** (serial 126122270802) IS connected and WORKING; wired into the real toolset (`camera.py`, tools `capture_image`/`get_scene_depth`, `GET /camera/snapshot`). `pyrealsense2`+`opencv` are in `.venv`. Camera works independent of the arm.
   - **Hardware VERIFIED WORKING (2026-07-24)**: real arm fully controlled end-to-end via the Asimov toolset — enable, gripper, and joint1 +10° move all confirmed on hardware (fps 3040, 200 Hz telemetry, Slave mode). Bring-up gotcha: after an arm power-cycle `can0` drops to DOWN/STOPPED and needs `sudo can_activate.sh can0 1000000` (SETUP.md step 3, needs user's sudo); a single-node bus (arm off) makes sends fail with `SEND_MESSAGE_FAILED (100017)` and fps=0. IMPORTANT: real arm joint zero-convention differs from the sim URDF (joint6 rests ~-174°); `backends.py` JOINT_LIMITS_DEG were widened to real guardrails so joint commands don't clamp into surprise moves. Always bring up observe-only (`PIPER_ALLOW_MOTION=0`) first.

Durable gotchas:
- **Repo shipped with CRLF line endings** — broke Docker shell entrypoints (`bash\r`). Converted to LF.
- **Host is X11** (`DISPLAY=:1`), not Wayland. On-screen MuJoCo viewer needs `MUJOCO_GL=glx` + `xhost +local:`; headless needs `MUJOCO_GL=egl`.
- **A live GLFW viewer cannot be hosted from the automated Bash tool** (uvicorn+GLFW → the tool call dies with exit 144). The user must run `./start_sim.sh` in their own terminal for the live window; automated runs use headless + `record_demo.py` for video.
- `sudo` needs the user (no passwordless). `docker` works via `sg docker -c '...'` in this session's shell.
- Local `qwen2.5:7b` is weak at multi-step tool selection — give it high-level one-shot tools (e.g. `pick_cube`) and steer the system prompt to them. [[feedback_no_gmail_draft]] unrelated.
