"""AgileX Piper REAL-hardware Agent — Asimov RobotAgent for the physical Piper.

Same toolset-agnostic RobotAgent pattern as the sim agent, but connects to the
REAL Piper toolset (piper_real__ tools over MCP) which drives the physical arm
over CAN. The system prompt is safety-first: this commands real hardware.

Run:  python agent.py   (needs the piper-real toolset running; see .config/.env)
"""
import logging
import os

from asimov.robot_agent.agents.orchestra import AgentOrchestra
from asimov.robot_agent.agents.telemetry import TelemetryProvider
from asimov.robot_agent.robot_agent import RobotAgent

logger = logging.getLogger(__name__)

TOOL_PREFIX = "piper_real"

PIPER_REAL_SYSTEM_PROMPT = """You are the AgileX Piper Real-Arm Agent, operating a PHYSICAL AgileX Piper 6-DOF robotic arm over a CAN bus. This is REAL hardware — your commands move a real robot that can hit people or objects.

Your tools are prefixed `piper_real__` (e.g. piper_real__get_status, piper_real__enable_arm, piper_real__move_to_pose, piper_real__move_to_joint_positions, piper_real__open_gripper, piper_real__close_gripper, piper_real__home, piper_real__emergency_stop).

Coordinate frame: x forward, y left, z up, in METERS from the arm base. Joint angles in DEGREES (joint1 base .. joint6 wrist). Gripper opening 0..1.

## Safety — read carefully
1. ALWAYS call piper_real__get_status() and piper_real__get_connection() first. If telemetry Hz is 0 or a warning is present, STOP and report it — the arm is probably in Master/teach mode and not ready (do not send motion).
2. Before any motion in real mode you must call piper_real__enable_arm() once. If motion tools return "OBSERVE-ONLY", tell the user motion is disabled (PIPER_ALLOW_MOTION=1 is required) and do NOT keep retrying.
3. Move deliberately and ONE motion at a time; wait for each to finish. Prefer small, verified moves. Never chain many large moves blindly.
4. If the user says "stop", "halt", or anything urgent, call piper_real__emergency_stop() immediately.
5. Report tool results verbatim, including failures (e.g. "target rejected/unreachable"). Never claim the arm moved if a tool reported an error.
6. Do not invent coordinates. If you don't know a reachable target, ask the user or read the current pose first.

Execute requested actions directly, but always within these safety rules."""


class PiperRealTelemetryProvider(TelemetryProvider):
    def __init__(self, url=None):
        self._url = url

    async def get_state(self) -> dict:
        return {"robot": "AgileX Piper (real)", "dof": 6, "embodiment": "piper-6dof-real", "status": "idle"}


class PiperRealAgent(RobotAgent):
    def __init__(self, toolset_url: str | None = None):
        self._toolset_url = toolset_url or os.getenv("PIPER_TOOLSET_URL")
        self._telemetry = PiperRealTelemetryProvider(self._toolset_url)
        super().__init__(name="PiperRealArm")
        logger.info("Piper Real Arm Agent initialized (toolset_url=%s)", self._toolset_url)

    def configure_telemetry_provider(self) -> TelemetryProvider:
        return self._telemetry

    def configure_system_prompt(self) -> str:
        return PIPER_REAL_SYSTEM_PROMPT

    def configure_function_tools(self) -> list:
        return []

    def configure_openapi_tools(self) -> list:
        return []

    def configure_mcp_tools(self) -> list:
        tools = []
        if self._toolset_url:
            from asimov.robot_agent.mcp.mcp_http_tool import PrefixedMCPStreamableHTTPTool

            url = self._toolset_url.rstrip("/")
            if not url.endswith("/mcp"):
                url = f"{url}/mcp"
            logger.info("Using Piper REAL toolset at %s", url)
            tools.append(PrefixedMCPStreamableHTTPTool(name=TOOL_PREFIX, url=url, eager_connect=True))
        else:
            logger.warning("PIPER_TOOLSET_URL not set — no toolset tools registered")
        return tools

    async def emergency_stop(self) -> None:
        """Route the agent-level e-stop to the toolset's REST/MCP stop path."""
        import httpx
        if self._toolset_url:
            base = self._toolset_url.rstrip("/").removesuffix("/mcp")
            try:
                async with httpx.AsyncClient(timeout=3.0) as c:
                    await c.post(f"{base}/stop")
            except Exception as e:  # noqa: BLE001
                logger.warning("emergency_stop passthrough failed: %s", e)
        await super().emergency_stop()

    def configure_agents(self, orchestra: AgentOrchestra) -> None:
        pass


def main():
    from dotenv import load_dotenv

    here = os.path.dirname(__file__)
    env_file, config_file = os.path.join(here, ".env"), os.path.join(here, ".config")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()
    if os.path.exists(config_file):
        load_dotenv(config_file, override=False)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for noisy in ("httpx", "httpcore", "uvicorn.access", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8213"))

    agent = PiperRealAgent()
    server = agent.create_server(
        host=host, port=port,
        title="AgileX Piper Real-Arm Agent API",
        description="Natural-language control of the physical AgileX Piper 6-DOF arm",
    )
    logger.info("Starting Piper Real Arm agent on %s:%s", host, port)
    server.run()


if __name__ == "__main__":
    main()
