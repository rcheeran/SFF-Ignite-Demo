"""The teleop HTTPS server as the headset sees it: the real serve() on a free port around a dry-run teleop
(no robot, real control thread), the POST /pose arming sequence, malformed packets, /status under load, WS.

    python -m pytest tests/test_teleop_server.py -q
"""
import asyncio
import json
import socket
import ssl
import time
import urllib.error
import urllib.request

import aiohttp
import pytest

from teleop.quest import QuestTeleop, serve

CTX = ssl.create_default_context()
CTX.check_hostname, CTX.verify_mode = False, ssl.CERT_NONE
HEAD = {"p": [0.0, 1.6, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}
BAD = {"error": "packet must be a JSON object"}


def http(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=3) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get(url):
    code, body = http(url)
    return code, json.loads(body)


def post(base, obj=None, raw=None):
    code, body = http(base + "/pose", json.dumps(obj).encode() if raw is None else raw,
                      {"Content-Type": "application/json"})
    return code, json.loads(body)


def drive(base, secs, **right):
    """What the page does over http: ~30 packets/s, one in flight."""
    t0, ack = time.monotonic(), None
    while (el := time.monotonic() - t0) < secs:
        pkt = {"p": [0.2, 1.2, -0.4], "q": [0, 0, 0, 1], "trigger": 0.0, "squeeze": 0.0, "a": False, "b": False,
               "thumb": False, "axes": [0, 0], **right}
        _, ack = post(base, {"t": el, "head": HEAD, "right": pkt, "left": None})
        time.sleep(1 / 30)
    return ack


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    teleop = QuestTeleop(None, scale=0.5).start()
    serve(teleop, port)
    base = f"https://localhost:{port}"
    for _ in range(50):
        try:
            get(base + "/status")
            break
        except OSError:
            time.sleep(0.1)
    yield teleop, base
    teleop.stop()


def test_status_and_page(server):
    teleop, base = server
    code, s = get(base + "/status")
    assert code == 200 and s["armed"] is False and s["status"] == "hold: no controller packets" and s["scale"] == 0.5
    assert s["arm_status"] is None and s["target_pose"] is not None and s["pkts"] == 0
    code, page = http(base + "/")
    assert code == 200 and b"<script>" in page and b"requestSession" in page


def test_malformed_packets_get_400_and_the_server_survives(server):
    teleop, base = server
    for raw in (b"not json", b"[1, 2]", b'"x"', b"42", b"null", b"\xff\xfe", b""):
        code, a = post(base, raw=raw)
        assert code == 400 and a == BAD, raw
    code, a = post(base, {"event": "hello"})
    assert code == 200 and a["transport"] == "http" and teleop.fault is None


def test_arming_sequence_over_post(server):
    teleop, base = server
    post(base, {"event": "session_start", "mode": "immersive-ar"})
    ack = drive(base, 1.0, a=True)
    assert not ack["armed"] and 0.2 < ack["arm_progress"] < 0.5 and ack["tracked"]
    ack = drive(base, 2.5, a=True)
    assert ack["armed"] and ack["status"] == "ARMED: squeeze to move"
    ack = drive(base, 0.3)
    assert ack["armed"] and not ack["engaged"]
    ack = drive(base, 0.5, squeeze=1.0)
    assert ack["engaged"] and ack["status"] == "ENGAGED"
    ack = drive(base, 0.5, squeeze=1.0, trigger=0.8)
    assert abs(ack["grip_cmd_mm"] - 14.0) < 0.5 and ack["engaged"]
    ack = drive(base, 0.3, trigger=0.8)
    assert ack["armed"] and not ack["engaged"] and ack["status"] == "ARMED: squeeze to move"
    code, s = get(base + "/status")                                   # /status while the control thread runs
    assert code == 200 and s["armed"] and s["rate_hz"] >= 10 and s["pkt_age_s"] < 0.5
    code, ack = post(base, {"event": "session_end"})                  # leaving XR disarms
    time.sleep(0.2)
    code, s = get(base + "/status")
    assert not s["armed"] and s["status"] == "DISARMED: hold A 3 s"
    time.sleep(0.5)                                                   # last frame older than the watchdog
    code, s = get(base + "/status")
    assert not s["armed"] and s["status"] == "hold: no controller packets" and teleop.fault is None


def test_ws_path_and_bad_ws_packet(server):
    teleop, base = server

    async def go():
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(base.replace("https", "wss") + "/ws", ssl=False) as ws:
                await ws.send_str("garbage")
                assert await ws.receive_json() == BAD
                await ws.send_str("[1, 2]")
                assert await ws.receive_json() == BAD
                await ws.send_json({"event": "hello"})
                ack = await ws.receive_json()
                assert ack["transport"] == "ws" and "status" in ack

    asyncio.run(go())
    assert teleop.fault is None
