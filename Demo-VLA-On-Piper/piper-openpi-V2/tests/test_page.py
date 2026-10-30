"""The WebXR page without a headset: extract its <script>, run it under gjs (GNOME's SpiderMonkey) with minimal
DOM stubs, feed every ack shape through onAck/updatePanel and check the rows and sound cues the operator gets.

    python -m pytest tests/test_page.py -q
    python tests/test_page.py            # print the full harness output
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run as a script, too
from piper import config as C  # noqa: E402

GJS = shutil.which("gjs")

PRELUDE = r"""
// gjs prelude: minimal browser stubs so the page script runs headless
globalThis.window = globalThis;
const _els = {};
globalThis.document = {
  getElementById: (id) => _els[id] || (_els[id] = { id: id, textContent: '', className: '', style: {}, disabled: false }),
  body: { classList: { add() {}, remove() {} } },
  createElement: () => ({}),
};
globalThis.performance = { now: () => Date.now() };
globalThis.location = { host: 'localhost:8443' };
globalThis.setInterval = () => 0;
globalThis.setTimeout = () => 0;
globalThis.fetch = () => Promise.resolve({ json: () => Promise.resolve({ armed: false, status: 'stub' }) });
globalThis.WebSocket = class { constructor() {} send() {} close() {} };
globalThis.navigator = {};   // no .xr: the page's init takes the "unsupported" early-return path
"""

DRIVER = r"""
// driver: feed every ack shape through onAck + updatePanel, dump visible rows and sound cues
let cues = [], curLabel = '';
beeps = (seq) => cues.push(curLabel + ' -> ' + seq.map((s) => s.join('/')).join(' '));
function dump(label) {
  updatePanel();
  let out = label + '\n  banner "' + bannerEl.textContent + '" [' + bannerEl.className + ']';
  const child = { grip: () => G.griptxt.textContent + ' fill=' + G.gripfill.style.width,
                  rec: () => G.recdot.textContent + G.rectxt.textContent + ' dot=[' + G.recdot.className + ']',
                  arm: () => G.armtxt.textContent + ' fill=' + G.armfill.style.width };
  for (const k of ['grip', 'lag', 'wrist', 'rec', 'net', 'arm', 'err']) {
    const el = R[k];
    if (el.style.display === 'none') continue;
    out += '\n  ' + k + ': "' + (child[k] ? child[k]() : el.textContent) + '" [' + el.className + ']';
  }
  print(out);
}
const base = { armed: false, engaged: false, arm_progress: 0.0, status: 'DISARMED: hold A 3 s', pkts: 5, transport: 'ws',
               rate_hz: 29, pkt_age_s: 0.03, tracked: true, squeeze: 0.0, trigger: 0.0, a: false, grip_cmd_mm: 70.0,
               scale: 1.0, target_pose: [84, 0, 206, -143.7, -18.2, -67] };
const t = (label, over) => { curLabel = label; onAck(Object.assign({}, base, over)); dump(label); };
t('1 old-server DISARMED (no new fields)', {});
t('2 old-server arming 60%', { arm_progress: 0.6 });
t('3 old-server ARMED', { armed: true, status: 'ARMED: squeeze to move' });
t('4 ENGAGED, all new fields benign', { armed: true, engaged: true, status: 'ENGAGED', err_mm: 12.3, err_deg: 4.2,
   lead_limited: false, stuck: false, no_feedback: false, j6_margin_deg: 45.0, arm_status: 'NORMAL', recording: null,
   episode: null, frames: null, grip_cmd_mm: 31.5 });
t('5 lead_limited (amber lag)', { armed: true, engaged: true, status: 'ENGAGED', err_mm: 34.0, err_deg: 11.0, lead_limited: true, j6_margin_deg: 22.0 });
t('6 wrist margin 12 (amber) + rec on', { armed: true, engaged: true, status: 'ENGAGED', err_mm: 5, err_deg: 2, j6_margin_deg: 12.0, recording: true, episode: 7, frames: 412 });
t('7 wrist margin 4 (red)', { armed: true, engaged: true, status: 'ENGAGED', err_mm: 5, err_deg: 2, j6_margin_deg: 4.0, recording: true, episode: 7, frames: 500 });
t('8 STUCK flips true (expect double-click cue)', { armed: true, engaged: false, status: 'IK REFUSED - release and re-squeeze', stuck: true, err_mm: 0, err_deg: 0, j6_margin_deg: 2.5, recording: false, episode: 7 });
t('9 NO ARM FEEDBACK', { armed: true, no_feedback: true, stuck: false });
t('10 recording null again (row hides)', { recording: null });
curLabel = 'garbage';
onAck(null); onAck('x'); onAck(42); onAck([1, 2]);
dump('11 after garbage acks (state unchanged)');
ackAt = Date.now() - 5000;
dump('12 stale ack (1 s timeout) -> NO SERVER');
t('13 fw NO_SOLUTION while engaged (informational)', { armed: true, engaged: true, status: 'ENGAGED', err_mm: 28, err_deg: 3, lead_limited: true, arm_status: 'NO_SOLUTION' });
t('14 COLLISION declutch (stuck + fw)', { armed: true, engaged: false, status: 'COLLISION_OCCURRED - release and re-squeeze', stuck: true, err_mm: 0, err_deg: 0, arm_status: 'COLLISION_OCCURRED' });
curLabel = '15 error ack';
onAck({ error: 'nope' });
dump('15 error ack (400 reply) ignored, state unchanged');
print('\ncues fired:\n  ' + (cues.length ? cues.join('\n  ') : 'none'));
print('\nlog:\n' + logEl.textContent);
print('HARNESS DONE');
"""

EXPECTED_ROWS = [
    '1 old-server DISARMED (no new fields)\n  banner "DISARMED" [s-grey]',
    '2 old-server arming 60%\n  banner "DISARMED" [s-grey]',
    '  arm: "arming 60% fill=60%" []',
    '3 old-server ARMED\n  banner "ARMED" [s-amber]',
    '4 ENGAGED, all new fields benign\n  banner "ENGAGED" [s-green]\n  grip: "grip 32 mm fill=45.0%" []\n  lag: "lag 12 mm 4°" [row dim]',
    '  lag: "lag 34 mm 11° — lead clamp" [row warn]',
    '  wrist: "WRIST ROLL LIMIT 12° — rotate gripper the other way ↺" [row warn]\n  rec: "● REC ep 7 · 412 frames dot=[on]" []',
    '  wrist: "WRIST ROLL LIMIT 4° — rotate gripper the other way ↺" [row bad]',
    '8 STUCK flips true (expect double-click cue)\n  banner "STUCK — RE-SQUEEZE" [s-red flash]',
    '9 NO ARM FEEDBACK\n  banner "NO ARM FEEDBACK" [s-red]',
    '11 after garbage acks (state unchanged)\n  banner "DISARMED" [s-grey]',
    '12 stale ack (1 s timeout) -> NO SERVER\n  banner "NO SERVER" [s-red]',
    '13 fw NO_SOLUTION while engaged (informational)\n  banner "ENGAGED" [s-green]',
    '  lag: "lag 28 mm 3° — lead clamp · fw NO_SOLUTION" [row warn]',
    '14 COLLISION declutch (stuck + fw)\n  banner "COLLISION_OCCURRED — RE-SQUEEZE" [s-red flash]',
    '  lag: "lag 0 mm 0° · fw COLLISION_OCCURRED" [row bad]',
    '15 error ack (400 reply) ignored, state unchanged\n  banner "COLLISION_OCCURRED — RE-SQUEEZE" [s-red flash]',
    '  err: "error: server: nope" []',
]
EXPECTED_CUES = [
    "3 old-server ARMED -> 660/60/0.08 990/90/0.08",
    "4 ENGAGED, all new fields benign -> 1500/10/0.03",
    "8 STUCK flips true (expect double-click cue) -> 1200/12/0.04 1200/12/0.04",
    "10 recording null again (row hides) -> 330/120/0.08",
    "13 fw NO_SOLUTION while engaged (informational) -> 660/60/0.08 990/90/0.08",
    "13 fw NO_SOLUTION while engaged (informational) -> 1500/10/0.03",
    "14 COLLISION declutch (stuck + fw) -> 1200/12/0.04 1200/12/0.04",
]


def page_script() -> str:
    html = open(os.path.join(C.STATIC_DIR, "index.html"), encoding="utf-8").read()
    return re.search(r"<script>(.*?)</script>", html, re.S).group(1)


def run_harness(out_dir) -> tuple[int, str]:
    path = os.path.join(out_dir, "page_harness.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PRELUDE + page_script() + DRIVER)
    r = subprocess.run([GJS, path], capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.skipif(GJS is None, reason="gjs not installed")
def test_page_harness(tmp_path):
    code, out = run_harness(str(tmp_path))
    print(out)
    assert code == 0 and "HARNESS DONE" in out, out
    assert [l for l in out.splitlines() if "ERROR" in l] == ["ERROR server: nope"]   # only the deliberate one
    for row in EXPECTED_ROWS:
        assert row in out, row
    cues = out.split("cues fired:\n")[1].split("\n\nlog:")[0]
    assert [c.strip() for c in cues.strip().splitlines()] == EXPECTED_CUES
    assert "wrist:" not in out.split("4 ENGAGED")[1].split("5 lead_limited")[0]      # 45 deg margin: no warning
    assert "rec:" not in out.split("10 recording null")[1].split("11 after")[0]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        rc, text = run_harness(d)
        print(text)
        raise SystemExit(rc)
