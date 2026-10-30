"""Cloud-training tooling without network, GPU or robot: the dataset contract check, the plan maths, local-file
verification, the Lambda API flow against the documented shapes, every script parses, status.sh parses a log."""
import hashlib
import json
import os
import subprocess

import pytest

from piper import config as C
from training.cloud import hub_check, lambda_api

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPE = (C.IMG_H, C.IMG_W, 3)
SCRIPTS = ["training/train.sh", "training/pull_checkpoint.sh", "training/cloud/launch.sh", "training/cloud/train.sh",
           "training/cloud/setup.sh", "training/cloud/status.sh", "training/cloud/terminate.sh"]


def contract_info(**over):
    img = {"dtype": "video", "shape": list(SHAPE), "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": [7], "names": list(C.STATE_NAMES)}
    info = {"fps": C.FPS, "total_episodes": 3, "total_frames": 180,
            "features": {**{f"observation.images.{c}": dict(img) for c in C.CAMERA_NAMES},
                         "observation.state": dict(vec), "action": dict(vec)}}
    info.update(over)
    return info


def problems(info):
    return hub_check.contract_problems(info, C.FPS, C.STATE_NAMES, C.CAMERA_NAMES, SHAPE)


def test_contract_accepts_the_contract():
    assert problems(contract_info()) == []


def test_contract_flags_every_deviation():
    assert problems(contract_info(fps=30)) == [f"fps 30 != {C.FPS}"]
    info = contract_info()
    del info["features"]["observation.images.wrist"]
    assert problems(info) == ["missing feature observation.images.wrist"]
    info = contract_info()
    info["features"]["action"] = {"dtype": "float32", "shape": [6], "names": C.STATE_NAMES[:6]}
    assert problems(info) == ["action: float32 [6] != float32 [7]"]
    info = contract_info()
    info["features"]["observation.state"]["names"] = ["a"] * 7
    assert problems(info) == [f"observation.state names {['a'] * 7} != {C.STATE_NAMES}"]
    assert problems(contract_info(total_frames=0)) == ["no frames"]


def test_plan_maths():
    epochs, hours, usd = hub_check.plan(3000, 16, 4500, step_s=1.0, overhead_min=15, usd_per_h=3.29)
    assert epochs == pytest.approx(3000 * 16 / 4500)
    assert hours == pytest.approx((3000 + 15 * 60) / 3600)
    assert usd == pytest.approx(hours * 3.29)


class Sibling:
    def __init__(self, name, size, sha=None):
        self.rfilename, self.size = name, size
        self.lfs = type("Lfs", (), {"sha256": sha})() if sha else None


def fake_hub(monkeypatch, siblings=(), user="royc1981", info_path=None):
    class Api:
        def whoami(self):
            return {"name": user}

        def model_info(self, repo, files_metadata):
            return type("Info", (), {"siblings": list(siblings)})()

    monkeypatch.setattr(hub_check, "HfApi", Api)
    monkeypatch.setattr(hub_check, "hf_hub_download", lambda repo, fn, repo_type: str(info_path))


def test_model_check_size_and_sha(tmp_path, monkeypatch, capsys):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"weights" * 1000)
    size, sha = f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest()
    fake_hub(monkeypatch, [Sibling("model.safetensors", size, sha)])
    hub_check.check_model("u/r", str(f))
    assert "size and sha256 match" in capsys.readouterr().out
    fake_hub(monkeypatch, [Sibling("model.safetensors", size + 1, sha)])
    with pytest.raises(SystemExit, match="incomplete"):
        hub_check.check_model("u/r", str(f))
    fake_hub(monkeypatch, [Sibling("model.safetensors", size, "0" * 64)])
    with pytest.raises(SystemExit, match="corrupt"):
        hub_check.check_model("u/r", str(f))
    fake_hub(monkeypatch, [Sibling("config.json", 10)])
    with pytest.raises(SystemExit, match="no model.safetensors"):
        hub_check.check_model("u/r")


def test_dataset_check_messages(tmp_path, monkeypatch, capsys):
    info = tmp_path / "info.json"
    info.write_text(json.dumps(contract_info()))
    fake_hub(monkeypatch, info_path=info)
    hub_check.check_dataset("u/d", 3000, 16)
    out = capsys.readouterr().out
    assert "token OK (royc1981)" in out and "3 episodes, 180 frames" in out
    assert "plan: 3000 steps x batch 16" in out and f"${C.H100_USD_PER_H}/h" in out
    info.write_text(json.dumps(contract_info(fps=30)))
    with pytest.raises(SystemExit, match="violates the contract"):
        hub_check.check_dataset("u/d")

    def missing(*a, **k):
        raise FileNotFoundError("404")

    monkeypatch.setattr(hub_check, "hf_hub_download", missing)
    with pytest.raises(SystemExit, match="not on the Hub"):
        hub_check.check_dataset("u/d")


def lambda_stub(monkeypatch, statuses, h100_regions):
    """Responses in the shapes of https://cloud.lambda.ai/api/v1/openapi.json; records every call."""
    calls, polls = [], iter(statuses)
    h100 = {"instance_type": {"name": "gpu_1x_h100_pcie", "price_cents_per_hour": 329},
            "regions_with_capacity_available": [{"name": r, "description": r} for r in h100_regions]}
    inst = {"id": "0920582c", "ip": "10.0.0.7", "region": {"name": "us-east-1"}, "instance_type": h100["instance_type"]}

    def call(path, body=None):
        calls.append((path, body))
        if path == "/ssh-keys":
            return [{"id": "k1", "name": "jetson", "public_key": "ssh-ed25519 AAAA"}]
        if path == "/instance-types":
            return {"gpu_1x_h100_pcie": h100, "gpu_1x_a100": {"instance_type": {}, "regions_with_capacity_available": []}}
        if path == "/instance-operations/launch":
            return {"instance_ids": ["0920582c"]}
        if path == "/instances/0920582c":
            return {**inst, "status": next(polls)}
        if path == "/instances":
            return [{**inst, "status": "active"}]
        if path == "/instance-operations/terminate":
            return {"terminated_instances": [{**inst, "status": "terminating"} for _ in body["instance_ids"]]}
        raise AssertionError(path)

    monkeypatch.setattr(lambda_api, "call", call)
    monkeypatch.setattr(lambda_api.time, "sleep", lambda s: None)
    return calls


def test_lambda_launch_waits_for_active(monkeypatch, capsys):
    calls = lambda_stub(monkeypatch, ["booting", "booting", "active"], ["us-east-1", "us-west-1"])
    lambda_api.launch()
    assert capsys.readouterr().out.split() == ["0920582c", "10.0.0.7"]   # id first (persisted before the boot poll), ip when active
    assert ("/instance-operations/launch", {"region_name": "us-east-1", "instance_type_name": "gpu_1x_h100_pcie",
                                            "ssh_key_names": ["jetson"], "name": "pi05-train"}) in calls
    assert sum(p == "/instances/0920582c" for p, _ in calls) == 3


def test_lambda_launch_fails_fast(monkeypatch):
    lambda_stub(monkeypatch, [], [])
    with pytest.raises(SystemExit, match="no gpu_1x_h100_pcie capacity"):
        lambda_api.launch()
    lambda_stub(monkeypatch, ["preempted"], ["us-east-1"])
    with pytest.raises(SystemExit, match="is preempted"):
        lambda_api.launch()
    lambda_stub(monkeypatch, [], ["us-east-1"])
    with pytest.raises(SystemExit, match="LAMBDA_SSH_KEY_NAME"):
        lambda_api.launch(ssh_key="not-a-key")
    with pytest.raises(SystemExit, match="unknown instance type"):
        lambda_api.launch("gpu_9x_nope")


def test_lambda_terminate_all(monkeypatch, capsys):
    calls = lambda_stub(monkeypatch, [], [])
    lambda_api.terminate("all")
    assert ("/instance-operations/terminate", {"instance_ids": ["0920582c"]}) in calls
    assert "terminating 0920582c" in capsys.readouterr().out


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse(script):
    subprocess.run(["bash", "-n", os.path.join(ROOT, script)], check=True)


SAMPLE_LOG = ("== 2026-08-22T02:00:00+00:00 dataset=u/d model=u/m steps=3000 batch=16 out=/home/ubuntu/outputs/m\n"
              "INFO 2026-08-22 02:20:00 ts/train.py:641 step:1K smpl:20K ep:30 epch:4.44 loss:0.123 grdn:1.2 "
              "lr:2.5e-05 updt_s:1.02 data_s:0.01\n"
              "Training:  42%|####      | 1250/3000 [21:10<29:40,  1.02s/step]\r"
              "Training:  42%|####      | 1251/3000 [21:11<29:39,  1.02s/step]")


def run_status(tmp_path, log, lines="2"):
    """status.sh through a fake `ssh` on PATH that prints the sample log."""
    (tmp_path / "train.log").write_text(log)
    ssh = tmp_path / "ssh"
    ssh.write_text(f"#!/bin/bash\ncat {tmp_path / 'train.log'}\n")
    ssh.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "LAMBDA_HOST": "ubuntu@fake"}
    return subprocess.run(["bash", "training/cloud/status.sh", lines], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout


def test_status_parses_progress(tmp_path):
    out = run_status(tmp_path, SAMPLE_LOG)
    assert out.endswith("-- step 1251/3000  elapsed 21:11  ETA 29:39  loss 0.123\n")
    assert out.count("\n") == 3


def test_status_reports_done(tmp_path):
    out = run_status(tmp_path, SAMPLE_LOG + "\nDONE (lerobot-train exit 0, upload verified): https://huggingface.co/u/m\n")
    assert out.endswith("-- DONE (lerobot-train exit 0, upload verified): https://huggingface.co/u/m\n")
