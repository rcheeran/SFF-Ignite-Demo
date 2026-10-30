"""Lambda Cloud API - only what launch.sh / terminate.sh need. Needs LAMBDA_API_KEY (https://cloud.lambda.ai/api-keys).
Request/response shapes follow https://cloud.lambda.ai/api/v1/openapi.json (read 2026-08-22; not exercised - no key).

python -m training.cloud.lambda_api launch [INSTANCE_TYPE=gpu_1x_h100_pcie]   one instance in the first region with
                                                 capacity, wait until active, print "<instance-id> <ip>" on stdout
python -m training.cloud.lambda_api wait [INSTANCE_TYPE] [TIMEOUT_S=14400]   poll until that type has capacity
                                                 somewhere (exit 0) or the timeout passes (exit 1). Nothing billed.
python -m training.cloud.lambda_api list
python -m training.cloud.lambda_api terminate ID|all
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://cloud.lambda.ai/api/v1"
INSTANCE_TYPE = "gpu_1x_h100_pcie"
BOOT_TIMEOUT_S = 600
USER_AGENT = "curl/8.5.0"


def call(path, body=None):
    """GET (body None) or POST JSON; returns the response's `data`."""
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['LAMBDA_API_KEY']}",
                                          "Content-Type": "application/json",
                                          # Cloudflare in front of the API blocks urllib's default agent
                                          # ("Python-urllib/3.x") with HTTP 403 error code 1010, while the
                                          # same key over curl works. Identify as curl.
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["data"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Lambda API {path}: HTTP {e.code} {e.read().decode(errors='replace')[:300]}")


def capacity(instance_type=INSTANCE_TYPE) -> list:
    """Regions with capacity for this type right now (empty = none). Free, read-only."""
    t = call("/instance-types").get(instance_type)
    return [r["name"] for r in t["regions_with_capacity_available"]] if t else []


def wait_capacity(instance_type=INSTANCE_TYPE, timeout_s=14400, poll_s=180) -> bool:
    """Block until the type is available. H100 capacity comes and goes in minutes, and launch() just
    aborts when there is none, so a run started at a bad moment would otherwise need babysitting."""
    t0 = time.time()
    while True:
        regs = capacity(instance_type)
        if regs:
            print(f"{instance_type} available in {regs}", file=sys.stderr)
            return True
        waited = time.time() - t0
        if waited + poll_s > timeout_s:
            print(f"no {instance_type} capacity after {waited / 60:.0f} min - giving up", file=sys.stderr)
            return False
        print(f"[{time.strftime('%H:%M:%S')}] no {instance_type} capacity, retrying in {poll_s}s "
              f"({waited / 60:.0f}/{timeout_s / 60:.0f} min)", file=sys.stderr, flush=True)
        time.sleep(poll_s)


def launch(instance_type=INSTANCE_TYPE, ssh_key=None):
    keys = [k["name"] for k in call("/ssh-keys")]
    key = ssh_key or (keys[0] if len(keys) == 1 else None)
    if key not in keys:
        raise SystemExit(f"LAMBDA_SSH_KEY_NAME must be one of your Lambda ssh keys {keys}")
    types = call("/instance-types")
    if instance_type not in types:
        raise SystemExit(f"unknown instance type {instance_type}; known: {sorted(types)}")
    regions = [r["name"] for r in types[instance_type]["regions_with_capacity_available"]]
    if not regions:
        raise SystemExit(f"no {instance_type} capacity in any region right now - retry later")
    iid = call("/instance-operations/launch", {"region_name": regions[0], "instance_type_name": instance_type,
                                               "ssh_key_names": [key], "name": "pi05-train"})["instance_ids"][0]
    print(f"launched {instance_type} in {regions[0]}: {iid} (billing started; terminate.sh {iid})", file=sys.stderr)
    print(iid, flush=True)                 # line 1 right away, so the caller can persist the id before the boot poll
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT_S:
        time.sleep(10)
        inst = call(f"/instances/{iid}")
        if inst["status"] == "active" and inst.get("ip"):
            print(inst["ip"], flush=True)  # line 2
            return
        if inst["status"] != "booting":
            raise SystemExit(f"instance {iid} is {inst['status']} - bash training/cloud/terminate.sh {iid}")
        print(f"  {inst['status']} ({time.time() - t0:.0f} s)", file=sys.stderr)
    raise SystemExit(f"instance {iid} not active after {BOOT_TIMEOUT_S} s - https://cloud.lambda.ai/instances (billing runs)")


def list_instances():
    rows = call("/instances")
    for i in rows:
        print(f"{i['id']}  {i['status']:11s} {i.get('ip') or '-':16s} {i['instance_type']['name']}  {i['region']['name']}  "
              f"${i['instance_type']['price_cents_per_hour'] / 100:.2f}/h")
    return rows


def terminate(which):
    ids = [i["id"] for i in call("/instances")] if which == "all" else [which]
    if not ids:
        print("no instances running")
        return
    for i in call("/instance-operations/terminate", {"instance_ids": ids})["terminated_instances"]:
        print(f"terminating {i['id']} ({i['status']})")


def main(argv):
    if not os.environ.get("LAMBDA_API_KEY"):
        raise SystemExit("LAMBDA_API_KEY missing (training/cloud/lambda.env; keys: https://cloud.lambda.ai/api-keys)")
    cmd = argv[0] if argv else ""
    if cmd == "launch" and len(argv) <= 2:
        launch(argv[1] if len(argv) == 2 else INSTANCE_TYPE, os.environ.get("LAMBDA_SSH_KEY_NAME"))
    elif cmd == "wait":
        sys.exit(0 if wait_capacity(argv[1] if len(argv) > 1 else INSTANCE_TYPE,
                                    float(argv[2]) if len(argv) > 2 else 14400) else 1)
    elif cmd == "list" and len(argv) == 1:
        list_instances()
    elif cmd == "terminate" and len(argv) == 2:
        terminate(argv[1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
