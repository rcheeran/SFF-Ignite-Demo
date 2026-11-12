#!/usr/bin/env bash
# Keep k3s on the Jetson's current network address.
#
# k3s records the node address when it starts (node INTERNAL-IP, the API
# server's advertised address, and the flannel VXLAN endpoint) and never
# updates it. If the Jetson gets a new address while k3s runs (new network or
# DHCP lease), the cluster keeps using an address the host no longer has.
# Restarting k3s picks up the new address. k3s.service uses KillMode=process,
# so the running pods (camera, hand, detection) keep running through it.
#
#   k3s-node-ip-sync.sh --check   print the status; exit 0 in sync, 1 out of
#                                 sync, 2 unknown (k3s API not reachable)
#   k3s-node-ip-sync.sh           as root: restart k3s if it is out of sync
#
# Installed as /usr/local/sbin/k3s-node-ip-sync.sh and run by
# k3s-node-ip-sync.service, which the networkd-dispatcher hook
# k3s-node-ip-sync.hook starts whenever a network interface becomes routable.
# See Demo-Ignite/EVENT_DAY.md.
set -uo pipefail

MIN_K3S_UPTIME_SECONDS=60   # a freshly started k3s may still report its previous address
SETTLE_SECONDS=5            # let DHCP finish configuring the interface
RESTART_TIMEOUT_SECONDS=180

if [ "$(id -u)" -eq 0 ]; then KUBECTL=(kubectl); else KUBECTL=(sudo -n kubectl); fi

node_ip() {
    "${KUBECTL[@]}" get node "$(hostname)" \
        -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null
}

# The address k3s picks when it starts: the first IPv4 address of the
# default-route interface.
default_route_ip() {
    local dev
    dev=$(ip -4 route show default | awk '{for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit }}')
    [ -n "$dev" ] || return 0
    ip -4 -o addr show dev "$dev" scope global | awk '{split($4, a, "/"); print a[1]; exit}'
}

host_has_ip() { ip -4 -o addr show | awk '{split($4, a, "/"); print a[1]}' | grep -qxF "$1"; }

# Set NODE_IP and HOST_IP; return 0 in sync, 1 out of sync, 2 unknown.
compare() {
    NODE_IP=$(node_ip)
    HOST_IP=$(default_route_ip)
    [ -n "$NODE_IP" ] || return 2
    host_has_ip "$NODE_IP" && return 0
    return 1
}

report() {
    case $1 in
        0) echo "in sync: k3s uses $NODE_IP, which this host has" ;;
        1) echo "OUT OF SYNC: k3s uses $NODE_IP, but this host has ${HOST_IP:-no network address}" ;;
        *) echo "unknown: cannot read the k3s node address (k3s not running, or sudo needs a password)" ;;
    esac
}

if [ "${1:-}" = "--check" ]; then
    compare; rc=$?
    report $rc
    exit $rc
fi
if [ $# -gt 0 ]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "run as root to fix, or use --check" >&2
    exit 2
fi

sleep "$SETTLE_SECONDS"
while :; do
    compare; rc=$?
    report $rc
    [ $rc -eq 1 ] || exit 0
    if [ -z "$HOST_IP" ]; then
        echo "no default-route address yet; leaving k3s alone until the network is up"
        exit 0
    fi
    if ! systemctl is-active --quiet k3s; then
        echo "k3s is not running; it picks up $HOST_IP when it starts"
        exit 0
    fi
    started=$(date -d "$(systemctl show k3s -p ActiveEnterTimestamp --value)" +%s) || started=0
    wait=$(( started + MIN_K3S_UPTIME_SECONDS - $(date +%s) ))
    [ $wait -gt 0 ] || break
    echo "k3s started less than ${MIN_K3S_UPTIME_SECONDS}s ago; checking again in ${wait}s"
    sleep "$wait"
done

echo "restarting k3s to move it to $HOST_IP (running pods keep running)"
if ! systemctl restart k3s; then
    echo "k3s restart failed; see: journalctl -u k3s -b" >&2
    exit 1
fi
deadline=$(( $(date +%s) + RESTART_TIMEOUT_SECONDS ))
while :; do
    compare; rc=$?
    if [ $rc -eq 0 ]; then
        report 0
        exit 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        report $rc
        echo "k3s did not move to a host address within ${RESTART_TIMEOUT_SECONDS}s" >&2
        exit 1
    fi
    sleep 5
done
