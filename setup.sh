#!/bin/bash
# Sets up an Ubuntu machine to control the AgileX Piper arm: system deps,
# a Python venv with pyAgxArm (+ piper_sdk for cross-checking), the
# gs_usb USB-CAN driver (only needed on Jetson Thor - see step 5), a stable
# name + auto-bring-up for the CAN adapter (step 6), and Intel RealSense
# camera dependencies.
# Tested on Ubuntu 24.04.4 / Python 3.12 (x86) and on the NVIDIA Jetson
# AGX Thor Developer Kit (aarch64, L4T kernel 6.8.x-tegra).
#
# Usage: bash setup.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venv}"
PIPER_IFACE="${PIPER_IFACE:-can_piper}"   # stable name pinned via udev (step 6)
CAN_BITRATE="${CAN_BITRATE:-1000000}"     # Piper runs at 1 Mbps
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# ensure_gs_usb: make sure the Geschwister Schneider / candleLight USB-CAN
# driver (gs_usb) is present, building + persistently installing it if not.
#
# WHY THIS EXISTS (Jetson Thor): the AgileX Piper ships with a gs_usb-based
# USB-CAN adapter (USB ID 1d50:606f). On a normal x86 Ubuntu box gs_usb is
# already in the kernel, so this is a no-op. But the NVIDIA Jetson AGX Thor
# Developer Kit's L4T kernel (6.8.x-tegra) ships with NO USB-CAN drivers at
# all - the whole drivers/net/can/usb/ directory is absent - so the adapter
# never enumerates as a canX interface until gs_usb is built and installed
# by hand. This function does that and wires it to auto-load on every boot.
#
# Requires kernel headers + a toolchain (installed in step 1). On Jetson the
# headers come from the nvidia-l4t-kernel-headers package.
# ---------------------------------------------------------------------------
ensure_gs_usb() {
    if modinfo gs_usb >/dev/null 2>&1; then
        echo "    gs_usb already available (in-kernel or previously installed) - skipping."
        return 0
    fi
    echo "    gs_usb NOT found - expected on Jetson Thor. Building from source."

    local krel kver build_dir work
    krel="$(uname -r)"
    kver="v$(echo "$krel" | grep -oE '^[0-9]+\.[0-9]+')"   # kernel major.minor -> git tag, e.g. v6.8
    build_dir="/lib/modules/$krel/build"

    if [ ! -d "$build_dir" ]; then
        echo "    ERROR: kernel headers not found at $build_dir" >&2
        echo "           Jetson:  sudo apt install nvidia-l4t-kernel-headers" >&2
        echo "           Generic: sudo apt install linux-headers-$krel" >&2
        return 1
    fi

    work="$(mktemp -d)"
    # Fetch the gs_usb source matching the running kernel's major.minor. Point
    # releases (6.8.x) don't touch this self-contained USB driver, so the vX.Y
    # tag builds cleanly against the -tegra headers.
    echo "    Fetching gs_usb.c for kernel $kver ..."
    curl -fsSL -o "$work/gs_usb.c" \
        "https://raw.githubusercontent.com/torvalds/linux/$kver/drivers/net/can/usb/gs_usb.c"
    printf 'obj-m += gs_usb.o\n' > "$work/Makefile"

    echo "    Building gs_usb.ko ..."
    make -C "$build_dir" M="$work" modules

    # Persistent install: drop the .ko in the kernel's updates/ dir, refresh
    # module deps, and register it for auto-load at boot via modules-load.d.
    echo "    Installing gs_usb.ko and enabling auto-load on boot ..."
    sudo install -D -m 644 "$work/gs_usb.ko" "/lib/modules/$krel/updates/gs_usb.ko"
    sudo depmod -a
    printf 'gs_usb\n' | sudo tee /etc/modules-load.d/gs_usb.conf >/dev/null
    sudo modprobe gs_usb
    rm -rf "$work"
    # NOTE: an out-of-tree module is tied to this exact kernel version. After a
    # kernel upgrade, re-run this (or manage it with DKMS) to rebuild gs_usb.
    echo "    gs_usb installed - the USB-CAN adapter should now appear as a canX interface."
}

# ---------------------------------------------------------------------------
# install_piper_can_service: pin the Piper's USB-CAN adapter to a stable name
# and bring it up automatically at 1 Mbps on every boot.
#
# WHY: the adapter's kernel interface name is a race with any other CAN
# controllers present. On the Jetson Thor there are 4 native mttcan
# controllers, so the adapter has come up as both can0 and can4 across
# reboots. A udev rule keyed on the adapter's USB ID (1d50:606f) pins it to a
# fixed name ($PIPER_IFACE, default can_piper) so SDK scripts always target
# the same interface. A systemd oneshot then brings it up, locating it by
# driver (gs_usb) so it works no matter what the kernel happened to name it.
# ---------------------------------------------------------------------------
install_piper_can_service() {
    echo "    udev rule -> stable name '$PIPER_IFACE' (USB 1d50:606f)"
    sudo tee /etc/udev/rules.d/80-can-piper.rules >/dev/null <<EOF
# AgileX Piper USB-CAN adapter (Geschwister Schneider / candleLight, gs_usb).
# Pin it to a stable interface name regardless of boot enumeration order,
# so it doesn't race with any native CAN controllers (e.g. Jetson mttcan).
SUBSYSTEM=="net", ACTION=="add", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", NAME="$PIPER_IFACE"
EOF

    echo "    bring-up script -> /usr/local/sbin/piper-can-up.sh"
    sudo tee /usr/local/sbin/piper-can-up.sh >/dev/null <<'EOF'
#!/bin/bash
# Bring up the AgileX Piper's USB-CAN adapter (gs_usb / candleLight) at 1 Mbps.
# Finds the interface by driver (not a fixed name), so it works regardless of
# what the kernel/udev named it.
set -u
BITRATE="${1:-1000000}"
found=""
for iface in $(ip -br link show type can 2>/dev/null | awk '{print $1}'); do
    drv=$(ethtool -i "$iface" 2>/dev/null | awk '/^driver:/{print $2}')
    if [ "$drv" = "gs_usb" ]; then
        found="$iface"
        ip link set "$iface" down 2>/dev/null || true
        ip link set "$iface" type can bitrate "$BITRATE"
        ip link set "$iface" up
        echo "piper-can-up: brought up $iface (gs_usb) @ ${BITRATE}bps"
    fi
done
[ -n "$found" ] || { echo "piper-can-up: no gs_usb CAN adapter found" >&2; exit 1; }
EOF
    sudo chmod +x /usr/local/sbin/piper-can-up.sh

    echo "    systemd service -> piper-can.service (auto bring-up at boot)"
    sudo tee /etc/systemd/system/piper-can.service >/dev/null <<'EOF'
[Unit]
Description=Bring up AgileX Piper USB-CAN (gs_usb) at 1 Mbps
# Run after kernel modules (gs_usb via modules-load.d) are loaded.
After=systemd-modules-load.service network-pre.target
Wants=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
# Retry: the USB adapter may enumerate a moment after boot.
ExecStart=/bin/bash -c 'for i in $(seq 1 15); do /usr/local/sbin/piper-can-up.sh 1000000 && exit 0; sleep 1; done; exit 1'

[Install]
WantedBy=multi-user.target
EOF

    sudo udevadm control --reload-rules
    sudo systemctl daemon-reload
    sudo systemctl enable piper-can.service
    # Apply the udev name now if possible, then bring the interface up for this
    # session. The stable name always takes effect on the next replug/reboot.
    sudo udevadm trigger --action=add --subsystem-match=net >/dev/null 2>&1 || true
    sudo systemctl restart piper-can.service || true
}

echo "==> 1. Installing system packages (python3-pip, python3-venv, can-utils, ethtool, build tools)"
sudo apt update
# build-essential + curl are needed to build gs_usb from source on Jetson (step 5).
sudo apt install -y python3-pip python3.12-venv can-utils ethtool build-essential curl

echo "==> 2. Creating Python virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> 3. Installing SDKs into the venv"
pip install --upgrade pip
pip install python-can
# pyAgxArm is not published on PyPI (pip install pyAgxArm will fail) -
# install straight from GitHub source.
pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"
# piper_sdk (older/more established SDK) is on PyPI - installed alongside
# for cross-checking behavior against pyAgxArm.
pip install piper_sdk

echo "==> 4. Installing camera/vision dependencies (Intel RealSense)"
pip install pyrealsense2 numpy opencv-python-headless

echo "==> 5. Ensuring the gs_usb USB-CAN driver is present (needed on Jetson Thor)"
ensure_gs_usb

echo "==> 6. Pinning a stable interface name + auto bring-up at boot (udev + systemd)"
install_piper_can_service
# For a one-off manual bring-up, or for non-Piper CAN adapters, the lower-level
# helpers scripts/find_all_can_port.sh and scripts/can_activate.sh are still
# available (can_activate.sh expects a single USB-CAN adapter; on a Jetson with
# native mttcan controllers pass a USB bus-address argument - see its usage).

echo "==> 7. Verifying the interface"
PIPER_IF=""
for i in $(ip -br link show type can 2>/dev/null | awk '{print $1}'); do
    if [ "$(ethtool -i "$i" 2>/dev/null | awk '/^driver:/{print $2}')" = gs_usb ]; then
        PIPER_IF="$i"; break
    fi
done
if [ -n "$PIPER_IF" ]; then
    ip -details link show "$PIPER_IF"
    echo "    Piper adapter is '$PIPER_IF' now; stable name '$PIPER_IFACE' applies from the next replug/reboot."
else
    echo "    WARNING: no gs_usb CAN adapter found - is the Piper's USB-CAN adapter plugged in?"
fi

echo
echo "Setup complete. To use the SDK later:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 scripts/check_connection.py $PIPER_IFACE"
