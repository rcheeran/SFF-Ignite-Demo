#!/bin/bash
# Sets up an Ubuntu machine to control the AgileX Piper arm: system deps,
# a Python venv with pyAgxArm (+ piper_sdk for cross-checking), and the
# single-adapter CAN interface. Tested on Ubuntu 24.04.4 / Python 3.12.
#
# Usage: bash setup.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venv}"
CAN_IFACE="${CAN_IFACE:-can0}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 1. Installing system packages (python3-pip, python3-venv, can-utils, ethtool)"
sudo apt update
sudo apt install -y python3-pip python3.12-venv can-utils ethtool

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

echo "==> 4. Activating the CAN interface"
echo "    Detected CAN adapters:"
bash "$SCRIPT_DIR/scripts/find_all_can_port.sh"
echo "    Bringing up a single CAN adapter as $CAN_IFACE @ ${CAN_BITRATE}bps"
echo "    (if you have more than one adapter plugged in, run"
echo "     scripts/can_activate.sh manually with a USB address argument instead)"
bash "$SCRIPT_DIR/scripts/can_activate.sh" "$CAN_IFACE" "$CAN_BITRATE"

echo "==> 5. Verifying the interface"
ip -details link show "$CAN_IFACE"

echo
echo "Setup complete. To use the SDK later:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 scripts/check_connection.py"
