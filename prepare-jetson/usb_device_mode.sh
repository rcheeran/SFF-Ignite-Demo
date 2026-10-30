#!/bin/bash
#
# usb_device_mode.sh — Force the USB-C 5A port back into device (peripheral) mode.
#
# This is the reverse of usb_host_mode.sh. Use it when you need the port to act
# as a USB device rather than a host — e.g. to put the board into recovery /
# flashing mode. Run manually on demand; this is intentionally NOT a boot
# service (the usb-host-mode.service will restore host mode on the next reboot).

set -euo pipefail

ROLE_SWITCH="/sys/class/usb_role/usb2-0-role-switch/role"
DESIRED_ROLE="device"

if [ ! -e "$ROLE_SWITCH" ]; then
    echo "Error: USB role switch not found at $ROLE_SWITCH" >&2
    exit 1
fi

current="$(cat "$ROLE_SWITCH")"
if [ "$current" = "$DESIRED_ROLE" ]; then
    echo "USB role switch already set to '$DESIRED_ROLE'; nothing to do."
    exit 0
fi

echo "Switching USB role from '$current' to '$DESIRED_ROLE'..."
echo "$DESIRED_ROLE" > "$ROLE_SWITCH"

new="$(cat "$ROLE_SWITCH")"
if [ "$new" != "$DESIRED_ROLE" ]; then
    echo "Error: failed to set role (still '$new')" >&2
    exit 1
fi

echo "USB role switch is now '$new'."
