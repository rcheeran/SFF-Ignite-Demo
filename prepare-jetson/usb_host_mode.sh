#!/bin/bash
#
# usb_host_mode.sh — Force the USB-C 5A port into host mode.
#
# The Type-C port exposes a USB role switch in sysfs. By default it comes up
# as "device" (peripheral); writing "host" makes the port act as a USB host
# so downstream devices (hubs, cameras, the arm's USB-CAN adapter, etc.)
# enumerate correctly. Run at boot via usb-host-mode.service.

set -euo pipefail

ROLE_SWITCH="/sys/class/usb_role/usb2-0-role-switch/role"
DESIRED_ROLE="host"

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
