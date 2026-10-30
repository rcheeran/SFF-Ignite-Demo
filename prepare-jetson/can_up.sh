#!/bin/bash
# Bring up the Piper USB-CAN adapter at 1 Mbps and show the bus state.
# The down/up also resets a bus left in ERROR-PASSIVE / BUS-OFF by a crash.
set -e
sudo /usr/local/sbin/piper-can-up.sh "${1:-1000000}"
ip -br link show can_piper
state=$(ip -details link show can_piper | sed -n 's/.*can state \([A-Z-]*\).*/\1/p')
echo "can_piper bus state: $state  (ERROR-ACTIVE = healthy; ERROR-PASSIVE after traffic = the arm is not answering)"
