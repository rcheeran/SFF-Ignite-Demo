#!/bin/bash
set -e

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$PROJECT_DIR/piper-openpi-V2/tools/can_up.sh" "$@"
