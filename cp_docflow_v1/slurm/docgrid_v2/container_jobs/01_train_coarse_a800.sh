#!/usr/bin/env bash
# Portal resources: 1x A800, 8 CPU, 64G host RAM, 48 hours.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_stage_a800.sh" stage1_coarse
