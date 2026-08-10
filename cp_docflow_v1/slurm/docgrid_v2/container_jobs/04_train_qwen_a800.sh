#!/usr/bin/env bash
# Portal resources: 1x A800, 8 CPU, 192G host RAM, 48 hours. Requires Gate 3 and Qwen receipt.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_stage_a800.sh" stage4_qwen
