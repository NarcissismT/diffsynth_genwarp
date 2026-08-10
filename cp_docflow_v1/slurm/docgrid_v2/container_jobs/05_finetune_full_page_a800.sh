#!/usr/bin/env bash
# Portal resources: 1x A800, 8 CPU, 192G host RAM, 48 hours. Requires full-page audit and Gate 4.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_stage_a800.sh" stage5_full_page
