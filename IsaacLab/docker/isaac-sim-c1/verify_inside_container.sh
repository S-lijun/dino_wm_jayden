#!/usr/bin/env bash
# Quick permission smoke test inside C1 custom image (run after ./run_c1_isaac.sh).
set -euo pipefail
echo "=== C1 Isaac Sim permission smoke test ==="
id
ls -la /isaac-sim/python.sh
ls -la /isaac-sim/kit/python/bin/python3
/isaac-sim/python.sh -c "print('python.sh ok')"
echo "=== PASSED ==="
