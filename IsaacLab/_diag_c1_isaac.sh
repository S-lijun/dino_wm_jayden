#!/bin/bash
# One-shot diagnostic: Isaac Sim 5.1.0 permissions on Compute1
set -x
echo "=== HOST ==="
hostname
id
echo "=== /isaac-sim top ==="
ls -la /isaac-sim 2>&1 | head -20
echo "=== python.sh ==="
ls -la /isaac-sim/python.sh 2>&1
echo "=== kit python dir ==="
ls -la /isaac-sim/kit/python/bin/ 2>&1 | head -10
echo "=== passwd isaac-sim user ==="
getent passwd isaac-sim 2>&1 || true
getent passwd 1234 2>&1 || true
echo "=== try execute ==="
/isaac-sim/python.sh -c "print('python.sh ok')" 2>&1
/isaac-sim/kit/python/bin/python3 -c "print('kit python ok')" 2>&1
echo "=== done ==="
