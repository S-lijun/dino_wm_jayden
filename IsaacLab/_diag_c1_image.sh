#!/bin/bash
echo "=== IMAGE DIAG: $1 ==="
id
echo "--- ls / ---"
ls -la / 2>&1 | head -20
echo "--- try /isaac-sim ---"
ls -la /isaac-sim 2>&1 | head -5
stat /isaac-sim 2>&1
echo "--- try python.sh ---"
ls -la /isaac-sim/python.sh 2>&1
/isaac-sim/python.sh -c "print('ok')" 2>&1
echo "=== END ==="
