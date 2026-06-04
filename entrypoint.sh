#!/bin/sh
# Entrypoint for the CPU image: run the pipeline, then make outputs host-readable.
set -e

/opt/venv/bin/python -m pipeline_v2.run "$@"
status=$?

# Best-effort: make any --out directory readable by the host user.
prev=""
for arg in "$@"; do
    if [ "$prev" = "--out" ]; then
        chmod -R a+rX "$arg" 2>/dev/null || true
    fi
    prev="$arg"
done

exit $status
