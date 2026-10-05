#!/usr/bin/env bash
# Stop every Modal app created by this project (names start with "infertutor-").
# Run this whenever you are done for the day; each app keeps min_containers GPUs
# warm and billing until stopped.
set -euo pipefail
apps=$(modal app list --json 2>/dev/null | python3 -c '
import json, sys
for a in json.load(sys.stdin):
    name = a.get("Name") or a.get("name") or ""
    state = (a.get("State") or a.get("state") or "").lower()
    if name.startswith("infertutor-") and state in ("deployed", "running", "ephemeral", ""):
        print(name)
')
if [ -z "$apps" ]; then echo "No running infertutor-* apps."; exit 0; fi
for app in $apps; do
  echo "Stopping $app"
  modal app stop "$app" || true
done
modal app list
