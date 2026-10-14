#!/usr/bin/env bash
# Runs `<ai-eda> selftest --dir "$RUNNER_TEMP/selftest"` for the install workflow; when it has not finished after
# SELFTEST_WATCHDOG_S seconds (default 600 - a selftest takes 1 to 3 minutes on the CI machines), lists the running
# processes (which tool hangs) and stops it. GitHub keeps no log of a job cancelled while a step hangs.
set -u
limit="${SELFTEST_WATCHDOG_S:-600}"
"$1" selftest --dir "$RUNNER_TEMP/selftest" &
pid=$!
waited=0
while kill -0 "$pid" 2>/dev/null; do
  if [ "$waited" -ge "$limit" ]; then
    echo "selftest still running after $limit s - the processes:"
    ps -ef 2>/dev/null || ls -l /proc/[0-9]*/exe 2>/dev/null
    kill "$pid" 2>/dev/null
    break
  fi
  sleep 5
  waited=$((waited + 5))
done
wait "$pid"
