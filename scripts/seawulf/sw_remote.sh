#!/bin/bash
# Runs inside WSL. Keeps ONE long-lived ssh session to SeaWulf and feeds it commands: SeaWulf asks for
# Duo on every new session (even over a shared connection), so this costs one approval per session start.
#
#   sw_remote.sh start            open the session (one Duo push; waits up to 120 s for approval)
#   sw_remote.sh run FILE [SECS]  run script FILE (a local path) in the session; print its output; exit with its status
#   sw_remote.sh status | stop
#
# Commands are appended to ~/sw/queue, which `tail -F` streams into the remote shell; output is appended
# to ~/sw/out.log and cut out between per-command markers. Each script is written to a remote temp file
# and run as its own `bash` with stdin from /dev/null, so `exit`, `cd` or `set -e` in a script can't end
# or change the session, and nothing in it can read the queue.
D=$HOME/sw
mkdir -p "$D"
alive() { [ -f "$D/pid" ] && kill -0 "$(cat "$D/pid")" 2>/dev/null; }

wait_for() {  # wait_for MARKER SECS -> 0 found, 1 timed out, 2 session gone
  local end=$((SECONDS + $2))
  while [ $SECONDS -lt $end ]; do
    grep -q -- "$1" "$D/out.log" 2>/dev/null && return 0
    alive || return 2
    sleep 1
  done
  return 1
}

case "$1" in
  start)
    if alive; then echo "session already open"; exit 0; fi
    : > "$D/queue"; : > "$D/out.log"
    setsid nohup bash -c "tail -n0 -F '$D/queue' 2>/dev/null | ssh -T -o BatchMode=yes -o ServerAliveInterval=60 seawulf 'bash -l -s' >> '$D/out.log' 2>&1" \
      < /dev/null > /dev/null 2>&1 &
    echo $! > "$D/pid"
    sleep 1
    echo 'echo __READY__' >> "$D/queue"
    echo "waiting for Duo approval (up to 120 s)..."
    if wait_for __READY__ 120; then echo "session open"; exit 0; fi
    echo "session did not open (Duo not approved in time?). Log:"; tail -5 "$D/out.log"
    "$0" stop > /dev/null; exit 1
    ;;
  run)
    alive || { echo "no session: run 'sw.sh start' first" >&2; exit 3; }
    [ -f "$2" ] || { echo "no script file $2" >&2; exit 2; }
    id=$(date +%s%N)
    {
      echo "echo __BEGIN_${id}__"
      echo "f=\$(mktemp)"
      echo "cat > \"\$f\" <<'__EOF_${id}__'"
      cat "$2"
      echo "__EOF_${id}__"
      echo "bash \"\$f\" < /dev/null 2>&1; rc=\$?; rm -f \"\$f\""
      echo "echo __END_${id}__ \$rc"
    } >> "$D/queue"
    wait_for "__END_${id}__" "${3:-600}"; w=$?
    awk -v b="__BEGIN_${id}__" -v e="__END_${id}__" '$0==b{on=1;next} index($0,e)==1{on=0} on' "$D/out.log"
    [ $w = 1 ] && { echo "sw: still running after ${3:-600}s (it keeps running remotely)" >&2; exit 124; }
    [ $w = 2 ] && { echo "sw: session closed" >&2; exit 3; }
    rc=$(grep -- "__END_${id}__" "$D/out.log" | awk '{print $2}')
    exit "${rc:-1}"
    ;;
  status) alive && echo "session open" || echo "no session" ;;
  stop)
    if alive; then pkill -P "$(cat "$D/pid")"; kill "$(cat "$D/pid")" 2>/dev/null; fi
    rm -f "$D/pid"; echo "session closed"
    ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
