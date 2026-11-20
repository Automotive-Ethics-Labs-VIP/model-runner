#!/usr/bin/env bash
# Run the script given on stdin on SeaWulf over the shared WSL ssh connection.
# Usage: sw.sh [timeout_seconds] <<'X' ... X
# Sessions on the shared connection sometimes fail before the script starts (~60 s, exit 1, no output).
# The script prints a start marker first; only attempts where the marker never arrived are retried,
# so a script that started is never run twice.
t=${1:-300}
dir=$(cd "$(dirname "$0")" && pwd)
f="$dir/sw_script_$$.sh"
out="$dir/sw_out_$$.txt"
{ echo 'echo __SW_START__'; cat; } > "$f"
to_wsl() { MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 --exec wslpath -u "$(cygpath -w "$1")" < /dev/null | tr -d '\0\r'; }
remote=$(to_wsl "$dir/sw_remote.sh"); script=$(to_wsl "$f")
for attempt in 1 2 3; do
  MSYS_NO_PATHCONV=1 timeout "$t" wsl.exe -d Ubuntu-24.04 --exec bash "$remote" "$script" < /dev/null > "$out" 2>&1
  rc=$?
  if grep -q __SW_START__ "$out"; then break; fi
  echo "sw.sh: attempt $attempt did not start (rc=$rc); retrying" >&2
done
tr -d '\0' < "$out" | grep -v __SW_START__
rm -f "$f" "$out"
[ $rc = 124 ] && echo "sw.sh: timed out after ${t}s" >&2
exit $rc
