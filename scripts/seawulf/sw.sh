#!/usr/bin/env bash
# Windows (Git Bash) side of sw_remote.sh: run commands on SeaWulf through one long-lived session.
#
#   sw.sh start                        open the session (approve ONE Duo push)
#   sw.sh [SECS] <<'X' ... X           run the script on stdin there (default wait 600 s)
#   sw.sh status | stop
#
# Never retries: a script must not run twice (e.g. sbatch).
dir=$(cd "$(dirname "$0")" && pwd)
to_wsl() { MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 --exec wslpath -u "$(cygpath -w "$1")" < /dev/null | tr -d '\0\r'; }
remote=$(to_wsl "$dir/sw_remote.sh")
wsl() { MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 --exec bash "$remote" "$@" < /dev/null | tr -d '\0'; return "${PIPESTATUS[0]}"; }
case "${1:-}" in
  start|status|stop) wsl "$1"; exit $? ;;
esac
secs=${1:-600}
f="$dir/sw_script_$$.sh"
cat > "$f"
sed -i 's/\r$//' "$f"
wsl run "$(to_wsl "$f")" "$secs"
rc=$?
rm -f "$f"
exit $rc
