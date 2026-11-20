#!/bin/bash
# Runs inside WSL. $1: a script file (Linux path). SeaWulf saves it to a temp file, then runs it in a
# login shell with stdin closed (a login shell reading its script from stdin does not work there).
exec ssh -o BatchMode=yes -o ConnectTimeout=15 seawulf 'f=$(mktemp); cat > "$f"; bash -l "$f" < /dev/null; rc=$?; rm -f "$f"; exit $rc' < "$1"
