#!/usr/bin/env zsh
# Rig a character without loading Kimodo or Python.
set -euo pipefail

tool_root=${0:A:h:h}
if [[ ! -x "$tool_root/out/bin/kimodo-autorig" ]]; then
  "$tool_root/scripts/build-autorig.zsh"
fi
exec "$tool_root/out/bin/kimodo-autorig" "$@"
