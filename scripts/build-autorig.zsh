#!/usr/bin/env zsh
# Build and publish the native autorig using a fresh executable inode.
set -euo pipefail

tool_root=${0:A:h:h}
export CARGO_HOME="$tool_root/out/cargo"
export CARGO_TARGET_DIR="$tool_root/out/native-build"
mkdir -p "$tool_root/out/bin"
cargo build --locked --release --manifest-path "$tool_root/native/Cargo.toml"
cp "$CARGO_TARGET_DIR/release/kimodo-autorig" "$tool_root/out/bin/kimodo-autorig.new"
mv "$tool_root/out/bin/kimodo-autorig.new" "$tool_root/out/bin/kimodo-autorig"
