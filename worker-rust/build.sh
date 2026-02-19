#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Building gaia-worker (release)..."
cargo build --release
echo "Done! Binary: target/release/gaia-worker"
ls -lh target/release/gaia-worker
