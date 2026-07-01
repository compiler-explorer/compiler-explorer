#!/bin/bash
# Tenstorrent TT-Metal compiler detection for Compiler Explorer

COMPILER_PATH="${1:-/opt/tenstorrent/bin/tt-metal-cc}"

if [ ! -f "$COMPILER_PATH" ]; then
    echo "TT-Metal compiler not found at $COMPILER_PATH"
    exit 1
fi

echo "Detecting Tenstorrent TT-Metal..."
VERSION=$("$COMPILER_PATH" --version 2>/dev/null || echo "unknown")
echo "  Version: $VERSION"
