#!/bin/bash
# Set UV cache directory to avoid disk quota issues
# Run this before using uv commands

export BASE_DIR="/research/d7/fyp25/yyyu2"

# UV uses UV_HOME for Python installations and UV_CACHE_DIR for package cache
export UV_HOME="$BASE_DIR/.cache/uv/home"
export UV_CACHE_DIR="$BASE_DIR/.cache/uv/cache"

# Create cache directories
mkdir -p "$UV_HOME"
mkdir -p "$UV_CACHE_DIR"

echo "UV_HOME set to: $UV_HOME"
echo "UV_CACHE_DIR set to: $UV_CACHE_DIR"

# Verify
echo ""
echo "Current uv cache location:"
uv cache dir 2>/dev/null || echo "Run 'uv cache dir' after sourcing this script"

# Add to your shell profile if you want it permanent
echo ""
echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
echo "export UV_HOME=\"$UV_HOME\""
echo "export UV_CACHE_DIR=\"$UV_CACHE_DIR\""

