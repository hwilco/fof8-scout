#!/bin/bash
set -e

# This script is called by postCreateCommand in devcontainer.json
# to finalize the environment setup.

echo "Running post-create setup script..."

# 1. Fix Git ownership and permissions
# This is crucial for volume mounts from Windows/WSL where permissions can be tricky
echo "Fixing Git directory ownership..."
if [ -d ".git" ]; then
    sudo chown -R vscode:vscode .git
fi
git config --global --add safe.directory '*'

# 2. Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA GPU not detected. Ensure the NVIDIA Container Toolkit is installed on your host."
else
    echo "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# 3. Sync dependencies for the experimentation package
echo "Syncing dependencies..."
uv sync --package fof8-ml

# 4. Setup Jupyter notebook stripping
echo "Configuring nbstripout..."
# We set these explicitly to ensure they use 'uv run' and don't fail
git config filter.nbstripout.clean "uv run nbstripout"
git config filter.nbstripout.smudge "cat"
git config filter.nbstripout.required true
git config diff.ipynb.textconv "uv run nbstripout -t"

# 5. Install pre-commit hooks and environments
# We use --overwrite to replace any Windows-formatted hooks with Linux ones
echo "Installing pre-commit hooks and environments..."
uv run pre-commit install --overwrite
uv run pre-commit install-hooks

echo "Setup complete!"
