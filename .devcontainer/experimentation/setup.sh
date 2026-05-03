#!/bin/bash
set -e

# This script is called by postCreateCommand in devcontainer.json
# to finalize the environment setup.

echo "Running post-create setup script..."

# Load optional repo-level env for local runs and postCreate usage.
# This allows identity/signing values to be declared once in .devcontainer/.env.
if [ -f ".devcontainer/.env" ]; then
    set -a
    source .devcontainer/.env
    set +a
fi

windows_personal_config=""
if [ -n "${HOST_USER:-}" ]; then
    candidate="/mnt/c/Users/${HOST_USER}/.gitconfig-personal"
    if [ -f "$candidate" ]; then
        windows_personal_config="$candidate"
    fi
fi

# 1. Fix Git ownership and permissions
# This is crucial for volume mounts from Windows/WSL where permissions can be tricky
echo "Fixing Git directory ownership..."
if [ -d ".git" ]; then
    sudo chown -R vscode:vscode .git
fi

# Configure repo-local identity without touching global git config.
git_name="${GIT_USER_NAME:-}"
git_email="${GIT_USER_EMAIL:-}"
if [ -z "$git_name" ]; then
    git_name="$(git config --global user.name 2>/dev/null || true)"
fi
if [ -z "$git_email" ]; then
    git_email="$(git config --global user.email 2>/dev/null || true)"
fi
if [ -z "$git_name" ] && [ -f "/home/vscode/.gitconfig-personal" ]; then
    git_name="$(git config --file /home/vscode/.gitconfig-personal user.name 2>/dev/null || true)"
fi
if [ -z "$git_email" ] && [ -f "/home/vscode/.gitconfig-personal" ]; then
    git_email="$(git config --file /home/vscode/.gitconfig-personal user.email 2>/dev/null || true)"
fi
if [ -z "$git_name" ] && [ -n "$windows_personal_config" ]; then
    git_name="$(git config --file "$windows_personal_config" user.name 2>/dev/null || true)"
fi
if [ -z "$git_email" ] && [ -n "$windows_personal_config" ]; then
    git_email="$(git config --file "$windows_personal_config" user.email 2>/dev/null || true)"
fi
if [ -z "$git_name" ] && [ -f "/home/vscode/.gitconfig" ]; then
    git_name="$(git config --file /home/vscode/.gitconfig user.name 2>/dev/null || true)"
fi
if [ -z "$git_email" ] && [ -f "/home/vscode/.gitconfig" ]; then
    git_email="$(git config --file /home/vscode/.gitconfig user.email 2>/dev/null || true)"
fi
if [ -n "$git_name" ] && [ -n "$git_email" ]; then
    git config --local user.name "$git_name"
    git config --local user.email "$git_email"
else
    echo "WARNING: Git user.name/user.email not found. Set .devcontainer/.env with:"
    echo "  GIT_USER_NAME=Your Name"
    echo "  GIT_USER_EMAIL=you@example.com"
fi

# 2. Configure SSH signing for commits in the container.
ssh_signing_key="${SSH_SIGNING_KEY:-}"
ssh_signing_pubkey_path="${SSH_SIGNING_PUBKEY_PATH:-}"
if [ -z "$ssh_signing_pubkey_path" ]; then
    for candidate in /home/vscode/.ssh-host/*.pub /home/vscode/.ssh/*.pub; do
        if [ -f "$candidate" ]; then
            ssh_signing_pubkey_path="$candidate"
            break
        fi
    done
fi

git config --local --unset-all gpg.program || true
git config --local --unset-all gpg.ssh.defaultKeyCommand || true

if [ -n "$ssh_signing_key" ]; then
    git config --local gpg.format ssh
    git config --local user.signingkey "$ssh_signing_key"
    git config --local commit.gpgsign true
elif [ -n "$ssh_signing_pubkey_path" ]; then
    git config --local gpg.format ssh
    git config --local user.signingkey "$ssh_signing_pubkey_path"
    git config --local commit.gpgsign true
elif ssh-add -L >/dev/null 2>&1; then
    git config --local gpg.format ssh
    git config --local --unset-all user.signingkey || true
    git config --local gpg.ssh.defaultKeyCommand "ssh-add -L"
    git config --local commit.gpgsign true
else
    git config --local --unset-all user.signingkey || true
    git config --local --unset-all gpg.format || true
    git config --local commit.gpgsign false
    echo "WARNING: SSH signing disabled (no signing key configured)."
    echo "Set one of these in .devcontainer/.env:"
    echo "  SSH_SIGNING_KEY='key::ssh-ed25519 AAAA... comment'"
    echo "  SSH_SIGNING_PUBKEY_PATH=/home/vscode/.ssh-host/id_ed25519.pub"
    echo "Or make a public key available under /home/vscode/.ssh-host and ensure the matching private key is available via ssh-agent."
fi

# 3. Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA GPU not detected. Ensure the NVIDIA Container Toolkit is installed on your host."
else
    echo "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# 4. Sync dependencies for the experimentation package
echo "Syncing dependencies..."
# Ensure the external venv path is writable by vscode even if mount
# ownership changed at container runtime.
sudo mkdir -p /workspaces/.venv
sudo chown -R vscode:vscode /workspaces/.venv
uv sync --package fof8-ml --group notebook --group viz

# 5. Setup Jupyter notebook stripping
echo "Configuring nbstripout..."
# We set these explicitly to ensure they use 'uv run' and don't fail
git config filter.nbstripout.clean "uv run nbstripout"
git config filter.nbstripout.smudge "cat"
git config filter.nbstripout.required true
git config diff.ipynb.textconv "uv run nbstripout -t"

# 6. Install pre-commit hooks and environments
# We use --overwrite to replace any Windows-formatted hooks with Linux ones
echo "Installing pre-commit hooks and environments..."
uv run pre-commit install --overwrite
uv run pre-commit install-hooks

echo "Setup complete!"
