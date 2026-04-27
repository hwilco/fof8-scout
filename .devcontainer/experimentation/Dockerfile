FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Install uv by copying the pre-built binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv defaults to hardlinks, which fail across filesystem boundaries
# (Windows NTFS host → Linux ext4 container). Copy mode is required.
ENV UV_LINK_MODE=copy

# Place the .venv outside the bind-mounted project directory so it is
# never synced back to the Windows host filesystem (avoids a large,
# slow directory appearing in the repo on the host).
# Set environment variables so VS Code and Jupyter automatically use the virtual environment
ENV UV_PROJECT_ENVIRONMENT="/workspaces/.venv"
ENV VIRTUAL_ENV="/workspaces/.venv"
ENV PATH="/workspaces/.venv/bin:$PATH"

# Pre-create the directory and give the vscode user ownership so that
# postCreateCommand (which runs as vscode) can write to it without permission errors.
RUN mkdir -p /workspaces/.venv && chown -R vscode:vscode /workspaces/.venv
