#!/bin/bash
# ACP launcher for agentknit on this machine: uses the venv python that has
# agentknit installed, with the deepseek-v4-flash face by default.
export AGENTKNIT_ACP_MODEL="${AGENTKNIT_ACP_MODEL:-deepseek-v4-flash}"
export AGENTKNIT_ACP_ENDPOINT="${AGENTKNIT_ACP_ENDPOINT:-https://api.deepseek.com/v1}"
export AGENTKNIT_ACP_KEY_NAME="${AGENTKNIT_ACP_KEY_NAME:-deepseek_api_key}"
exec "$HOME/.venvs/agentknit/bin/python" "$HOME/workspace/prototypes/harness-remote/agentknit/agentknit-acp.py" "$@"
